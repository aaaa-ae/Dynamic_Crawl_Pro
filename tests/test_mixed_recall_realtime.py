# -*- coding: utf-8 -*-
"""
Mixed recall v3 (cold-start recall stage):

Sources:
  - Wikipedia external links (clean extraction: external-links -> references -> a.external)
  - GDELT realtime articles (staged timespans + cache)
  - RSS feeds (keyword-filtered; includes simple CN->EN expansions)
  - NEW v3: promote entrance candidates from evidence URLs
  - NEW v3: if promoted entrances are too few (typical for root-level slugs),
            fetch a SMALL sample of evidence pages and extract:
              * RSS/Atom links (<link rel="alternate" type="application/rss+xml">)
              * tag/topic/category/archive entrances from on-page links

Output:
  output/mixed_recall_realtime_test.json

This script is intentionally recall-heavy (no strict filtering); your second script
will do seed selection / stream split (event/anchor/evidence).
"""

import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set
from urllib.parse import urlparse, urlunparse, urljoin

import requests
from bs4 import BeautifulSoup

USER_AGENT = "DynamicCrawlPro/1.2 (mixed-recall realtime test v3)"
DEFAULT_TIMEOUT = 12

# -----------------------------
# RSS sources (examples; can extend)
# -----------------------------
RSS_SOURCES = [
    "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    "https://www.jpl.nasa.gov/feeds/news/",
    "https://www.esa.int/rssfeed/Our_Activities/Space_Science",
    "https://www.science.org/rss/news_current.xml",
    "https://www.nature.com/nature.rss",
    "https://spectrum.ieee.org/rss/fulltext",
]

# Keyword expansions for RSS matching (extend gradually)
RSS_KEYWORD_TRANSLATIONS = {
    "马克思主义": ["marxism", "marxist", "karl marx"],
    "算法": ["algorithm", "algorithms"],
    "机器学习": ["machine learning", "ml"],
}

# Hints for “entrance-like” paths
ENTRANCE_HINTS = (
    "news", "press", "pressroom", "releases", "release", "updates", "update",
    "announcements", "announcement", "notices", "notice", "bulletin", "blog",
    "topic", "topics", "tag", "tags", "category", "categories", "archive", "archives"
)

# Wikipedia / Wikimedia family domains to block when extracting external links
WIKI_BLOCKED_SUFFIXES = [
    "wikipedia.org",
    "wikimedia.org",
    "wikidata.org",
    "mediawiki.org",
    "wiktionary.org",
    "wikisource.org",
    "wikibooks.org",
    "wikiquote.org",
    "wikinews.org",
    "wikivoyage.org",
]

# Promote & scrape controls (keep small to avoid heavy crawling)
PROMOTE_MAX_PARENTS_PER_URL = 2
PROMOTE_MIN_ENTRANCES_PER_KEYWORD = 4  # if below this, we try HTML-based entrance discovery
HTML_DISCOVERY_MAX_PAGES = 6           # fetch at most N evidence pages per keyword
HTML_DISCOVERY_MAX_BYTES = 250_000     # stop after ~250KB
HTML_DISCOVERY_TIMEOUT = 10


# -----------------------------
# Utilities
# -----------------------------
def safe_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff\-\.]+", "_", s)
    return s[:120] if len(s) > 120 else s


def normalize_url(url: str) -> str:
    """Normalize URL for better dedup (strip fragment, trim trailing slash)."""
    try:
        p = urlparse(url)
        p = p._replace(fragment="")
        norm = urlunparse(p)
        if norm.endswith("/") and len(p.path) > 1:
            norm = norm[:-1]
        return norm
    except Exception:
        return url


def merge_dedup(*lists: List[str]) -> List[str]:
    seen: Set[str] = set()
    merged: List[str] = []
    for lst in lists:
        for u in lst:
            u = normalize_url(u)
            if not u or u in seen:
                continue
            seen.add(u)
            merged.append(u)
    return merged


def is_external_candidate(url: str) -> bool:
    if not url or not url.startswith("http"):
        return False
    domain = urlparse(url).netloc.lower()
    if not domain:
        return False
    if any(domain.endswith(b) for b in WIKI_BLOCKED_SUFFIXES):
        return False
    # Drop heavy binary assets as wikipedia externals
    if re.search(r"\.(pdf|jpg|jpeg|png|gif|svg|mp4|mp3|zip|rar)$", url, re.IGNORECASE):
        return False
    return True


# -----------------------------
# Ranking (coarse, recall-stage)
# -----------------------------
def rank_candidates(urls: List[str], url_sources: Dict[str, List[str]] | None = None) -> List[Dict]:
    """
    Coarse ranking to help debugging; not a strict filter.
    Important: Do NOT treat .org as default high trust.
    """
    def score(url: str) -> int:
        p = urlparse(url)
        domain = (p.netloc or "").lower()
        path = (p.path or "").lower()
        s = 0

        # Mild trust signals
        if domain.endswith((".edu", ".ac.uk", ".ac.cn", ".edu.cn", ".gov", ".gov.cn")):
            s += 10

        # Entrance-like path bonus
        if any(f"/{h}/" in path or path.endswith(f"/{h}") for h in ENTRANCE_HINTS):
            s += 5

        # News-like paths (small)
        if "/news" in path or "/press" in path or "/releases" in path or "/updates" in path:
            s += 2

        # Date patterns suggest detail page; small score (keep it but not dominant)
        if re.search(r"/20\d{2}/", url):
            s += 1

        return s

    ranked = []
    for u in urls:
        item = {"url": u, "score": score(u)}
        if url_sources is not None:
            item["sources"] = sorted(set(url_sources.get(u, [])))
        ranked.append(item)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


# -----------------------------
# Keyword matching (RSS)
# -----------------------------
def keyword_match(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False
    text_l = text.lower()
    keyword_l = keyword.lower()
    has_cn = any("\u4e00" <= ch <= "\u9fff" for ch in keyword)
    if has_cn:
        return keyword in text
    pattern = r"\b" + re.escape(keyword_l) + r"\b"
    return re.search(pattern, text_l) is not None


def build_rss_keywords(keyword: str) -> List[str]:
    if keyword in RSS_KEYWORD_TRANSLATIONS:
        return RSS_KEYWORD_TRANSLATIONS[keyword]
    return [keyword]


# -----------------------------
# Wikipedia (clean external link extraction)
# -----------------------------
def search_wikipedia(session: requests.Session, keyword: str, language: str = "zh", limit: int = 1):
    url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": keyword,
        "format": "json",
        "srlimit": limit,
        "srprop": "",
    }
    resp = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("query", {}).get("search", [])


def get_wikipedia_page_url(title: str, language: str = "zh") -> str:
    title_escaped = title.replace(" ", "_")
    return f"https://{language}.wikipedia.org/wiki/{title_escaped}"


def extract_external_links_from_wikipedia_html(html: str) -> List[str]:
    """
    Prefer "External links/外部链接" section.
    Fallback: references external links.
    Fallback: a.external.
    NO fallback to all <a>.
    """
    soup = BeautifulSoup(html, "lxml")
    links: List[str] = []

    # 1) External links section
    external_section = None
    headline_targets = {"external links", "外部链接"}
    for header in soup.find_all(["h2", "h3"]):
        span = header.find("span", {"class": "mw-headline"})
        if not span:
            continue
        if span.get_text(strip=True).lower() in headline_targets:
            external_section = header
            break

    if external_section:
        nxt = external_section.find_next_sibling()
        while nxt and nxt.name in {"p", "ul", "ol", "div"}:
            for a in nxt.find_all("a", href=True):
                links.append(a["href"])
            nxt = nxt.find_next_sibling()

    # 2) References external links
    if not links:
        for a in soup.select("ol.references a.external, .reference a.external"):
            href = a.get("href")
            if href:
                links.append(href)

    # 3) External links anywhere
    if not links:
        for a in soup.select("a.external"):
            href = a.get("href")
            if href:
                links.append(href)

    seen = set()
    candidates = []
    for href in links:
        href = href.strip()
        if not is_external_candidate(href):
            continue
        href = normalize_url(href)
        if href in seen:
            continue
        seen.add(href)
        candidates.append(href)

    return candidates


# -----------------------------
# GDELT (staged, with cache)
# -----------------------------
def _parse_retry_after_seconds(resp: requests.Response):
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    ra = ra.strip()
    if ra.isdigit():
        return float(ra)
    return None


def _sleep_with_jitter(base_seconds: float, jitter_ratio: float = 0.25):
    jitter = base_seconds * jitter_ratio
    sleep_s = max(0.0, base_seconds + random.uniform(-jitter, jitter))
    time.sleep(sleep_s)


def search_gdelt(
    session: requests.Session,
    keyword: str,
    timespan: str = "3m",
    maxrecords: int = 20,
    retries: int = 4,
    base_backoff: float = 2.0,
    max_backoff: float = 30.0,
) -> List[str]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": keyword,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": maxrecords,
        "sort": "DateDesc",
        "timespan": timespan,
    }

    backoff = base_backoff
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        except requests.RequestException:
            wait_s = min(max_backoff, backoff)
            _sleep_with_jitter(wait_s)
            backoff = min(max_backoff, backoff * 2)
            continue

        if resp.status_code == 200:
            try:
                data = resp.json()
            except ValueError:
                return []
            urls = []
            for item in data.get("articles", []) or []:
                u = item.get("url")
                if u:
                    urls.append(normalize_url(u))
            return urls

        if resp.status_code == 429:
            ra = _parse_retry_after_seconds(resp)
            wait_s = min(max_backoff, ra if ra is not None else backoff)
            _sleep_with_jitter(wait_s)
            backoff = min(max_backoff, backoff * 2)
            continue

        if 500 <= resp.status_code < 600:
            wait_s = min(max_backoff, backoff)
            _sleep_with_jitter(wait_s)
            backoff = min(max_backoff, backoff * 2)
            continue

        break

    return []


def search_gdelt_staged(
    session: requests.Session,
    keyword: str,
    maxrecords: int = 20,
    timespans: Tuple[str, ...] = ("1m", "3m", "6m"),
    cache_dir: Path | None = None,
) -> List[str]:
    all_urls: List[str] = []
    for ts in timespans:
        cache_file = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"gdelt_{safe_filename(keyword)}_{ts}_{maxrecords}.json"

        if cache_file is not None and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                urls = cached.get("urls", [])
            except Exception:
                urls = []
        else:
            urls = search_gdelt(session, keyword, timespan=ts, maxrecords=maxrecords)
            if cache_file is not None:
                try:
                    cache_file.write_text(
                        json.dumps(
                            {"keyword": keyword, "timespan": ts, "maxrecords": maxrecords, "urls": urls},
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                except Exception:
                    pass

        all_urls = merge_dedup(all_urls, urls)
        if len(all_urls) >= maxrecords:
            break

        time.sleep(1.0)

    return all_urls


# -----------------------------
# RSS (keyword-filtered)
# -----------------------------
def fetch_rss_items(session: requests.Session, rss_url: str, max_items: int = 30):
    try:
        resp = session.get(rss_url, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    items = []
    try:
        soup = BeautifulSoup(resp.text, "xml")
        for item in soup.find_all("item")[:max_items]:
            title = item.find("title")
            desc = item.find("description")
            link = item.find("link")
            items.append(
                {
                    "title": title.text.strip() if title and title.text else "",
                    "description": desc.text.strip() if desc and desc.text else "",
                    "link": link.text.strip() if link and link.text else "",
                }
            )
    except Exception:
        return []

    return items


def search_rss(session: requests.Session, keyword: str, max_per_feed: int = 15) -> List[str]:
    all_urls: List[str] = []
    rss_keywords = build_rss_keywords(keyword)
    for feed in RSS_SOURCES:
        items = fetch_rss_items(session, feed, max_items=max_per_feed)
        for it in items:
            text = (it.get("title", "") + " " + it.get("description", "")).strip()
            if not text:
                continue
            matched = any(keyword_match(text, k) for k in rss_keywords)
            if not matched:
                continue
            link = it.get("link", "")
            if link:
                all_urls.append(normalize_url(link))
    return merge_dedup(all_urls)


# -----------------------------
# Promote entrances (string-based)
# -----------------------------
def promote_entrances(urls: List[str], max_parents_per_url: int = 2) -> List[str]:
    """
    Derive likely entrance/listing URLs:
      - root of domain
      - parent path levels
      - /news, /press, /blog near root if segments match
    """
    promoted: List[str] = []
    for u in urls:
        try:
            p = urlparse(u)
            if not p.scheme.startswith("http") or not p.netloc:
                continue

            domain_root = urlunparse((p.scheme, p.netloc, "/", "", "", ""))
            promoted.append(domain_root)

            path = (p.path or "").strip("/")
            if not path:
                continue

            segs = path.split("/")
            # promote parents (a/b/c/d -> a/b/c, a/b)
            count = 0
            for k in range(len(segs) - 1, 0, -1):
                parent_segs = segs[:k]
                parent_path = "/" + "/".join(parent_segs) + "/"
                parent_url = urlunparse((p.scheme, p.netloc, parent_path, "", "", ""))
                promoted.append(parent_url)
                count += 1
                if count >= max_parents_per_url:
                    break

            # if early segments look like entrance hints, add that entrance
            for s in segs[:3]:
                s_low = s.lower()
                if s_low in {"news", "press", "blog", "updates", "releases"}:
                    entrance = urlunparse((p.scheme, p.netloc, f"/{s_low}/", "", "", ""))
                    promoted.append(entrance)

        except Exception:
            continue

    return merge_dedup(promoted)


# -----------------------------
# v3: HTML-based entrance discovery from evidence pages
# -----------------------------
def _fetch_html_limited(session: requests.Session, url: str) -> str:
    """
    Fetch a page but limit bytes to keep it light.
    """
    try:
        resp = session.get(url, timeout=HTML_DISCOVERY_TIMEOUT, stream=True)
        resp.raise_for_status()
        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total >= HTML_DISCOVERY_MAX_BYTES:
                break
        content = b"".join(chunks)
        return content.decode(errors="ignore")
    except Exception:
        return ""


def _is_probable_feed_url(u: str) -> bool:
    u_low = u.lower()
    if not u_low.startswith("http"):
        return False
    if "feedback" in u_low:
        return False
    # typical feed indicators
    return any(x in u_low for x in ("rss", "atom", "feed")) or u_low.endswith((".rss", ".xml"))


def discover_entrances_from_html(session: requests.Session, evidence_urls: List[str]) -> List[str]:
    """
    From a small sample of evidence URLs, extract:
      - RSS/Atom from <link rel="alternate" type="application/rss+xml" ...>
      - candidate entrance links containing tag/topic/category/archive/news/press/blog etc.
    """
    discovered: List[str] = []

    sample = evidence_urls[:HTML_DISCOVERY_MAX_PAGES]
    for u in sample:
        html = _fetch_html_limited(session, u)
        if not html:
            continue

        base = u
        soup = BeautifulSoup(html, "lxml")

        # 1) <link rel="alternate" ...> feeds
        for link in soup.find_all("link"):
            rel = " ".join(link.get("rel", [])).lower() if isinstance(link.get("rel"), list) else (link.get("rel") or "").lower()
            typ = (link.get("type") or "").lower()
            href = link.get("href")
            if not href:
                continue
            abs_u = urljoin(base, href)
            if "alternate" in rel and ("rss" in typ or "atom" in typ or _is_probable_feed_url(abs_u)):
                discovered.append(abs_u)

        # 2) On-page anchors for entrances
        for a in soup.find_all("a", href=True):
            href = a.get("href") or ""
            if not href:
                continue
            abs_u = urljoin(base, href)
            if not abs_u.startswith("http"):
                continue
            p = urlparse(abs_u)
            # keep only same-site or same root-ish? Here we allow cross-site entrances too, but can be noisy.
            # We'll keep it conservative by requiring entrance hints in path.
            path = (p.path or "").lower()
            if any(f"/{h}/" in path or path.endswith(f"/{h}") for h in ENTRANCE_HINTS):
                # avoid obvious share/login links
                if any(x in abs_u.lower() for x in ("login", "signup", "subscribe", "share", "facebook.com", "twitter.com", "t.co")):
                    continue
                discovered.append(abs_u)

    return merge_dedup(discovered)


# -----------------------------
# Main
# -----------------------------
def main():
    # Your test keywords (keep as you like)
    test_keywords = [
        ("马克思主义", "zh"),
        ("算法", "zh"),
        ("machine learning", "en"),
    ]

    results = []
    cache_dir = Path(".cache") / "gdelt"

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})

        for keyword, lang in test_keywords:
            print("=" * 70)
            print(f"Keyword: {keyword} | Language: {lang}")

            url_sources: Dict[str, List[str]] = defaultdict(list)

            # ---- Wikipedia external links
            wiki_urls: List[str] = []
            title = ""
            page_url = ""

            try:
                wiki_results = search_wikipedia(session, keyword, language=lang, limit=1)
            except requests.RequestException as e:
                print(f"  - Wikipedia search error: {e.__class__.__name__}")
                wiki_results = []

            if wiki_results:
                title = wiki_results[0].get("title", "")
                page_url = get_wikipedia_page_url(title, language=lang)
                try:
                    resp = session.get(page_url, timeout=DEFAULT_TIMEOUT)
                    resp.raise_for_status()
                    wiki_urls = extract_external_links_from_wikipedia_html(resp.text)
                except requests.RequestException as e:
                    print(f"  - Wikipedia page fetch error: {e.__class__.__name__}")
                    wiki_urls = []

            for u in wiki_urls:
                url_sources[normalize_url(u)].append("wikipedia_external")

            # ---- GDELT staged
            gdelt_urls = search_gdelt_staged(
                session,
                keyword,
                maxrecords=20,
                timespans=("1m", "3m", "6m"),
                cache_dir=cache_dir,
            )
            for u in gdelt_urls:
                url_sources[normalize_url(u)].append("gdelt")

            # ---- RSS keyword-filtered
            rss_urls = search_rss(session, keyword, max_per_feed=15)
            for u in rss_urls:
                url_sources[normalize_url(u)].append("rss")

            # Evidence-ish urls (mostly detail pages)
            evidence_like = merge_dedup(gdelt_urls, rss_urls)

            # ---- v2: promote entrances (string-only)
            promoted = promote_entrances(evidence_like, max_parents_per_url=PROMOTE_MAX_PARENTS_PER_URL)
            for u in promoted:
                url_sources[normalize_url(u)].append("promoted_entrance")

            # ---- v3: if promoted is too small, do HTML-based entrance discovery
            discovered_html: List[str] = []
            if len(promoted) < PROMOTE_MIN_ENTRANCES_PER_KEYWORD and evidence_like:
                # sample evidence urls (prefer same-domain diversity)
                # take unique by root domain
                by_domain = {}
                for u in evidence_like:
                    d = urlparse(u).netloc.lower()
                    if d and d not in by_domain:
                        by_domain[d] = u
                sample_urls = list(by_domain.values())[:HTML_DISCOVERY_MAX_PAGES]

                discovered_html = discover_entrances_from_html(session, sample_urls)
                for u in discovered_html:
                    url_sources[normalize_url(u)].append("html_discovered_entrance")

            merged = merge_dedup(wiki_urls, gdelt_urls, rss_urls, promoted, discovered_html)
            ranked = rank_candidates(merged, url_sources=url_sources)

            print(f"  - Wikipedia external links: {len(wiki_urls)}")
            print(f"  - GDELT articles (unique): {len(gdelt_urls)}")
            print(f"  - RSS urls (keyword-filtered): {len(rss_urls)}")
            print(f"  - Promoted entrances (string): {len(promoted)}")
            print(f"  - HTML discovered entrances  : {len(discovered_html)}")
            print(f"  - Merged candidates          : {len(merged)}")

            for i, item in enumerate(ranked[:12], 1):
                src = ",".join(item.get("sources", [])[:6])
                print(f"    {i}. [{item['score']}] {item['url']} ({src})")

            results.append(
                {
                    "keyword": keyword,
                    "language": lang,
                    "wikipedia": {
                        "page_title": title,
                        "page_url": page_url,
                        "candidate_count": len(wiki_urls),
                    },
                    "gdelt": {
                        "candidate_count": len(gdelt_urls),
                    },
                    "rss": {
                        "candidate_count": len(rss_urls),
                    },
                    "promoted": {
                        "candidate_count": len(promoted),
                    },
                    "html_discovered": {
                        "candidate_count": len(discovered_html),
                    },
                    "merged_count": len(merged),
                    "ranked_candidates": ranked,
                }
            )

            time.sleep(1.2)

    output_file = Path("output/mixed_recall_realtime_test.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[OK] Results saved to:", output_file)


if __name__ == "__main__":
    main()
