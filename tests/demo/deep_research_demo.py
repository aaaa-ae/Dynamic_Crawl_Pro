"""
Deep Research MVP (no search API) - Quality-Enhanced

What this version fixes (based on your logs/results):
1) MAIN-CONTENT extraction (site-aware selectors) -> title no longer null, snippet from real article
2) Topic-bound expansion: path allowlist per domain -> prevents cp-algorithms "whole-site drift"
3) Link parsing bug: markdown links like (url "title") -> url
4) Retry with exponential backoff for flaky sources (e.g., oi-wiki) -> failures won't break the run
5) Lower link noise: extract outgoing links primarily from the main-content container

Requirements:
  pip install crawl4ai beautifulsoup4 lxml

If bs4/lxml are not available, the script falls back to the crawler's .text/.markdown outputs.
"""

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse



# -----------------------------
# Config
# -----------------------------

@dataclass
class ResearchConfig:
    topic: str
    seed_urls: List[str]

    # budget / control
    max_depth: int = 2
    max_pages: int = 80
    max_pages_per_domain: int = 25
    concurrency: int = 6

    # scope control
    allow_external_domains: bool = True
    allowed_domains: Optional[Set[str]] = None  # if provided, only keep these domains
    blocked_domains: Optional[Set[str]] = None  # drop these domains

    # relevance filter
    min_keyword_hits: int = 1

    # crawl robustness
    max_retries: int = 3
    retry_backoff_base_sec: float = 1.5

    # output
    output_jsonl: str = "deep_research_results.jsonl"


# -----------------------------
# Helpers
# -----------------------------

def norm_url(u: str) -> str:
    u = (u or "").strip()

    # remove fragments
    u = re.sub(r"#.*$", "", u)

    # if url accidentally contains a markdown title part: https://... "RSS"
    u = u.split()[0].strip('"').strip("'")

    # drop trailing slash duplicates (keep root)
    if len(u) > 8 and u.endswith("/"):
        u = u[:-1]
    return u

def get_domain(u: str) -> str:
    return urlparse(u).netloc.lower()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def build_keywords(topic: str) -> List[str]:
    base = (topic or "").lower().strip()
    kws = {base}
    for extra in ["tutorial", "editorial", "proof", "implementation", "complexity"]:
        kws.add(extra)
    for tok in re.split(r"[\s\-_/]+", base):
        if tok:
            kws.add(tok)
    return sorted(kws)

def keyword_score(text: str, keywords: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for kw in keywords if kw and kw in t)

def _clean_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _safe_import_bs4():
    try:
        from bs4 import BeautifulSoup  # type: ignore
        return BeautifulSoup
    except Exception:
        return None

def _get_html_from_result(r) -> str:
    """
    crawl4ai Result object can vary by version/config.
    We try common attributes defensively.
    """
    for attr in ["cleaned_html", "html", "raw_html", "content", "page_content"]:
        v = getattr(r, attr, None)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def extract_links_from_markdown(md: str, base_url: str) -> Set[str]:
    """
    Parse markdown links: [text](url "optional title")
    """
    links: Set[str] = set()
    if not md:
        return links

    for m in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", md):
        href = (m.group(1) or "").strip()
        href = href.split()[0].strip('"').strip("'")  # strip optional title
        if not href:
            continue
        if href.startswith(("mailto:", "javascript:")):
            continue
        abs_url = urljoin(base_url, href)
        abs_url = norm_url(abs_url)
        if abs_url.startswith(("http://", "https://")):
            links.add(abs_url)
    return links

# -----------------------------
# Site-aware extraction
# -----------------------------

SITE_MAIN_SELECTORS: Dict[str, List[str]] = {
    # cp-algorithms pages are typically in <article> (works well in most cases)
    "cp-algorithms.com": ["article", "main article", "div#content", "div.main-content"],
    # GfG uses <article> but sometimes has viewer content containers
    "www.geeksforgeeks.org": ["article", "div.article--viewer_content", "div.text"],
    # OI-Wiki commonly uses article / md-content
    "oi-wiki.org": ["article", "div.md-content", "div#content"],
}

def extract_main_text_and_links(url: str, domain: str, html: str) -> Tuple[Optional[str], str, Set[str]]:
    """
    Returns (title, main_text, outgoing_links) extracted from main content.
    Falls back to whole-page extraction if selectors fail.
    """
    BeautifulSoup = _safe_import_bs4()
    if not BeautifulSoup or not html:
        return None, "", set()

    soup = BeautifulSoup(html, "lxml")

    # remove obvious noise blocks
    for tag in soup.select("nav, header, footer, aside, form, script, style"):
        tag.decompose()

    # pick main container
    selectors = SITE_MAIN_SELECTORS.get(domain, ["article", "main", "body"])
    container = None
    for sel in selectors:
        found = soup.select_one(sel)
        if found:
            container = found
            break
    if container is None:
        container = soup.body or soup

    # title: prefer h1 within container, then document title
    title = None
    h1 = container.select_one("h1")
    if h1:
        title = _clean_text(h1.get_text(" "))
    if not title and soup.title:
        title = _clean_text(soup.title.get_text(" "))

    # main text: join paragraphs/list items + headings (keeps structure-ish)
    parts: List[str] = []
    for node in container.select("h1, h2, h3, p, li, pre, code, blockquote"):
        txt = _clean_text(node.get_text(" "))
        if txt:
            parts.append(txt)
    main_text = _clean_text(" ".join(parts))

    # outgoing links from container only
    out_links: Set[str] = set()
    for a in container.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href.startswith(("mailto:", "javascript:")):
            continue
        abs_url = urljoin(url, href)
        abs_url = norm_url(abs_url)
        if abs_url.startswith(("http://", "https://")):
            out_links.add(abs_url)

    return title, main_text, out_links


# -----------------------------
# Topic-bound expansion rules
# -----------------------------

def is_relevant_url(url: str, topic: str) -> bool:
    """
    Keep this strict for quality. You can widen later.
    For 'dynamic programming' topic, we:
      - allow cp-algorithms DP + sequences
      - allow only specific graph pages (shortest paths) if desired
      - allow oi-wiki /dp
      - allow geeksforgeeks dp-like pages
    """
    u = (url or "").lower()

    # global exclusions
    if any(x in u for x in [
        "feed_rss", "rss", "sitemap", "privacy", "terms", "about", "contact",
        "login", "signup", "register", "subscribe",
    ]):
        return False

    if "cp-algorithms.com" in u:
        # DP core sections
        if any(p in u for p in ["/dynamic_programming/", "/sequences/"]):
            return True
        # optional: shortest path pages you seeded (keep narrow)
        allow_graph = [
            "/graph/bellman_ford",
            "/graph/all-pair-shortest-path-floyd-warshall",
            "/graph/dijkstra",
            "/graph/finding-negative-cycle-in-graph",
        ]
        return any(p in u for p in allow_graph)

    if "oi-wiki.org" in u:
        return "/dp" in u  # allow /dp and /dp/xxx

    if "geeksforgeeks.org" in u:
        return any(p in u for p in [
            "/dynamic-programming",
            "/dynamic-programming/",
            "-dp-",
            "/dsa/",
        ])

    return False

def keep_page(topic: str, title: Optional[str], text: str, keyword_hits: int, min_hits: int) -> bool:
    """
    Two-stage keep rule:
      1) keyword hits >= min_hits
      2) topic-core check (prevents keeping pure navigation pages)
    """
    if keyword_hits < min_hits:
        return False

    t = f"{title or ''}\n{text or ''}".lower()

    # DP-specific core signals (tight on purpose for demo quality)
    core = [
        "dynamic programming", " dp ", "dp-", "knapsack", "lcs", "lis",
        "edit distance", "state", "transition", "subproblem",
        "bellman", "floyd", "warshall",
    ]
    return any(k in t for k in core)


# -----------------------------
# Deep Research Loop
# -----------------------------

@dataclass
class PageRecord:
    url: str
    depth: int
    domain: str
    title: Optional[str]
    keyword_hits: int
    content_hash: str
    text_snippet: str
    extracted_links_count: int


async def deep_research(cfg: ResearchConfig) -> List[PageRecord]:
    keywords = build_keywords(cfg.topic)

    seen_urls: Set[str] = set()
    seen_content_hash: Set[str] = set()
    per_domain_count: Dict[str, int] = {}

    frontier: List[Tuple[str, int]] = [(norm_url(u), 0) for u in cfg.seed_urls]
    results: List[PageRecord] = []

    open(cfg.output_jsonl, "w", encoding="utf-8").close()

    sem = asyncio.Semaphore(cfg.concurrency)

    async with AsyncWebCrawler(verbose=False) as crawler:

        async def crawl_one(url: str, depth: int) -> Tuple[str, int, Optional[PageRecord], Set[str]]:
            async with sem:
                url_n = norm_url(url)
                if not url_n:
                    return url_n, depth, None, set()

                dom = get_domain(url_n)

                # scope checks
                if cfg.blocked_domains and dom in cfg.blocked_domains:
                    return url_n, depth, None, set()
                if cfg.allowed_domains and dom not in cfg.allowed_domains:
                    return url_n, depth, None, set()

                # domain budget
                if per_domain_count.get(dom, 0) >= cfg.max_pages_per_domain:
                    return url_n, depth, None, set()

                last_exc: Optional[Exception] = None
                for attempt in range(cfg.max_retries):
                    try:
                        r = await crawler.arun(url=url_n)
                        last_exc = None

                        # Prefer HTML main-content extraction
                        html = _get_html_from_result(r)
                        title2, main_text, main_links = extract_main_text_and_links(url_n, dom, html)

                        # Fall back to crawler outputs if needed
                        title = title2 or (getattr(r, "title", None) or None)
                        md = getattr(r, "markdown", None) or ""
                        text = main_text or (getattr(r, "text", None) or "")

                        if not text and md:
                            # crude markdown stripping fallback
                            text = re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]\([^)]+\)", " ", md)).strip()

                        # outgoing links: prefer main_links, else markdown parse
                        out_links = main_links if main_links else extract_links_from_markdown(md, base_url=url_n)

                        combined = f"{title or ''}\n{text}"
                        hits = keyword_score(combined, keywords)

                        # content dedup
                        ch = sha256_text((text or "")[:20000])
                        if ch in seen_content_hash:
                            return url_n, depth, None, set()

                        # keep decision
                        if keep_page(cfg.topic, title, text, hits, cfg.min_keyword_hits):
                            snippet = _clean_text((text or "")[:400])
                            rec = PageRecord(
                                url=url_n,
                                depth=depth,
                                domain=dom,
                                title=title,
                                keyword_hits=hits,
                                content_hash=ch,
                                text_snippet=snippet,
                                extracted_links_count=len(out_links),
                            )
                            return url_n, depth, rec, out_links

                        # not kept but still return links for discovery (controlled by relevance filter later)
                        return url_n, depth, None, out_links

                    except Exception as e:
                        last_exc = e
                        await asyncio.sleep(cfg.retry_backoff_base_sec * (attempt + 1))

                # all retries failed
                _ = last_exc
                return url_n, depth, None, set()

        while frontier and len(results) < cfg.max_pages:
            batch: List[Tuple[str, int]] = []
            while frontier and len(batch) < cfg.concurrency * 2:
                u, d = frontier.pop(0)
                u = norm_url(u)
                if not u or u in seen_urls:
                    continue
                if d > cfg.max_depth:
                    continue
                seen_urls.add(u)
                batch.append((u, d))

            if not batch:
                break

            crawled = await asyncio.gather(*[crawl_one(u, d) for (u, d) in batch])

            new_urls_added = 0

            for url_n, depth, rec, out_links in crawled:
                if not url_n:
                    continue

                dom = get_domain(url_n)
                per_domain_count[dom] = per_domain_count.get(dom, 0) + 1

                if rec is not None:
                    seen_content_hash.add(rec.content_hash)
                    results.append(rec)
                    with open(cfg.output_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

                # expand frontier with strict relevance control
                if depth < cfg.max_depth and out_links:
                    for link in out_links:
                        link = norm_url(link)
                        if not link or link in seen_urls:
                            continue

                        ldom = get_domain(link)

                        # allow/disallow external domains
                        if not cfg.allow_external_domains and ldom != dom:
                            continue

                        # domain allow/block
                        if cfg.blocked_domains and ldom in cfg.blocked_domains:
                            continue
                        if cfg.allowed_domains and ldom not in cfg.allowed_domains:
                            continue

                        # IMPORTANT: topic/site-aware relevance filter
                        if not is_relevant_url(link, cfg.topic):
                            continue

                        frontier.append((link, depth + 1))
                        new_urls_added += 1

            if new_urls_added == 0 and all(x[2] is None for x in crawled):
                break

    return results


# -----------------------------
# Run Example
# -----------------------------

if __name__ == "__main__":
    cfg = ResearchConfig(
        topic="dynamic programming",
        seed_urls=[
            "https://cp-algorithms.com/dynamic_programming/intro-to-dp.html",
            "https://oi-wiki.org/dp/",
            "https://www.geeksforgeeks.org/competitive-programming/dynamic-programming/",
            "https://cp-algorithms.com/dynamic_programming/knapsack.html",
            "https://oi-wiki.org/dp/knapsack/",
            "https://cp-algorithms.com/sequences/longest_increasing_subsequence.html",
            "https://www.geeksforgeeks.org/dsa/longest-common-subsequence-dp-4/",
            "https://www.geeksforgeeks.org/dsa/longest-increasing-subsequence-dp-3/",
            "https://cp-algorithms.com/graph/bellman_ford.html",
            "https://cp-algorithms.com/graph/all-pair-shortest-path-floyd-warshall.html",
            "https://www.geeksforgeeks.org/dsa/edit-distance-dp-5/",
        ],

        max_depth=2,
        max_pages=80,
        max_pages_per_domain=30,
        concurrency=6,

        allow_external_domains=False,
        allowed_domains={
            "cp-algorithms.com",
            "oi-wiki.org",
            "www.geeksforgeeks.org",
        },

        blocked_domains={
            "accounts.google.com",
            "support.google.com",
        },

        min_keyword_hits=1,
        max_retries=3,
        retry_backoff_base_sec=1.5,
        output_jsonl="deep_research_results.jsonl",
    )

    out = asyncio.run(deep_research(cfg))
    print(f"Done. Kept {len(out)} pages. Saved to {cfg.output_jsonl}")
