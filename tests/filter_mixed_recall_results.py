# -*- coding: utf-8 -*-
"""
filter_mixed_recall_results.py (v4)

Purpose:
  Take output/mixed_recall_realtime_test.json (mixed recall candidates)
  and produce output/mixed_recall_realtime_filtered.json with three streams:
    - event_seeds    : continuously-updating entrances (feeds, taxonomy listings, shallow newsroom entrances, pagination listings)
    - anchor_seeds   : stable background anchors (docs/guide/course/wiki + content pages + knowledge bases)
    - evidence_pages : terminal detail articles/news pages (kept small)

v4 Improvements (on top of your v3):
  1) Feed-derived entrance expansion:
     If we detect a feed URL as an event_seed, we derive a few likely entrance URLs
     (e.g., /feeds/topic/X.rss -> /topic/X, /topics/X, /tag/X, /category/X, etc.)
     then re-classify them and add to event_seeds/anchor_seeds accordingly.
     This increases "entrance diversity" without loosening the event_seed definition.

  2) Keep event_seed strict:
     - detail articles NEVER become event_seeds
     - content pages / knowledge bases NEVER become event_seeds
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import urlparse, parse_qs, urlunparse

INPUT_FILE = Path("output/mixed_recall_realtime_test.json")
OUTPUT_FILE = Path("output/mixed_recall_realtime_filtered.json")

# -----------------------------
# Optional: better root domain parsing
# -----------------------------
try:
    import tldextract  # type: ignore
except Exception:
    tldextract = None


# -----------------------------
# Blocks / allowlists
# -----------------------------
BLOCK_DOMAINS = {
    "web.archive.org",
    "archive.org",
    "books.google.com",
    "books.google.co.uk",
    "books.googleusercontent.com",
    "worldcat.org",
    "id.worldcat.org",
    "bnf.fr",
    "catalogue.bnf.fr",
    "data.bnf.fr",
    "loc.gov",
    "id.loc.gov",
    "d-nb.info",
    "ndl.go.jp",
    "id.ndl.go.jp",
    "aleph.nkp.cz",
    "kopkatalogs.lv",
    "nli.org.il",
    "hdl.handle.net",
    "doi.org",
    "api.semanticscholar.org",

    # academic/paper portals (usually not good "event seeds")
    "arxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "ui.adsabs.harvard.edu",
    "jstor.org",
    "citeseer.ist.psu.edu",
    "citeseerx.ist.psu.edu",
    "projecteuclid.org",
    "eprints.ecs.soton.ac.uk",
    "research-portal.uws.ac.uk",
    "research-information.bris.ac.uk",
    "dblp.org",
    "semanticscholar.org",
}

SOFT_BLOCK_TOKENS = {
    "lifeandstyle",
    "celebrity",
    "entertainment",
    "fashion",
    "gossip",
    "lifestyle",
    "tabloid",
}

# High-trust boost (still not bypassing hard blocks)
ALLOWLIST_DOMAINS = {
    "nasa.gov",
    "jpl.nasa.gov",
    "mars.nasa.gov",
    "esa.int",
    "jaxa.jp",
    "cnsa.gov.cn",

    "science.org",
    "nature.com",
    "ieee.org",
    "acm.org",

    "mit.edu",
    "stanford.edu",
    "harvard.edu",
    "berkeley.edu",
    "cornell.edu",
    "ox.ac.uk",
    "cam.ac.uk",
}

# Knowledge base / identifier sites:
# allow as anchors (optional), but never as event seeds
KNOWLEDGE_BASE_ROOT_DOMAINS = {
    "getty.edu",          # vocab.getty.edu
    "wikidata.org",
    "wikipedia.org",
    "wikiversity.org",
    "wikisource.org",
    "wikimedia.org",
    "loc.gov",
    "ndl.go.jp",
}

KNOWLEDGE_BASE_SUBDOMAINS = {
    "vocab.getty.edu",
    "id.loc.gov",
    "id.ndl.go.jp",
}


# -----------------------------
# Heuristics: URL type detection
# -----------------------------
FEED_HINTS = ("rss", "atom", "feed", ".rss", ".xml")

TAXONOMY_LIST_HINTS = (
    "/tag/", "/tags/",
    "/topic/", "/topics/",
    "/category/", "/categories/",
    "/archive/", "/archives/",
)

ANCHOR_PATH_HINTS = (
    "/docs/", "/documentation/", "/doc/", "/manual/", "/handbook/",
    "/guide/", "/tutorial/", "/reference/", "/api/", "/spec/", "/standards/",
    "/course/", "/courses/", "/syllabus/", "/lecture/", "/lectures/", "/teaching/",
    "/learn/", "/learning/", "/kb/", "/wiki/",
)

FORUM_HINTS = (
    "stackoverflow.com", "stackexchange.com", "reddit.com", "quora.com",
    "zhihu.com", "tieba.baidu.com", "discuss", "forum", "bbs"
)

TRACKING_PARAMS_PREFIX = ("utm_", "gclid", "fbclid", "mc_cid", "mc_eid")

NEWSROOM_HINTS = (
    "/news", "/press", "/pressroom", "/press-releases", "/releases",
    "/updates", "/announcements", "/notices", "/bulletin", "/blog"
)

CONTENT_FILE_EXTS = (".html", ".htm", ".pdf", ".doc", ".docx", ".ppt", ".pptx")


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class ClassifiedURL:
    url: str
    domain: str
    root_domain: str
    kind: str  # event_seed / anchor_seed / evidence / discard
    score: int
    reasons: List[str]


# -----------------------------
# Helpers
# -----------------------------
def normalize_url(url: str) -> str:
    """Strip fragment, normalize trivial trailing slash."""
    try:
        p = urlparse(url)
        p = p._replace(fragment="")
        out = urlunparse(p)
        if out.endswith("/") and len(p.path) > 1:
            out = out[:-1]
        return out
    except Exception:
        return url


def root_domain(domain: str) -> str:
    domain = (domain or "").lower().strip(".")
    if not domain:
        return ""
    if tldextract is None:
        parts = domain.split(".")
        return domain if len(parts) <= 2 else ".".join(parts[-2:])
    ext = tldextract.extract(domain)
    if not ext.suffix:
        return domain
    return f"{ext.domain}.{ext.suffix}".lower()


def blocked(domain: str, url: str) -> bool:
    d = (domain or "").lower()
    rd = root_domain(d)
    if d in BLOCK_DOMAINS or rd in BLOCK_DOMAINS:
        return True
    u = (url or "").lower()
    if any(tok in u for tok in SOFT_BLOCK_TOKENS):
        return True
    return False


def allowed_boost(domain: str) -> bool:
    d = (domain or "").lower()
    rd = root_domain(d)
    return (d in ALLOWLIST_DOMAINS) or (rd in ALLOWLIST_DOMAINS) or d.endswith(
        (".edu", ".ac.uk", ".ac.cn", ".edu.cn", ".gov", ".gov.cn")
    )
    # NOTE: no default .org trust


def is_knowledge_base(domain: str) -> bool:
    d = (domain or "").lower()
    rd = root_domain(d)
    return (d in KNOWLEDGE_BASE_SUBDOMAINS) or (rd in KNOWLEDGE_BASE_ROOT_DOMAINS)


def has_many_tracking_params(url: str) -> bool:
    try:
        qs = parse_qs(urlparse(url).query)
        if not qs:
            return False
        keys = [k.lower() for k in qs.keys()]
        tracking = sum(1 for k in keys if k.startswith(TRACKING_PARAMS_PREFIX))
        return (len(keys) >= 6 and tracking >= 1) or (tracking >= 2)
    except Exception:
        return False


def is_feed_url(url: str) -> bool:
    u = (url or "").lower()
    if any(h in u for h in FEED_HINTS):
        if "feedback" in u:
            return False
        return True
    return False


def is_detail_article(url: str) -> bool:
    """
    Terminal pages (single news/article). These should NOT be seeds.
    Intentionally over-detect to protect event_seeds quality.
    """
    u = (url or "").lower()
    p = urlparse(url)
    path = (p.path or "").lower()

    # Date patterns: /YYYY/MM/DD/
    if re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", u):
        return True

    # /archives/2026/02/09/...
    if re.search(r"/archives?/20\d{2}/\d{1,2}/\d{1,2}/", u):
        return True

    # /news/xxx usually detail unless clear listing
    if "/news/" in path:
        if any(h in path for h in TAXONOMY_LIST_HINTS) or "/page/" in path:
            return False
        qs = parse_qs(p.query)
        if any(k.lower() in ("page", "p", "offset") for k in qs.keys()):
            return False
        return True

    # very long slug-like
    if len(path) > 90 and path.count("/") >= 3:
        return True

    return False


def is_taxonomy_listing(url: str) -> bool:
    path = (urlparse(url).path or "").lower()
    return any(h in path for h in TAXONOMY_LIST_HINTS)


def has_pagination(url: str) -> bool:
    p = urlparse(url)
    path = (p.path or "").lower()
    if "/page/" in path:
        return True
    qs = parse_qs(p.query)
    if any(k.lower() in ("page", "p", "offset") for k in qs.keys()):
        return True
    return False


def is_newsroom_entrance(url: str) -> bool:
    """
    Conservative detector for newsroom/press entrance pages.
    Must be shallow and NOT a detail article.
    """
    p = urlparse(url)
    path = (p.path or "").lower().rstrip("/")
    if is_detail_article(url):
        return False

    if not any(path == h or path.startswith(h + "/") for h in NEWSROOM_HINTS):
        return False

    segs = [s for s in path.split("/") if s]
    return len(segs) <= 2


def is_anchor_like(url: str) -> bool:
    path = (urlparse(url).path or "").lower()
    if any(h in path for h in ANCHOR_PATH_HINTS):
        return True
    if any(x in path for x in ("/introduction", "/overview", "/getting-started", "/getting_started")):
        return True
    return False


def is_content_file_page(url: str) -> bool:
    """
    Content pages rather than listing entrances:
      - endswith .html/.htm/.pdf etc.
      - deep paths
      - strong library patterns
    """
    p = urlparse(url)
    path = (p.path or "").lower()

    if path.endswith(CONTENT_FILE_EXTS):
        return True

    if re.search(r"/works?/20?\d{2}/", path):
        return True
    if "/fulltext" in path or "/full-text" in path:
        return True

    segs = [s for s in path.split("/") if s]
    if len(segs) >= 5:
        return True

    return False


# -----------------------------
# Feed -> entrance derivation (v4)
# -----------------------------
def derive_entrances_from_feed(feed_url: str) -> List[str]:
    """
    Derive likely entrance/listing URLs from a feed URL without fetching the web.

    Examples:
      https://spectrum.ieee.org/feeds/topic/the-institute.rss
        -> https://spectrum.ieee.org/topic/the-institute
        -> https://spectrum.ieee.org/topics/the-institute
        -> https://spectrum.ieee.org/tag/the-institute
        -> https://spectrum.ieee.org/category/the-institute

      https://example.com/rss
        -> https://example.com/
    """
    derived: List[str] = []
    u = normalize_url(feed_url)
    p = urlparse(u)
    if not p.scheme.startswith("http") or not p.netloc:
        return derived

    domain_root = urlunparse((p.scheme, p.netloc, "/", "", "", ""))
    path = (p.path or "").strip("/")

    # Always include root as a weak candidate (filter will decide)
    derived.append(domain_root)

    if not path:
        return list({normalize_url(x) for x in derived})

    # Remove common feed endings
    path_no_ext = re.sub(r"(\.rss|\.xml)$", "", path, flags=re.IGNORECASE)

    # Common patterns: /feeds/topic/{slug}
    m = re.search(r"(?:^|/)feeds/(?:topic|topics|tag|tags|category|categories)/([^/]+)$", path_no_ext, re.IGNORECASE)
    if m:
        slug = m.group(1)
        for kind in ("topic", "topics", "tag", "tags", "category", "categories"):
            derived.append(urlunparse((p.scheme, p.netloc, f"/{kind}/{slug}", "", "", "")))
        return list({normalize_url(x) for x in derived})

    # /feeds/{something}
    m2 = re.search(r"(?:^|/)feeds/([^/]+)$", path_no_ext, re.IGNORECASE)
    if m2:
        slug = m2.group(1)
        for kind in ("topic", "topics", "tag", "tags", "category", "categories", "news", "blog"):
            derived.append(urlunparse((p.scheme, p.netloc, f"/{kind}/{slug}", "", "", "")))
        return list({normalize_url(x) for x in derived})

    # /rss or /feed or /atom -> likely root entrance only
    if re.search(r"(?:^|/)(rss|feed|atom)$", path_no_ext, re.IGNORECASE):
        return list({normalize_url(x) for x in derived})

    # If feed path is like /something/rss or /something/feed
    m3 = re.search(r"^(.*?)/(rss|feed|atom)$", path_no_ext, re.IGNORECASE)
    if m3:
        base = m3.group(1).strip("/")
        derived.append(urlunparse((p.scheme, p.netloc, f"/{base}", "", "", "")))
        return list({normalize_url(x) for x in derived})

    return list({normalize_url(x) for x in derived})


# -----------------------------
# Core classifier
# -----------------------------
def classify(url: str) -> ClassifiedURL:
    url = normalize_url(url)
    p = urlparse(url)
    domain = (p.netloc or "").lower()
    rd = root_domain(domain)
    reasons: List[str] = []
    score = 0

    # basic validity
    if not url or not p.scheme.startswith("http") or not domain:
        return ClassifiedURL(url=url, domain=domain, root_domain=rd, kind="discard", score=-999, reasons=["invalid_url"])

    if blocked(domain, url):
        return ClassifiedURL(url=url, domain=domain, root_domain=rd, kind="discard", score=-999, reasons=["blocked_domain_or_token"])

    # forum-like
    u_low = url.lower()
    if any(h in domain for h in FORUM_HINTS) or any(h in u_low for h in ("/forum", "/forums", "/discuss", "/thread")):
        return ClassifiedURL(url=url, domain=domain, root_domain=rd, kind="discard", score=-50, reasons=["forum_like"])

    if has_many_tracking_params(url):
        reasons.append("many_tracking_params")
        score -= 8

    if allowed_boost(domain):
        reasons.append("trusted_domain")
        score += 8

    kb = is_knowledge_base(domain)
    if kb:
        reasons.append("knowledge_base")
        score += 1

    # detail pages: NEVER event seeds
    if is_detail_article(url):
        reasons.append("detail_article")
        score += 2
        return ClassifiedURL(url=url, domain=domain, root_domain=rd, kind="evidence", score=score, reasons=reasons)

    # content pages: never event seeds (anchor or discard)
    content_page = is_content_file_page(url)
    if content_page:
        reasons.append("content_page")
        score += 3

    feed = is_feed_url(url)
    taxonomy = is_taxonomy_listing(url)
    pagination = has_pagination(url)
    newsroom = is_newsroom_entrance(url)
    anchor = is_anchor_like(url)

    # event seeds
    if feed:
        reasons.append("feed_url")
        score += 30
        return ClassifiedURL(url=url, domain=domain, root_domain=rd, kind="event_seed", score=score, reasons=reasons)

    # never allow kb/content pages to become event seeds
    if content_page or kb:
        kind = "anchor_seed" if (anchor or content_page or kb) else "discard"
        if kind == "anchor_seed":
            reasons.append("anchor_like" if anchor else "anchor_from_content_or_kb")
            score += 6 if anchor else 4
        else:
            reasons.append("discard_content_or_kb")
            score -= 2
        # anchor boosts
        path = (p.path or "").lower()
        if kind == "anchor_seed":
            if any(h in path for h in ("/docs", "/documentation", "/manual", "/handbook", "/reference")):
                score += 3
                reasons.append("docs_hint")
            if any(h in path for h in ("/course", "/courses", "/syllabus", "/lectures")):
                score += 2
                reasons.append("course_hint")
        return ClassifiedURL(url=url, domain=domain, root_domain=rd, kind=kind, score=score, reasons=reasons)

    # entrance types
    if taxonomy:
        reasons.append("taxonomy_listing")
        score += 16
        if allowed_boost(domain):
            score += 3
        kind = "event_seed"
    elif pagination:
        reasons.append("pagination_listing")
        score += 12
        if allowed_boost(domain):
            score += 2
        kind = "event_seed"
    elif newsroom:
        reasons.append("newsroom_entrance")
        score += 10
        if allowed_boost(domain):
            score += 2
        kind = "event_seed"
    elif anchor:
        reasons.append("anchor_like")
        score += 10
        kind = "anchor_seed"
    else:
        reasons.append("unclear_type")
        score -= 2
        kind = "discard"

    # mild boosts
    path = (p.path or "").lower()
    if kind == "event_seed":
        if any(x in path for x in ("/changelog", "/release-notes", "/security", "/advisory")):
            score += 4
            reasons.append("update_security_hint")

    if kind == "anchor_seed":
        if any(h in path for h in ("/docs", "/documentation", "/manual", "/handbook", "/reference")):
            score += 3
            reasons.append("docs_hint")
        if any(h in path for h in ("/course", "/courses", "/syllabus", "/lectures")):
            score += 2
            reasons.append("course_hint")

    return ClassifiedURL(url=url, domain=domain, root_domain=rd, kind=kind, score=score, reasons=reasons)


# -----------------------------
# Main
# -----------------------------
def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    out_results = []
    for item in data.get("results", []):
        keyword = item.get("keyword")
        language = item.get("language")
        ranked = item.get("ranked_candidates", [])

        event_seeds: List[Dict] = []
        anchor_seeds: List[Dict] = []
        evidence_pages: List[Dict] = []

        seen_urls: Set[str] = set()

        # First pass: classify given candidates
        classified_events: List[ClassifiedURL] = []
        for r in ranked:
            raw = (r.get("url") or "").strip()
            if not raw:
                continue
            url = normalize_url(raw)
            if url in seen_urls:
                continue
            seen_urls.add(url)

            c = classify(url)
            if c.kind == "discard":
                continue

            if c.kind == "event_seed":
                classified_events.append(c)

            obj = {
                "url": c.url,
                "score": c.score,
                "domain": c.domain,
                "root_domain": c.root_domain,
                "reasons": c.reasons,
            }

            if c.kind == "event_seed" and c.score >= 8:
                event_seeds.append(obj)
            elif c.kind == "anchor_seed" and c.score >= 6:
                anchor_seeds.append(obj)
            elif c.kind == "evidence" and c.score >= 0:
                evidence_pages.append(obj)

        # v4: expand entrances from feed event_seeds
        derived_candidates: List[str] = []
        for c in classified_events:
            if "feed_url" in c.reasons:
                derived_candidates.extend(derive_entrances_from_feed(c.url))

        for durl in derived_candidates:
            durl = normalize_url(durl)
            if not durl or durl in seen_urls:
                continue
            seen_urls.add(durl)

            dc = classify(durl)
            if dc.kind == "discard":
                continue

            obj = {
                "url": dc.url,
                "score": dc.score,
                "domain": dc.domain,
                "root_domain": dc.root_domain,
                "reasons": (dc.reasons + ["derived_from_feed"]),
            }

            if dc.kind == "event_seed" and dc.score >= 8:
                event_seeds.append(obj)
            elif dc.kind == "anchor_seed" and dc.score >= 6:
                anchor_seeds.append(obj)
            elif dc.kind == "evidence" and dc.score >= 0:
                evidence_pages.append(obj)

        # Sort & trim
        event_seeds.sort(key=lambda x: x["score"], reverse=True)
        anchor_seeds.sort(key=lambda x: x["score"], reverse=True)
        evidence_pages.sort(key=lambda x: x["score"], reverse=True)

        out_results.append({
            "keyword": keyword,
            "language": language,
            "event_seed_count": len(event_seeds),
            "anchor_seed_count": len(anchor_seeds),
            "evidence_count": len(evidence_pages),
            "event_seeds": event_seeds[:60],
            "anchor_seeds": anchor_seeds[:40],
            "evidence_pages": evidence_pages[:20],
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps({"results": out_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Filtered results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
