# -*- coding: utf-8 -*-
# Wikipedia external-link seed discovery test script
# Purpose: search Wikipedia, then extract external links as seed candidates

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


USER_AGENT = "DynamicCrawlPro/1.0 (seed-discovery test)"


def search_wikipedia(keyword: str, language: str = "zh", limit: int = 3):
    url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": keyword,
        "format": "json",
        "srlimit": limit,
        "srprop": "",
    }
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("query", {}).get("search", [])


def get_page_url(title: str, language: str = "zh") -> str:
    title_escaped = title.replace(" ", "_")
    return f"https://{language}.wikipedia.org/wiki/{title_escaped}"


def is_external_candidate(url: str) -> bool:
    if not url:
        return False
    if not url.startswith("http"):
        return False

    domain = urlparse(url).netloc.lower()
    if not domain:
        return False

    blocked = [
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
    if any(domain.endswith(b) for b in blocked):
        return False

    # filter obvious media files
    if re.search(r"\.(pdf|jpg|jpeg|png|gif|svg|mp4|mp3|zip|rar)$", url, re.IGNORECASE):
        return False

    return True


def extract_external_links(html: str):
    soup = BeautifulSoup(html, "lxml")

    links = []
    external_section = None
    headline_targets = {"external links", "\u5916\u90e8\u94fe\u63a5"}

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

    if not links:
        for a in soup.find_all("a", href=True):
            links.append(a["href"])

    seen = set()
    candidates = []
    for href in links:
        href = href.strip()
        if not is_external_candidate(href):
            continue
        if href in seen:
            continue
        seen.add(href)
        candidates.append(href)

    return candidates


def rank_candidates(urls):
    def score(url):
        domain = urlparse(url).netloc.lower()
        s = 0
        if domain.endswith((".edu", ".ac.uk", ".ac.cn", ".edu.cn", ".gov", ".gov.cn", ".org")):
            s += 10
        if "/research" in url or "/publication" in url or "/course" in url:
            s += 3
        if re.search(r"/20\d{2}/", url):
            s += 2
        return s

    ranked = [{"url": u, "score": score(u)} for u in urls]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def main():
    test_keywords = [
        ("\u9a6c\u514b\u601d\u4e3b\u4e49", "zh"),
        ("\u7b97\u6cd5", "zh"),
        ("machine learning", "en"),
    ]

    results = []

    for keyword, lang in test_keywords:
        print("=" * 70)
        print(f"Keyword: {keyword} | Language: {lang}")

        search_results = search_wikipedia(keyword, language=lang, limit=1)
        if not search_results:
            print("  - No search results")
            continue

        title = search_results[0]["title"]
        page_url = get_page_url(title, language=lang)
        print(f"  - Title: {title}")
        print(f"  - URL: {page_url}")

        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(page_url, headers=headers, timeout=10)
        resp.raise_for_status()
        candidates = extract_external_links(resp.text)
        ranked = rank_candidates(candidates)

        print(f"  - External link candidates: {len(candidates)}")
        for i, item in enumerate(ranked[:10], 1):
            print(f"    {i}. [{item['score']}] {item['url']}")

        results.append({
            "keyword": keyword,
            "language": lang,
            "page_title": title,
            "page_url": page_url,
            "candidates_count": len(candidates),
            "ranked_candidates": ranked,
        })

    output_file = Path("output/wikipedia_external_links_test.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    print("\n[OK] Results saved to:", output_file)


if __name__ == "__main__":
    main()
