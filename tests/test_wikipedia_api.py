"""
Wikipedia API 测试脚本

用于测试 Wikipedia API 是否可用于发现中文教育网站的种子 URLs

API 文档: https://zh.wikipedia.org/w/api.php
特点:
- 免费，无需 API key
- 支持中文
- 返回结构化数据
"""

import requests
import json
from typing import List, Dict
from pathlib import Path


def search_wikipedia(keyword: str, language: str = "zh") -> Dict:
    """
    搜索 Wikipedia

    Args:
        keyword: 搜索关键词
        language: 语言代码 (zh=中文, en=英文)

    Returns:
        API 响应数据
    """
    url = f"https://{language}.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": keyword,
        "format": "json",
        "srlimit": 10,  # 返回结果数量
        "srprop": "titlesnippet|snippet|titles|wordcount",  # 返回的属性
    }

    print(f"\n{'='*70}")
    print(f"测试关键词: {keyword} (语言: {language})")
    print(f"{'='*70}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        return data

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API 请求失败: {e}")
        return {}


def get_page_url(title: str, language: str = "zh") -> str:
    """
    根据页面标题获取完整 URL

    Args:
        title: 页面标题
        language: 语言代码

    Returns:
        完整 URL
    """
    # 将空格替换为下划线
    title_escaped = title.replace(" ", "_")
    return f"https://{language}.wikipedia.org/wiki/{title_escaped}"


def extract_urls_from_wiki_results(data: Dict, language: str = "zh") -> List[Dict]:
    """
    从 Wikipedia API 响应中提取 URLs

    Args:
        data: API 响应数据
        language: 语言代码

    Returns:
        提取的 URL 列表
    """
    urls = []

    search_results = data.get("query", {}).get("search", [])

    for result in search_results:
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        wordcount = result.get("wordcount", 0)
        timestamp = result.get("timestamp", "")

        # 构建完整 URL
        url = get_page_url(title, language)

        urls.append({
            "url": url,
            "title": title,
            "snippet": snippet,
            "wordcount": wordcount,
            "timestamp": timestamp,
            "source": f"Wikipedia ({language})",
        })

    return urls


def main():
    """主测试函数"""

    # 测试关键词列表
    test_keywords = [
        ("马克思主义", "zh"),  # 中文
        ("算法", "zh"),       # 中文
        ("machine learning", "en"),  # 英文
    ]

    print("="*70)
    print("Wikipedia API 测试")
    print("="*70)

    all_discovered_urls = []

    for keyword, language in test_keywords:
        # 1. 调用 API
        data = search_wikipedia(keyword, language)

        if not data:
            print(f"[WARNING] 关键词 '{keyword}' 未获得结果")
            continue

        # 2. 打印响应基本信息
        search_info = data.get("query", {}).get("searchinfo", {})
        total_hits = search_info.get("totalhits", 0)

        print(f"\n[响应信息]")
        print(f"  - 总命中数: {total_hits}")

        # 3. 提取 URLs
        urls = extract_urls_from_wiki_results(data, language)

        if not urls:
            print(f"[!] 未从响应中提取到 URLs")
            continue

        print(f"\n[URL] 提取到 {len(urls)} 个 URLs:")

        # 4. 打印结果
        for i, url_info in enumerate(urls, 1):
            print(f"\n  {i}. {url_info['url']}")
            print(f"     标题: {url_info['title']}")
            print(f"     字数: {url_info['wordcount']}")
            # snippet 包含 HTML 标签和特殊字符，可能导致编码问题
            # print(f"     摘要: {url_info['snippet'][:80]}...")

        all_discovered_urls.extend(urls)

    # 5. 总结
    print(f"\n{'='*70}")
    print("[统计] 测试总结")
    print(f"{'='*70}")
    print(f"测试关键词数: {len(test_keywords)}")
    print(f"发现 URLs 总数: {len(all_discovered_urls)}")

    # 按语言分类统计
    zh_urls = [u for u in all_discovered_urls if "zh.wikipedia.org" in u["url"]]
    en_urls = [u for u in all_discovered_urls if "en.wikipedia.org" in u["url"]]

    print(f"中文 Wikipedia URLs: {len(zh_urls)}")
    print(f"英文 Wikipedia URLs: {len(en_urls)}")

    # 6. 推荐的种子 URLs
    print(f"\n{'='*70}")
    print("[推荐] 推荐的种子 URLs")
    print(f"{'='*70}")

    for i, url_info in enumerate(all_discovered_urls, 1):
        print(f"\n{i}. {url_info['url']}")
        print(f"   标题: {url_info['title']}")
        print(f"   来源: {url_info['source']}")

    # 7. 保存结果
    output_file = "tests/output/wikipedia_discovery_test.json"
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_keywords": [kw for kw, _ in test_keywords],
            "total_discovered": len(all_discovered_urls),
            "zh_urls": len(zh_urls),
            "en_urls": len(en_urls),
            "all_urls": all_discovered_urls,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] 测试结果已保存到: {output_file}")

    # 8. 结论和建议
    print(f"\n{'='*70}")
    print("[总结] 结论和建议")
    print(f"{'='*70}")
    print(f"""
1. Wikipedia API 特点:
   [+] 免费，无需 API key
   [+] 支持中文（zh.wikipedia.org）
   [+] 返回结构化数据
   [+] 结果质量高，适合作为种子
   [!] 结果数量有限（通常 10 条）

2. 适用场景:
   [+] 发现权威的百科条目
   [+] 获取概念的官方解释
   [+] 作为冷启动的种子来源
   [-] 不适合大规模 URL 发现

3. 对比 DuckDuckGo:
   [+] Wikipedia 对中文支持更好
   [+] DuckDuckGo 对英文更好
   [+] 两者可以结合使用

4. 建议策略:
   a) 对中文关键词，使用 Wikipedia API
   b) 对英文关键词，使用 DuckDuckGo 或 Wikipedia
   c) 将发现的 URLs 作为种子，用深度爬虫扩展
   d) 建立种子 URLs 库供复用
""")


if __name__ == "__main__":
    main()
