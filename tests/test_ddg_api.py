"""
DuckDuckGo Instant Answer API 测试脚本

用于测试 DuckDuckGo Instant Answer API 是否可用于发现教育网站的种子 URLs

API 文档: https://api.duckduckgo.com/
参数说明:
- q: 查询关键词
- format: 返回格式 (json/xml)
- no_html: 1 (不返回 HTML)
- skip_disambig: 1 (跳过消歧页面)

特点:
- 免费，无需 API key
- 返回结构化数据
- 包含官方 URL、摘要、相关主题等
"""

import requests
import json
from typing import List, Dict
from urllib.parse import urlparse
from pathlib import Path


def test_ddg_instant_answer(keyword: str) -> Dict:
    """
    测试 DuckDuckGo Instant Answer API

    Args:
        keyword: 搜索关键词

    Returns:
        API 响应数据
    """
    url = "https://api.duckduckgo.com/"

    params = {
        "q": keyword,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
    }

    print(f"\n{'='*70}")
    print(f"测试关键词: {keyword}")
    print(f"{'='*70}")

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        return data

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API 请求失败: {e}")
        return {}


def extract_urls_from_ddg_response(data: Dict) -> List[Dict]:
    """
    从 DuckDuckGo API 响应中提取 URLs

    Args:
        data: API 响应数据

    Returns:
        提取的 URL 列表，包含类型和来源信息
    """
    urls = []

    # 1. 主要结果 (AbstractURL)
    if data.get("AbstractURL"):
        urls.append({
            "url": data["AbstractURL"],
            "type": "primary",
            "source": "AbstractURL",
            "title": data.get("Abstract", "")[:100],
        })

    # 2. 官方网站 (AbstractURL - 当它是官方站点时)
    if data.get("AbstractURL") and data.get("AbstractSource"):
        urls.append({
            "url": data["AbstractURL"],
            "type": "official",
            "source": data["AbstractSource"],
            "title": data.get("Heading", ""),
        })

    # 3. 相关主题 (RelatedTopics)
    related_topics = data.get("RelatedTopics", [])
    for topic in related_topics:
        if isinstance(topic, dict):
            if topic.get("FirstURL"):
                urls.append({
                    "url": topic["FirstURL"],
                    "type": "related",
                    "source": "RelatedTopics",
                    "title": topic.get("Text", "")[:100],
                })

    # 4. 结果列表 (Results)
    results = data.get("Results", [])
    for result in results:
        if result.get("FirstURL"):
            urls.append({
                "url": result["FirstURL"],
                "type": "result",
                "source": "Results",
                "title": result.get("Text", "")[:100],
            })

    return urls


def is_educational_url(url: str) -> bool:
    """
    判断是否是教育类 URL

    Args:
        url: URL 字符串

    Returns:
        是否是教育类 URL
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    educational_domains = [
        ".edu",           # 美国教育机构
        ".ac.uk",         # 英国教育机构
        ".edu.cn",        # 中国教育机构
        ".ac.cn",         # 中国科学院等
        ".gov",           # 政府机构
        ".gov.cn",        # 中国政府机构
        ".org",           # 非营利组织
    ]

    return any(domain.endswith(suffix) for suffix in educational_domains)


def filter_and_rank_urls(urls: List[Dict], keyword: str) -> List[Dict]:
    """
    过滤和排序 URLs

    优先级:
    1. 教育类域名
    2. 官方网站
    3. 主要结果
    4. 其他相关结果

    Args:
        urls: URL 列表
        keyword: 关键词（用于相关度判断）

    Returns:
        排序后的 URL 列表
    """
    # 为每个 URL 计算优先级分数
    for url_info in urls:
        score = 0
        url = url_info["url"]

        # 1. 教育类域名加分
        if is_educational_url(url):
            score += 10

        # 2. 官方网站加分
        if url_info["type"] == "official":
            score += 5

        # 3. 主要结果加分
        if url_info["type"] == "primary":
            score += 3

        url_info["priority_score"] = score

    # 按分数降序排序
    urls.sort(key=lambda x: x["priority_score"], reverse=True)

    return urls


def main():
    """主测试函数"""

    # 测试关键词列表
    test_keywords = [
        "马克思主义",
        "算法",
        "machine learning",
    ]

    print("="*70)
    print("DuckDuckGo Instant Answer API 测试")
    print("="*70)

    all_discovered_urls = []

    for keyword in test_keywords:
        # 1. 调用 API
        data = test_ddg_instant_answer(keyword)

        if not data:
            print(f"[WARNING] 关键词 '{keyword}' 未获得结果")
            continue

        # 2. 打印响应基本信息
        print(f"\n[响应信息]")
        print(f"  - Heading: {data.get('Heading', 'N/A')}")
        print(f"  - AbstractSource: {data.get('AbstractSource', 'N/A')}")
        print(f"  - AbstractURL: {data.get('AbstractURL', 'N/A')}")
        print(f"  - Answer: {data.get('Answer', 'N/A')}")
        print(f"  - AnswerType: {data.get('AnswerType', 'N/A')}")

        # 3. 提取 URLs
        urls = extract_urls_from_ddg_response(data)

        if not urls:
            print(f"[!] 未从响应中提取到 URLs")
            continue

        print(f"\n[URL] 提取到 {len(urls)} 个 URLs:")

        # 4. 过滤和排序
        ranked_urls = filter_and_rank_urls(urls, keyword)

        # 5. 打印结果
        for i, url_info in enumerate(ranked_urls, 1):
            url = url_info["url"]
            is_edu = "[EDU]" if is_educational_url(url) else "[WEB]"
            score = url_info["priority_score"]

            print(f"\n  {i}. {is_edu} [优先级: {score}]")
            print(f"     URL: {url}")
            print(f"     类型: {url_info['type']}")
            print(f"     标题: {url_info['title']}")

        all_discovered_urls.extend(ranked_urls)

    # 6. 总结
    print(f"\n{'='*70}")
    print("[统计] 测试总结")
    print(f"{'='*70}")
    print(f"测试关键词数: {len(test_keywords)}")
    print(f"发现 URLs 总数: {len(all_discovered_urls)}")

    edu_urls = [u for u in all_discovered_urls if is_educational_url(u["url"])]
    print(f"教育类 URLs: {len(edu_urls)}")

    # 7. 推荐的种子 URLs (高优先级)
    print(f"\n{'='*70}")
    print("[推荐] 推荐的种子 URLs (优先级分数 >= 5)")
    print(f"{'='*70}")

    recommended = [u for u in all_discovered_urls if u["priority_score"] >= 5]
    for i, url_info in enumerate(recommended, 1):
        print(f"\n{i}. {url_info['url']}")
        print(f"   类型: {url_info['type']}")
        print(f"   来源: {url_info['source']}")

    # 8. 保存结果
    output_file = "tests/output/ddg_discovery_test.json"
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_keywords": test_keywords,
            "total_discovered": len(all_discovered_urls),
            "educational_urls": len(edu_urls),
            "recommended_count": len(recommended),
            "all_urls": all_discovered_urls,
            "recommended_urls": recommended,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] 测试结果已保存到: {output_file}")

    # 9. 结论和建议
    print(f"\n{'='*70}")
    print("[总结] 结论和建议")
    print(f"{'='*70}")
    print(f"""
1. DuckDuckGo Instant Answer API 特点:
   [+] 免费，无需 API key
   [+] 返回结构化数据
   [+] 包含官方网站和权威来源
   [!] 结果数量有限（通常 5-10 条）
   [!] 主要针对英文关键词优化

2. 适用场景:
   [+] 发现某个概念的官方网站或权威解释
   [+] 获取知名教育资源的首页 URL
   [+] 作为种子 URLs 的补充来源
   [-] 不适合大规模 URL 发现

3. 建议策略:
   a) 对每个新类别，先用 DDG API 获取 3-5 个权威种子 URLs
   b) 然后用深度爬虫从这些种子 URLs 扩展发现更多页面
   c) 对于中文关键词，考虑结合百度/搜狗等国内搜索引擎
   d) 建立种子 URLs 库（preset_seed_urls.json）供复用

4. 后续优化方向:
   - 集成多个搜索 API (DDG + 百度)
   - 使用 Wikipedia API 作为知识库种子
   - 利用学术资源库（DOAJ、CSSN）
""")


if __name__ == "__main__":
    main()
