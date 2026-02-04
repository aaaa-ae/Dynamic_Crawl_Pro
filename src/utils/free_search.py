"""
免费URL搜索工具
使用DuckDuckGo HTML版本搜索，无需API Key，完全免费
"""

import asyncio
import aiohttp
from typing import List
from urllib.parse import quote, urljoin
from bs4 import BeautifulSoup
from loguru import logger


async def search_urls_free(
    query: str,
    max_results: int = 10,
    timeout: int = 15,
    retries: int = 2
) -> List[str]:
    """
    使用DuckDuckGo搜索URL（完全免费）

    Args:
        query: 搜索查询
        max_results: 最大结果数
        timeout: 超时时间
        retries: 重试次数

    Returns:
        URL列表
    """
    search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    urls = []

    for attempt in range(retries + 1):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(search_url, headers=headers) as response:
                    if response.status != 200:
                        logger.warning(f"Search failed (attempt {attempt + 1}): HTTP {response.status}")
                        if attempt < retries:
                            await asyncio.sleep(1)
                            continue
                        return []

                    html = await response.text()

                    # 检查是否被拦截
                    if len(html) < 1000:
                        logger.warning(f"Search response too short (attempt {attempt + 1}), possibly blocked")
                        if attempt < retries:
                            await asyncio.sleep(2)
                            continue
                        return []

                    soup = BeautifulSoup(html, 'html.parser')

                    # DuckDuckGo HTML结果解析 - 尝试多种选择器
                    results = soup.find_all('a', class_='result__a', href=True)

                    # 如果找不到结果，尝试其他选择器
                    if not results:
                        results = soup.select('div.result__body a.result__a')

                    if not results:
                        results = soup.select('a[href*="http"]')

                    for result in results[:max_results]:
                        href = result.get('href', '')

                        # DDG返回的是重定向URL，需要解析
                        if '/l/?uddg=' in href:
                            try:
                                # 提取真实URL
                                real_url = href.split('/l/?uddg=')[1].split('&')[0]
                                # URL decode
                                from urllib.parse import unquote
                                real_url = unquote(real_url)
                                urls.append(real_url)
                            except:
                                continue
                        elif href.startswith('http'):
                            urls.append(href)

                    if urls:
                        break  # 成功获取到结果，退出重试循环

        except asyncio.TimeoutError:
            logger.warning(f"Search timeout (attempt {attempt + 1}): {query}")
            if attempt < retries:
                await asyncio.sleep(1)
                continue
        except Exception as e:
            logger.error(f"Search error (attempt {attempt + 1}): {e}")
            if attempt < retries:
                await asyncio.sleep(1)
                continue

    # 去重
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    logger.info(f"搜索 '{query}' 找到 {len(unique_urls)} 个URL")
    return unique_urls[:max_results]


async def search_resource_urls(knowledge_points: List[dict]) -> dict:
    """
    为所有知识点搜索相关资源URL

    Args:
        knowledge_points: 知识点列表

    Returns:
        知识点到URL的映射
    """
    from collections import defaultdict

    kp_urls_map = defaultdict(list)

    for kp in knowledge_points:
        kp_id = kp['kp_id']
        title = kp['title']

        # 构建搜索查询
        query = f"{title} 算法应用案例 教程 blog"

        logger.info(f"为 {title} 搜索资源...")

        # 搜索URL
        urls = await search_urls_free(query, max_results=5)

        if urls:
            kp_urls_map[kp_id].extend(urls)

    return dict(kp_urls_map)


# 使用示例
if __name__ == "__main__":
    async def test():
        results = await search_urls_free("动态规划 算法应用")
        for url in results:
            print(f"  - {url}")

    asyncio.run(test())
