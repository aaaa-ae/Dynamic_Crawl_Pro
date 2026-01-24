"""
URL 工具模块
处理 URL 解析、规范化、过滤等操作
"""

import re
from urllib.parse import urlparse, urljoin, urlunparse
from typing import List, Optional, Set
from dataclasses import dataclass


@dataclass
class ParsedLink:
    """解析后的链接"""
    url: str
    text: str
    title: Optional[str] = None


def normalize_url(url: str, base_url: str = None) -> Optional[str]:
    """
    规范化 URL

    Args:
        url: 原始 URL
        base_url: 基础 URL（用于相对链接）

    Returns:
        规范化后的绝对 URL，如果无效则返回 None
    """
    if not url or url.startswith(("mailto:", "tel:", "javascript:", "data:", "#")):
        return None

    # 如果有 base_url，解析相对链接
    if base_url:
        url = urljoin(base_url, url)

    try:
        parsed = urlparse(url)

        # 移除 fragment
        parsed = parsed._replace(fragment="")

        # 规范化 scheme 和 netloc
        scheme = parsed.scheme.lower() if parsed.scheme else "https"
        netloc = parsed.netloc.lower() if parsed.netloc else ""

        if not netloc:
            return None

        # 移除端口号（如果是标准端口）
        if (scheme == "http" and netloc.endswith(":80")) or \
           (scheme == "https" and netloc.endswith(":443")):
            netloc = netloc.rsplit(":", 1)[0]

        # 移除多余的斜杠
        path = re.sub(r'/+', '/', parsed.path) or "/"

        # 规范化查询参数（按字母排序）
        query = "&".join(sorted(parsed.query.split("&"))) if parsed.query else ""

        normalized = urlunparse((scheme, netloc, path, parsed.params, query, ""))
        return normalized

    except Exception:
        return None


def get_domain(url: str) -> Optional[str]:
    """
    从 URL 提取域名

    Args:
        url: URL 字符串

    Returns:
        域名，如果无效则返回 None
    """
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        # 移除 www. 前缀
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return None


def is_allowed_domain(
    url: str,
    allowed_domains: List[str],
    blocked_domains: List[str]
) -> bool:
    """
    检查 URL 的域名是否在允许列表中且不在阻止列表中

    Args:
        url: URL 字符串
        allowed_domains: 允许的域名列表（空表示允许所有）
        blocked_domains: 阻止的域名列表

    Returns:
        是否允许爬取
    """
    domain = get_domain(url)
    if not domain:
        return False

    # 检查阻止列表
    for blocked in blocked_domains:
        if blocked in domain or domain.endswith(blocked):
            return False

    # 检查允许列表
    if not allowed_domains:
        return True

    for allowed in allowed_domains:
        if allowed in domain or domain.endswith(allowed):
            return True

    return False


def extract_links_from_markdown(
    markdown_text: str,
    base_url: str = None
) -> List[ParsedLink]:
    """
    从 Markdown 文本中提取链接

    Args:
        markdown_text: Markdown 格式的文本
        base_url: 基础 URL（用于相对链接）

    Returns:
        解析后的链接列表
    """
    links = []

    # 匹配 Markdown 链接: [text](url "title")
    # 支持带标题和不带标题的情况
    link_pattern = re.compile(
        r'\[([^\]]+)\]\(([^\)]+?)(?:\s+"([^"]+)")?\)',
        re.MULTILINE
    )

    for match in link_pattern.finditer(markdown_text):
        text = match.group(1)
        url_str = match.group(2)
        title = match.group(3) if match.group(3) else None

        normalized = normalize_url(url_str, base_url)
        if normalized:
            links.append(ParsedLink(url=normalized, text=text, title=title))

    return links


def extract_links_from_html(
    html_content: str,
    base_url: str = None
) -> List[ParsedLink]:
    """
    从 HTML 内容中提取链接

    Args:
        html_content: HTML 内容
        base_url: 基础 URL（用于相对链接）

    Returns:
        解析后的链接列表
    """
    from bs4 import BeautifulSoup

    links = []
    soup = BeautifulSoup(html_content, 'lxml')

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        text = a_tag.get_text(strip=True) or ""
        title = a_tag.get('title')

        normalized = normalize_url(href, base_url)
        if normalized:
            links.append(ParsedLink(url=normalized, text=text, title=title))

    return links


def should_crawl_url(
    url: str,
    visited_urls: Set[str],
    allowed_domains: List[str],
    blocked_domains: List[str],
    same_domain: bool = False,
    referrer_domain: str = None
) -> bool:
    """
    判断是否应该爬取该 URL

    Args:
        url: 目标 URL
        visited_urls: 已访问的 URL 集合
        allowed_domains: 允许的域名列表
        blocked_domains: 阻止的域名列表
        same_domain: 是否只爬取同一域名的链接
        referrer_domain: 引用页面的域名

    Returns:
        是否应该爬取
    """
    # 检查是否已访问
    if url in visited_urls:
        return False

    # 检查文件扩展名（跳过媒体文件）
    from urllib.parse import urlparse
    parsed = urlparse(url)
    path = parsed.path.lower()
    blocked_extensions = [
        '.webp', '.jpg', '.jpeg', '.png', '.gif', '.svg',
        '.pdf', '.doc', '.docx', '.mp4', '.mp3', '.zip', '.rar'
    ]
    if any(path.endswith(ext) for ext in blocked_extensions):
        return False

    # 检查域名限制
    if not is_allowed_domain(url, allowed_domains, blocked_domains):
        return False

    # 检查同域名限制
    if same_domain and referrer_domain:
        target_domain = get_domain(url)
        if target_domain != referrer_domain:
            return False

    return True


def is_valid_url(url: str) -> bool:
    """
    简单检查 URL 是否有效

    Args:
        url: URL 字符串

    Returns:
        URL 是否有效
    """
    if not url:
        return False

    # 检查协议
    if not url.startswith(("http://", "https://")):
        return False

    # 检查域名
    parsed = urlparse(url)
    if not parsed.netloc:
        return False

    return True
