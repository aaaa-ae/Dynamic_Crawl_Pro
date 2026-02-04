"""
ExtractorAgent - 内容提取 Agent
负责从爬取的页面中提取正文内容和链接
"""

import re
from typing import Any, Dict, List, Optional, Set
from bs4 import BeautifulSoup
from loguru import logger

from .base import BaseAgent
from ..utils import (
    DataManager,
    normalize_url,
    get_domain,
    extract_links_from_markdown,
    extract_links_from_html,
    count_keyword_hits,
    extract_headings,
    get_text_snippet,
    compute_content_hash,
    extract_main_content_from_markdown,
    ParsedLink,
)


class ExtractorAgent(BaseAgent):
    """
    内容提取 Agent
    从 HTML/Markdown 中提取正文内容和链接
    """

    def __init__(
        self,
        name: str = "ExtractorAgent",
        config: Dict[str, Any] = None,
        data_manager: DataManager = None,
        keywords: List[str] = None,
    ):
        """
        初始化 ExtractorAgent

        Args:
            name: Agent 名称
            config: 配置字典，包含:
                - extract_max_length: 最大提取长度
                - extract_text_snippet_length: 文本摘要长度
                - extract_headings_max: 最多提取的标题数
                - extract_links_max: 最多提取的链接数
                - filter_same_domain: 是否只提取同域名链接
            data_manager: 数据管理器实例
            keywords: 关键词列表（用于匹配统计）
        """
        super().__init__(name, config)
        self.data_manager = data_manager
        self.keywords = keywords or []

        # 配置参数
        self.extract_max_length = config.get("extract_max_length", 10000)
        self.extract_text_snippet_length = config.get("extract_text_snippet_length", 300)
        self.extract_headings_max = config.get("extract_headings_max", 10)
        self.extract_links_max = config.get("extract_links_max", 50)
        self.filter_same_domain = config.get("filter_same_domain", True)

        logger.info(f"[{self.name}] Initialized with {len(self.keywords)} keywords")

    def can_process(self, input_data: Any) -> bool:
        """
        检查是否能够处理该输入

        Args:
            input_data: 输入数据，应为包含 data_id 或直接包含 html/markdown 的字典

        Returns:
            是否能够处理
        """
        if not isinstance(input_data, dict):
            return False

        # 方式1: 通过 data_id 获取
        if "data_id" in input_data:
            return True

        # 方式2: 直接提供内容
        if "html" in input_data or "markdown" in input_data:
            return True

        return False

    def extract_links(
        self,
        html: str,
        markdown: str,
        base_url: str,
        allowed_domains: List[str] = None,
        blocked_domains: List[str] = None,
    ) -> List[ParsedLink]:
        """
        从页面中提取链接

        Args:
            html: HTML 内容
            markdown: Markdown 内容
            base_url: 基础 URL
            allowed_domains: 允许的域名列表
            blocked_domains: 阻止的域名列表

        Returns:
            提取的链接列表
        """
        links = []

        # 从 Markdown 中提取链接
        markdown_links = extract_links_from_markdown(markdown, base_url)
        links.extend(markdown_links)

        # 从 HTML 中提取链接
        html_links = extract_links_from_html(html, base_url)
        links.extend(html_links)

        # 去重（基于 URL）
        seen_urls: Set[str] = set()
        unique_links = []
        for link in links:
            if link.url not in seen_urls:
                seen_urls.add(link.url)
                unique_links.append(link)

        # 应用域名过滤
        if allowed_domains or blocked_domains:
            filtered_links = []
            for link in unique_links:
                domain = get_domain(link.url)
                if domain:
                    # 检查阻止列表
                    blocked = any(b in domain or domain.endswith(b) for b in (blocked_domains or []))
                    if blocked:
                        continue

                    # 检查允许列表
                    if allowed_domains:
                        allowed = any(a in domain or domain.endswith(a) for a in allowed_domains)
                        if not allowed:
                            continue

                    filtered_links.append(link)
            unique_links = filtered_links

        # 限制数量
        if len(unique_links) > self.extract_links_max:
            unique_links = unique_links[:self.extract_links_max]

        return unique_links

    def extract_main_content(
        self,
        html: str,
        markdown: str
    ) -> Dict[str, Any]:
        """
        提取主要内容

        Args:
            html: HTML 内容
            markdown: Markdown 内容

        Returns:
            包含提取内容的字典
        """
        # 优先使用 Markdown
        if markdown:
            main_content = extract_main_content_from_markdown(
                markdown,
                self.extract_max_length
            )
        else:
            # 从 HTML 提取纯文本
            soup = BeautifulSoup(html, 'lxml')
            # 移除脚本和样式
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            main_content = soup.get_text(separator=' ', strip=True)
            main_content = main_content[:self.extract_max_length]

        return {
            "main_content": main_content,
            "length": len(main_content),
        }

    def process(self, input_data: Any) -> Optional[Dict[str, Any]]:
        """
        处理输入数据，提取内容和链接

        Args:
            input_data: 输入数据字典，包含:
                - data_id: 数据 ID（优先）
                - html: HTML 内容（备选）
                - markdown: Markdown 内容（备选）
                - url: 页面 URL
                - depth: 深度
                - allowed_domains: 允许的域名列表
                - blocked_domains: 阻止的域名列表

        Returns:
            提取的结果字典，格式:
            {
                "success": bool,
                "data_id": str,
                "url": str,
                "title": str,
                "depth": int,
                "main_content": str,
                "headings": List[str],
                "keyword_hits": int,
                "content_hash": str,
                "text_snippet": str,
                "extracted_links": List[Dict],
                "extracted_links_count": int,
                "error": str (可选)
            }
        """
        # 获取输入参数
        data_id = input_data.get("data_id")
        html = input_data.get("html")
        markdown = input_data.get("markdown")
        url = input_data.get("url")
        depth = input_data.get("depth", 0)
        allowed_domains = input_data.get("allowed_domains", [])
        blocked_domains = input_data.get("blocked_domains", [])

        # 优化：统一获取 page_data（只获取一次）
        page_data = None
        if data_id and self.data_manager:
            page_data = self.data_manager.get_page(data_id)
            if page_data:
                html = page_data.html
                markdown = page_data.markdown
                url = page_data.url
            else:
                logger.error(f"[{self.name}] Page data not found for data_id: {data_id}")
                return {
                    "success": False,
                    "error": f"Page data not found for data_id: {data_id}",
                }

        if not html and not markdown:
            logger.error(f"[{self.name}] No content provided (html/markdown)")
            return {
                "success": False,
                "error": "No content provided",
            }

        if not url:
            logger.error(f"[{self.name}] No URL provided")
            return {
                "success": False,
                "error": "No URL provided",
            }

        logger.info(f"[{self.name}] Extracting from: {url}")

        try:
            # 优先使用 clean_content（trafilatura 提取的干净正文）
            if page_data and page_data.clean_content:
                # 使用 trafilatura 提取的干净正文
                main_content = page_data.clean_content
                logger.info(f"[{self.name}] Using trafilatura clean_content ({len(main_content)} chars)")
            else:
                # 回退到旧的提取方式
                content_data = self.extract_main_content(html, markdown)
                main_content = content_data["main_content"]

            # 提取标题（从 markdown 或 html）
            headings = extract_headings(markdown or html, self.extract_headings_max)

            # 统计关键词匹配（main_content 已经是干净的文本）
            keyword_hits = count_keyword_hits(main_content, self.keywords)

            # 计算内容哈希
            content_hash = compute_content_hash(main_content)

            # 获取文本摘要（直接使用 main_content，因为已经是干净的）
            text_snippet = get_text_snippet(main_content, self.extract_text_snippet_length)

            # 提取链接
            links = self.extract_links(
                html=html or "",
                markdown=markdown or "",
                base_url=url,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
            )

            # 转换链接为字典格式
            extracted_links = [
                {
                    "url": link.url,
                    "text": link.text,
                    "title": link.title,
                }
                for link in links
            ]

            # 组装结果
            result = {
                "success": True,
                "data_id": data_id,
                "url": url,
                "depth": depth,
                "title": input_data.get("title", ""),
                "main_content": main_content,
                "headings": headings,
                "keyword_hits": keyword_hits,
                "content_hash": content_hash,
                "text_snippet": text_snippet,
                "extracted_links": extracted_links,
                "extracted_links_count": len(extracted_links),
                "domain": get_domain(url),
            }

            # 存储提取的数据到数据管理器
            if self.data_manager and data_id:
                self.data_manager.store_extracted_data(data_id, result)

            logger.info(
                f"[{self.name}] Extraction complete for {url}: "
                f"{len(main_content)} chars, {keyword_hits} keyword hits, {len(extracted_links)} links"
            )

            return result

        except Exception as e:
            logger.error(f"[{self.name}] Error extracting from {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url,
            }

    def extract_link_urls(self, extracted_data: Dict[str, Any]) -> List[str]:
        """
        从提取的数据中获取 URL 列表

        Args:
            extracted_data: process 方法的输出数据

        Returns:
            URL 列表
        """
        if not extracted_data or not extracted_data.get("success"):
            return []

        return [link["url"] for link in extracted_data.get("extracted_links", [])]


def create_extractor_agent(
    name: str = "ExtractorAgent",
    config: Dict[str, Any] = None,
    data_manager: DataManager = None,
    keywords: List[str] = None,
) -> ExtractorAgent:
    """
    工厂函数：创建 ExtractorAgent 实例

    Args:
        name: Agent 名称
        config: 配置字典
        data_manager: 数据管理器
        keywords: 关键词列表

    Returns:
        ExtractorAgent 实例
    """
    return ExtractorAgent(
        name=name,
        config=config or {},
        data_manager=data_manager,
        keywords=keywords or [],
    )
