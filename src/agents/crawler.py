"""
CrawlerAgent - 爬虫 Agent
负责使用 AsyncWebCrawler 异步爬取页面，带重试和指数退避
"""

import asyncio
from typing import Any, Dict, Optional
from crawl4ai import AsyncWebCrawler
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import BaseAgent, Message


class CrawlerAgent(BaseAgent):
    """
    爬虫 Agent
    使用 AsyncWebCrawler 异步爬取页面
    """

    def __init__(
        self,
        name: str = "CrawlerAgent",
        config: Dict[str, Any] = None,
        data_manager: Any = None
    ):
        """
        初始化 CrawlerAgent

        Args:
            name: Agent 名称
            config: 配置字典，包含:
                - timeout: 请求超时时间（秒）
                - max_retries: 最大重试次数
                - retry_delay: 重试延迟基数（秒）
                - browser_type: 浏览器类型
                - headless: 是否无头模式
            data_manager: 数据管理器实例
        """
        super().__init__(name, config)
        self.data_manager = data_manager

        # 配置参数
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.browser_type = config.get("browser_type", "chromium")
        self.headless = config.get("headless", True)

        # AsyncWebCrawler 实例池
        self._crawler = None

        logger.info(f"[{self.name}] Initialized with timeout={self.timeout}, browser={self.browser_type}")

    async def _get_crawler(self) -> AsyncWebCrawler:
        """
        获取或创建 AsyncWebCrawler 实例

        Returns:
            AsyncWebCrawler 实例
        """
        if self._crawler is None:
            self._crawler = AsyncWebCrawler(
                headless=self.headless,
                browser_type=self.browser_type,
            )
        return self._crawler

    def can_process(self, input_data: Any) -> bool:
        """
        检查是否能够处理该输入

        Args:
            input_data: 输入数据，应为包含 url 的字典或 URL 字符串

        Returns:
            是否能够处理
        """
        if isinstance(input_data, str):
            return input_data.startswith(("http://", "https://"))
        elif isinstance(input_data, dict):
            return "url" in input_data
        return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def _crawl_single_page(self, url: str) -> Dict[str, Any]:
        """
        爬取单个页面（带重试）

        Args:
            url: 要爬取的 URL

        Returns:
            包含爬取结果的字典
        """
        crawler = await self._get_crawler()

        result = await crawler.arun(
            url=url,
            timeout=self.timeout,
            magic=True,  # 启用智能提取
        )

        if result.success:
            # 提取 markdown 文本（crawl4ai 返回的是 MarkdownGenerationResult 对象）
            # 该对象有 .markdown 属性包含实际文本
            markdown_obj = result.markdown
            if hasattr(markdown_obj, 'markdown'):
                markdown_text = markdown_obj.markdown
            else:
                markdown_text = str(markdown_obj)

            return {
                "success": True,
                "html": result.html,
                "markdown": markdown_text,
                "title": result.metadata.get("title", ""),
                "status_code": result.status_code,
                "url": url,
            }
        else:
            return {
                "success": False,
                "error": result.error_message,
                "status_code": result.status_code,
                "url": url,
            }

    async def process(self, input_data: Any) -> Optional[Dict[str, Any]]:
        """
        处理爬取请求

        Args:
            input_data: 输入数据，可以是 URL 字符串或包含 url 的字典

        Returns:
            包含爬取结果的字典，格式:
            {
                "success": bool,
                "data_id": str,  # 数据管理器中的 ID
                "url": str,
                "title": str,
                "status_code": int,
                "error": str (可选)
            }
        """
        # 解析输入
        if isinstance(input_data, str):
            url = input_data
            depth = input_data.get("depth", 0) if isinstance(input_data, dict) else 0
        else:
            url = input_data.get("url")
            depth = input_data.get("depth", 0)

        if not url:
            logger.error(f"[{self.name}] Invalid input: no URL found")
            return None

        logger.info(f"[{self.name}] Crawling: {url} (depth={depth})")

        try:
            # 执行爬取（带重试）
            crawl_result = await self._crawl_single_page(url)

            if not crawl_result["success"]:
                logger.error(f"[{self.name}] Failed to crawl {url}: {crawl_result.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "url": url,
                    "error": crawl_result.get("error", "Unknown error"),
                    "status_code": crawl_result.get("status_code", 0),
                }

            # 存储到数据管理器
            if self.data_manager:
                data_id = self.data_manager.store_page(
                    url=url,
                    html=crawl_result["html"],
                    markdown=crawl_result["markdown"],
                    title=crawl_result["title"],
                    status_code=crawl_result["status_code"],
                )
            else:
                # 如果没有数据管理器，直接返回内容（不推荐，用于测试）
                data_id = None
                logger.warning(f"[{self.name}] No data_manager provided, returning content directly")

            logger.info(f"[{self.name}] Successfully crawled: {url}")

            return {
                "success": True,
                "data_id": data_id,
                "url": url,
                "title": crawl_result["title"],
                "status_code": crawl_result["status_code"],
            }

        except Exception as e:
            logger.error(f"[{self.name}] Exception while crawling {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "status_code": 0,
            }

    async def crawl_batch(self, urls: list) -> list:
        """
        批量爬取多个 URL

        Args:
            urls: URL 列表

        Returns:
            爬取结果列表
        """
        logger.info(f"[{self.name}] Starting batch crawl of {len(urls)} URLs")

        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(self.config.get("concurrency", 5))

        async def crawl_with_semaphore(url):
            async with semaphore:
                return await self.process(url)

        # 并发爬取
        tasks = [crawl_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        processed_results = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.error(f"[{self.name}] Exception for {url}: {result}")
                processed_results.append({
                    "success": False,
                    "url": url,
                    "error": str(result),
                })
            else:
                processed_results.append(result)

        success_count = sum(1 for r in processed_results if r and r.get("success"))
        logger.info(f"[{self.name}] Batch crawl completed: {success_count}/{len(urls)} successful")

        return processed_results

    async def cleanup(self):
        """清理资源"""
        if self._crawler:
            await self._crawler.close()
            self._crawler = None
            logger.info(f"[{self.name}] Cleanup completed")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()


def create_crawler_agent(
    name: str = "CrawlerAgent",
    config: Dict[str, Any] = None,
    data_manager: Any = None
) -> CrawlerAgent:
    """
    工厂函数：创建 CrawlerAgent 实例

    Args:
        name: Agent 名称
        config: 配置字典
        data_manager: 数据管理器

    Returns:
        CrawlerAgent 实例
    """
    return CrawlerAgent(
        name=name,
        config=config or {},
        data_manager=data_manager,
    )
