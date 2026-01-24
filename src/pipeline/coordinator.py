"""
任务协调器
负责协调多个 Agent 之间的任务流转
"""

import asyncio
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from loguru import logger

from ..agents import (
    CrawlerAgent,
    ExtractorAgent,
    QualityGateAgent,
    create_crawler_agent,
    create_extractor_agent,
    create_quality_gate_agent,
)
from ..utils import (
    DataManager,
    OutputWriter,
    PageRecord,
    get_domain,
)
from ..config import CrawlConfig, LLMConfig, AgentConfig


@dataclass
class CrawlTask:
    """爬取任务"""
    url: str
    depth: int
    priority: str = "medium"
    source_url: str = None


class TaskCoordinator:
    """
    任务协调器
    协调 CrawlerAgent、ExtractorAgent 和 QualityGateAgent 之间的任务流转
    """

    def __init__(
        self,
        crawl_config: CrawlConfig,
        llm_config: LLMConfig,
        agent_config: AgentConfig,
    ):
        """
        初始化任务协调器

        Args:
            crawl_config: 爬虫配置
            llm_config: LLM 配置
            agent_config: Agent 配置
        """
        self.crawl_config = crawl_config
        self.llm_config = llm_config
        self.agent_config = agent_config

        # 数据管理器
        self.data_manager = DataManager(
            cache_dir=crawl_config.cache_dir,
            enable_persistence=crawl_config.enable_cache,
        )

        # 输出写入器
        self.output_writer = OutputWriter(crawl_config.output_file)

        # 创建 Agents
        self.crawler_agent = create_crawler_agent(
            config={
                "timeout": crawl_config.request_timeout,
                "max_retries": crawl_config.max_retries,
                "concurrency": crawl_config.concurrency,
            },
            data_manager=self.data_manager,
        )

        self.extractor_agent = create_extractor_agent(
            config={
                "extract_max_length": agent_config.extract_max_length,
                "extract_text_snippet_length": agent_config.extract_text_snippet_length,
                "extract_headings_max": agent_config.extract_headings_max,
                "extract_links_max": agent_config.extract_links_max,
                "filter_same_domain": agent_config.filter_same_domain,
            },
            data_manager=self.data_manager,
            keywords=crawl_config.keywords,
        )

        self.quality_gate_agent = create_quality_gate_agent(
            config={
                "fast_filter_min_length": llm_config.fast_filter_min_length,
                "fast_filter_min_keyword_hits": llm_config.fast_filter_min_keyword_hits,
                "quality_decision_threshold": agent_config.quality_decision_threshold,
            },
            keywords=crawl_config.keywords,
            llm_config={
                "enabled": llm_config.enabled,
                "api_key": llm_config.api_key,
                "model": llm_config.model,
                "base_url": llm_config.base_url,
                "temperature": llm_config.temperature,
                "max_tokens": llm_config.max_tokens,
            },
        )

        # 任务队列
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.visited_urls: Set[str] = set()
        self.domain_page_counts: Dict[str, int] = defaultdict(int)

        # 统计
        self.stats = {
            "total_tasks": 0,
            "processed_tasks": 0,
            "kept_pages": 0,
            "discarded_pages": 0,
            "expanded_links": 0,
            "start_time": None,
            "end_time": None,
        }

        # 运行标志
        self._running = False

        logger.info("[Coordinator] Initialized with all agents")

    def should_crawl(self, url: str, depth: int) -> bool:
        """
        判断是否应该爬取该 URL

        Args:
            url: 目标 URL
            depth: 深度

        Returns:
            是否应该爬取
        """
        # 检查是否已访问
        if url in self.visited_urls:
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

        # 检查深度限制
        if depth > self.crawl_config.max_depth:
            return False

        # 检查总页面数限制
        if self.stats["processed_tasks"] >= self.crawl_config.max_pages:
            return False

        # 检查域名页面数限制
        domain = get_domain(url)
        if domain:
            if self.domain_page_counts[domain] >= self.crawl_config.max_pages_per_domain:
                return False

        return True

    async def add_seed_urls(self, urls: List[str]):
        """
        添加种子 URL 到任务队列

        Args:
            urls: URL 列表
        """
        for url in urls:
            task = CrawlTask(url=url, depth=0, priority="high")
            await self.task_queue.put(task)
            self.stats["total_tasks"] += 1

        logger.info(f"[Coordinator] Added {len(urls)} seed URLs to queue")

    async def _process_single_task(self, task: CrawlTask) -> Optional[PageRecord]:
        """
        处理单个爬取任务

        Args:
            task: 爬取任务

        Returns:
            PageRecord（如果决定保存）
        """
        url = task.url
        depth = task.depth

        # 检查是否应该爬取
        if not self.should_crawl(url, depth):
            logger.debug(f"[Coordinator] Skipping: {url} (depth={depth})")
            return None

        # 标记为已访问
        self.visited_urls.add(url)
        self.stats["processed_tasks"] += 1

        # 更新域名计数
        domain = get_domain(url)
        if domain:
            self.domain_page_counts[domain] += 1

        logger.info(f"[Coordinator] Processing task {self.stats['processed_tasks']}/{self.stats['total_tasks']}: {url}")

        # 步骤1: 爬取页面
        crawl_result = await self.crawler_agent.process({"url": url, "depth": depth})
        if not crawl_result or not crawl_result.get("success"):
            logger.error(f"[Coordinator] Crawl failed for {url}")
            return None

        data_id = crawl_result.get("data_id")

        # 步骤2: 提取内容和链接
        extract_input = {
            "data_id": data_id,
            "url": url,
            "depth": depth,
            "title": crawl_result.get("title", ""),
            "allowed_domains": self.crawl_config.allowed_domains,
            "blocked_domains": self.crawl_config.blocked_domains,
        }
        extracted_data = self.extractor_agent.process(extract_input)
        if not extracted_data or not extracted_data.get("success"):
            logger.error(f"[Coordinator] Extraction failed for {url}")
            return None

        # 步骤3: 质量判断
        quality_result = await self.quality_gate_agent.process(extracted_data)
        if not quality_result:
            quality_result = {
                "decision": "discard",
                "expand": False,
                "priority": "low",
                "reasons": "Quality evaluation failed",
                "relevance_score": 0.0,
                "method": "error",
            }

        # 步骤4: 根据决策处理
        decision = quality_result.get("decision", "discard")
        expand = quality_result.get("expand", False)
        priority = quality_result.get("priority", "medium")
        reasons = quality_result.get("reasons", "")

        if decision == "keep":
            self.stats["kept_pages"] += 1
            # 创建 PageRecord
            record = PageRecord(
                url=url,
                depth=depth,
                domain=domain or "",
                title=extracted_data.get("title", ""),
                keyword_hits=extracted_data.get("keyword_hits", 0),
                content_hash=extracted_data.get("content_hash", ""),
                text_snippet=extracted_data.get("text_snippet", ""),
                extracted_links_count=extracted_data.get("extracted_links_count", 0),
                headings=extracted_data.get("headings", []),
                decision=decision,
                priority=priority,
                reasons=reasons,
                main_content=extracted_data.get("main_content", ""),
            )
            self.output_writer.write_record(record)
        else:
            self.stats["discarded_pages"] += 1
            record = None

        # 步骤5: 如果需要扩展，添加新任务
        if expand:
            new_urls = self.extractor_agent.extract_link_urls(extracted_data)
            new_tasks_added = 0

            for new_url in new_urls:
                # 检查是否应该爬取
                if self.should_crawl(new_url, depth + 1):
                    new_task = CrawlTask(
                        url=new_url,
                        depth=depth + 1,
                        priority=priority,
                        source_url=url,
                    )
                    await self.task_queue.put(new_task)
                    self.stats["total_tasks"] += 1
                    new_tasks_added += 1

            if new_tasks_added > 0:
                self.stats["expanded_links"] += new_tasks_added
                logger.info(f"[Coordinator] Added {new_tasks_added} new tasks from {url}")

        return record

    async def _worker(self, worker_id: int):
        """
        工作协程，从队列中取出任务并处理

        Args:
            worker_id: 工作器 ID
        """
        logger.info(f"[Coordinator] Worker {worker_id} started")

        while self._running:
            try:
                # 获取任务（带超时，避免无限等待）
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # 队列为空，检查是否应该继续
                    if self.stats["processed_tasks"] >= self.crawl_config.max_pages:
                        break
                    continue

                # 处理任务
                await self._process_single_task(task)

                # 标记任务完成
                self.task_queue.task_done()

            except Exception as e:
                logger.error(f"[Coordinator] Worker {worker_id} error: {e}")

        logger.info(f"[Coordinator] Worker {worker_id} stopped")

    async def run(self):
        """
        运行协调器，执行爬取任务
        """
        logger.info("[Coordinator] Starting crawl execution")
        self.stats["start_time"] = datetime.now()
        self._running = True

        # 创建工作协程
        workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.crawl_config.concurrency)
        ]

        try:
            # 等待所有任务完成
            await self.task_queue.join()

            # 停止工作协程
            self._running = False

            # 等待工作协程停止
            await asyncio.gather(*workers, return_exceptions=True)

        finally:
            self.stats["end_time"] = datetime.now()

        # 写入输出文件
        self.output_writer.flush()

        # 打印统计信息
        self._log_stats()

    def _log_stats(self):
        """打印统计信息"""
        elapsed = (
            self.stats["end_time"] - self.stats["start_time"]
            if self.stats["end_time"] and self.stats["start_time"]
            else None
        )

        logger.info("=" * 60)
        logger.info("[Coordinator] Crawl Statistics:")
        logger.info(f"  Total tasks: {self.stats['total_tasks']}")
        logger.info(f"  Processed: {self.stats['processed_tasks']}")
        logger.info(f"  Kept pages: {self.stats['kept_pages']}")
        logger.info(f"  Discarded: {self.stats['discarded_pages']}")
        logger.info(f"  Expanded links: {self.stats['expanded_links']}")
        logger.info(f"  Unique domains: {len(self.domain_page_counts)}")
        if elapsed:
            logger.info(f"  Elapsed time: {elapsed.total_seconds():.2f}s")
            logger.info(
                f"  Pages per second: "
                f"{self.stats['processed_tasks'] / max(elapsed.total_seconds(), 1):.2f}"
            )
        logger.info(f"  Output file: {self.crawl_config.output_file}")
        logger.info("=" * 60)

    async def cleanup(self):
        """清理资源"""
        logger.info("[Coordinator] Cleaning up...")

        await self.crawler_agent.cleanup()
        await self.quality_gate_agent.cleanup()

        self.data_manager.clear_cache()

        logger.info("[Coordinator] Cleanup completed")

    def get_agent_metrics(self) -> Dict[str, Any]:
        """
        获取所有 Agent 的统计指标

        Returns:
            指标字典
        """
        return {
            "crawler": self.crawler_agent.get_metrics(),
            "extractor": self.extractor_agent.get_metrics(),
            "quality_gate": self.quality_gate_agent.get_metrics(),
        }


def create_coordinator(
    crawl_config: CrawlConfig,
    llm_config: LLMConfig,
    agent_config: AgentConfig,
) -> TaskCoordinator:
    """
    工厂函数：创建 TaskCoordinator 实例

    Args:
        crawl_config: 爬虫配置
        llm_config: LLM 配置
        agent_config: Agent 配置

    Returns:
        TaskCoordinator 实例
    """
    return TaskCoordinator(
        crawl_config=crawl_config,
        llm_config=llm_config,
        agent_config=agent_config,
    )
