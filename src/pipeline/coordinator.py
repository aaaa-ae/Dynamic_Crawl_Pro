"""
任务协调器
负责协调多个 Agent 之间的任务流转
支持对话式协调机制
"""

import asyncio
import json
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from loguru import logger

# CAMEL imports for conversation coordinator
from camel.agents import ChatAgent
from camel.types import ModelType, ModelPlatformType
from camel.models import ModelFactory

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
    支持对话式协调机制
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

        # 对话式协调器
        self.conversation_coordinator: ChatAgent = None
        if llm_config.enable_conversation_coordinator and llm_config.api_key:
            self._init_conversation_coordinator()

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

    def _map_model_name(self, model_name: str) -> ModelType:
        """
        将模型名称映射到 CAMEL 的 ModelType 枚举

        Args:
            model_name: 模型名称字符串

        Returns:
            CAMEL ModelType 枚举值
        """
        model_mapping = {
            "gpt-4": ModelType.GPT_4,
            "gpt-4-turbo": ModelType.GPT_4_TURBO,
            "gpt-4-turbo-preview": ModelType.GPT_4_TURBO,
            "gpt-4o": ModelType.GPT_4O,
            "gpt-4o-mini": ModelType.GPT_4O_MINI,
            "gpt-3.5-turbo": ModelType.GPT_3_5_TURBO,
            "gpt-35-turbo": ModelType.GPT_3_5_TURBO,
        }

        # 移除可能的前缀
        clean_name = model_name.lower()
        if clean_name.startswith("openai/"):
            clean_name = clean_name[7:]

        return model_mapping.get(clean_name, ModelType.GPT_3_5_TURBO)

    def _init_conversation_coordinator(self):
        """初始化对话式协调器"""
        try:
            # 系统消息使用字符串格式
            system_message = """你是爬虫系统的任务协调器，负责综合多个 Agent 的结果，做出最终决策。

你的职责：
1. 协调 CrawlerAgent（爬取）、ExtractorAgent（提取）、QualityGateAgent（质量判断）的结果
2. 提供平衡的决策建议，而不是简单地覆盖原决策
3. 考虑整体爬取策略，避免重复和低效

决策原则：
- 优先尊重 QualityGateAgent 的专业判断
- 只有在有明显冲突时才进行调整
- 保持决策的一致性和可解释性

请严格按照 JSON 格式返回协调结果。"""

            # 获取配置
            model_name = self.llm_config.coordinator_model if self.llm_config.coordinator_model else "gpt-3.5-turbo"
            base_url = self.llm_config.base_url
            api_key = self.llm_config.api_key

            # 如果使用自定义 base_url，使用 OPENAI_COMPATIBLE_MODEL
            is_custom_base_url = "openai.com" not in base_url

            # 创建模型实例 (使用 url 参数而不是 base_url)
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL if is_custom_base_url else ModelPlatformType.OPENAI,
                model_type=self._map_model_name(model_name),
                api_key=api_key,
                url=base_url,
            )

            # 创建 ChatAgent
            self.conversation_coordinator = ChatAgent(
                system_message=system_message,
                model=model,
            )

            logger.info(f"[Coordinator] Conversation coordinator initialized: "
                       f"{model_name} @ {base_url}")
        except Exception as e:
            logger.warning(f"[Coordinator] Failed to initialize conversation coordinator: {e}")
            self.conversation_coordinator = None

    async def _conversation_based_decision(
        self,
        crawl_result: Dict[str, Any],
        extracted_data: Dict[str, Any],
        quality_result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        使用对话式协调器进行决策协调

        Args:
            crawl_result: CrawlerAgent 的结果
            extracted_data: ExtractorAgent 的结果
            quality_result: QualityGateAgent 的结果

        Returns:
            协调后的决策结果
        """
        if not self.conversation_coordinator:
            return None

        # 汇总信息
        url = extracted_data.get("url", "")
        title = extracted_data.get("title", "")
        keyword_hits = extracted_data.get("keyword_hits", 0)
        links_count = extracted_data.get("extracted_links_count", 0)
        headings = extracted_data.get("headings", [])[:5]

        quality_decision = quality_result.get("decision", "discard")
        quality_expand = quality_result.get("expand", False)
        quality_priority = quality_result.get("priority", "medium")
        quality_score = quality_result.get("relevance_score", 0.0)
        quality_reasons = quality_result.get("reasons", "")
        quality_method = quality_result.get("method", "unknown")

        # 构建协调提示词
        headings_str = ", ".join(headings) if headings else "N/A"

        prompt = f"""请综合以下信息，给出协调后的决策：

页面信息：
- URL: {url}
- 标题: {title}
- 关键词命中数: {keyword_hits}
- 提取链接数: {links_count}
- 标题列表: {headings_str}

QualityGateAgent 评估结果：
- 决策: {quality_decision}
- 是否扩展链接: {quality_expand}
- 优先级: {quality_priority}
- 相关性评分: {quality_score:.2f}
- 评估方法: {quality_method}
- 理由: {quality_reasons}

请严格按照以下 JSON 格式返回协调结果（不要添加任何其他文本）:
{{
    "final_decision": "keep" | "discard",
    "final_expand": true | false,
    "final_priority": "high" | "medium" | "low",
    "final_reasons": "简短说明协调理由",
    "should_override": true | false,
    "coordination_notes": "协调说明"
}}

字段说明：
- final_decision: 最终决策（keep=保留，discard=丢弃）
- final_expand: 是否继续爬取链接
- final_priority: 页面优先级
- final_reasons: 最终决策理由
- should_override: 是否覆盖原决策（true=有调整，false=维持原决策）
- coordination_notes: 协调过程的说明

注意：默认情况下应该尊重 QualityGateAgent 的专业判断，只在有明显问题时才进行调整。"""

        try:
            # 使用 asyncio.to_thread 包装同步调用
            response = await asyncio.to_thread(
                self.conversation_coordinator.step,
                prompt  # 直接传递字符串作为用户消息
            )

            # 提取响应内容
            content = response.msgs[0].content if response.msgs else ""

            # 尝试解析 JSON
            try:
                # 提取 JSON 部分（处理可能的额外文本）
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    result = json.loads(json_content)

                    logger.info(f"[Coordinator] Conversation coordination for {url}: "
                               f"override={result.get('should_override')}, "
                               f"notes={result.get('coordination_notes', '')}")

                    return {
                        "decision": result.get("final_decision", quality_decision),
                        "expand": result.get("final_expand", quality_expand),
                        "priority": result.get("final_priority", quality_priority),
                        "reasons": result.get("final_reasons", quality_reasons),
                        "should_override": result.get("should_override", False),
                        "coordination_notes": result.get("coordination_notes", ""),
                    }
                else:
                    return json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"[Coordinator] Failed to parse coordination response: {content}")
                return None

        except Exception as e:
            logger.error(f"[Coordinator] Conversation coordination error: {e}")
            # 返回 None，表示协调失败，将使用原始决策
            return None

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
        import json  # Import here for coordination

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

        # 步骤4: 对话式协调（如果启用）
        if self.conversation_coordinator:
            coordination_result = await self._conversation_based_decision(
                crawl_result, extracted_data, quality_result
            )

            if coordination_result:
                # 如果协调成功且需要覆盖
                if coordination_result.get("should_override", False):
                    logger.info(f"[Coordinator] Applying coordinated decision for {url}")
                    quality_result.update({
                        "decision": coordination_result["decision"],
                        "expand": coordination_result["expand"],
                        "priority": coordination_result["priority"],
                        "reasons": coordination_result["reasons"],
                        "method": "coordinated",
                    })
                else:
                    # 记录协调建议但保持原决策
                    logger.debug(f"[Coordinator] Coordination suggested keeping original decision for {url}")
                    quality_result["coordination_notes"] = coordination_result.get("coordination_notes", "")

        # 步骤5: 根据决策处理
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

        # 步骤6: 如果需要扩展，添加新任务
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
