"""
任务协调器（修正版）
- URL 入队去重（canonical + queued_urls + visited_urls）
- 扩链过滤（只入队“像正文页”的链接）
- ✅ 评估级去重（content_hash + keyword + context_signature）
  注意：不会误杀“同内容但不同关键词/不同上下文”的评估
- 将 AgentConfig 新字段传递给 QualityGateAgent
"""

import asyncio
import json
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from loguru import logger
from urllib.parse import urlparse, parse_qs
import re

from camel.agents import ChatAgent
from camel.types import ModelType, ModelPlatformType
from camel.models import ModelFactory

from ..agents import (
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
from ..utils.url_utils import normalize_url
from ..config import CrawlConfig, LLMConfig, AgentConfig


@dataclass
class CrawlTask:
    url: str
    depth: int
    priority: str = "medium"
    source_url: str = None


class TaskCoordinator:
    def __init__(self, crawl_config: CrawlConfig, llm_config: LLMConfig, agent_config: AgentConfig):
        self.crawl_config = crawl_config
        self.llm_config = llm_config
        self.agent_config = agent_config

        # Data + output
        self.data_manager = DataManager(
            cache_dir=crawl_config.cache_dir,
            enable_persistence=crawl_config.enable_cache,
        )
        self.output_writer = OutputWriter(crawl_config.output_file)

        # Agents
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

        # Slide context
        context = getattr(crawl_config, "context", {}) or {}

        # ✅ Pass AgentConfig knobs into QualityGate
        self.quality_gate_agent = create_quality_gate_agent(
            config={
                "quality_decision_threshold": agent_config.quality_decision_threshold,
                "context": context,

                "enable_expand": getattr(agent_config, "enable_expand", True),
                "expand_min_score": getattr(agent_config, "expand_min_score", 0.45),
                "expand_priority_allow": getattr(agent_config, "expand_priority_allow", ["high", "medium"]),

                "llm_on_zero_hits": getattr(agent_config, "llm_on_zero_hits", True),
                "zero_hits_llm_min_rule_score": getattr(agent_config, "zero_hits_llm_min_rule_score", 0.15),
                "zero_hits_llm_min_chars": getattr(agent_config, "zero_hits_llm_min_chars", 250),

                "fast_discard_min_chars": getattr(agent_config, "fast_discard_min_chars", 80),
                "high_hit_threshold": getattr(agent_config, "high_hit_threshold", 3),
                "llm_hit_range": getattr(agent_config, "llm_hit_range", [1, 2]),
            },
            keywords=crawl_config.keywords,
            llm_config={
                "enabled": llm_config.enabled,
                "api_key": llm_config.api_key,
                "model": llm_config.model,
                "base_url": llm_config.base_url,
                "temperature": llm_config.temperature,
                "max_tokens": llm_config.max_tokens,
                "context": context,
            },
        )

        # Optional conversation coordinator (disabled by default)
        self.conversation_coordinator: Optional[ChatAgent] = None
        if getattr(llm_config, "enable_conversation_coordinator", False) and llm_config.api_key:
            self._init_conversation_coordinator()

        # Queue & dedup
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.visited_urls: Set[str] = set()   # processed
        self.queued_urls: Set[str] = set()    # enqueued

        # ✅ IMPORTANT: evaluation-level dedup keys
        # Key = (content_hash, keyword_signature, context_signature)
        self.seen_eval_keys: Set[Tuple[str, str, str]] = set()

        self._dedup_lock = asyncio.Lock()

        # Domain quota
        self.domain_page_counts: Dict[str, int] = defaultdict(int)

        # Stats
        self.stats = {
            "total_tasks": 0,
            "processed_tasks": 0,
            "kept_pages": 0,
            "discarded_pages": 0,
            "expanded_links": 0,
            "start_time": None,
            "end_time": None,
        }

        self._running = False

    # -------------------------
    # Conversation coordinator
    # -------------------------
    def _init_conversation_coordinator(self):
        try:
            system_message = "你是一个任务协调专家，负责在爬虫-抽取-质量判断之间做最终仲裁。只返回 JSON。"
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                api_key=self.llm_config.api_key,
                url=self.llm_config.base_url,
            )
            self.conversation_coordinator = ChatAgent(system_message=system_message, model=model)
            logger.info("[Coordinator] Conversation coordinator initialized")
        except Exception as e:
            logger.warning(f"[Coordinator] Failed to init conversation coordinator: {e}")
            self.conversation_coordinator = None

    # -------------------------
    # URL canonical + filter
    # -------------------------
    def _canonical(self, url: str, base_url: Optional[str] = None) -> Optional[str]:
        return normalize_url(url, base_url=base_url)

    def _should_enqueue_url(self, url: str) -> bool:
        """
        轻量过滤：只让“像正文页”的链接进入队列
        你可以按站点继续加规则。
        """
        if not url:
            return False

        parsed = urlparse(url)
        path = (parsed.path or "").lower()

        # Block media/attachments
        blocked_ext = (
            ".webp", ".jpg", ".jpeg", ".png", ".gif", ".svg",
            ".pdf", ".doc", ".docx", ".mp4", ".mp3", ".zip", ".rar",
        )
        if any(path.endswith(ext) for ext in blocked_ext):
            return False

        # Root path (usually homepage)
        if path in ("", "/"):
            return False

        # Filter obvious index pages
        if path.endswith(("/index.htm", "/index.html", "/index.shtml")):
            return False

        # Filter obvious list/channel patterns (contains is more reliable than endswith)
        bad_tokens = ("node_", "list_", "channel_", "column_", "search", "login", "register")
        if any(tok in path for tok in bad_tokens):
            return False

        # Filter tracking/pagination queries (extra safety)
        qs = parse_qs(parsed.query or "")
        for k in qs.keys():
            lk = k.lower()
            if lk.startswith("utm_") or lk in {"spm", "from", "ref", "source", "share"}:
                return False
            if lk in {"page", "p", "pagesize"}:
                return False

        # Strong signals: html-like
        if path.endswith((".shtml", ".html", ".htm")):
            return True

        # Strong signals: date-like
        if re.search(r"/20\d{2}[-/]\d{1,2}[-/]\d{1,2}/", path) or re.search(r"/20\d{6}/", path):
            return True

        # Weak signal: deeper path more likely article
        depth = path.strip("/").count("/")
        return depth >= 2

    async def _mark_queued_if_new(self, url: str) -> bool:
        async with self._dedup_lock:
            if url in self.queued_urls or url in self.visited_urls:
                return False
            self.queued_urls.add(url)
            return True

    # -------------------------
    # Crawl policy
    # -------------------------
    def should_crawl(self, url: str, depth: int) -> bool:
        if not url:
            return False
        if depth > self.crawl_config.max_depth:
            return False
        if self.stats["processed_tasks"] >= self.crawl_config.max_pages:
            return False
        domain = get_domain(url)
        if domain and self.domain_page_counts[domain] >= self.crawl_config.max_pages_per_domain:
            return False
        return True

    async def add_seed_urls(self, urls: List[str]):
        added = 0
        for raw in urls:
            cu = self._canonical(raw)
            if not cu:
                continue
            if not self.should_crawl(cu, 0):
                continue
            if not await self._mark_queued_if_new(cu):
                continue

            await self.task_queue.put(CrawlTask(url=cu, depth=0, priority="high"))
            self.stats["total_tasks"] += 1
            added += 1

        logger.info(f"[Coordinator] Added {added} seed URLs to queue (deduped)")

    # -------------------------
    # Optional arbitration (rarely needed)
    # -------------------------
    async def _conversation_based_decision(self, extracted_data: Dict[str, Any], quality_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.conversation_coordinator:
            return None
        try:
            payload = {
                "url": extracted_data.get("url", ""),
                "title": extracted_data.get("title", ""),
                "keyword_hits": extracted_data.get("keyword_hits", 0),
                "snippet": (extracted_data.get("text_snippet", "") or "")[:400],
                "quality_decision": quality_result,
            }
            msg = (
                "请综合信息决定是否覆盖 keep/discard 与 expand。只返回 JSON："
                "{should_override, decision, expand, priority, reasons}\n\n"
                f"{json.dumps(payload, ensure_ascii=False)}"
            )
            resp = await asyncio.to_thread(self.conversation_coordinator.step, msg)
            content = resp.msgs[0].content if resp.msgs else ""
            s = content.find("{")
            e = content.rfind("}") + 1
            if s >= 0 and e > s:
                return json.loads(content[s:e])
            return json.loads(content)
        except Exception as e:
            logger.warning(f"[Coordinator] Conversation arbitration failed: {e}")
            return None

    # -------------------------
    # ✅ Evaluation dedup key
    # -------------------------
    def _current_keyword_signature(self) -> str:
        """
        你现在是一轮只跑一个关键词（crawl_config.keywords=[keyword]），这里取第一项。
        如果未来变成多关键词联合爬，这里会把 keywords 合并成稳定签名。
        """
        kws = self.crawl_config.keywords or []
        if not kws:
            return ""
        if len(kws) == 1:
            return kws[0]
        return "|".join(sorted(kws))

    def _context_signature(self) -> str:
        """
        用 slide_title + slide_reason 作为上下文签名。
        如果 context 为空，则返回空串。
        """
        ctx = getattr(self.crawl_config, "context", {}) or {}
        title = (ctx.get("slide_title") or "").strip()
        reason = (ctx.get("slide_reason") or "").strip()
        if not title and not reason:
            return ""
        return f"{title}||{reason}"

    async def _eval_dedup_hit(self, content_hash: str) -> bool:
        """
        返回 True 表示：同 content_hash 在同 keyword + 同 context 下已经评估过，可跳过评估。
        返回 False 表示：需要评估（不会误杀不同关键词/不同上下文）。
        """
        if not content_hash:
            return False

        key_sig = self._current_keyword_signature()
        ctx_sig = self._context_signature()
        dedup_key = (content_hash, key_sig, ctx_sig)

        async with self._dedup_lock:
            if dedup_key in self.seen_eval_keys:
                return True
            self.seen_eval_keys.add(dedup_key)
            return False

    # -------------------------
    # Core processing
    # -------------------------
    async def _process_single_task(self, task: CrawlTask) -> Optional[PageRecord]:
        url = task.url
        depth = task.depth

        if not self.should_crawl(url, depth):
            return None

        # visited dedup (worker-safe)
        async with self._dedup_lock:
            if url in self.visited_urls:
                return None
            self.visited_urls.add(url)

        self.stats["processed_tasks"] += 1

        domain = get_domain(url)
        if domain:
            self.domain_page_counts[domain] += 1

        logger.info(f"[Coordinator] Processing task {self.stats['processed_tasks']}/{self.stats['total_tasks']}: {url}")

        # 1) Crawl
        crawl_result = await self.crawler_agent.process({"url": url, "depth": depth})
        if not crawl_result or not crawl_result.get("success"):
            logger.error(f"[Coordinator] Crawl failed for {url}")
            return None

        data_id = crawl_result.get("data_id")

        # 2) Extract
        extracted_data = self.extractor_agent.process({
            "data_id": data_id,
            "url": url,
            "depth": depth,
            "title": crawl_result.get("title", ""),
            "allowed_domains": self.crawl_config.allowed_domains,
            "blocked_domains": self.crawl_config.blocked_domains,
        })
        if not extracted_data or not extracted_data.get("success"):
            logger.error(f"[Coordinator] Extraction failed for {url}")
            return None

        content_hash = extracted_data.get("content_hash") or ""

        # ✅ 2.5) 评估级去重：仅当 (hash + keyword + context) 全相同才跳过
        # 这样不会误杀“同内容不同关键词/不同上下文”的评估。
        if content_hash:
            hit = await self._eval_dedup_hit(content_hash)
            if hit:
                # 注意：这里跳过的是“本次评估与输出”，避免重复 LLM/重复写入
                logger.info(
                    f"[Coordinator] Eval dedup hit (same content+keyword+context), skip: {url} "
                    f"(hash={content_hash[:8]})"
                )
                self.stats["discarded_pages"] += 1
                return None

        # 3) Quality gate
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

        # Optional conversation override
        if self.conversation_coordinator:
            coord = await self._conversation_based_decision(extracted_data, quality_result)
            if coord and coord.get("should_override", False):
                quality_result.update({
                    "decision": coord.get("decision", quality_result.get("decision")),
                    "expand": coord.get("expand", quality_result.get("expand")),
                    "priority": coord.get("priority", quality_result.get("priority")),
                    "reasons": coord.get("reasons", quality_result.get("reasons")),
                    "method": "coordinated",
                })

        decision = quality_result.get("decision", "discard")
        expand = bool(quality_result.get("expand", False))
        priority = quality_result.get("priority", "medium")
        reasons = quality_result.get("reasons", "")
        score = float(quality_result.get("relevance_score", 0.0) or 0.0)

        record: Optional[PageRecord] = None

        # 4) Output only if keep
        if decision == "keep":
            self.stats["kept_pages"] += 1
            record = PageRecord(
                url=url,
                depth=depth,
                domain=domain or "",
                title=extracted_data.get("title", ""),
                keyword_hits=extracted_data.get("keyword_hits", 0),
                content_hash=content_hash,
                text_snippet=extracted_data.get("text_snippet", ""),
                extracted_links_count=extracted_data.get("extracted_links_count", 0),
                headings=extracted_data.get("headings", []),
                decision=decision,
                priority=priority,
                reasons=reasons,
                relevance_score=score,
                expand=expand,
                main_content=extracted_data.get("main_content", ""),
            )
            self.output_writer.write_record(record)
        else:
            self.stats["discarded_pages"] += 1

        # 5) Expand (filtered + deduped)
        if expand:
            new_urls = self.extractor_agent.extract_link_urls(extracted_data)
            new_added = 0

            for raw in new_urls:
                cu = self._canonical(raw, base_url=url)
                if not cu:
                    continue
                if not self._should_enqueue_url(cu):
                    continue
                if not self.should_crawl(cu, depth + 1):
                    continue
                if not await self._mark_queued_if_new(cu):
                    continue

                await self.task_queue.put(CrawlTask(
                    url=cu,
                    depth=depth + 1,
                    priority=priority,
                    source_url=url,
                ))
                self.stats["total_tasks"] += 1
                new_added += 1

            if new_added:
                self.stats["expanded_links"] += new_added
                logger.info(f"[Coordinator] Added {new_added} new tasks from {url} (filtered+deduped)")

        return record

    async def _worker(self, worker_id: int):
        logger.info(f"[Coordinator] Worker {worker_id} started")
        while self._running:
            try:
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if self.stats["processed_tasks"] >= self.crawl_config.max_pages:
                        break
                    continue

                await self._process_single_task(task)
                self.task_queue.task_done()

            except Exception as e:
                logger.error(f"[Coordinator] Worker {worker_id} error: {e}")

        logger.info(f"[Coordinator] Worker {worker_id} stopped")

    async def run(self):
        logger.info("[Coordinator] Starting crawl execution")
        self.stats["start_time"] = datetime.now()
        self._running = True

        workers = [asyncio.create_task(self._worker(i)) for i in range(self.crawl_config.concurrency)]

        try:
            await self.task_queue.join()
            self._running = False
            await asyncio.gather(*workers, return_exceptions=True)
        finally:
            self.stats["end_time"] = datetime.now()

        self.output_writer.flush()
        self._log_stats()

    def _log_stats(self):
        elapsed = None
        if self.stats["start_time"] and self.stats["end_time"]:
            elapsed = self.stats["end_time"] - self.stats["start_time"]

        logger.info("=" * 60)
        logger.info("[Coordinator] Crawl Statistics:")
        logger.info(f"  Total tasks: {self.stats['total_tasks']}")
        logger.info(f"  Processed: {self.stats['processed_tasks']}")
        logger.info(f"  Kept pages: {self.stats['kept_pages']}")
        logger.info(f"  Discarded: {self.stats['discarded_pages']}")
        logger.info(f"  Expanded links: {self.stats['expanded_links']}")
        logger.info(f"  Unique domains: {len(self.domain_page_counts)}")
        logger.info(f"  Visited URLs: {len(self.visited_urls)}")
        logger.info(f"  Queued URLs: {len(self.queued_urls)}")
        logger.info(f"  Eval dedup keys: {len(self.seen_eval_keys)}")
        if elapsed:
            sec = max(elapsed.total_seconds(), 1.0)
            logger.info(f"  Elapsed time: {elapsed.total_seconds():.2f}s")
            logger.info(f"  Pages per second: {self.stats['processed_tasks'] / sec:.2f}")
        logger.info(f"  Output file: {self.crawl_config.output_file}")
        logger.info("=" * 60)

    async def cleanup(self):
        logger.info("[Coordinator] Cleaning up...")
        await self.crawler_agent.cleanup()
        await self.quality_gate_agent.cleanup()
        self.output_writer.close()
        logger.info("[Coordinator] Cleanup completed")
