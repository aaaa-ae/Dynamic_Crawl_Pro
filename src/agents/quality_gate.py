"""
QualityGateAgent - 质量判断 Agent
使用 LLM 对页面进行质量判断和决策
使用 CAMEL 框架集成

【2026-02 修订版】
- 允许 expand_links=true，使 Coordinator 能进行深度扩链
- keyword_hits==0 场景：规则粗筛后可调用 LLM（预算可控）
- 更强可观测日志：decision/priority/score/expand/len/hits
"""

import json
import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger

# CAMEL imports
from camel.agents import ChatAgent
from camel.types import ModelType, ModelPlatformType
from camel.models import ModelFactory

from .base import BaseAgent
from ..utils import (
    calculate_content_quality_score,
    truncate_text,
)


class QualityGateAgent(BaseAgent):
    """
    质量判断 Agent
    基于规则和 LLM 对页面进行质量评估和决策
    使用 CAMEL 框架进行 LLM 集成
    """

    def __init__(
        self,
        name: str = "QualityGateAgent",
        config: Dict[str, Any] = None,
        keywords: List[str] = None,
        llm_config: Dict[str, Any] = None,
    ):
        super().__init__(name, config)
        self.keywords = keywords or []
        self.llm_config = llm_config or {}

        # =========================
        # 过滤与路由相关阈值
        # =========================
        self.fast_discard_min_chars = self.config.get("fast_discard_min_chars", 80)

        # keyword_hits 路由
        self.high_hit_threshold = self.config.get("high_hit_threshold", 3)  # >=3 直接高相关 keep
        self.llm_hit_range = self.config.get("llm_hit_range", [1, 2])       # hits=1/2 调 LLM

        # hits==0 是否允许调 LLM（强烈建议 True）
        self.llm_on_zero_hits = self.config.get("llm_on_zero_hits", True)
        self.zero_hits_llm_min_rule_score = self.config.get("zero_hits_llm_min_rule_score", 0.15)
        self.zero_hits_llm_min_chars = self.config.get("zero_hits_llm_min_chars", 250)

        # 决策阈值（fallback）
        self.quality_decision_threshold = self.config.get("quality_decision_threshold", 0.35)

        # =========================
        # 扩链策略
        # =========================
        # expand 由 Gate 决定，Coordinator 依赖 expand 才会入队出链
        self.enable_expand = self.config.get("enable_expand", True)
        self.expand_min_score = self.config.get("expand_min_score", 0.45)
        self.expand_priority_allow = set(self.config.get("expand_priority_allow", ["high", "medium"]))

        # =========================
        # 幻灯片上下文
        # =========================
        self.slide_context = self.config.get("context", {}) or self.llm_config.get("context", {})
        self.slide_title = self.slide_context.get("slide_title", "")
        self.slide_reason = self.slide_context.get("slide_reason", "")

        # =========================
        # LLM 配置
        # =========================
        self.llm_enabled = self.llm_config.get("enabled", True)
        self.camel_agent = None
        if self.llm_enabled and self.llm_config.get("api_key"):
            self._init_camel_agent()

        logger.info(
            f"[{self.name}] Initialized | llm={'on' if self.llm_enabled else 'off'} | "
            f"keywords={len(self.keywords)} | expand={'on' if self.enable_expand else 'off'}"
        )

    def _map_model_name(self, model_name: str) -> ModelType:
        model_mapping = {
            "gpt-4": ModelType.GPT_4,
            "gpt-4-turbo": ModelType.GPT_4_TURBO,
            "gpt-4-turbo-preview": ModelType.GPT_4_TURBO,
            "gpt-4o": ModelType.GPT_4O,
            "gpt-4o-mini": ModelType.GPT_4O_MINI,
            "gpt-3.5-turbo": ModelType.GPT_3_5_TURBO,
            "gpt-35-turbo": ModelType.GPT_3_5_TURBO,
        }

        clean_name = model_name.lower()
        if clean_name.startswith("openai/"):
            clean_name = clean_name[7:]

        return model_mapping.get(clean_name, ModelType.GPT_3_5_TURBO)

    def _init_camel_agent(self):
        try:
            system_message = (
                "你是一个内容质量评估专家，专门评估网页内容与目标知识点的相关性与教学价值。"
                "请严格按照 JSON 格式返回评估结果。"
            )

            model_name = self.llm_config.get("model", "gpt-3.5-turbo")
            base_url = self.llm_config.get("base_url", "https://api.openai.com/v1")
            api_key = self.llm_config.get("api_key")

            is_custom_base_url = "openai.com" not in base_url

            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL if is_custom_base_url else ModelPlatformType.OPENAI,
                model_type=self._map_model_name(model_name),
                api_key=api_key,
                url=base_url,
            )

            self.camel_agent = ChatAgent(
                system_message=system_message,
                model=model,
            )

            logger.info(f"[{self.name}] CAMEL ChatAgent initialized: {model_name} @ {base_url}")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to initialize CAMEL agent: {e}")
            self.llm_enabled = False

    def _build_evaluation_prompt(self, extracted_data: Dict[str, Any]) -> str:
        url = extracted_data.get("url", "")
        title = extracted_data.get("title", "")
        main_content = extracted_data.get("main_content", "")
        headings = extracted_data.get("headings", [])
        keyword_hits = extracted_data.get("keyword_hits", 0)
        extracted_links = extracted_data.get("extracted_links", [])[:8]

        content_preview = truncate_text(main_content, 1400)

        link_samples = "\n".join([
            f"- {link.get('text', '')[:40]}: {link.get('url', '')}"
            for link in extracted_links
        ]) if extracted_links else "(无)"

        headings_str = "\n".join([f"- {h}" for h in headings[:10]]) if headings else "(无)"

        # 上下文是否有意义（保留你的过滤策略）
        context_section = ""
        reason_has_value = False
        if self.slide_reason and len(self.slide_reason) > 10:
            meaningless_reasons = ["无内容", "封面页", "目录页", "不需要优化"]
            if not any(r in self.slide_reason for r in meaningless_reasons):
                reason_has_value = True

        title_has_value = False
        if self.slide_title and len(self.slide_title) > 5:
            meaningless_patterns = ["第一讲", "第二讲", "第三讲", "第四讲",
                                   "第五讲", "第六讲", "第七讲", "第八讲",
                                   "第九讲", "第十讲", "第十一讲", "第十二讲",
                                   "第十三讲", "第十四讲", "第十五讲", "第十六讲",
                                   "导言", "目录", "封面", "封底"]
            if not any(pattern in self.slide_title for pattern in meaningless_patterns):
                title_has_value = True

        if reason_has_value or title_has_value:
            context_section = "\n\n【教学上下文】\n"
            if title_has_value:
                context_section += f"- 当前幻灯片: {self.slide_title}\n"
            if reason_has_value:
                context_section += f"- 优化原因: {self.slide_reason}\n"
            context_section += (
                "请结合教学上下文理解关键词的语义关联：即使网页标题不含关键词，只要能解释幻灯片主题也应视为相关。\n"
            )

        # ✅ 关键：不再写死 expand_links=false；允许模型建议扩链
        prompt = f"""你是一个智能内容质量评估助手，专门评估网页内容与目标关键词的语义相关性及教学价值。{context_section}

【重要：跨语言语义理解】
- 目标关键词可能是中文，但网页内容可能是英文（或反之）
- 请判断**语义关联度**，而非字面关键词匹配
- 即使关键词在文本中完全没有出现（keyword_hits=0），只要语义相关就保留

【评估原则】
1. 语义优先：判断主题与概念是否相关
2. 理解翻译对应：主动理解中文关键词的英文表达（及反向）
3. 隐性关联：标题不含关键词但讨论相关主题也可保留
4. 教学价值：是否有助于解释、举例、扩展该知识点

目标关键词: {", ".join(self.keywords[:8])}

请评估以下页面：

页面 URL: {url}
页面标题: {title}

页面标题列表:
{headings_str}

正文内容摘要（前1400字）:
{content_preview}

⚠️ 关键词匹配次数: {keyword_hits}

出链样例（最多8条）:
{link_samples}

请严格按 JSON 返回（不要输出其它任何文本）:
{{
  "keep_page": true/false,
  "expand_links": true/false,
  "priority": "high"/"medium"/"low",
  "relevance_score": 0.0-1.0,
  "reasons": "简短说明决策理由，需解释语义相关依据"
}}

建议：
- keep_page: 语义相关且有参考/教学价值
- expand_links: 如果该页像“目录/专题/聚合页”或能继续发现大量相关内容，可设为 true
- priority: high=高度相关且高质量；medium=相关；low=弱相关但可参考
- relevance_score: 0.0-1.0（0.3以上为相关，0.6以上为高相关）
"""
        return prompt

    async def _camel_evaluate(self, extracted_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.camel_agent or not self.llm_enabled:
            return None

        prompt = self._build_evaluation_prompt(extracted_data)

        try:
            response = await asyncio.to_thread(self.camel_agent.step, prompt)
            content = response.msgs[0].content if response.msgs else ""

            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    return json.loads(content[json_start:json_end])
                return json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"[{self.name}] Failed to parse LLM response: {content[:200]}...")
                return None

        except Exception as e:
            logger.error(f"[{self.name}] CAMEL evaluation error: {e}")
            return None

    def can_process(self, input_data: Any) -> bool:
        if not isinstance(input_data, dict):
            return False
        required_fields = ["main_content", "keyword_hits", "url", "headings"]
        return all(field in input_data for field in required_fields)

    def _should_expand(self, decision: str, priority: str, score: float) -> bool:
        if not self.enable_expand:
            return False
        if decision != "keep":
            return False
        if priority in self.expand_priority_allow:
            return True
        return score >= self.expand_min_score

    async def process(self, input_data: Any) -> Optional[Dict[str, Any]]:
        url = input_data.get("url", "")
        main_content = input_data.get("main_content", "") or ""
        keyword_hits = int(input_data.get("keyword_hits", 0) or 0)
        headings = input_data.get("headings", []) or []

        content_len = len(main_content)

        logger.info(
            f"[{self.name}] Evaluating | hits={keyword_hits} | len={content_len} | url={url}"
        )

        try:
            # 层级0：极短垃圾页直接丢
            if content_len < self.fast_discard_min_chars:
                return {
                    "success": True,
                    "decision": "discard",
                    "expand": False,
                    "priority": "low",
                    "reasons": f"too short ({content_len} chars)",
                    "relevance_score": 0.0,
                    "method": "fast_discard",
                    "url": url,
                }

            # 层级1：hits 很高直接通过（省钱）
            if keyword_hits >= self.high_hit_threshold:
                score = min(0.92, 0.35 + keyword_hits * 0.10)
                decision = "keep"
                priority = "high"
                expand = self._should_expand(decision, priority, score)
                return {
                    "success": True,
                    "decision": decision,
                    "expand": expand,
                    "priority": priority,
                    "reasons": f"keyword_hits={keyword_hits} (fast pass)",
                    "relevance_score": score,
                    "method": "keyword_fast_pass",
                    "url": url,
                }

            # 先算规则分，作为预算门槛/兜底
            rule_score = calculate_content_quality_score(
                text=main_content,
                keywords=self.keywords,
                headings=headings,
            )

            # 层级2：hits=1/2 → LLM 精判
            if keyword_hits in self.llm_hit_range and self.llm_enabled:
                llm_result = await self._camel_evaluate(input_data)
                if llm_result:
                    decision = "keep" if llm_result.get("keep_page", False) else "discard"
                    priority = llm_result.get("priority", "medium")
                    score = float(llm_result.get("relevance_score", 0.0) or 0.0)
                    # LLM 若没给 expand_links，按 Gate 策略补全
                    llm_expand = bool(llm_result.get("expand_links", False))
                    expand = llm_expand or self._should_expand(decision, priority, score)

                    reasons = llm_result.get("reasons", "") or f"llm judged (rule_score={rule_score:.2f})"

                    logger.info(
                        f"[{self.name}] LLM | decision={decision} | prio={priority} | score={score:.2f} | "
                        f"expand={expand} | url={url}"
                    )

                    return {
                        "success": True,
                        "decision": decision,
                        "expand": expand,
                        "priority": priority,
                        "reasons": reasons,
                        "relevance_score": score,
                        "method": "llm",
                        "url": url,
                    }

                # LLM 失败：fallback 到规则
                decision = "keep" if rule_score >= self.quality_decision_threshold else "discard"
                priority = "medium" if decision == "keep" else "low"
                expand = self._should_expand(decision, priority, rule_score)
                return {
                    "success": True,
                    "decision": decision,
                    "expand": expand,
                    "priority": priority,
                    "reasons": f"llm_failed fallback rule_score={rule_score:.2f}",
                    "relevance_score": rule_score,
                    "method": "llm_fallback",
                    "url": url,
                }

            # 层级3：hits==0 → 规则粗筛后可 LLM（关键改动）
            if keyword_hits == 0 and self.llm_enabled and self.llm_on_zero_hits:
                if rule_score >= self.zero_hits_llm_min_rule_score and content_len >= self.zero_hits_llm_min_chars:
                    llm_result = await self._camel_evaluate(input_data)
                    if llm_result:
                        decision = "keep" if llm_result.get("keep_page", False) else "discard"
                        priority = llm_result.get("priority", "low")
                        score = float(llm_result.get("relevance_score", 0.0) or 0.0)
                        llm_expand = bool(llm_result.get("expand_links", False))
                        expand = llm_expand or self._should_expand(decision, priority, score)
                        reasons = llm_result.get("reasons", "") or f"llm judged (rule_score={rule_score:.2f})"

                        logger.info(
                            f"[{self.name}] LLM@0hits | decision={decision} | prio={priority} | score={score:.2f} | "
                            f"expand={expand} | url={url}"
                        )

                        return {
                            "success": True,
                            "decision": decision,
                            "expand": expand,
                            "priority": priority,
                            "reasons": reasons,
                            "relevance_score": score,
                            "method": "llm_zero_hits",
                            "url": url,
                        }

                    # LLM 失败：fallback 规则
                    decision = "keep" if rule_score >= self.quality_decision_threshold else "discard"
                    priority = "low" if decision == "keep" else "low"
                    expand = self._should_expand(decision, priority, rule_score)
                    return {
                        "success": True,
                        "decision": decision,
                        "expand": expand,
                        "priority": priority,
                        "reasons": f"llm@0hits failed fallback rule_score={rule_score:.2f}",
                        "relevance_score": rule_score,
                        "method": "llm_zero_hits_fallback",
                        "url": url,
                    }

            # 层级4：纯规则决策（兜底）
            decision = "keep" if rule_score >= self.quality_decision_threshold else "discard"
            priority = "medium" if rule_score >= 0.6 else ("low" if decision == "keep" else "low")
            expand = self._should_expand(decision, priority, rule_score)

            logger.info(
                f"[{self.name}] RULE | decision={decision} | prio={priority} | score={rule_score:.2f} | "
                f"expand={expand} | url={url}"
            )

            return {
                "success": True,
                "decision": decision,
                "expand": expand,
                "priority": priority,
                "reasons": f"rule_score={rule_score:.2f}",
                "relevance_score": rule_score,
                "method": "rule_based",
                "url": url,
            }

        except Exception as e:
            logger.error(f"[{self.name}] Error evaluating {url}: {e}")
            return {
                "success": False,
                "decision": "discard",
                "expand": False,
                "priority": "low",
                "reasons": f"Evaluation error: {str(e)}",
                "relevance_score": 0.0,
                "method": "error",
                "url": url,
            }

    async def cleanup(self):
        if self.camel_agent:
            logger.info(f"[{self.name}] CAMEL ChatAgent cleanup completed")


def create_quality_gate_agent(
    name: str = "QualityGateAgent",
    config: Dict[str, Any] = None,
    keywords: List[str] = None,
    llm_config: Dict[str, Any] = None,
) -> QualityGateAgent:
    return QualityGateAgent(
        name=name,
        config=config or {},
        keywords=keywords or [],
        llm_config=llm_config or {},
    )
