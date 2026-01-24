"""
QualityGateAgent - 质量判断 Agent
使用 LLM 对页面进行质量判断和决策
"""

import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from openai import AsyncOpenAI

from .base import BaseAgent
from ..utils import (
    fast_filter,
    calculate_content_quality_score,
    truncate_text,
)


class QualityGateAgent(BaseAgent):
    """
    质量判断 Agent
    基于规则和 LLM 对页面进行质量评估和决策
    """

    def __init__(
        self,
        name: str = "QualityGateAgent",
        config: Dict[str, Any] = None,
        keywords: List[str] = None,
        llm_config: Dict[str, Any] = None,
    ):
        """
        初始化 QualityGateAgent

        Args:
            name: Agent 名称
            config: 配置字典
            keywords: 关键词列表
            llm_config: LLM 配置，包含:
                - enabled: 是否启用 LLM
                - api_key: API Key
                - model: 模型名称
                - base_url: API 基础 URL
                - temperature: 温度参数
                - max_tokens: 最大 Token 数
        """
        super().__init__(name, config)
        self.keywords = keywords or []
        self.llm_config = llm_config or {}

        # 配置参数
        self.fast_filter_min_length = config.get("fast_filter_min_length", 500)
        self.fast_filter_min_keyword_hits = config.get("fast_filter_min_keyword_hits", 1)
        self.quality_decision_threshold = config.get("quality_decision_threshold", 0.6)

        # LLM 配置
        self.llm_enabled = self.llm_config.get("enabled", True)
        self.llm_client = None

        if self.llm_enabled and self.llm_config.get("api_key"):
            self._init_llm_client()

        logger.info(
            f"[{self.name}] Initialized with LLM={'enabled' if self.llm_enabled else 'disabled'}, "
            f"{len(self.keywords)} keywords"
        )

    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        try:
            self.llm_client = AsyncOpenAI(
                api_key=self.llm_config.get("api_key"),
                base_url=self.llm_config.get("base_url", "https://api.openai.com/v1"),
            )
            logger.info(f"[{self.name}] LLM client initialized: {self.llm_config.get('model')}")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to initialize LLM client: {e}")
            self.llm_enabled = False

    def can_process(self, input_data: Any) -> bool:
        """
        检查是否能够处理该输入

        Args:
            input_data: 输入数据，应为包含提取数据的字典

        Returns:
            是否能够处理
        """
        if not isinstance(input_data, dict):
            return False

        # 检查必要的字段
        required_fields = ["main_content", "keyword_hits", "url", "headings"]
        return all(field in input_data for field in required_fields)

    async def _llm_evaluate(self, extracted_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用 LLM 进行深度评估

        Args:
            extracted_data: ExtractorAgent 提取的数据

        Returns:
            LLM 评估结果
        """
        if not self.llm_client or not self.llm_enabled:
            return None

        # 准备 LLM 输入
        url = extracted_data.get("url", "")
        title = extracted_data.get("title", "")
        main_content = extracted_data.get("main_content", "")
        headings = extracted_data.get("headings", [])
        keyword_hits = extracted_data.get("keyword_hits", 0)
        extracted_links = extracted_data.get("extracted_links", [])[:5]  # 最多 5 个链接样例

        # 截断内容（控制在合理范围内）
        content_preview = truncate_text(main_content, 1200)

        # 构造链接样例
        link_samples = "\n".join([
            f"- {link['text']}: {link['url']}"
            for link in extracted_links
        ]) if extracted_links else "(无)"

        # 构造标题列表
        headings_str = "\n".join([f"- {h}" for h in headings[:10]]) if headings else "(无)"

        # 构造提示词
        prompt = f"""你是一个智能内容质量评估助手，专门评估网页内容与目标知识点的相关性。

目标知识点: {", ".join(self.keywords[:5])}

请评估以下页面的质量：

页面 URL: {url}
页面标题: {title}

页面标题列表:
{headings_str}

正文内容摘要（前1200字）:
{content_preview}

关键词匹配次数: {keyword_hits}

出链样例:
{link_samples}

请严格按照以下 JSON 格式返回评估结果（不要添加任何其他文本）:
{{
    "keep_page": true/false,
    "expand_links": true/false,
    "priority": "high"/"medium"/"low",
    "relevance_score": 0.0-1.0,
    "reasons": "简短说明决策理由，例如：'包含详细的算法讲解和代码示例，与目标知识点高度相关'"
}}

评估标准：
- keep_page: 页面是否值得保存（内容是否与目标知识点相关、是否有价值）
- expand_links: 是否应该继续爬取该页面中的链接
- priority: 页面优先级（high=核心内容/质量很高，medium=相关但非核心，low=弱相关）
- relevance_score: 相关性评分（0-1，1 表示完全相关）
- reasons: 决策理由，50-100字

请只返回 JSON 格式的结果，不要添加任何其他文字或解释。
"""

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": "你是一个内容质量评估专家，只返回 JSON 格式的评估结果。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.llm_config.get("temperature", 0.3),
                max_tokens=self.llm_config.get("max_tokens", 500),
            )

            content = response.choices[0].message.content.strip()

            # 尝试解析 JSON
            try:
                # 提取 JSON 部分（处理可能的额外文本）
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    return json.loads(json_content)
                else:
                    return json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"[{self.name}] Failed to parse LLM response: {content}")
                return None

        except Exception as e:
            logger.error(f"[{self.name}] LLM evaluation error: {e}")
            return None

    async def process(self, input_data: Any) -> Optional[Dict[str, Any]]:
        """
        处理输入数据，进行质量判断

        Args:
            input_data: 输入数据字典，包含 ExtractorAgent 的输出

        Returns:
            质量判断结果，格式:
            {
                "success": bool,
                "decision": "keep" | "discard",
                "expand": bool,
                "priority": "high" | "medium" | "low",
                "reasons": str,
                "relevance_score": float,
                "method": "fast_filter" | "llm" | "rule_based",
                "url": str,
            }
        """
        # 提取字段
        url = input_data.get("url", "")
        main_content = input_data.get("main_content", "")
        keyword_hits = input_data.get("keyword_hits", 0)
        headings = input_data.get("headings", [])

        logger.info(f"[{self.name}] Evaluating: {url} (keyword_hits={keyword_hits})")

        try:
            # 步骤1: 快速过滤器（基于规则）
            passed, filter_reason = fast_filter(
                text=main_content,
                keywords=self.keywords,
                min_length=self.fast_filter_min_length,
                min_keyword_hits=self.fast_filter_min_keyword_hits,
            )

            if not passed:
                logger.info(f"[{self.name}] Fast filter discarded: {url} - {filter_reason}")
                return {
                    "success": True,
                    "decision": "discard",
                    "expand": False,
                    "priority": "low",
                    "reasons": filter_reason,
                    "relevance_score": 0.0,
                    "method": "fast_filter",
                    "url": url,
                }

            # 步骤2: 如果 LLM 未启用，使用基于规则的判断
            if not self.llm_enabled or not self.llm_client:
                quality_score = calculate_content_quality_score(
                    text=main_content,
                    keywords=self.keywords,
                    headings=headings,
                )

                if quality_score >= self.quality_decision_threshold:
                    decision = "keep"
                    expand = quality_score >= 0.7  # 高分页面才继续扩展
                    priority = "high" if quality_score >= 0.8 else "medium"
                else:
                    decision = "discard"
                    expand = False
                    priority = "low"

                reasons = f"Rule-based decision: quality_score={quality_score:.2f}"

                logger.info(
                    f"[{self.name}] Rule-based decision for {url}: {decision}, "
                    f"priority={priority}, score={quality_score:.2f}"
                )

                return {
                    "success": True,
                    "decision": decision,
                    "expand": expand,
                    "priority": priority,
                    "reasons": reasons,
                    "relevance_score": quality_score,
                    "method": "rule_based",
                    "url": url,
                }

            # 步骤3: LLM 深度评估
            llm_result = await self._llm_evaluate(input_data)

            if llm_result:
                decision = "keep" if llm_result.get("keep_page", False) else "discard"
                expand = llm_result.get("expand_links", False)
                priority = llm_result.get("priority", "medium")
                reasons = llm_result.get("reasons", "")
                relevance_score = llm_result.get("relevance_score", 0.0)

                logger.info(
                    f"[{self.name}] LLM decision for {url}: {decision}, "
                    f"expand={expand}, priority={priority}, score={relevance_score:.2f}"
                )

                return {
                    "success": True,
                    "decision": decision,
                    "expand": expand,
                    "priority": priority,
                    "reasons": reasons,
                    "relevance_score": relevance_score,
                    "method": "llm",
                    "url": url,
                }
            else:
                # LLM 调用失败，回退到基于规则的判断
                quality_score = calculate_content_quality_score(
                    text=main_content,
                    keywords=self.keywords,
                    headings=headings,
                )

                decision = "keep" if quality_score >= self.quality_decision_threshold else "discard"
                expand = decision == "keep" and quality_score >= 0.7
                priority = "high" if quality_score >= 0.8 else ("medium" if quality_score >= 0.5 else "low")
                reasons = f"LLM fallback: quality_score={quality_score:.2f}"

                logger.warning(
                    f"[{self.name}] LLM failed, using fallback for {url}: {decision}"
                )

                return {
                    "success": True,
                    "decision": decision,
                    "expand": expand,
                    "priority": priority,
                    "reasons": reasons,
                    "relevance_score": quality_score,
                    "method": "llm_fallback",
                    "url": url,
                }

        except Exception as e:
            logger.error(f"[{self.name}] Error evaluating {url}: {e}")
            # 出错时保守处理：保存页面但不扩展
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
        """清理资源"""
        if self.llm_client:
            await self.llm_client.close()
            logger.info(f"[{self.name}] LLM client closed")


def create_quality_gate_agent(
    name: str = "QualityGateAgent",
    config: Dict[str, Any] = None,
    keywords: List[str] = None,
    llm_config: Dict[str, Any] = None,
) -> QualityGateAgent:
    """
    工厂函数：创建 QualityGateAgent 实例

    Args:
        name: Agent 名称
        config: 配置字典
        keywords: 关键词列表
        llm_config: LLM 配置

    Returns:
        QualityGateAgent 实例
    """
    return QualityGateAgent(
        name=name,
        config=config or {},
        keywords=keywords or [],
        llm_config=llm_config or {},
    )
