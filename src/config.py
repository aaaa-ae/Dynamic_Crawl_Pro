"""
配置模块
集中管理所有配置参数
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CrawlConfig:
    """爬虫配置"""

    # 核心配置
    topic: str = "dynamic programming"
    keywords: List[str] = field(default_factory=lambda: ["dynamic programming", "dp", "动态规划"])
    seed_urls: List[str] = field(default_factory=list)

    # 上下文配置（用于 LLM 判断）
    context: dict = field(default_factory=dict)  # 幻灯片上下文 {slide_title, slide_reason}

    # 预算参数
    max_depth: int = 3
    max_pages: int = 100
    max_pages_per_domain: int = 20
    concurrency: int = 5

    # 域名控制
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)

    # 超时和重试
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1

    # 输出配置
    output_file: str = "output/crawl_results.jsonl"

    # 缓存配置
    enable_cache: bool = True
    cache_dir: str = ".cache"


@dataclass
class LLMConfig:
    """LLM 配置"""

    enabled: bool = True
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.3
    max_tokens: int = 500

    # 预过滤阈值
    fast_filter_min_length: int = 500
    fast_filter_min_keyword_hits: int = 1

    # 质量判断阈值
    min_relevance_score: float = 0.5

    # 对话式协调配置
    enable_conversation_coordinator: bool = False
    coordinator_model: str = "gpt-3.5-turbo"


@dataclass
class AgentConfig:
    """Agent 配置"""

    # 提取配置
    extract_max_length: int = 10000
    extract_text_snippet_length: int = 300
    extract_headings_max: int = 10

    # 质量判断配置
    quality_decision_threshold: float = 0.6

    # 链接提取配置
    extract_links_max: int = 50
    filter_same_domain: bool = True

    # 过滤与路由配置
    fast_discard_min_chars: int = 80
    high_hit_threshold: int = 3
    llm_hit_range: List[int] = field(default_factory=lambda: [1, 2])

    # 零命中LLM评估配置
    llm_on_zero_hits: bool = True
    zero_hits_llm_min_rule_score: float = 0.15
    zero_hits_llm_min_chars: int = 250

    # 扩链策略配置
    enable_expand: bool = True
    expand_min_score: float = 0.45
    expand_priority_allow: List[str] = field(default_factory=lambda: ["high", "medium"])


# 默认配置实例
DEFAULT_CRAWL_CONFIG = CrawlConfig()
DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_AGENT_CONFIG = AgentConfig()


def get_config():
    """获取合并后的配置字典（用于命令行参数覆盖）"""
    return {
        "crawl": DEFAULT_CRAWL_CONFIG,
        "llm": DEFAULT_LLM_CONFIG,
        "agent": DEFAULT_AGENT_CONFIG,
    }
