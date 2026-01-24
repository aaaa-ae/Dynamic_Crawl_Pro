"""
主入口文件
知识点驱动的深度内容采集系统
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CrawlConfig, LLMConfig, AgentConfig
from pipeline import create_coordinator


# ========================================
# 核心配置（在此处修改以适配您的需求）
# ========================================

# 目标知识点配置
TOPIC = "dynamic programming"
KEYWORDS = ["dynamic programming", "dp", "动态规划", "memoization", "tabulation", "optimal substructure"]

# 种子 URL 列表
SEED_URLS = [
    "https://en.wikipedia.org/wiki/Dynamic_programming",
    "https://leetcode.com/tag/dynamic-programming/",
]

# 预算参数
MAX_DEPTH = 2                # 最大爬取深度（建议：2-3）
MAX_PAGES = 50               # 最大页面数（建议：50-200）
MAX_PAGES_PER_DOMAIN = 20    # 每个域名最大页面数
CONCURRENCY = 3              # 并发爬取数（建议：3-5）

# 域名控制
ALLOWED_DOMAINS = ["wikipedia.org", "leetcode.com", "geeksforgeeks.org"]
BLOCKED_DOMAINS = [
    "youtube.com",
    "twitter.com",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
]

# LLM 配置（用于 QualityGateAgent）
# 设置为空字符串或 None 以禁用 LLM（仅使用规则判断）
LLM_API_KEY = ""  # 替换为您的 OpenAI API Key
LLM_MODEL = "gpt-3.5-turbo"
LLM_BASE_URL = "https://api.openai.com/v1"
ENABLE_LLM = False  # 设置为 True 以启用 LLM

# 输出配置
OUTPUT_FILE = "output/dp_crawl_results.jsonl"
CACHE_DIR = ".cache"

# ========================================
# 配置加载
# ========================================


def load_config_from_args(args) -> tuple:
    """
    从命令行参数加载配置

    Args:
        args: 命令行参数

    Returns:
        (CrawlConfig, LLMConfig, AgentConfig)
    """
    # 爬虫配置
    crawl_config = CrawlConfig(
        topic=args.topic or TOPIC,
        keywords=args.keywords or KEYWORDS,
        seed_urls=args.seed_url or SEED_URLS,
        max_depth=args.max_depth or MAX_DEPTH,
        max_pages=args.max_pages or MAX_PAGES,
        max_pages_per_domain=args.max_pages_per_domain or MAX_PAGES_PER_DOMAIN,
        concurrency=args.concurrency or CONCURRENCY,
        allowed_domains=args.allowed_domains or ALLOWED_DOMAINS,
        blocked_domains=args.blocked_domains or BLOCKED_DOMAINS,
        output_file=args.output or OUTPUT_FILE,
        cache_dir=args.cache_dir or CACHE_DIR,
    )

    # LLM 配置
    llm_config = LLMConfig(
        enabled=args.enable_llm or ENABLE_LLM,
        api_key=args.api_key or LLM_API_KEY,
        model=args.model or LLM_MODEL,
        base_url=args.base_url or LLM_BASE_URL,
    )

    # Agent 配置
    agent_config = AgentConfig()

    return crawl_config, llm_config, agent_config


def setup_logging(verbose: bool = False):
    """
    设置日志配置

    Args:
        verbose: 是否启用详细日志
    """
    log_level = "DEBUG" if verbose else "INFO"

    logger.remove()  # 移除默认处理器

    # 控制台输出
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # 文件输出
    logger.add(
        "logs/crawler_{time:YYYYMMDD_HHmmss}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="50 MB",
        retention="7 days",
    )


def print_config_summary(
    crawl_config: CrawlConfig,
    llm_config: LLMConfig,
):
    """
    打印配置摘要

    Args:
        crawl_config: 爬虫配置
        llm_config: LLM 配置
    """
    logger.info("=" * 60)
    logger.info("Dynamic Crawl Pro - Knowledge-Driven Deep Content Crawler")
    logger.info("=" * 60)
    logger.info(f"Topic: {crawl_config.topic}")
    logger.info(f"Keywords: {', '.join(crawl_config.keywords[:5])}...")
    logger.info(f"Seed URLs: {len(crawl_config.seed_urls)}")
    logger.info(f"Max Depth: {crawl_config.max_depth}")
    logger.info(f"Max Pages: {crawl_config.max_pages}")
    logger.info(f"Concurrency: {crawl_config.concurrency}")
    logger.info(f"LLM Enabled: {llm_config.enabled} ({llm_config.model})")
    logger.info(f"Output: {crawl_config.output_file}")
    logger.info("=" * 60)


async def main_async(args):
    """
    异步主函数

    Args:
        args: 命令行参数
    """
    # 设置日志
    setup_logging(verbose=args.verbose)

    # 加载配置
    crawl_config, llm_config, agent_config = load_config_from_args(args)

    # 打印配置摘要
    print_config_summary(crawl_config, llm_config)

    # 创建协调器
    coordinator = create_coordinator(
        crawl_config=crawl_config,
        llm_config=llm_config,
        agent_config=agent_config,
    )

    # 添加种子 URL
    await coordinator.add_seed_urls(crawl_config.seed_urls)

    try:
        # 运行爬取
        await coordinator.run()

    except KeyboardInterrupt:
        logger.warning("[Main] Interrupted by user")

    except Exception as e:
        logger.error(f"[Main] Error: {e}")
        raise

    finally:
        # 清理资源
        await coordinator.cleanup()


def parse_arguments():
    """
    解析命令行参数

    Returns:
        命令行参数命名空间
    """
    parser = argparse.ArgumentParser(
        description="Knowledge-Driven Deep Content Crawler"
    )

    # 核心参数
    parser.add_argument(
        "--topic",
        type=str,
        help="Target knowledge topic",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        help="Keywords for matching",
    )
    parser.add_argument(
        "--seed-url",
        type=str,
        action="append",
        help="Seed URL (can be specified multiple times)",
    )

    # 预算参数
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Maximum crawl depth",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Maximum number of pages to crawl",
    )
    parser.add_argument(
        "--max-pages-per-domain",
        type=int,
        help="Maximum pages per domain",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Number of concurrent crawlers",
    )

    # 域名控制
    parser.add_argument(
        "--allowed-domains",
        type=str,
        nargs="+",
        help="Allowed domains",
    )
    parser.add_argument(
        "--blocked-domains",
        type=str,
        nargs="+",
        help="Blocked domains",
    )

    # LLM 配置
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable LLM-based quality evaluation",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="LLM model name",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="LLM API base URL",
    )

    # 输出配置
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Cache directory",
    )

    # 其他
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    """
    主函数入口
    """
    # 解析命令行参数
    args = parse_arguments()

    # 创建事件循环并运行
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("[Main] Interrupted")
    except Exception as e:
        logger.error(f"[Main] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
