"""
启动文件 - 在 PyCharm 中直接运行此文件即可
"""

import asyncio
from pathlib import Path
import sys
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import CrawlConfig, LLMConfig, AgentConfig
from src.pipeline import create_coordinator


# ========================================
# 在这里修改配置，不需要命令行参数
# ========================================

# 目标知识点配置
TOPIC = "dynamic programming"
KEYWORDS = ["dynamic programming", "dp", "动态规划", "memoization", "tabulation"]

# 种子 URL 列表（添加更多起点以体现 Deep Research）
SEED_URLS = [
    "https://en.wikipedia.org/wiki/Dynamic_programming",
    "https://www.geeksforgeeks.org/dynamic-programming/",
    "https://cp-algorithms.com/dynamic_programming/intro-to-dp.html",
    "https://oi-wiki.org/dp/",
]

# 预算参数
MAX_DEPTH = 3                # 最大爬取深度（增加以体现 Deep Research）
MAX_PAGES = 50               # 最大页面数（增加以爬取更多链接）
MAX_PAGES_PER_DOMAIN = 15    # 每个域名最大页面数
CONCURRENCY = 3              # 并发爬取数

# 域名控制（只允许英文站点）
ALLOWED_DOMAINS = [
    "en.wikipedia.org",      # 只允许英文 Wikipedia
    "leetcode.com",
    "geeksforgeeks.org",
]
BLOCKED_DOMAINS = [
    "youtube.com",
    "twitter.com",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
]

# LLM 配置（从 .env 文件读取）
ENABLE_LLM = True           # 设置为 True 启用 LLM
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 质量判断阈值（调整这些值来控制保留多少页面）
MIN_KEYWORD_HITS = 0        # 最小关键词匹配数（设为 0 不要求关键词）
QUALITY_THRESHOLD = 0.3     # 质量分数阈值（0.0-1.0，越低保留越多）
SNIPPET_LENGTH = 500         # 摘要长度（字符数）

# 输出配置
OUTPUT_FILE = "output/dp_crawl_results.jsonl"
CACHE_DIR = ".cache"


# ========================================
# 主程序（无需修改）
# ========================================

def setup_logging():
    """设置日志"""
    from loguru import logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    logger.add(
        "logs/crawler_{time:YYYYMMDD_HHmmss}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="50 MB",
        retention="7 days",
    )
    return logger


async def main():
    """主函数"""
    logger = setup_logging()

    # 创建配置
    crawl_config = CrawlConfig(
        topic=TOPIC,
        keywords=KEYWORDS,
        seed_urls=SEED_URLS,
        max_depth=MAX_DEPTH,
        max_pages=MAX_PAGES,
        max_pages_per_domain=MAX_PAGES_PER_DOMAIN,
        concurrency=CONCURRENCY,
        allowed_domains=ALLOWED_DOMAINS,
        blocked_domains=BLOCKED_DOMAINS,
        output_file=OUTPUT_FILE,
        cache_dir=CACHE_DIR,
    )

    llm_config = LLMConfig(
        enabled=ENABLE_LLM,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        fast_filter_min_keyword_hits=MIN_KEYWORD_HITS,
    )

    agent_config = AgentConfig(
        quality_decision_threshold=QUALITY_THRESHOLD,
        extract_text_snippet_length=SNIPPET_LENGTH,
    )

    # 打印配置摘要
    logger.info("=" * 60)
    logger.info("Dynamic Crawl Pro - Starting")
    logger.info("=" * 60)
    logger.info(f"Topic: {TOPIC}")
    logger.info(f"Seed URLs: {len(SEED_URLS)}")
    logger.info(f"Max Depth: {MAX_DEPTH}")
    logger.info(f"Max Pages: {MAX_PAGES}")
    logger.info(f"Concurrency: {CONCURRENCY}")
    logger.info(f"LLM Enabled: {ENABLE_LLM}")
    logger.info(f"Output: {OUTPUT_FILE}")
    logger.info("=" * 60)

    # 创建协调器
    coordinator = create_coordinator(
        crawl_config=crawl_config,
        llm_config=llm_config,
        agent_config=agent_config,
    )

    # 添加种子 URL
    await coordinator.add_seed_urls(SEED_URLS)

    try:
        # 运行爬取
        await coordinator.run()

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise

    finally:
        # 清理资源
        await coordinator.cleanup()


if __name__ == "__main__":
    print("=" * 60)
    print("Dynamic Crawl Pro")
    print("=" * 60)
    print("在 PyCharm 中右键此文件，选择 'Run run_crawler'")
    print("或者在上方配置完成后，点击绿色运行按钮")
    print("=" * 60)
    print()

    # 运行主程序
    asyncio.run(main())
