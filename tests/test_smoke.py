"""
烟雾测试
快速验证项目是否能够正常运行
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CrawlConfig, LLMConfig, AgentConfig
from src.pipeline import create_coordinator


def setup_logging():
    """设置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )


async def smoke_test():
    """
    烟雾测试
    使用最小配置测试系统是否能够正常运行
    """
    logger.info("Starting smoke test...")

    # 最小配置
    crawl_config = CrawlConfig(
        topic="test",
        keywords=["test", "python"],
        seed_urls=["https://example.com"],
        max_depth=1,
        max_pages=2,
        max_pages_per_domain=5,
        concurrency=1,
        allowed_domains=[],
        blocked_domains=[],
        output_file="output/smoke_test_results.jsonl",
        cache_dir=".cache/smoke_test",
    )

    llm_config = LLMConfig(
        enabled=False,  # 禁用 LLM
    )

    agent_config = AgentConfig()

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

        logger.info("Smoke test completed successfully!")

        # 检查输出文件是否存在
        output_path = Path(crawl_config.output_file)
        if output_path.exists():
            logger.info(f"Output file created: {output_path}")
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                logger.info(f"Output contains {len(lines)} records")
        else:
            logger.warning(f"Output file not found: {output_path}")

    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        raise

    finally:
        # 清理资源
        await coordinator.cleanup()


if __name__ == "__main__":
    setup_logging()
    asyncio.run(smoke_test())
