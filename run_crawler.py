"""
启动文件 - 根据幻灯片关键词深度爬取相关内容

使用CAMEL Agent架构，基于预设种子URLs进行深度爬取

流程：
1. 读取 decision_result.json
2. 对每个 should_optimize=true 的 slide：
   - 从 preset_seed_urls.json 加载种子 URLs
   - 使用 coordinator 深度爬取（2-3层）
   - 用 QualityGateAgent 评分排序
   - 选出每个关键词最相关的 3 条
3. 输出 keywords_information
"""

import asyncio
from pathlib import Path
import sys
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from loguru import logger

# 加载 .env 文件
load_dotenv()

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.agents import QualityGateAgent, create_quality_gate_agent
from src.pipeline import create_coordinator
from src.utils.data_manager import DataManager


# ========================================
# 配置区域
# ========================================

# 输入输出配置
INPUT_FILE = "input/decision_result.json"
PRESET_URLS_FILE = "config/preset_seed_urls.json"
OUTPUT_DIR = "output"

# 深度爬虫配置
MAX_DEPTH = 2                  # 爬取深度（2-3层）
MAX_PAGES = 30                # 最大爬取页面数
MAX_PAGES_PER_DOMAIN = 10     # 每个域名最多爬多少页
CONCURRENCY = 5               # 并发数
REQUEST_TIMEOUT = 30          # 请求超时

# 质量筛选配置
TOP_CONTENTS_PER_KEYWORD = 3  # 每个关键词最终保留多少条最相关的内容
MIN_RELEVANCE_SCORE = 0.3     # 最低相关性分数

# LLM配置
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ========================================
# 工具函数
# ========================================

def setup_logging():
    """设置日志"""
    import warnings
    import logging

    warnings.filterwarnings('ignore', category=ResourceWarning)
    logging.basicConfig(level=logging.WARNING)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    logger.add(
        "logs/slide_content_fetcher_{time:YYYYMMDD_HHmmss}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="50 MB",
        retention="7 days",
    )
    return logger


def load_seed_urls(preset_file: str) -> list:
    """
    从配置文件加载预设的种子 URLs
    TODO: 后续可以根据输入文件的主题动态选择类别

    Args:
        preset_file: 配置文件路径

    Returns:
        URL 列表
    """
    try:
        with open(preset_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        preset_urls_dict = config.get("preset_urls", {})

        # TODO: 暂时写死加载马克思主义类 URLs
        # 后续可以根据 input/decision_result.json 的主题动态选择
        target_category = "马克思主义"

        if target_category not in preset_urls_dict:
            logger.error(f"配置文件中没有找到类别: {target_category}")
            return []

        data = preset_urls_dict[target_category]
        if data.get("status") not in ["approved", "pending"]:
            logger.warning(f"类别 [{target_category}] 状态为: {data.get('status')}")
            return []

        urls = data.get("urls", [])
        logger.info(f"加载 [{target_category}]: {len(urls)} 个 URLs")
        logger.info(f"总计加载 {len(urls)} 个种子 URLs")

        return urls

    except FileNotFoundError:
        logger.error(f"预设URL配置文件不存在: {preset_file}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"预设URL配置文件格式错误: {e}")
        return []


async def fetch_content_for_keyword(
    keyword: str,
    seed_urls: list,
    slide_context: dict,
    llm_config: dict
) -> list:
    """
    为单个关键词深度爬取并筛选最相关的 3 条内容

    Args:
        keyword: 当前关键词（来自 suggested_keywords）
        seed_urls: 种子 URLs
        slide_context: 幻灯片上下文 {title, reason, all_keywords}
        llm_config: LLM配置

    Returns:
        最相关的 3 条内容列表
    """
    title = slide_context.get('title', '')
    reason = slide_context.get('reason', '')

    logger.info(f"  关键词: {keyword}")
    logger.info(f"    幻灯片上下文: {title}")

    # 创建临时输出文件
    temp_output_file = f"output/temp_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    # 创建爬虫配置（使用所有种子 URLs）
    # 注意：这里不设置 keywords 过滤，让 QualityGateAgent 用 LLM 评分
    crawl_config = {
        "topic": keyword,
        "keywords": [keyword],  # 只用当前关键词判断
        "context": {
            "slide_title": title,
            "slide_reason": reason,
        },
        "seed_urls": seed_urls,
        "max_depth": MAX_DEPTH,
        "max_pages": MAX_PAGES,
        "max_pages_per_domain": MAX_PAGES_PER_DOMAIN,
        "concurrency": CONCURRENCY,
        "allowed_domains": [],  # 允许所有域名
        "blocked_domains": ["youtube.com", "twitter.com", "facebook.com", "instagram.com"],
        "request_timeout": REQUEST_TIMEOUT,
        "max_retries": 2,
        "retry_delay": 1,
        "output_file": temp_output_file,
        "cache_dir": ".cache",
        "enable_cache": True,
    }

    # LLM配置
    llm_config_full = {
        "enabled": True,
        "api_key": llm_config.get("api_key", ""),
        "model": llm_config.get("model", LLM_MODEL),
        "base_url": llm_config.get("base_url", "https://api.openai.com/v1"),
        "temperature": 0.3,
        "max_tokens": 500,
        "fast_filter_min_length": 100,
        "fast_filter_min_keyword_hits": 0,  # 不做关键词过滤，完全靠 LLM 评分
    }

    # Agent配置
    agent_config = {
        # 允许扩链：让 max_depth=2 真正发生
        "enable_expand": True,
        "expand_min_score": 0.45,
        "expand_priority_allow": ["high", "medium"],

        # hits==0 也允许 LLM（预算门槛）
        "llm_on_zero_hits": True,
        "zero_hits_llm_min_rule_score": 0.15,
        "zero_hits_llm_min_chars": 250,

        # 规则兜底阈值
        "quality_decision_threshold": 0.35,

        # 其它
        "extract_text_snippet_length": 500,
    }

    # 创建 Coordinator（深度爬虫）
    from src.config import CrawlConfig, LLMConfig as LLMCfg, AgentConfig as AgentCfg

    coordinator = create_coordinator(
        crawl_config=CrawlConfig(**crawl_config),
        llm_config=LLMCfg(**llm_config_full),
        agent_config=AgentCfg(**agent_config),
    )

    # 添加种子 URLs
    await coordinator.add_seed_urls(seed_urls)

    # 运行深度爬虫
    logger.info(f"    开始深度爬取...")
    await coordinator.run()

    # 读取爬取结果
    crawled_pages = []
    try:
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    # 只保留有 relevance_score 的页面
                    if "relevance_score" in record:
                        crawled_pages.append(record)
    except FileNotFoundError:
        logger.warning(f"    爬取输出文件不存在")

    # 清理
    await coordinator.cleanup()

    # 删除临时文件
    try:
        os.remove(temp_output_file)
    except:
        pass

    logger.info(f"    爬取完成，获得 {len(crawled_pages)} 个页面")

    if not crawled_pages:
        return []

    # 按 relevance_score 排序，选出前 3 条
    crawled_pages.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
    top_pages = crawled_pages[:TOP_CONTENTS_PER_KEYWORD]

    logger.info(f"    ✓ 选中 {len(top_pages)} 条最相关内容")
    for i, page in enumerate(top_pages, 1):
        score = page.get("relevance_score", 0.0)
        title = page.get("title", "无标题")[:50]
        logger.info(f"      {i}. [{score:.2f}] {title}")

    # 格式化输出
    results = []
    for page in top_pages:
        results.append({
            "url": page.get("url", ""),
            "title": page.get("title", ""),
            "summary": page.get("text_snippet", ""),
            "relevance_score": page.get("relevance_score", 0.0),
            "reason": page.get("reasons", ""),
        })

    return results


async def process_slide(slide: dict, seed_urls: list, llm_config: dict) -> dict:
    """
    处理单个幻灯片，为所有关键词深度爬取内容

    Args:
        slide: 幻灯片数据
        seed_urls: 种子 URLs
        llm_config: LLM配置

    Returns:
        增强后的幻灯片数据
    """
    slide_number = slide["slide_number"]
    title = slide["title"]
    should_optimize = slide.get("should_optimize", False)
    keywords = slide.get("suggested_keywords", [])
    reason = slide.get("reason", "")

    # 复制原数据
    enhanced_slide = slide.copy()

    if not should_optimize or not keywords:
        enhanced_slide["keywords_information"] = {}
        return enhanced_slide

    logger.info(f"\n处理幻灯片 {slide_number}: {title}")
    logger.info(f"  优化原因: {reason}")
    logger.info(f"  关键词数量: {len(keywords)}")

    # 为每个关键词深度爬取
    keywords_information = {}

    # 准备幻灯片上下文
    slide_context = {
        "title": title,
        "reason": reason,
    }

    for keyword in keywords:
        contents = await fetch_content_for_keyword(
            keyword=keyword,
            seed_urls=seed_urls,
            slide_context=slide_context,
            llm_config=llm_config
        )

        keywords_information[keyword] = contents

    # 添加到结果中
    enhanced_slide["keywords_information"] = keywords_information
    enhanced_slide["crawl_timestamp"] = datetime.now().isoformat()

    return enhanced_slide


async def main():
    """主函数"""
    logger = setup_logging()

    # ==================== 读取输入文件 ====================
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        decisions = input_data.get("decisions", [])
        total_slides = len(decisions)
        to_process = sum(1 for d in decisions if d.get("should_optimize"))

        logger.info("=" * 70)
        logger.info("幻灯片内容深度爬取工具")
        logger.info("=" * 70)
        logger.info(f"输入文件: {INPUT_FILE}")
        logger.info(f"总幻灯片数: {total_slides}")
        logger.info(f"需要处理: {to_process}")
        logger.info(f"爬取深度: {MAX_DEPTH} 层")
        logger.info(f"最大页面数: {MAX_PAGES}")
        logger.info("=" * 70)
        logger.info("")

    except FileNotFoundError:
        logger.error(f"输入文件不存在: {INPUT_FILE}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"输入文件 JSON 格式错误: {e}")
        return

    # ==================== 加载种子 URLs ====================
    logger.info("加载预设种子 URLs...")
    seed_urls = load_seed_urls(PRESET_URLS_FILE)

    if not seed_urls:
        logger.error("没有可用的种子 URLs，程序退出")
        return

    logger.info("")

    # ==================== LLM 配置 ====================
    llm_config = {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": LLM_MODEL,
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    }

    # ==================== 处理所有幻灯片 ====================
    enhanced_decisions = []

    for slide in decisions:
        enhanced_slide = await process_slide(slide, seed_urls, llm_config)
        enhanced_decisions.append(enhanced_slide)

    # ==================== 保存结果 ====================
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_file = f"{OUTPUT_DIR}/slide_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_data = {
        "input_file": input_data.get("input_file"),
        "total_pages": input_data.get("total_pages"),
        "should_optimize_count": input_data.get("should_optimize_count"),
        "teaching_chain": input_data.get("teaching_chain"),
        "crawl_config": {
            "max_depth": MAX_DEPTH,
            "max_pages": MAX_PAGES,
            "seed_urls_count": len(seed_urls),
        },
        "crawl_timestamp": datetime.now().isoformat(),
        "decisions": enhanced_decisions
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # ==================== 统计结果 ====================
    total_keywords = sum(len(d.get("keywords_information", {})) for d in enhanced_decisions)
    total_contents = sum(
        sum(len(v) for v in d.get("keywords_information", {}).values())
        for d in enhanced_decisions
    )

    logger.info("")
    logger.info("=" * 70)
    logger.success("深度爬取完成！")
    logger.info("=" * 70)
    logger.info(f"输出文件: {output_file}")
    logger.info(f"处理的关键词数: {total_keywords}")
    logger.info(f"爬取的内容总数: {total_contents}")
    logger.info("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("幻灯片内容深度爬取工具")
    print("=" * 70)
    print("基于预设种子 URLs 进行深度爬取")
    print(f"爬取深度: {MAX_DEPTH} 层")
    print(f"最大页面数: {MAX_PAGES}")
    print("")
    asyncio.run(main())
