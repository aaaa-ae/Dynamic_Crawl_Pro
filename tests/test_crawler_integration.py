"""
测试项目集成 - CrawlerAgent + trafilatura + ExtractorAgent

验证完整的爬取和提取流程是否正常工作
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import create_crawler_agent, create_extractor_agent
from src.utils import DataManager
from loguru import logger


async def test_full_pipeline():
    """测试完整的爬取和提取流程"""

    # 测试 URL
    test_url = "http://cpc.people.com.cn/"
    test_keywords = ["马克思主义", "中国共产党", "理论"]

    logger.info("=" * 80)
    logger.info("测试项目集成：CrawlerAgent → trafilatura → ExtractorAgent")
    logger.info("=" * 80)
    logger.info(f"测试 URL: {test_url}")
    logger.info(f"测试关键词: {test_keywords}")
    logger.info("")

    # 创建数据管理器
    data_manager = DataManager(cache_dir=".cache/test", enable_persistence=True)

    # 创建 CrawlerAgent
    crawler_agent = create_crawler_agent(
        config={
            "timeout": 30,
            "max_retries": 2,
        },
        data_manager=data_manager,
    )

    # 创建 ExtractorAgent
    extractor_agent = create_extractor_agent(
        config={
            "extract_text_snippet_length": 300,
        },
        data_manager=data_manager,
        keywords=test_keywords,
    )

    try:
        # ============================================================
        # 步骤1: CrawlerAgent 爬取页面（使用 trafilatura）
        # ============================================================
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("步骤1: CrawlerAgent 爬取页面")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        crawl_result = await crawler_agent.process({"url": test_url})

        if not crawl_result or not crawl_result.get("success"):
            logger.error(f"❌ 爬取失败: {crawl_result.get('error', 'Unknown error')}")
            return

        data_id = crawl_result.get("data_id")
        logger.success(f"✅ 爬取成功，data_id: {data_id}")
        logger.info(f"   标题: {crawl_result.get('title', 'N/A')}")

        # 检查是否有 clean_content
        page_data = data_manager.get_page(data_id)
        if page_data:
            if page_data.clean_content:
                logger.success(f"✅ trafilatura 提取成功，正文长度: {len(page_data.clean_content)} 字符")
                logger.info(f"   正文预览（前300字）:")
                logger.info(f"   {page_data.clean_content[:300]}...")
            else:
                logger.warning(f"⚠️  clean_content 为空，trafilatura 可能提取失败")
                logger.info(f"   markdown 长度: {len(page_data.markdown)} 字符")

        # ============================================================
        # 步骤2: ExtractorAgent 提取内容和关键词
        # ============================================================
        logger.info("")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("步骤2: ExtractorAgent 提取内容和关键词")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        extract_result = extractor_agent.process({
            "data_id": data_id,
            "url": test_url,
            "depth": 0,
            "title": crawl_result.get("title", ""),
        })

        if not extract_result or not extract_result.get("success"):
            logger.error(f"❌ 提取失败: {extract_result.get('error', 'Unknown error')}")
            return

        logger.success(f"✅ 提取成功")
        logger.info(f"   main_content 长度: {len(extract_result.get('main_content', ''))} 字符")
        logger.info(f"   headings 数量: {len(extract_result.get('headings', []))}")
        logger.info(f"   keyword_hits: {extract_result.get('keyword_hits', 0)}")
        logger.info(f"   extracted_links: {extract_result.get('extracted_links_count', 0)}")

        # ============================================================
        # 步骤3: 详细结果分析
        # ============================================================
        logger.info("")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("步骤3: 详细结果分析")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        logger.info(f"\n【标题】")
        logger.info(f"  {extract_result.get('title', 'N/A')}")

        logger.info(f"\n【提取的标题（headings）】")
        for i, heading in enumerate(extract_result.get('headings', [])[:10], 1):
            logger.info(f"  {i}. {heading}")

        logger.info(f"\n【关键词匹配统计】")
        for keyword in test_keywords:
            count = extract_result.get('main_content', '').count(keyword)
            logger.info(f"  '{keyword}': {count} 次")

        logger.info(f"\n【文本摘要（text_snippet）】")
        logger.info(f"  {extract_result.get('text_snippet', 'N/A')[:300]}...")

        logger.info(f"\n【内容哈希】")
        logger.info(f"  {extract_result.get('content_hash', 'N/A')}")

        # ============================================================
        # 步骤4: 质量检查
        # ============================================================
        logger.info("")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("步骤4: 质量检查")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        checks = []

        # 检查1: clean_content 是否存在
        if page_data and page_data.clean_content:
            checks.append(("✅", "trafilatura 提取成功", True))
        else:
            checks.append(("❌", "trafilatura 提取失败", False))

        # 检查2: main_content 长度
        main_content_len = len(extract_result.get('main_content', ''))
        if main_content_len > 500:
            checks.append(("✅", f"main_content 长度充足 ({main_content_len} 字符)", True))
        else:
            checks.append(("❌", f"main_content 长度不足 ({main_content_len} 字符)", False))

        # 检查3: keyword_hits
        keyword_hits = extract_result.get('keyword_hits', 0)
        if keyword_hits > 0:
            checks.append(("✅", f"关键词匹配成功 ({keyword_hits} 次)", True))
        else:
            checks.append(("⚠️ ", f"关键词未匹配 (keyword_hits=0)，可能是跨语言内容", False))

        # 检查4: text_snippet 质量
        text_snippet = extract_result.get('text_snippet', '')
        if text_snippet and len(text_snippet) > 100:
            # 检查是否包含 markdown 语法
            has_markdown = any(x in text_snippet for x in ["[](", "**", "__", "# "])
            if not has_markdown:
                checks.append(("✅", "text_snippet 干净（无 markdown 标签）", True))
            else:
                checks.append(("⚠️ ", "text_snippet 包含 markdown 标签", False))
        else:
            checks.append(("❌", "text_snippet 为空或过短", False))

        # 检查5: 提取链接数量
        links_count = extract_result.get('extracted_links_count', 0)
        if links_count > 0:
            checks.append(("✅", f"成功提取 {links_count} 个链接", True))
        else:
            checks.append(("⚠️ ", f"未提取到链接", False))

        # 打印检查结果
        all_passed = True
        for icon, message, passed in checks:
            status = "通过" if passed else "失败"
            logger.info(f"  {icon} {message} [{status}]")
            if not passed and icon == "❌":
                all_passed = False

        # ============================================================
        # 最终结论
        # ============================================================
        logger.info("")
        logger.info("=" * 80)
        if all_passed:
            logger.success("🎉 所有检查通过！项目集成正常工作。")
        else:
            logger.warning("⚠️  部分检查未通过，需要进一步检查。")
        logger.info("=" * 80)

    finally:
        # 清理资源
        await crawler_agent.cleanup()

        # 清理测试缓存
        import shutil
        cache_dir = Path(".cache/test")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info("\n已清理测试缓存")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
