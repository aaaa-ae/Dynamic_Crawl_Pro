"""
工具模块
"""

from .url_utils import (
    normalize_url,
    get_domain,
    is_allowed_domain,
    extract_links_from_markdown,
    extract_links_from_html,
    should_crawl_url,
    is_valid_url,
    ParsedLink,
)

from .text_utils import (
    count_keyword_hits,
    extract_headings,
    get_text_snippet,
    compute_content_hash,
    fast_filter,
    clean_text,
    extract_main_content_from_markdown,
    calculate_content_quality_score,
    truncate_text,
)

from .data_manager import (
    PageRecord,
    CrawledPage,
    DataManager,
    OutputWriter,
)

__all__ = [
    # URL utils
    "normalize_url",
    "get_domain",
    "is_allowed_domain",
    "extract_links_from_markdown",
    "extract_links_from_html",
    "should_crawl_url",
    "is_valid_url",
    "ParsedLink",

    # Text utils
    "count_keyword_hits",
    "extract_headings",
    "get_text_snippet",
    "compute_content_hash",
    "fast_filter",
    "clean_text",
    "extract_main_content_from_markdown",
    "calculate_content_quality_score",
    "truncate_text",

    # Data manager
    "PageRecord",
    "CrawledPage",
    "DataManager",
    "OutputWriter",
]
