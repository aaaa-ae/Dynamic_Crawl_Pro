"""
文本工具模块
处理文本处理、关键词匹配、哈希计算等操作
"""

import hashlib
import re
from typing import List, Dict, Set, Tuple


def count_keyword_hits(text: str, keywords: List[str]) -> int:
    """
    统计文本中关键词匹配次数

    Args:
        text: 待搜索的文本
        keywords: 关键词列表

    Returns:
        匹配次数总和
    """
    if not text or not keywords:
        return 0

    text_lower = text.lower()
    total_hits = 0

    for keyword in keywords:
        if not keyword:
            continue
        keyword_lower = keyword.lower()
        # 使用正则进行单词边界匹配，提高准确性
        pattern = r'\b' + re.escape(keyword_lower) + r'\b'
        matches = re.findall(pattern, text_lower)
        total_hits += len(matches)

    return total_hits


def extract_headings(text: str, max_headings: int = 10) -> List[str]:
    """
    从 Markdown 文本中提取标题

    Args:
        text: Markdown 文本
        max_headings: 最多提取的标题数量

    Returns:
        标题列表
    """
    headings = []

    # 匹配 Markdown 标题: # 至 ######
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    for match in heading_pattern.finditer(text):
        level = len(match.group(1))
        heading = match.group(2).strip()
        headings.append(heading)

        if len(headings) >= max_headings:
            break

    return headings


def get_text_snippet(text: str, max_length: int = 300) -> str:
    """
    获取文本摘要片段（跳过导航菜单）

    Args:
        text: 原始文本
        max_length: 最大长度

    Returns:
        文本摘要（截断并添加省略号）
    """
    if not text:
        return ""

    # 移除多余空白
    text = re.sub(r'\s+', ' ', text).strip()

    # 策略：跳过常见的导航模式
    skip_patterns = [
        r'\[Main menu\]',
        r'\[Contents\]',
        r'\[Jump to content\]',
        r'\[Current events\]',
        r'\[Random article\]',
        r'\[About Wikipedia\]',
        r'\[Contact us\]',
        r'\[Contribute',
        r'\[Help\]',
        r'\[Community portal\]',
        r'\[Recent changes\]',
        r'\[Upload file\]',
        r'\[Special pages\]',
        r'\[Donate',
        r'\[Create account\]',
        r'\[Log in\]',
        r'\[Appearance',
        r'\[Search',
        r'Sign In',
        r'Register',
        r'Courses',
        r'\* Courses',
        r'\* Tutorials',
        r'\* Interview Prep',
        r'\* DSA Tutorial',
        r'\* Interview Questions',
        r'\* Quizzes',
        r'\* Must Do',
        r'\* Advanced DSA',
        r'\* System Design',
        r'\* Aptitude',
        r'\* Puzzles',
        r'\* Interview Corner',
        r'\* DSA Python',
        r'## Contents',
    ]

    # 方法1：跳过导航部分（从第一个实际内容开始）
    lines = text.split('\n')
    content_start = 0
    for i, line in enumerate(lines):
        # 跳过导航行（包含特定链接的行）
        if any(pattern in line for pattern in skip_patterns):
            continue
        # 找到正文开始（包含实际内容而非链接）
        if line.strip() and not line.startswith('*') and not line.startswith('['):
            content_start = i
            break

    # 从正文开始位置提取
    if content_start > 0:
        text = '\n'.join(lines[content_start:])
    else:
        text = lines[0] if lines else ""

    # 限制长度
    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] + "..."

    return text


def compute_content_hash(text: str) -> str:
    """
    计算文本内容的哈希值（用于去重）

    Args:
        text: 文本内容

    Returns:
        MD5 哈希字符串
    """
    if not text:
        return ""

    # 移除空白后再计算哈希，避免格式差异影响
    normalized = re.sub(r'\s+', ' ', text).strip().encode('utf-8')
    return hashlib.md5(normalized).hexdigest()


def fast_filter(
    text: str,
    keywords: List[str],
    min_length: int = 500,
    min_keyword_hits: int = 1
) -> Tuple[bool, str]:
    """
    快速过滤器：基于规则的预判

    Args:
        text: 待评估的文本
        keywords: 关键词列表
        min_length: 最小文本长度
        min_keyword_hits: 最小关键词匹配数

    Returns:
        (是否通过过滤, 原因)
    """
    if not text:
        return False, "Empty text"

    text_length = len(text)

    # 规则1: 文本长度太短
    if text_length < min_length:
        return False, f"Text too short ({text_length} < {min_length})"

    # 规则2: 关键词匹配数不足
    keyword_hits = count_keyword_hits(text, keywords)
    if keyword_hits < min_keyword_hits:
        return False, f"Insufficient keyword hits ({keyword_hits} < {min_keyword_hits})"

    # 规则3: 内容密度检查（非空字符比例）
    non_space_chars = len(re.sub(r'\s', '', text))
    density = non_space_chars / text_length if text_length > 0 else 0
    if density < 0.1:
        return False, f"Low content density ({density:.2f} < 0.1)"

    return True, "Passed fast filter"


def clean_text(text: str) -> str:
    """
    清理文本：移除多余空白、特殊字符等

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    if not text:
        return ""

    # 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 移除多余空白
    text = re.sub(r'[ \t]+', ' ', text)  # 多个空格/制表符
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # 多个空行

    return text.strip()


def extract_main_content_from_markdown(markdown: str, max_length: int = 10000) -> str:
    """
    从 Markdown 中提取主要内容

    Args:
        markdown: Markdown 文本
        max_length: 最大长度

    Returns:
        提取的主要内容
    """
    if not markdown:
        return ""

    # 简单实现：移除代码块、引用等，保留正文
    lines = markdown.split('\n')
    main_content = []

    skip_next = False

    for i, line in enumerate(lines):
        # 跳过代码块
        if line.strip().startswith('```'):
            skip_next = not skip_next
            continue

        if skip_next:
            continue

        # 跳过 HTML 注释
        if '<!--' in line and '-->' in line:
            continue

        # 跳过纯引用块
        if line.strip().startswith('>'):
            continue

        main_content.append(line)

    content = '\n'.join(main_content).strip()

    if len(content) > max_length:
        content = content[:max_length]

    return content


def calculate_content_quality_score(
    text: str,
    keywords: List[str],
    headings: List[str] = None
) -> float:
    """
    计算内容质量分数（0-1）

    Args:
        text: 文本内容
        keywords: 关键词列表
        headings: 标题列表

    Returns:
        质量分数
    """
    if not text:
        return 0.0

    score = 0.0

    # 1. 关键词匹配分数 (0-0.4)
    keyword_hits = count_keyword_hits(text, keywords)
    if keywords:
        expected_hits = len(keywords)
        keyword_score = min(keyword_hits / max(expected_hits, 1), 1.0)
        score += keyword_score * 0.4

    # 2. 文本长度分数 (0-0.3)
    text_length = len(text)
    if text_length > 3000:
        length_score = 1.0
    elif text_length > 1000:
        length_score = 0.7
    elif text_length > 500:
        length_score = 0.5
    else:
        length_score = 0.2
    score += length_score * 0.3

    # 3. 标题质量分数 (0-0.3)
    if headings and len(headings) > 0:
        # 检查标题中是否包含关键词
        heading_keyword_hits = sum(
            1 for h in headings if any(k.lower() in h.lower() for k in keywords)
        )
        heading_score = min(heading_keyword_hits / max(len(headings), 1), 1.0)
        score += heading_score * 0.3
    else:
        score += 0.1  # 没有标题但有内容，给一点分数

    return round(score, 3)


def truncate_text(text: str, max_length: int) -> str:
    """
    截断文本到指定长度

    Args:
        text: 原始文本
        max_length: 最大长度

    Returns:
        截断后的文本
    """
    if not text or len(text) <= max_length:
        return text

    return text[:max_length].rsplit(' ', 1)[0] + "..."
