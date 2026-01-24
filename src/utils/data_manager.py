"""
数据管理器模块
用于在 Agent 之间共享数据，避免直接传递大型 HTML 内容
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Dict
from dataclasses import dataclass, asdict


@dataclass
class PageRecord:
    """页面记录数据结构"""
    url: str
    depth: int
    domain: str
    title: str
    keyword_hits: int
    content_hash: str
    text_snippet: str
    extracted_links_count: int
    headings: list = None
    decision: str = "pending"
    priority: str = "medium"
    reasons: str = ""
    main_content: str = ""  # 完整的正文内容

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def to_jsonl(self) -> str:
        """转换为 JSONL 格式字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageRecord":
        """从字典创建实例"""
        return cls(**data)


@dataclass
class CrawledPage:
    """爬取的页面原始数据"""
    url: str
    html: str
    markdown: str
    title: str
    status_code: int
    error: Optional[str] = None
    timestamp: Optional[str] = None


class DataManager:
    """
    数据管理器
    用于在 Agent 之间共享数据，避免直接传递大型内容
    """

    def __init__(self, cache_dir: str = ".cache", enable_persistence: bool = True):
        """
        初始化数据管理器

        Args:
            cache_dir: 缓存目录
            enable_persistence: 是否持久化到磁盘
        """
        self.cache_dir = Path(cache_dir)
        self.enable_persistence = enable_persistence

        # 内存缓存
        self._pages: Dict[str, CrawledPage] = {}
        self._extracted_data: Dict[str, Dict[str, Any]] = {}

        # 如果启用持久化，创建缓存目录
        if self.enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "pages").mkdir(exist_ok=True)
            (self.cache_dir / "extracted").mkdir(exist_ok=True)

    def generate_id(self, url: str) -> str:
        """
        生成数据 ID（基于 URL 的哈希）

        Args:
            url: URL 字符串

        Returns:
            数据 ID
        """
        return hashlib.md5(url.encode('utf-8')).hexdigest()

    def store_page(self, url: str, html: str, markdown: str,
                   title: str, status_code: int, error: str = None) -> str:
        """
        存储爬取的页面数据

        Args:
            url: 页面 URL
            html: HTML 内容
            markdown: Markdown 内容
            title: 页面标题
            status_code: HTTP 状态码
            error: 错误信息（如果有）

        Returns:
            数据 ID
        """
        from datetime import datetime

        data_id = self.generate_id(url)

        page = CrawledPage(
            url=url,
            html=html,
            markdown=markdown,
            title=title,
            status_code=status_code,
            error=error,
            timestamp=datetime.now().isoformat()
        )

        # 存储到内存
        self._pages[data_id] = page

        # 持久化到磁盘（如果启用）
        if self.enable_persistence:
            self._persist_page(data_id, page)

        return data_id

    def get_page(self, data_id: str) -> Optional[CrawledPage]:
        """
        获取页面数据

        Args:
            data_id: 数据 ID

        Returns:
            CrawledPage 实例，如果不存在则返回 None
        """
        # 先从内存获取
        if data_id in self._pages:
            return self._pages[data_id]

        # 如果内存中没有，尝试从磁盘加载
        if self.enable_persistence:
            return self._load_page(data_id)

        return None

    def store_extracted_data(self, data_id: str, data: Dict[str, Any]) -> None:
        """
        存储提取的数据

        Args:
            data_id: 数据 ID
            data: 提取的数据字典
        """
        self._extracted_data[data_id] = data

        if self.enable_persistence:
            self._persist_extracted_data(data_id, data)

    def get_extracted_data(self, data_id: str) -> Optional[Dict[str, Any]]:
        """
        获取提取的数据

        Args:
            data_id: 数据 ID

        Returns:
            数据字典，如果不存在则返回 None
        """
        if data_id in self._extracted_data:
            return self._extracted_data[data_id]

        if self.enable_persistence:
            return self._load_extracted_data(data_id)

        return None

    def _persist_page(self, data_id: str, page: CrawledPage) -> None:
        """持久化页面数据到磁盘"""
        file_path = self.cache_dir / "pages" / f"{data_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(page), f, ensure_ascii=False, indent=2)

    def _load_page(self, data_id: str) -> Optional[CrawledPage]:
        """从磁盘加载页面数据"""
        file_path = self.cache_dir / "pages" / f"{data_id}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                page = CrawledPage(**data)
                self._pages[data_id] = page  # 缓存到内存
                return page
        except Exception:
            return None

    def _persist_extracted_data(self, data_id: str, data: Dict[str, Any]) -> None:
        """持久化提取的数据到磁盘"""
        file_path = self.cache_dir / "extracted" / f"{data_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_extracted_data(self, data_id: str) -> Optional[Dict[str, Any]]:
        """从磁盘加载提取的数据"""
        file_path = self.cache_dir / "extracted" / f"{data_id}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._extracted_data[data_id] = data  # 缓存到内存
                return data
        except Exception:
            return None

    def clear_cache(self) -> None:
        """清除缓存"""
        self._pages.clear()
        self._extracted_data.clear()

        if self.enable_persistence and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "pages").mkdir(exist_ok=True)
            (self.cache_dir / "extracted").mkdir(exist_ok=True)

    def get_cache_size(self) -> int:
        """获取缓存大小（页面数量）"""
        return len(self._pages)

    def get_cache_dir(self) -> str:
        """获取缓存目录路径"""
        return str(self.cache_dir)


class OutputWriter:
    """
    输出写入器
    将 PageRecord 写入 JSONL 文件
    """

    def __init__(self, output_file: str):
        """
        初始化输出写入器

        Args:
            output_file: 输出文件路径
        """
        self.output_file = Path(output_file)
        self._records: list = []

        # 确保输出目录存在
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def write_record(self, record: PageRecord) -> None:
        """
        写入一条记录

        Args:
            record: PageRecord 实例
        """
        self._records.append(record)

    def flush(self) -> None:
        """将所有记录写入文件"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for record in self._records:
                f.write(record.to_jsonl() + '\n')

    def append_record(self, record: PageRecord) -> None:
        """
        追加一条记录到文件

        Args:
            record: PageRecord 实例
        """
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(record.to_jsonl() + '\n')

    def get_record_count(self) -> int:
        """获取已写入的记录数量"""
        return len(self._records)
