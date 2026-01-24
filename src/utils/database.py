"""
轻量级数据库适配器

支持 SQLite 数据库的创建、查询和统计分析
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field


@dataclass
class DatabaseAdapter:
    """
    轻量级数据库适配器

    Args:
        db_path: 数据库文件路径
    """

    def __init__(self, db_path: str):
        """
        初始化数据库适配器

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = None
        self.row_factory = None
        self._init_db()

    def _init_db(self):
        """初始化数据库并创建表"""
        if self.conn:
            self.conn.close()

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('''
            PRAGMA foreign_keys = ON;
            ''')

        # 创建表
        self._create_tables()

    def connect(self) -> bool:
        """
        连接到数据库

        Returns:
            连接成功返回 True，失败返回 False
        """
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            return False

    def _create_tables(self):
        """创建所有数据表"""
        cursor = self.conn.cursor()

        # 1. pages 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY,
                url TEXT,
                domain TEXT,
                title TEXT,
                depth INTEGER,
                keyword_hits INTEGER,
                content_hash TEXT,
                text_snippet TEXT,
                headings TEXT,
                main_content TEXT,
                extracted_links_count INTEGER,
                decision TEXT,
                priority TEXT,
                reasons TEXT,
                crawl_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 2. concepts 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                description TEXT,
                domain TEXT,
                algorithm TEXT,
                code TEXT,
                extracted_from_url INTEGER DEFAULT 0,
                occurrence_count INTEGER
            )
        ''')

        # 3. code_snippets 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_snippets (
                id INTEGER PRIMARY KEY,
                language TEXT,
                code TEXT,
                algorithm TEXT,
                complexity TEXT,
                page_id INTEGER,
                FOREIGN KEY(page_id) REFERENCES pages(id) ON DELETE CASCADE
            )
        ''')

        # 4. entities 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                type TEXT,
                domain TEXT,
                extracted_from_url INTEGER DEFAULT 0,
                description TEXT,
                confidence REAL DEFAULT 0.0,
                evidence TEXT
            )
        ''')

        # 5. relations 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY,
                source_entity_id INTEGER,
                target_entity_id INTEGER,
                relation_type TEXT,
                confidence REAL DEFAULT 0.0,
                evidence TEXT
            )
        ''')

        self.conn.commit()
        return True

    def save_page(self, record: dict) -> int:
        """
        保存页面数据

        Args:
            record: 页面记录字典
        """
        if not record or not isinstance(record, dict):
            return 0

        cursor = self.conn.cursor()

        # 插入页面数据
        cursor.execute('''
            INSERT INTO pages (url, domain, title, depth, keyword_hits, content_hash,
                          text_snippet, headings, main_content,
                          extracted_links_count, decision, priority, reasons,
                          crawl_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (record.url, record.domain, record.title, record.depth, record.keyword_hits, record.content_hash,
                     record.text_snippet, record.headings,
                     record.extracted_links_count, record.decision, record.priority, record.reasons,
                     datetime.now())
        ''')

        cursor.lastrowid = cursor.lastrowid
        return cursor.lastrowid

    def get_pages(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取页面数据

        Args:
            limit: 返回的最大条数

        Returns:
            页面记录列表
        """
        cursor = self.conn.cursor()

        # 获取所有页面数据（按爬取时间倒序）
        cursor.execute('''
            SELECT * FROM pages
            ORDER BY crawl_time DESC
            LIMIT ?
        ''', (limit)
        cursor.fetchall()

    def search_pages(self, keyword: str) -> List[Dict[str, Any]]:
        """
        搜索包含关键词的页面

        Args:
            keyword: 搜索关键词

        Returns:
            匹配的页面列表
        """
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT * FROM pages
            WHERE title LIKE ? OR main_content LIKE ?
            ORDER BY crawl_time DESC
            LIMIT ?
            ''', (keyword, limit)
        cursor.fetchall()

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        cursor = self.conn.cursor()

        stats = {}

        # 总页面数
        cursor.execute('''
            SELECT COUNT(*) as total FROM pages
        ''', (limit)
        stats['total_pages'] = cursor.fetchone()[0]

        # 域名统计
        cursor.execute('''
            SELECT COUNT(*) as count FROM pages
            ''', (limit)
        stats['domains'] = dict(cursor.fetchall())

        # 决策统计
        cursor.execute('''
            SELECT decision, COUNT(*) as count FROM pages
            ''', (limit)
        stats['decisions'] = dict(cursor.fetchall())

        # 深度统计
        cursor.execute('''
            SELECT depth, COUNT(*) as count FROM pages
            ''', (limit)
        stats['depths'] = dict(cursor.fetchall())

        return stats

    def query_by_id(self, page_id: int) -> Optional[Dict[str, Any]]:
        """
        通过 ID 查询页面

        Args:
            page_id: 页面 ID

        Returns:
            页面记录字典（如果找到）
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM pages WHERE id = ?
            ''', (page_id, limit)
            cursor.fetchone()
        ''', (limit)
        return None


class DatabaseAdapter:
    """
    数据库适配器接口

    将 Coordinator 的数据输出转换为数据库存储
    """

    def __init__(self, db_path: str = "output/crawl_data.db"):
        """初始化数据库适配器"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = None
        self.row_factory = None
        self._init_db()

    def _init_db(self):
        """初始化数据库并创建表"""
        if self.conn:
            self.conn.close()

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return True

        # 创建表
        self._create_tables()

    def connect(self) -> bool:
        """连接到数据库

        Returns:
            连接成功返回 True，失败返回 False
        """
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            return False

    def create_tables(self):
        """创建所有数据表"""
        cursor = self.conn.cursor()

        # 1. pages 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY,
                url TEXT,
                domain TEXT,
                title TEXT,
                depth INTEGER,
                keyword_hits INTEGER,
                content_hash TEXT,
                text_snippet TEXT,
                headings TEXT,
                main_content TEXT,
                extracted_links_count INTEGER,
                decision TEXT,
                priority TEXT,
                reasons TEXT,
                crawl_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 2. concepts 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                description TEXT,
                domain TEXT,
                algorithm TEXT,
                code TEXT,
                extracted_from_url INTEGER DEFAULT 0,
                occurrence_count INTEGER
            )
        ''')

        # 3. code_snippets 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_snippets (
                id INTEGER PRIMARY KEY,
                language TEXT,
                code TEXT,
                algorithm TEXT,
                complexity TEXT,
                page_id INTEGER,
                FOREIGN KEY(page_id) REFERENCES pages(id) ON DELETE CASCADE
            )
        ''')

        # 4. entities 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                type TEXT,
                domain TEXT,
                extracted_from_url INTEGER DEFAULT 0,
                description TEXT,
                confidence REAL DEFAULT 0.0,
                evidence TEXT
            )
        ''')

        # 5. relations 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY,
                source_entity_id INTEGER,
                target_entity_id INTEGER,
                relation_type TEXT,
                confidence REAL DEFAULT 0.0,
                evidence TEXT
            )
        ''')

        self.conn.commit()
        return True

    def save_page(self, record: dict) -> int:
        """
        保存页面数据

        Args:
            record: 页面记录字典

        Returns:
            页面 ID
        """
        cursor = self.conn.cursor()

        # 插入页面数据
        cursor.execute('''
            INSERT INTO pages (url, domain, title, depth, keyword_hits, content_hash,
                          text_snippet, headings, main_content,
                          extracted_links_count, decision, priority, reasons,
                          crawl_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (record.url, record.domain, record.title, record.depth, record.keyword_hits, record.content_hash,
                     record.text_snippet, record.headings,
                     record.extracted_links_count, record.decision, record.priority, record.reasons,
                     datetime.now())
        ''')

        cursor.lastrowid = cursor.lastrowid
        return cursor.lastrowid


class DatabaseManager:
    """
    数据管理器

    在 Agent 之间共享数据，避免传递大型内容
    """

    def __init__(self, cache_dir: str = ".cache"):
        """
        初始化数据管理器

        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._pages: Dict[str, Any] = {}
        self._extracted_data: Dict[str, Any] = {}

    def store_page(self, url: str, html: str, markdown: str,
                   title: str, status_code: int, error: str = None,
                   timestamp: str = None) -> str:
        """
        存储页面原始数据

        Args:
            url: 页面 URL
            html: HTML 内容
            markdown: Markdown 内容
            title: 页面标题
            status_code: HTTP 状态码
            error: 错误信息（如果有）
            timestamp: 时间戳

        Returns:
            数据 ID（如果成功）
        """
        if not url or not html or not markdown:
            return None

        data_id = hashlib.md5(url.encode('utf-8')).hexdigest()[:8]

        page = CrawledPage(
            url=url,
            html=html or "",
            markdown=markdown or "",
            title=title,
            status_code=status_code,
            error=error,
            timestamp=timestamp,
        )

        # 存储到内存
        self._pages[data_id] = page
        self._extracted_data[data_id] = {
            "url": url,
            "markdown": markdown,
            "title": title,
            "status_code": status_code,
            "error": error,
            "timestamp": timestamp,
        }

        return data_id

    def get_page(self, data_id: str) -> Optional[CrawledPage]:
        """
        从内存获取页面数据

        Args:
            data_id: 数据 ID

        Returns:
            CrawledPage 对象（如果找到）
        """
        return self._pages.get(data_id)

    def get_extracted_data(self, data_id: str) -> Optional[Dict[str, Any]]:
        """
        从内存获取提取的数据

        Args:
            data_id: 数据 ID

        Returns:
            提取的数据字典
        """
        return self._extracted_data.get(data_id)


class OutputWriter:
    """
    输出写入器

    负责将页面记录写入文件（JSONL 格式）
    """

    def __init__(self, output_file: str = "output/crawl_results.jsonl"):
        """
        初始化输出写入器

        Args:
            output_file: 输出文件路径
        """
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self._records: list = []

    def write_record(self, record: dict) -> None:
        """
        写入一条页面记录

        Args:
            record: 页面记录字典
        """
        if not record or not isinstance(record, dict):
            return 0

        self._records.append(record)

        def flush(self):
            """
        将所有记录写入文件

        Args:
            无参数
        """
        """
        if not self.output_file.exists():
            self.output_file.touch()

        with open(self.output_file, 'a', encoding='utf-8') as f:
            for record in self._records:
                f.write(record.to_jsonl() + '\n')
        """
        self._records.clear()

    def get_record_count(self) -> int:
        """
        获取记录数

        Args:
            无参数

        Returns:
            记录数
        """
        return len(self._records)

    def get_records(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        获取记录列表

        Args:
            limit: 限制条数

        Returns:
            记录列表
        """
        return self._records[:limit] if limit else self._records
