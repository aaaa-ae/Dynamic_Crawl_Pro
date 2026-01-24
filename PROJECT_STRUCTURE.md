# Project Structure

```
Dynamic_Crawl_Pro/
├── src/                              # 源代码目录
│   ├── __init__.py                   # 包初始化文件
│   ├── config.py                     # 配置模块（核心、LLM、Agent配置）
│   ├── main.py                       # 主入口文件（支持命令行参数）
│   ├── agents/                       # Agent 实现目录
│   │   ├── __init__.py
│   │   ├── base.py                   # 基础 Agent 抽象类
│   │   ├── crawler.py                # CrawlerAgent - 爬虫 Agent
│   │   ├── extractor.py              # ExtractorAgent - 内容提取 Agent
│   │   └── quality_gate.py           # QualityGateAgent - 质量判断 Agent
│   ├── pipeline/                     # 任务协调目录
│   │   ├── __init__.py
│   │   └── coordinator.py            # TaskCoordinator - 任务协调器
│   └── utils/                        # 工具函数目录
│       ├── __init__.py
│       ├── url_utils.py              # URL 工具
│       ├── text_utils.py             # 文本工具
│       └── data_manager.py           # 数据管理器
│
├── tests/                            # 测试目录
│   ├── __init__.py
│   └── test_smoke.py                 # 烟雾测试
│
├── examples/                         # 示例目录
│   ├── __init__.py
│   └── example_configs.py           # 配置示例
│
├── run_crawler.py                    # 启动文件（Pycharm/本地运行推荐）
├── deep_research_demo.py             # 深度研究演示脚本
├── .env                              # 环境变量配置
├── requirements.txt                  # Python 依赖
├── README.md                         # 项目说明文档
├── PROJECT_STRUCTURE.md              # 本文档
│
├── output/                           # 输出目录（运行时自动创建）
├── logs/                             # 日志目录（运行时自动创建）
└── .cache/                           # 缓存目录（运行时自动创建）
```

## 核心模块说明

### 1. 配置模块 (`src/config.py`)

定义三个核心配置类：
- `CrawlConfig`: 爬虫基础配置（topic, keywords, max_depth, max_pages 等）
- `LLMConfig`: LLM 相关配置（enabled, api_key, model, base_url 等）
- `AgentConfig`: 各个 Agent 的具体参数（提取长度、质量评分阈值等）

### 2. Agent 模块 (`src/agents/`)

- **BaseAgent**: 抽象基类，提供统一的任务处理接口和指标统计功能。
- **CrawlerAgent**: 基于 `crawl4ai` 的 `AsyncWebCrawler` 实现，支持并发爬取和重试。
- **ExtractorAgent**: 负责从原始页面（HTML/Markdown）提取标题、正文、链接和关键词命中统计。
- **QualityGateAgent**: 系统核心，结合简单规则（Fast Filter）和 LLM 评估页面的相关性和质量。

### 3. 工具模块 (`src/utils/`)

- **url_utils.py**: 处理 URL 规范化、域名提取及过滤。
- **text_utils.py**: 提供文本清洗、关键词统计及内容哈希计算。
- **data_manager.py**: 实现数据共享层，提供内存和磁盘缓存，管理 `PageRecord` 输出。

### 4. 任务协调器 (`src/pipeline/coordinator.py`)

核心调度逻辑：
- 管理异步任务队列。
- 协调 Agent 间的数据流转。
- 控制整体预算（最大页面数、域名限额、并发数等）。
- 实时统计爬取指标。

## 数据流

```
Seed URLs
    │
    ▼
[asyncio.Queue] ──┐
    │              │
    ▼              │
CrawlerAgent       │
    │              │
    ▼              │
DataManager        │
(data_id)          │
    │              │
    ▼              │
ExtractorAgent ───┘
    │
    ▼
QualityGateAgent
    │
    ├─► keep ──► PageRecord ──► OutputWriter ──► JSONL
    │
    └─► expand ──► new URLs ──► [asyncio.Queue]
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `src/main.py` 中的配置，或通过命令行参数指定：

```bash
python -m src.main --topic "machine learning" --max-pages 50
```

### 3. 运行

```bash
# 推荐用法：直接运行启动脚本
python run_crawler.py

# 命令行模式（支持动态参数覆盖）
python -m src.main --topic "machine learning" --max-pages 50
```

### 4. 查看输出

输出文件位于 `output/` 目录，每行一个 JSON 记录：

```bash
cat output/dp_crawl_results.jsonl | head -5
```

## 配置参数详解

### 爬虫配置 (`CrawlConfig`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `topic` | 目标知识点 | "dynamic programming" |
| `keywords` | 关键词列表 | ["dynamic programming", "dp"] |
| `seed_urls` | 种子 URL 列表 | [] |
| `max_depth` | 最大爬取深度 | 3 |
| `max_pages` | 最大页面数 | 100 |
| `max_pages_per_domain` | 每域名最大页面数 | 20 |
| `concurrency` | 并发爬取数 | 5 |
| `allowed_domains` | 允许的域名 | [] |
| `blocked_domains` | 阻止的域名 | [] |

### LLM 配置 (`LLMConfig`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `enabled` | 是否启用 LLM | False |
| `api_key` | OpenAI API Key | "" |
| `model` | 模型名称 | "gpt-3.5-turbo" |
| `base_url` | API 基础 URL | "https://api.openai.com/v1" |

## 扩展和定制

### 添加新的 Agent

继承 `BaseAgent` 类并实现 `process` 和 `can_process` 方法：

```python
from .base import BaseAgent

class CustomAgent(BaseAgent):
    async def process(self, input_data):
        # 处理逻辑
        pass

    def can_process(self, input_data):
        # 判断是否能够处理
        return True
```

### 自定义 URL 过滤规则

修改 `url_utils.py` 中的 `should_crawl_url` 函数，或在协调器中添加自定义逻辑。

### 自定义质量判断标准

修改 `quality_gate.py` 中的提示词（prompt）或评分逻辑。

## 故障排查

### 爬取失败

1. 检查网络连接
2. 确认目标网站可访问
3. 查看 `logs/` 目录中的日志文件
4. 检查 `crawl4ai` 是否正确安装

### LLM 调用失败

1. 确认 API Key 正确
2. 检查 API 额度
3. 确认 BASE_URL 正确
4. 可以禁用 LLM（`ENABLE_LLM = False`）

## 性能优化建议

1. **调整并发数**: 根据网络和 CPU 性能调整 `CONCURRENCY`
2. **限制域名**: 设置 `allowed_domains` 减少无关内容
3. **深度控制**: 设置合理的 `max_depth` 避免无限扩展
4. **禁用缓存**: 如果不需要缓存，设置 `enable_cache=False`
5. **使用快速过滤**: 调整 `fast_filter_min_length` 减少 LLM 调用
