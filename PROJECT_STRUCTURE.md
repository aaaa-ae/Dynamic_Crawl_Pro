# Project Structure

```
Dynamic_Crawl_Pro/
├── docs/
│   └── Pipeline1_and_2.md     # Pipeline 1（爬news） 和 Pipeline 2（深入爬知识点） 理解说明
├── src/                              # 源代码目录
│   ├── __init__.py                   # 包初始化文件
│   ├── config.py                     # 配置模块（核心、LLM、Agent配置）
│   ├── main.py                       # 主入口文件（支持命令行参数）
│   ├── agents/                       # Agent 实现目录
│   │   ├── __init__.py
│   │   ├── base.py                   # 基础 Agent 抽象类
│   │   ├── crawler.py                # CrawlerAgent - 爬虫 Agent (无需 LLM)
│   │   ├── extractor.py              # ExtractorAgent - 内容提取 Agent (无需 LLM)
│   │   └── quality_gate.py           # QualityGateAgent - 质量判断 Agent (CAMEL LLM)
│   ├── pipeline/                     # 任务协调目录
│   │   ├── __init__.py
│   │   └── coordinator.py            # TaskCoordinator - 任务协调器 (可选对话式协调)
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

#### CrawlConfig
爬虫基础配置：
- `topic`: 目标知识点
- `keywords`: 关键词列表
- `seed_urls`: 种子 URL 列表
- `max_depth`: 最大爬取深度
- `max_pages`: 最大页面数
- `max_pages_per_domain`: 每域名最大页面数
- `concurrency`: 并发爬取数
- `allowed_domains`: 允许的域名
- `blocked_domains`: 阻止的域名
- `output_file`: 输出文件路径
- `enable_cache`: 是否启用缓存
- `cache_dir`: 缓存目录

#### LLMConfig
LLM 相关配置：
- `enabled`: 是否启用 LLM
- `api_key`: OpenAI API Key
- `model`: 模型名称 (如 `gpt-4o-mini`, `gpt-3.5-turbo`)
- `base_url`: API 基础 URL (支持自定义代理)
- `temperature`: 温度参数
- `max_tokens`: 最大 Token 数
- `fast_filter_min_length`: 快速过滤最小长度
- `fast_filter_min_keyword_hits`: 快速过滤最小关键词命中数
- `min_relevance_score`: 最小相关性评分
- `enable_conversation_coordinator`: 是否启用对话式协调器（默认 `False`）
- `coordinator_model`: 对话式协调器使用的模型

#### AgentConfig
各个 Agent 的具体参数：
- `extract_max_length`: 提取最大长度
- `extract_text_snippet_length`: 文本摘要长度
- `extract_headings_max`: 提取标题最大数
- `quality_decision_threshold`: 质量判断阈值
- `extract_links_max`: 提取链接最大数
- `filter_same_domain`: 是否过滤同域链接

### 2. Agent 模块 (`src/agents/`)

#### BaseAgent
抽象基类，提供统一的任务处理接口和指标统计功能：
- `process(input_data)`: 处理输入数据（抽象方法）
- `can_process(input_data)`: 判断是否能够处理
- `get_metrics()`: 获取统计指标

#### CrawlerAgent
基于 `crawl4ai` 的 `AsyncWebCrawler` 实现：
- **LLM 使用**：❌ 否（纯技术任务）
- 异步并发爬取
- 重试机制（指数退避）
- 获取 HTML 内容并存储到 DataManager

#### ExtractorAgent
负责从原始页面（HTML/Markdown）提取结构化数据：
- **LLM 使用**：❌ 否（基于规则提取）
- 提取标题、正文、链接
- 关键词匹配统计
- 链接规范化和过滤

#### QualityGateAgent
系统核心，结合简单规则和 LLM 评估页面的相关性和质量：
- **技术栈**：**CAMEL ChatAgent** + OpenAI API
- **LLM 使用**：✅ 是
- Fast Filter：在调用 LLM 前进行关键词命中和文本长度预判（低成本）
- CAMEL LLM Deep Evaluation：根据页面摘要、标题、出链样例等进行深度语义评估
- 动态决策：决定页面是否 `keep` (保存) 以及是否 `expand` (进一步爬取出链)

### 3. 工具模块 (`src/utils/`)

#### url_utils.py
处理 URL 相关功能：
- URL 规范化
- 域名提取
- 链接过滤（媒体文件、已访问等）

#### text_utils.py
文本处理工具：
- 文本清洗
- 关键词统计
- 内容哈希计算（MD5）
- 文本截断

#### data_manager.py
实现数据共享层：
- 内存缓存（避免重复存储）
- 磁盘持久化（可选）
- PageRecord 输出管理

### 4. 任务协调器 (`src/pipeline/coordinator.py`)

核心调度逻辑：
- 管理异步任务队列
- 协调 Agent 间的数据流转
- 控制整体预算（最大页面数、域名限额、并发数等）
- 实时统计爬取指标
- **对话式协调（可选）**：使用 CAMEL ChatAgent 综合多个 Agent 的结果

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

可选对话式协调：
QualityGateAgent ──► TaskCoordinator (CAMEL) ──► 最终决策
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `.env` 文件或在 `run_crawler.py` 中配置：

```env
# .env 文件配置
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai-proxy.org/v1
```

```python
# run_crawler.py 中配置
TOPIC = "dynamic programming"
KEYWORDS = ["dynamic programming", "dp"]
SEED_URLS = ["https://en.wikipedia.org/wiki/Dynamic_programming"]
MAX_DEPTH = 3
MAX_PAGES = 50
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
| `enable_cache` | 是否启用缓存 | True |
| `cache_dir` | 缓存目录 | ".cache" |
| `output_file` | 输出文件路径 | "output/crawl_results.jsonl" |

### LLM 配置 (`LLMConfig`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `enabled` | 是否启用 LLM | True |
| `api_key` | OpenAI API Key | "" |
| `model` | 模型名称 | "gpt-3.5-turbo" |
| `base_url` | API 基础 URL | "https://api.openai.com/v1" |
| `temperature` | 温度参数 | 0.3 |
| `max_tokens` | 最大 Token 数 | 500 |
| `fast_filter_min_length` | 快速过滤最小长度 | 500 |
| `fast_filter_min_keyword_hits` | 快速过滤最小关键词命中数 | 1 |
| `min_relevance_score` | 最小相关性评分 | 0.5 |
| `enable_conversation_coordinator` | 是否启用对话式协调器 | False |
| `coordinator_model` | 对话式协调器使用的模型 | "gpt-3.5-turbo" |

### Agent 配置 (`AgentConfig`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `extract_max_length` | 提取最大长度 | 10000 |
| `extract_text_snippet_length` | 文本摘要长度 | 300 |
| `extract_headings_max` | 提取标题最大数 | 10 |
| `quality_decision_threshold` | 质量判断阈值 | 0.6 |
| `extract_links_max` | 提取链接最大数 | 50 |
| `filter_same_domain` | 是否过滤同域链接 | True |

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

### 启用对话式协调

在 `run_crawler.py` 中添加：

```python
llm_config = LLMConfig(
    enabled=True,
    api_key=LLM_API_KEY,
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    enable_conversation_coordinator=True,  # 启用对话式协调
    coordinator_model="gpt-3.5-turbo",      # 协调器使用的模型
)
```

## 故障排查

### 爬取失败

1. 检查网络连接
2. 确认目标网站可访问
3. 查看 `logs/` 目录中的日志文件
4. 检查 `crawl4ai` 是否正确安装
5. 运行 `playwright install` 安装浏览器依赖

### LLM 调用失败

1. 确认 API Key 正确
2. 检查 API 额度
3. 确认 BASE_URL 正确
4. 检查 `.env` 文件是否存在
5. 可以禁用 LLM（`ENABLE_LLM = False`）
6. 检查 CAMEL 初始化日志中的错误信息

### CAMEL 初始化问题

如果看到 `[QualityGateAgent] Failed to initialize CAMEL agent` 警告：

1. 检查 `OPENAI_API_KEY` 是否正确设置
2. 检查 `OPENAI_BASE_URL` 是否可访问
3. 检查模型名称是否受支持（如 `gpt-4o-mini`, `gpt-3.5-turbo` 等）
4. 如果使用自定义代理 URL，确认 URL 格式正确

## 性能优化建议

1. **调整并发数**: 根据网络和 CPU 性能调整 `CONCURRENCY`
2. **限制域名**: 设置 `allowed_domains` 减少无关内容
3. **深度控制**: 设置合理的 `max_depth` 避免无限扩展
4. **禁用缓存**: 如果不需要缓存，设置 `enable_cache=False`
5. **使用快速过滤**: 调整 `fast_filter_min_length` 减少 LLM 调用
6. **选择合适的模型**: 根据任务复杂度选择 `gpt-4o-mini` 或 `gpt-3.5-turbo`
7. **对话式协调**: 只在需要复杂决策时启用，会增加额外 LLM 调用成本
