# Dynamic Crawl Pro - 知识点驱动的深度内容采集系统

一个基于 crawl4ai 异步爬虫和 CAMEL 多智能体框架的深度内容采集系统，专注于特定知识点的主题式爬取。

## 功能特性

- **异步并发爬取**：基于 crawl4ai 的 AsyncWebCrawler 实现高效爬取
- **多智能体协作**：使用 CAMEL 框架协调多个 Agent 协同工作
- **智能质量判断**：基于 LLM 的页面质量评估和决策
- **成本优化**：基于规则的预过滤，减少不必要的 LLM 调用
- **深度爬取控制**：支持最大深度、页面数、域名限制等配置
- **去重机制**：基于内容哈希的去重
- **结构化输出**：JSONL 格式输出，便于后续处理

## 项目结构

```
Dynamic_Crawl_Pro/
├── src/
│   ├── __init__.py
│   ├── config.py              # 配置文件
│   ├── main.py                # 入口文件
│   ├── agents/                # Agent 实现
│   │   ├── __init__.py
│   │   ├── base.py           # 基础 Agent 类
│   │   ├── crawler.py        # CrawlerAgent - 爬虫 Agent
│   │   ├── extractor.py      # ExtractorAgent - 内容提取 Agent
│   │   └── quality_gate.py   # QualityGateAgent - 质量判断 Agent
│   ├── pipeline/              # 任务协调
│   │   ├── __init__.py
│   │   └── coordinator.py    # 协调器
│   └── utils/                 # 工具函数
│       ├── __init__.py
│       ├── data_manager.py   # 数据管理器
│       ├── url_utils.py      # URL 工具
│       └── text_utils.py     # 文本工具
├── tests/
│   └── test_smoke.py         # 烟雾测试
├── requirements.txt          # 依赖
└── README.md                 # 本文档
```

## 安装

1. 克隆或下载本项目
2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 配置

### 基本配置

编辑 `src/main.py` 中的配置部分：

```python
# 核心配置
TOPIC = "dynamic programming"                    # 目标知识点
KEYWORDS = ["dynamic programming", "dp", "动态规划", "memoization", "tabulation"]
SEED_URLS = [
    "https://en.wikipedia.org/wiki/Dynamic_programming",
    "https://leetcode.com/tag/dynamic-programming/",
]

# 预算参数
MAX_DEPTH = 3                                    # 最大爬取深度
MAX_PAGES = 100                                  # 最大页面数
MAX_PAGES_PER_DOMAIN = 20                        # 每个域名最大页面数
CONCURRENCY = 5                                  # 并发爬取数

# 域名控制
ALLOWED_DOMAINS = ["wikipedia.org", "leetcode.com", "geeksforgeeks.org"]
BLOCKED_DOMAINS = ["youtube.com", "twitter.com", "facebook.com"]

# LLM 配置（用于 QualityGateAgent）
LLM_API_KEY = "your-api-key"                    # OpenAI API Key
LLM_MODEL = "gpt-3.5-turbo"                      # 或 gpt-4
LLM_BASE_URL = "https://api.openai.com/v1"       # 可选：自定义 API 地址

# 输出配置
OUTPUT_FILE = "output/dp_crawl_results.jsonl"    # 输出文件路径
```

### LLM 配置选项

系统支持多种 LLM 提供商：

1. **OpenAI**（默认）
   ```python
   LLM_API_KEY = "sk-..."
   LLM_MODEL = "gpt-4"
   LLM_BASE_URL = "https://api.openai.com/v1"
   ```

2. **Azure OpenAI**
   ```python
   LLM_API_KEY = "your-azure-key"
   LLM_MODEL = "gpt-4"
   LLM_BASE_URL = "https://your-resource.openai.azure.com/"
   ```

3. **其他兼容 OpenAI API 的服务**（如 DeepSeek、Moonshot 等）
   ```python
   LLM_API_KEY = "your-key"
   LLM_MODEL = "deepseek-chat"
   LLM_BASE_URL = "https://api.deepseek.com/v1"
   ```

## 运行

### 基本用法

```bash
# 从项目根目录运行
python -m src.main
```

### 自定义配置运行

修改 `src/main.py` 中的配置参数，然后运行：

```bash
python -m src.main --topic "binary search"
```

### 命令行参数（可选）

```bash
python -m src.main \
  --topic "binary search" \
  --seed-url "https://en.wikipedia.org/wiki/Binary_search_algorithm" \
  --max-depth 3 \
  --max-pages 50 \
  --concurrency 5
```

## 输出格式

输出为 JSONL 文件，每行一条记录：

```json
{
  "url": "https://example.com/page",
  "depth": 1,
  "domain": "example.com",
  "title": "Page Title",
  "keyword_hits": 3,
  "content_hash": "a1b2c3d4e5f6...",
  "text_snippet": "This is the first 300 characters of the main content...",
  "extracted_links_count": 15,
  "headings": ["H1", "H2", ...],
  "decision": "keep",
  "priority": "high",
  "reasons": "Contains detailed explanation of the concept with code examples"
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `url` | string | 页面 URL |
| `depth` | int | 爬取深度 |
| `domain` | string | 域名 |
| `title` | string | 页面标题 |
| `keyword_hits` | int | 关键词匹配次数 |
| `content_hash` | string | 内容 MD5 哈希（用于去重） |
| `text_snippet` | string | 正文前 300 字符 |
| `extracted_links_count` | int | 提取的有效链接数 |
| `headings` | array | 页面标题列表 |
| `decision` | string | 决策：`keep` / `discard` |
| `priority` | string | 优先级：`high` / `medium` / `low` |
| `reasons` | string | 决策理由 |

## Agent 架构

### 1. CrawlerAgent（爬虫 Agent）

- **职责**：使用 AsyncWebCrawler 拉取页面
- **功能**：
  - 异步并发爬取
  - 重试机制（指数退避）
  - 获取 HTML 内容
- **输出**：`data_id`（传递给 ExtractorAgent）

### 2. ExtractorAgent（内容提取 Agent）

- **职责**：提取页面正文和链接
- **功能**：
  - 提取 main-content（正文内容）
  - 提取外部链接
  - 链接规范化
  - 关键词匹配统计
- **输出**：提取的结构化数据

### 3. QualityGateAgent（质量判断 Agent）

- **职责**：基于 LLM 的质量判断
- **功能**：
  - 基于规则的预过滤（fast_filter）
  - LLM 深度评估
  - 决策：keep/discard、expand/don't-expand、priority
- **输入**：title、正文片段、headings、关键词匹配数、出链样例
- **输出**：决策和理由

## 任务协调流程

```
┌─────────────┐
│  Seed URLs  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│CrawlerAgent │────▶│ ExtractorAgent   │────▶│QualityGateAgent │
│(爬取页面)   │     │ (提取内容和链接)  │     │  (质量判断)      │
└──────┬──────┘     └────────┬─────────┘     └────────┬─────────┘
       │                     │                        │
       ▼                     ▼                        ▼
  data_id            结构化数据                决策 + 新URL
   (共享存储)           (LLM评估)              (加入队列)
```

## 成本优化

### 快速过滤器 (fast_filter)

在调用 LLM 之前，系统会先执行快速规则过滤：

1. 如果关键词匹配数为 0 且文本长度 < 500 字符 → 直接丢弃
2. 只有通过预判的页面才会发送给 LLM 评估

这可以显著减少 LLM Token 消耗。

### 数据流优化

- CrawlerAgent 不直接传递 HTML 内容，而是传递 `data_id`
- 其他 Agent 通过 `DataManager` 从共享存储读取数据
- 减少内存使用和消息传递开销

## 示例：爬取不同主题

### 1. 爬取 "Machine Learning" 相关内容

```python
TOPIC = "machine learning"
KEYWORDS = ["machine learning", "ml", "neural network", "deep learning"]
SEED_URLS = [
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://scikit-learn.org/stable/",
]
ALLOWED_DOMAINS = ["wikipedia.org", "scikit-learn.org", "pytorch.org"]
```

### 2. 爬取 "React Framework" 相关内容

```python
TOPIC = "react framework"
KEYWORDS = ["react", "reactjs", "hooks", "jsx", "virtual dom"]
SEED_URLS = [
    "https://react.dev/",
    "https://legacy.reactjs.org/",
]
ALLOWED_DOMAINS = ["react.dev", "legacy.reactjs.org", "reactjs.org"]
```

## 测试

运行烟雾测试：

```bash
python tests/test_smoke.py
```

## 常见问题

### 1. 爬取失败怎么办？

检查：
- 网络连接是否正常
- 目标网站是否有反爬机制
- `crawl4ai` 是否正确安装（`pip install crawl4ai` 并运行 `playwright install`）

### 2. LLM 调用报错

检查：
- API Key 是否正确
- API 额度是否充足
- BASE_URL 是否正确配置

### 3. 如何禁用 LLM 功能？

将 `ENABLE_LLM` 设置为 `False`，系统将只使用规则判断：

```python
ENABLE_LLM = False  # 仅使用 fast_filter 进行判断
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
