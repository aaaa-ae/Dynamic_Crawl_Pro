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
│   ├── config.py              # 集中式配置
│   ├── main.py                # 支持命令行参数的入口
│   ├── agents/                # 核心智能体实现
│   │   ├── __init__.py
│   │   ├── base.py           # 基础抽象类
│   │   ├── crawler.py        # 爬虫 Agent (AsyncWebCrawler)
│   │   ├── extractor.py      # 提取 Agent (内容 & 链接)
│   │   └── quality_gate.py   # 质量门控 Agent (LLM 评估)
│   ├── pipeline/              # 任务流控
│   │   ├── __init__.py
│   │   └── coordinator.py    # 任务协调器 (队列管理)
│   └── utils/                 # 工具集
│       ├── __init__.py
│       ├── data_manager.py   # 数据共享 & 缓存
│       ├── url_utils.py      # URL 处理
│       └── text_utils.py     # 文本处理
├── run_crawler.py             # 便捷启动脚本
├── tests/
│   └── test_smoke.py         # 基础连通性测试
├── requirements.txt
└── README.md
```

## 安装

1. 克隆或下载本项目
2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 配置

系统提供两种配置方式：

### 1. 修改 `run_crawler.py` (推荐)
适用于 PyCharm 或本地快速开发。在 `run_crawler.py` 顶部的配置区域直接修改：

```python
TOPIC = "dynamic programming"
ENABLE_LLM = True
# ... 其他配置
```

### 2. 通过 `.env` 文件
将敏感信息（如 API Key）放入 `.env` 文件中：

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4
```

### 3. 命令行参数
`src/main.py` 支持丰富的命令行参数：

```bash
python -m src.main --topic "binary search" --max-pages 50 --enable-llm
```

## 运行

### 方法 A：使用启动脚本 (推荐)
```bash
python run_crawler.py
```

### 方法 B：命令行模块模式
```bash
python -m src.main --topic "machine learning"
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

- **职责**：基于 LLM 和规则的质量评估
- **功能**：
  - **Fast Filter**：在调用 LLM 前进行关键词命中和文本长度预判（低成本）
  - **LLM Deep Evaluation**：根据页面摘要、标题、出链样例等进行深度语义评估
  - **动态决策**：决定页面是否 `keep` (保存) 以及是否 `expand` (进一步爬取出链)
- **输入**：ExtractorAgent 的结构化输出
- **输出**：评估决策 (Keep/Discard, Expand/Don't-Expand, Priority)

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

## 贡献

欢迎提交 Issue 和 Pull Request！
