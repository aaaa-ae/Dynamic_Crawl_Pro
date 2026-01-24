"""
示例配置文件
展示不同知识点的配置示例
"""

# ========================================
# 示例 1: 爬取 "Binary Search" 相关内容
# ========================================

BINARY_SEARCH_CONFIG = {
    "topic": "binary search",
    "keywords": ["binary search", "二分搜索", "binary search algorithm", "divide and conquer"],
    "seed_urls": [
        "https://en.wikipedia.org/wiki/Binary_search_algorithm",
        "https://leetcode.com/tag/binary-search/",
    ],
    "max_depth": 2,
    "max_pages": 50,
    "concurrency": 3,
    "allowed_domains": ["wikipedia.org", "leetcode.com", "geeksforgeeks.org"],
    "blocked_domains": ["youtube.com", "twitter.com", "facebook.com"],
    "output_file": "output/binary_search_results.jsonl",
}

# ========================================
# 示例 2: 爬取 "Machine Learning" 相关内容
# ========================================

MACHINE_LEARNING_CONFIG = {
    "topic": "machine learning",
    "keywords": [
        "machine learning",
        "ml",
        "neural network",
        "deep learning",
        "supervised learning",
        "unsupervised learning",
        "tensorflow",
        "pytorch",
    ],
    "seed_urls": [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://scikit-learn.org/stable/",
    ],
    "max_depth": 2,
    "max_pages": 100,
    "concurrency": 5,
    "allowed_domains": ["wikipedia.org", "scikit-learn.org", "pytorch.org", "tensorflow.org"],
    "blocked_domains": ["youtube.com", "twitter.com", "facebook.com"],
    "output_file": "output/ml_results.jsonl",
}

# ========================================
# 示例 3: 爬取 "React Framework" 相关内容
# ========================================

REACT_CONFIG = {
    "topic": "react framework",
    "keywords": [
        "react",
        "reactjs",
        "react.js",
        "react component",
        "hooks",
        "jsx",
        "virtual dom",
    ],
    "seed_urls": [
        "https://react.dev/",
        "https://legacy.reactjs.org/",
    ],
    "max_depth": 2,
    "max_pages": 50,
    "concurrency": 3,
    "allowed_domains": ["react.dev", "legacy.reactjs.org", "reactjs.org"],
    "blocked_domains": [],
    "output_file": "output/react_results.jsonl",
}

# ========================================
# 示例 4: 爬取 "Docker" 相关内容
# ========================================

DOCKER_CONFIG = {
    "topic": "docker",
    "keywords": [
        "docker",
        "docker container",
        "docker compose",
        "kubernetes",
        "containerization",
        "dockerfile",
    ],
    "seed_urls": [
        "https://docs.docker.com/",
    ],
    "max_depth": 2,
    "max_pages": 80,
    "concurrency": 4,
    "allowed_domains": ["docs.docker.com", "kubernetes.io"],
    "blocked_domains": [],
    "output_file": "output/docker_results.jsonl",
}

# ========================================
# 示例 5: 爬取 "Graph Theory" 相关内容
# ========================================

GRAPH_THEORY_CONFIG = {
    "topic": "graph theory",
    "keywords": [
        "graph theory",
        "graph algorithm",
        "dfs",
        "bfs",
        "dijkstra",
        "shortest path",
        "minimum spanning tree",
        "拓扑排序",
    ],
    "seed_urls": [
        "https://en.wikipedia.org/wiki/Graph_theory",
        "https://leetcode.com/tag/graph/",
    ],
    "max_depth": 2,
    "max_pages": 60,
    "concurrency": 4,
    "allowed_domains": ["wikipedia.org", "leetcode.com", "geeksforgeeks.org"],
    "blocked_domains": [],
    "output_file": "output/graph_theory_results.jsonl",
}


# ========================================
# 使用示例
# ========================================

def get_config(topic: str) -> dict:
    """
    根据主题获取配置

    Args:
        topic: 主题名称

    Returns:
        配置字典
    """
    configs = {
        "binary search": BINARY_SEARCH_CONFIG,
        "machine learning": MACHINE_LEARNING_CONFIG,
        "react": REACT_CONFIG,
        "docker": DOCKER_CONFIG,
        "graph theory": GRAPH_THEORY_CONFIG,
    }

    return configs.get(topic.lower())


if __name__ == "__main__":
    # 打印所有可用的配置
    import sys

    if len(sys.argv) > 1:
        config = get_config(sys.argv[1])
        if config:
            print(f"Config for '{sys.argv[1]}':")
            for key, value in config.items():
                print(f"  {key}: {value}")
        else:
            print(f"No config found for '{sys.argv[1]}'")
            print("Available topics: binary search, machine learning, react, docker, graph theory")
    else:
        print("Available topics: binary search, machine learning, react, docker, graph theory")
        print(f"Usage: python example_configs.py <topic>")
