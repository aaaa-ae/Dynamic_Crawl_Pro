"""
基础 Agent 类
定义所有 Agent 的通用接口和功能
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import asyncio
from loguru import logger


class BaseAgent(ABC):
    """
    基础 Agent 抽象类
    所有具体的 Agent 都需要继承此类并实现 process 方法
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化基础 Agent

        Args:
            name: Agent 名称
            config: 配置字典
        """
        self.name = name
        self.config = config or {}
        self._metrics = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "total_time": 0.0,
        }

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """
        处理输入数据并返回结果

        Args:
            input_data: 输入数据

        Returns:
            处理结果
        """
        pass

    @abstractmethod
    def can_process(self, input_data: Any) -> bool:
        """
        检查是否能够处理该输入数据

        Args:
            input_data: 输入数据

        Returns:
            是否能够处理
        """
        pass

    async def process_with_metrics(self, input_data: Any) -> Optional[Any]:
        """
        带指标统计的处理方法

        Args:
            input_data: 输入数据

        Returns:
            处理结果，如果失败则返回 None
        """
        import time

        self._metrics["processed"] += 1
        start_time = time.time()

        try:
            if not self.can_process(input_data):
                logger.warning(f"[{self.name}] Cannot process input: {input_data}")
                self._metrics["failed"] += 1
                return None

            result = await self.process(input_data)
            self._metrics["success"] += 1

            elapsed = time.time() - start_time
            self._metrics["total_time"] += elapsed

            logger.info(f"[{self.name}] Processed successfully in {elapsed:.2f}s")
            return result

        except Exception as e:
            logger.error(f"[{self.name}] Error processing: {e}")
            self._metrics["failed"] += 1
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """
        获取 Agent 的统计指标

        Returns:
            指标字典
        """
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        """重置统计指标"""
        self._metrics = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "total_time": 0.0,
        }

    def log_metrics(self) -> None:
        """打印统计指标"""
        logger.info(f"[{self.name}] Metrics: {self._metrics}")


class Message:
    """
    Agent 之间传递的消息对象
    """

    def __init__(
        self,
        sender: str,
        receiver: str,
        content: Any,
        message_type: str = "default",
        metadata: Dict[str, Any] = None
    ):
        """
        初始化消息

        Args:
            sender: 发送者名称
            receiver: 接收者名称
            content: 消息内容
            message_type: 消息类型
            metadata: 元数据
        """
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.message_type = message_type
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "message_type": self.message_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建消息"""
        return cls(
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            message_type=data.get("message_type", "default"),
            metadata=data.get("metadata", {}),
        )
