"""
Manga Insight 配置模块

提供配置数据模型和序列化工具。
"""

from .serialization import SerializableMixin, create_default_factory

__all__ = [
    "SerializableMixin",
    "create_default_factory"
]
