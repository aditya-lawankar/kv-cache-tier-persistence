"""
Serialization and compression for KV cache tensors.
"""

from .serializer import CacheSerializer
from .safetensors_serde import SafetensorsSerializer
from .raw_binary_serde import RawBinarySerializer
from .compression import (
    Compressor,
    LZ4Compressor,
    ZstdCompressor,
    NoCompressor,
    CompressedSerializer,
    create_compressor
)

def create_serializer(format_name: str) -> CacheSerializer:
    """Factory method to create a serializer by name."""
    if format_name == "safetensors":
        return SafetensorsSerializer()
    elif format_name == "raw_binary":
        return RawBinarySerializer()
    else:
        raise ValueError(f"Unknown serialization format: {format_name}")

__all__ = [
    "CacheSerializer",
    "SafetensorsSerializer",
    "RawBinarySerializer",
    "Compressor",
    "LZ4Compressor",
    "ZstdCompressor",
    "NoCompressor",
    "CompressedSerializer",
    "create_compressor",
    "create_serializer"
]
