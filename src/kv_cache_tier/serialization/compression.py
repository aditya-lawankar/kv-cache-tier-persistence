"""
Compression wrappers for serializers.
"""

import lz4.frame
import zstandard as zstd
from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np

from ..config import ModelConfig
from .serializer import CacheSerializer

class Compressor(ABC):
    """Abstract interface for a compression algorithm."""
    
    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        pass
        
    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        pass
        
    @property
    @abstractmethod
    def name(self) -> str:
        pass

class NoCompressor(Compressor):
    def compress(self, data: bytes) -> bytes:
        return data
        
    def decompress(self, data: bytes) -> bytes:
        return data
        
    @property
    def name(self) -> str:
        return "none"

class LZ4Compressor(Compressor):
    """Fast compression for hot/warm transfers."""
    
    def compress(self, data: bytes) -> bytes:
        # lz4 frame format for compatibility and safety
        return lz4.frame.compress(data, compression_level=1)
        
    def decompress(self, data: bytes) -> bytes:
        return lz4.frame.decompress(data)
        
    @property
    def name(self) -> str:
        return "lz4"

class ZstdCompressor(Compressor):
    """Better ratio compression for cold storage."""
    
    def __init__(self, level: int = 1):
        self.level = level
        
    def compress(self, data: bytes) -> bytes:
        cctx = zstd.ZstdCompressor(level=self.level)
        return cctx.compress(data)
        
    def decompress(self, data: bytes) -> bytes:
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
        
    @property
    def name(self) -> str:
        return "zstd"

def create_compressor(name: str, level: int = 1) -> Compressor:
    if name == "none":
        return NoCompressor()
    elif name == "lz4":
        return LZ4Compressor()
    elif name == "zstd":
        return ZstdCompressor(level=level)
    else:
        raise ValueError(f"Unknown compression: {name}")

class CompressedSerializer(CacheSerializer):
    """
    Wraps another CacheSerializer and applies compression to the resulting bytes.
    """
    
    def __init__(self, base_serializer: CacheSerializer, compressor: Compressor):
        self.base_serializer = base_serializer
        self.compressor = compressor
        
    def serialize(self, kv_data: Dict[int, Tuple[np.ndarray, np.ndarray]], model_config: ModelConfig) -> bytes:
        raw_bytes = self.base_serializer.serialize(kv_data, model_config)
        return self.compressor.compress(raw_bytes)
        
    def deserialize(self, data: bytes) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], ModelConfig]:
        decompressed_bytes = self.compressor.decompress(data)
        return self.base_serializer.deserialize(decompressed_bytes)
        
    @property
    def format_name(self) -> str:
        return f"{self.base_serializer.format_name}+{self.compressor.name}"
