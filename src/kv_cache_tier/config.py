"""
Configuration dataclasses for the KV Cache Tier Persistence project.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    block_size: int = 16
    dtype: str = "float16"

    @property
    def cache_block_size_bytes(self) -> int:
        """Calculate bytes per block (factor 2 for K and V)."""
        bytes_per_element = 2 if self.dtype in ("float16", "bfloat16") else 4
        return 2 * self.num_heads * self.head_dim * self.block_size * bytes_per_element

    def session_size_bytes(self, token_count: int) -> int:
        """Calculate total memory of KV cache for a given token count across all layers."""
        bytes_per_element = 2 if self.dtype in ("float16", "bfloat16") else 4
        # 2 tensors (K, V) * num_layers * num_heads * token_count * head_dim * bytes_per_element
        return 2 * self.num_layers * self.num_heads * token_count * self.head_dim * bytes_per_element


@dataclass
class TierConfig:
    hot_capacity_mb: int = 8192
    warm_capacity_mb: int = 65536
    cold_capacity_mb: int = 0  # 0 = unlimited
    warm_storage_path: str = "./data/warm"
    cold_storage_path: str = "./data/cold"
    cold_backend: str = "local"  # "local" or "minio"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "kv-cache"


@dataclass
class EvictionConfig:
    policy: str = "lru"
    ttl_seconds: int = 3600
    hot_ttl_seconds: int = 300
    warm_ttl_seconds: int = 3600
    cold_ttl_seconds: int = 86400
    check_interval_seconds: int = 60
    predictive_alpha: float = 0.4
    predictive_beta: float = 0.4
    predictive_gamma: float = 0.2


@dataclass
class SerializationConfig:
    format: str = "safetensors"
    compression: str = "lz4"
    compression_level: int = 1


@dataclass
class SystemConfig:
    model: ModelConfig
    tiers: TierConfig
    eviction: EvictionConfig
    serialization: SerializationConfig

    @classmethod
    def default(cls) -> "SystemConfig":
        """Create a default configuration."""
        return cls(
            model=ModelConfig(),
            tiers=TierConfig(),
            eviction=EvictionConfig(),
            serialization=SerializationConfig()
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SystemConfig":
        """Create configuration from a dictionary."""
        return cls(
            model=ModelConfig(**d.get("model", {})),
            tiers=TierConfig(**d.get("tiers", {})),
            eviction=EvictionConfig(**d.get("eviction", {})),
            serialization=SerializationConfig(**d.get("serialization", {}))
        )
