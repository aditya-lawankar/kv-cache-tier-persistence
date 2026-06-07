"""
Abstract base class for serialization formats.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np

from ..config import ModelConfig

class CacheSerializer(ABC):
    """Abstract interface for a KV cache serialization format."""
    
    @abstractmethod
    def serialize(self, kv_data: Dict[int, Tuple[np.ndarray, np.ndarray]], model_config: ModelConfig) -> bytes:
        """
        Serialize KV cache data to bytes.
        
        Args:
            kv_data: Dict mapping layer index to (key_tensor, value_tensor)
            model_config: Configuration of the model that generated this cache
            
        Returns:
            bytes containing the serialized data
        """
        pass
        
    @abstractmethod
    def deserialize(self, data: bytes) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], ModelConfig]:
        """
        Deserialize bytes to KV cache data and model config.
        
        Args:
            data: bytes containing the serialized data
            
        Returns:
            Tuple of (kv_data, model_config)
        """
        pass
        
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the name of this serialization format."""
        pass
