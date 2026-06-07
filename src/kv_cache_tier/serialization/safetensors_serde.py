"""
Safetensors serialization format.
"""

import json
from typing import Dict, Tuple
import numpy as np
from safetensors.numpy import save, load

from ..config import ModelConfig
from .serializer import CacheSerializer

class SafetensorsSerializer(CacheSerializer):
    """
    Safetensors-based serialization.
    Very fast, zero-copy loading when mmapped, rich metadata.
    """
    
    @property
    def format_name(self) -> str:
        return "safetensors"
        
    def serialize(self, kv_data: Dict[int, Tuple[np.ndarray, np.ndarray]], model_config: ModelConfig) -> bytes:
        # Convert hierarchical dict to flat dict for safetensors
        flat_tensors = {}
        for layer, (k, v) in kv_data.items():
            flat_tensors[f"layer_{layer}_key"] = k
            flat_tensors[f"layer_{layer}_value"] = v
            
        # We can pass metadata directly to safetensors.numpy.save in newer versions
        # For compatibility, we'll serialize model_config into metadata
        metadata = {
            "num_layers": str(model_config.num_layers),
            "num_heads": str(model_config.num_heads),
            "head_dim": str(model_config.head_dim),
            "block_size": str(model_config.block_size),
            "dtype": model_config.dtype
        }
        
        # In safetensors.numpy, save() takes dict[str, ndarray] and optional metadata
        return save(flat_tensors, metadata=metadata)
        
    def deserialize(self, data: bytes) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], ModelConfig]:
        # Load from bytes
        # safetensors.numpy.load returns a dict of tensors
        # But we also need the metadata. Wait, load() from bytes doesn't return metadata.
        # Let's read the header directly to get metadata.
        
        # The safetensors format is: 8 bytes length of header (uint64), followed by JSON header
        header_len = np.frombuffer(data[:8], dtype=np.uint64)[0]
        header_json = data[8:8+int(header_len)].decode('utf-8')
        header = json.loads(header_json)
        
        metadata = header.get("__metadata__", {})
        
        model_config = ModelConfig(
            num_layers=int(metadata.get("num_layers", 32)),
            num_heads=int(metadata.get("num_heads", 32)),
            head_dim=int(metadata.get("head_dim", 128)),
            block_size=int(metadata.get("block_size", 16)),
            dtype=metadata.get("dtype", "float16")
        )
        
        flat_tensors = load(data)
        
        kv_data = {}
        for layer in range(model_config.num_layers):
            k_key = f"layer_{layer}_key"
            v_key = f"layer_{layer}_value"
            
            if k_key in flat_tensors and v_key in flat_tensors:
                kv_data[layer] = (flat_tensors[k_key], flat_tensors[v_key])
                
        return kv_data, model_config
