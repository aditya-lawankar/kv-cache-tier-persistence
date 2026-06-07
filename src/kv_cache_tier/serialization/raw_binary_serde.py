"""
Custom raw binary serialization format.
"""

import struct
from typing import Dict, Tuple
import numpy as np

from ..config import ModelConfig
from .serializer import CacheSerializer

# Header layout:
# magic (8 bytes): 'KVCACHE\x00'
# version (2 bytes): 1
# num_layers (2 bytes)
# num_heads (2 bytes)
# head_dim (2 bytes)
# token_count (4 bytes)
# dtype enum (1 byte): 0=float16, 1=float32
# padding (43 bytes) to 64 bytes total
HEADER_FORMAT = "<8s H H H H I B 43s"
HEADER_SIZE = 64
MAGIC = b'KVCACHE\x00'
VERSION = 1

class RawBinarySerializer(CacheSerializer):
    """
    Custom raw binary format.
    Maximum speed for hot <-> warm tier transfers by skipping complex parsing.
    """
    
    @property
    def format_name(self) -> str:
        return "raw_binary"
        
    def serialize(self, kv_data: Dict[int, Tuple[np.ndarray, np.ndarray]], model_config: ModelConfig) -> bytes:
        if not kv_data:
            raise ValueError("Cannot serialize empty kv_data")
            
        # Get token_count from first tensor
        layer_0_k = kv_data[0][0]
        token_count = layer_0_k.shape[1]
        
        dtype_enum = 0 if model_config.dtype in ("float16", "fp16") else 1
        
        # Pack header
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            VERSION,
            model_config.num_layers,
            model_config.num_heads,
            model_config.head_dim,
            token_count,
            dtype_enum,
            b'\x00' * 43
        )
        
        # Collect raw bytes in order
        parts = [header]
        for layer in range(model_config.num_layers):
            if layer in kv_data:
                k, v = kv_data[layer]
                parts.append(k.tobytes())
                parts.append(v.tobytes())
                
        return b"".join(parts)
        
    def deserialize(self, data: bytes) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], ModelConfig]:
        if len(data) < HEADER_SIZE:
            raise ValueError("Data too small to contain valid header")
            
        # Parse header
        magic, version, num_layers, num_heads, head_dim, token_count, dtype_enum, _ = struct.unpack(
            HEADER_FORMAT, data[:HEADER_SIZE]
        )
        
        if magic != MAGIC:
            raise ValueError(f"Invalid magic signature: {magic}")
            
        dtype_str = "float16" if dtype_enum == 0 else "float32"
        np_dtype = np.float16 if dtype_enum == 0 else np.float32
        
        model_config = ModelConfig(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype_str
        )
        
        # Calculate tensor sizes
        tensor_elements = num_heads * token_count * head_dim
        tensor_bytes = tensor_elements * np_dtype().itemsize
        
        kv_data = {}
        offset = HEADER_SIZE
        
        for layer in range(num_layers):
            # Key tensor
            k_bytes = data[offset : offset + tensor_bytes]
            offset += tensor_bytes
            # Value tensor
            v_bytes = data[offset : offset + tensor_bytes]
            offset += tensor_bytes
            
            k_tensor = np.frombuffer(k_bytes, dtype=np_dtype).reshape(num_heads, token_count, head_dim)
            v_tensor = np.frombuffer(v_bytes, dtype=np_dtype).reshape(num_heads, token_count, head_dim)
            
            kv_data[layer] = (k_tensor.copy(), v_tensor.copy()) # copy to allow free underlying buffer
            
        return kv_data, model_config
