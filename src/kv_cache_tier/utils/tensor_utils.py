"""
Utilities for simulating and validating KV cache tensors.
"""

import numpy as np
from typing import Dict, Tuple

from ..config import ModelConfig

def generate_random_kv_cache(model_config: ModelConfig, token_count: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Generate a dictionary of random K/V tensors simulating an LLM KV cache."""
    kv_cache = {}
    dtype = np.float16 if model_config.dtype in ("float16", "fp16") else np.float32
    shape = (model_config.num_heads, token_count, model_config.head_dim)
    
    for layer in range(model_config.num_layers):
        k_tensor = np.zeros(shape, dtype=dtype)
        v_tensor = np.zeros(shape, dtype=dtype)
        kv_cache[layer] = (k_tensor, v_tensor)
        
    return kv_cache

def validate_kv_cache(kv_data: Dict[int, Tuple[np.ndarray, np.ndarray]], model_config: ModelConfig) -> bool:
    """Validate that the KV data matches the expected shapes and dtypes."""
    if not isinstance(kv_data, dict):
        return False
    if len(kv_data) != model_config.num_layers:
        return False
        
    expected_dtype = np.float16 if model_config.dtype in ("float16", "fp16") else np.float32
    
    for layer, (k, v) in kv_data.items():
        if k.dtype != expected_dtype or v.dtype != expected_dtype:
            return False
        if len(k.shape) != 3 or len(v.shape) != 3:
            return False
        if k.shape != v.shape:
            return False
        if k.shape[0] != model_config.num_heads or k.shape[2] != model_config.head_dim:
            return False
            
    return True

def kv_cache_size_bytes(kv_data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> int:
    """Calculate the exact size in bytes of the KV cache data."""
    total_bytes = 0
    for k, v in kv_data.values():
        total_bytes += k.nbytes + v.nbytes
    return total_bytes

def kv_caches_equal(a: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                    b: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                    rtol: float = 1e-5) -> bool:
    """Check if two KV caches are approximately equal."""
    if a.keys() != b.keys():
        return False
        
    for layer in a:
        k_a, v_a = a[layer]
        k_b, v_b = b[layer]
        
        if k_a.shape != k_b.shape or v_a.shape != v_b.shape:
            return False
        if not np.allclose(k_a, k_b, rtol=rtol):
            return False
        if not np.allclose(v_a, v_b, rtol=rtol):
            return False
            
    return True

def quantize_kv_cache(kv_data: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                      target_dtype: str = 'float16') -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Convert a KV cache to a target dtype for simple quantization simulation."""
    dtype_map = {
        'float32': np.float32,
        'float16': np.float16,
        'int8': np.int8
    }
    target_np_dtype = dtype_map.get(target_dtype, np.float16)
    
    quantized = {}
    for layer, (k, v) in kv_data.items():
        if target_dtype == 'int8':
            # Simple symmetric min-max quantization
            k_max = np.max(np.abs(k))
            v_max = np.max(np.abs(v))
            k_scale = 127.0 / (k_max + 1e-9)
            v_scale = 127.0 / (v_max + 1e-9)
            
            k_q = np.clip(np.round(k * k_scale), -127, 127).astype(np.int8)
            v_q = np.clip(np.round(v * v_scale), -127, 127).astype(np.int8)
            quantized[layer] = (k_q, v_q)
        else:
            quantized[layer] = (k.astype(target_np_dtype), v.astype(target_np_dtype))
            
    return quantized
