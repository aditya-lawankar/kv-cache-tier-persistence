"""
Cache block abstraction, mirroring vLLM's PagedAttention blocks.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np

from ..config import ModelConfig

@dataclass
class CacheBlock:
    """Represents a single block of KV cache (e.g., 16 tokens)."""
    block_id: int
    # Dict mapping layer_idx to (k_tensor, v_tensor) for this block
    layer_data: Dict[int, Tuple[np.ndarray, np.ndarray]]
    token_count: int
    block_index: int  # Logical position in the sequence

class CacheBlockTable:
    """A collection of blocks forming a full session's KV cache."""
    
    def __init__(self, blocks: List[CacheBlock] = None):
        self.blocks = blocks or []
        
    def add_block(self, block: CacheBlock) -> None:
        self.blocks.append(block)
        
    @property
    def total_tokens(self) -> int:
        return sum(b.token_count for b in self.blocks)
        
    @property
    def total_size_bytes(self) -> int:
        if not self.blocks:
            return 0
        total = 0
        for block in self.blocks:
            for k, v in block.layer_data.values():
                total += k.nbytes + v.nbytes
        return total
        
    def to_kv_dict(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Merge all blocks into a contiguous KV dict per layer."""
        if not self.blocks:
            return {}
            
        # Ensure ordered by logical index
        ordered_blocks = sorted(self.blocks, key=lambda b: b.block_index)
        
        # Determine layers
        first_block = ordered_blocks[0]
        layers = list(first_block.layer_data.keys())
        
        result = {}
        for layer in layers:
            k_list = []
            v_list = []
            for b in ordered_blocks:
                k, v = b.layer_data[layer]
                k_list.append(k)
                v_list.append(v)
                
            # Concatenate along the token dimension (axis=1)
            k_concat = np.concatenate(k_list, axis=1)
            v_concat = np.concatenate(v_list, axis=1)
            result[layer] = (k_concat, v_concat)
            
        return result
        
    @classmethod
    def from_kv_dict(cls, kv_data: Dict[int, Tuple[np.ndarray, np.ndarray]], block_size: int = 16) -> 'CacheBlockTable':
        """Split a contiguous KV dict into blocks."""
        if not kv_data:
            return cls([])
            
        # Get total tokens from layer 0
        layer_0_k = kv_data[0][0]
        total_tokens = layer_0_k.shape[1]
        
        num_blocks = (total_tokens + block_size - 1) // block_size
        blocks = []
        
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min(start_idx + block_size, total_tokens)
            tokens_in_block = end_idx - start_idx
            
            block_data = {}
            for layer, (k, v) in kv_data.items():
                # Slice along token dimension (axis=1)
                k_slice = k[:, start_idx:end_idx, :].copy()
                v_slice = v[:, start_idx:end_idx, :].copy()
                block_data[layer] = (k_slice, v_slice)
                
            block = CacheBlock(
                block_id=i,  # arbitrary ID for this simulated block
                layer_data=block_data,
                token_count=tokens_in_block,
                block_index=i
            )
            blocks.append(block)
            
        return cls(blocks)
        
    def validate(self, model_config: ModelConfig) -> bool:
        """Validate shapes and dtypes against model config."""
        if not self.blocks:
            return True
            
        expected_dtype = np.float16 if model_config.dtype in ("float16", "fp16") else np.float32
        
        for b in self.blocks:
            if len(b.layer_data) != model_config.num_layers:
                return False
                
            for layer, (k, v) in b.layer_data.items():
                if k.dtype != expected_dtype or v.dtype != expected_dtype:
                    return False
                if k.shape[0] != model_config.num_heads or k.shape[2] != model_config.head_dim:
                    return False
                if k.shape[1] != b.token_count:
                    return False
                    
        return True
