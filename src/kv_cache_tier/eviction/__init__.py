"""
Eviction policies.
"""

from .base import EvictionPolicy
from .lru import LRUEvictionPolicy
from .ttl import TTLEvictionPolicy
from .predictive import PredictiveEvictionPolicy

def create_eviction_policy(name: str, **kwargs) -> EvictionPolicy:
    """Factory method to create an eviction policy."""
    if name == "lru":
        return LRUEvictionPolicy()
    elif name == "ttl":
        return TTLEvictionPolicy(kwargs.get("ttl_seconds", {}))
    elif name == "predictive":
        return PredictiveEvictionPolicy(
            alpha=kwargs.get("alpha", 0.4),
            beta=kwargs.get("beta", 0.4),
            gamma=kwargs.get("gamma", 0.2),
            decay_half_life=kwargs.get("decay_half_life", 1800)
        )
    else:
        raise ValueError(f"Unknown eviction policy: {name}")

__all__ = [
    "EvictionPolicy",
    "LRUEvictionPolicy",
    "TTLEvictionPolicy", 
    "PredictiveEvictionPolicy",
    "create_eviction_policy"
]
