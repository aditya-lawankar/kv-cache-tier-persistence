"""
Time-To-Live (TTL) eviction policy.
"""

import time
from typing import Dict, Optional, Any

from .lru import LRUEvictionPolicy

class TTLEvictionPolicy(LRUEvictionPolicy):
    """
    Time-To-Live eviction policy.
    Evicts entries that have not been accessed within a given TTL.
    Falls back to LRU when capacity is reached but no entries are expired.
    """
    
    def __init__(self, ttl_seconds: Dict[str, int]):
        super().__init__()
        # Default TTLs per tier
        self.ttl_seconds = {
            "hot": ttl_seconds.get("hot", 300),      # 5 min
            "warm": ttl_seconds.get("warm", 3600),    # 1 hour
            "cold": ttl_seconds.get("cold", 86400)    # 24 hours
        }
        
    @property
    def policy_name(self) -> str:
        return "ttl"
        
    def should_evict(self, session_id: str, entry: Any) -> bool:
        """Check if an entry has expired based on its tier's TTL."""
        now = time.time()
        ttl = self.ttl_seconds.get(entry.tier, 3600)
        
        # 0 means never expire
        if ttl <= 0:
            return False
            
        return (now - entry.last_accessed) > ttl
        
    def select_victim(self, entries: Dict[str, Any]) -> Optional[str]:
        """
        Select victim by checking for expired entries first.
        If none expired, fallback to LRU.
        """
        if not entries:
            return None
            
        now = time.time()
        
        # 1. Look for oldest expired entry
        oldest_expired = None
        oldest_access_time = float('inf')
        
        for session_id, entry in entries.items():
            ttl = self.ttl_seconds.get(entry.tier, 3600)
            if ttl > 0 and (now - entry.last_accessed) > ttl:
                if entry.last_accessed < oldest_access_time:
                    oldest_expired = session_id
                    oldest_access_time = entry.last_accessed
                    
        if oldest_expired:
            return oldest_expired
            
        # 2. If nothing is strictly expired, find the one closest to expiry
        closest_to_expiry = None
        min_time_remaining = float('inf')
        
        for session_id, entry in entries.items():
            ttl = self.ttl_seconds.get(entry.tier, 3600)
            if ttl > 0:
                time_remaining = ttl - (now - entry.last_accessed)
                if time_remaining < min_time_remaining:
                    min_time_remaining = time_remaining
                    closest_to_expiry = session_id
                    
        if closest_to_expiry:
            return closest_to_expiry
            
        # 3. Fallback to strict LRU
        return super().select_victim(entries)
