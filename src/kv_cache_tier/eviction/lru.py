"""
Least Recently Used (LRU) eviction policy.
"""

from collections import OrderedDict
from typing import Dict, Optional, Any
import threading

from .base import EvictionPolicy

class LRUEvictionPolicy(EvictionPolicy):
    """
    Least Recently Used eviction policy.
    Evicts the entry that has not been accessed for the longest time.
    """
    
    def __init__(self):
        self._access_order = OrderedDict()
        self._lock = threading.Lock()
        
    @property
    def policy_name(self) -> str:
        return "lru"
        
    def on_access(self, session_id: str, entry: Any) -> None:
        with self._lock:
            if session_id in self._access_order:
                self._access_order.move_to_end(session_id)
            else:
                self._access_order[session_id] = True
                
    def on_insert(self, session_id: str, entry: Any) -> None:
        with self._lock:
            self._access_order[session_id] = True
            
    def on_remove(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._access_order:
                del self._access_order[session_id]
                
    def select_victim(self, entries: Dict[str, Any]) -> Optional[str]:
        """Return the least recently used entry from the provided entries."""
        if not entries:
            return None
            
        with self._lock:
            # Iterate through access order (oldest first)
            for session_id in self._access_order:
                if session_id in entries:
                    return session_id
                    
            # Fallback if entries not in access order
            return next(iter(entries.keys()))
            
    def should_evict(self, session_id: str, entry: Any) -> bool:
        """LRU never evicts based on time alone, only when capacity is reached."""
        return False
