"""
Hot Tier: In-memory storage simulating GPU VRAM.
"""

from typing import Dict, List, Optional, Any
import logging

from .base import StorageTier, TierUsage

logger = logging.getLogger(__name__)

class HotTier(StorageTier):
    """
    In-memory storage simulating GPU VRAM.
    Fastest tier, limited capacity.
    """
    
    def __init__(self, capacity_bytes: int = 8 * 1024**3):
        super().__init__("hot", capacity_bytes)
        self._storage: Dict[str, bytes] = {}
        self._sizes: Dict[str, int] = {}
        self._current_bytes = 0
        
    def put(self, key: str, data: bytes, metadata: Any) -> None:
        with self._lock:
            data_size = len(data)
            
            # If the key already exists, subtract its old size
            old_size = self._sizes.get(key, 0)
            
            if self.capacity_bytes > 0 and (self._current_bytes - old_size + data_size) > self.capacity_bytes:
                raise MemoryError(f"Hot tier is full. Cannot add {data_size} bytes. Usage: {self._current_bytes}/{self.capacity_bytes}")
                
            self._storage[key] = data
            self._sizes[key] = data_size
            self._current_bytes = self._current_bytes - old_size + data_size
            
    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            return self._storage.get(key)
            
    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._storage:
                size = self._sizes.pop(key, 0)
                del self._storage[key]
                self._current_bytes -= size
                return True
            return False
            
    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._storage
            
    def usage(self) -> TierUsage:
        with self._lock:
            return TierUsage(
                name=self.name,
                current_bytes=self._current_bytes,
                capacity_bytes=self.capacity_bytes,
                entry_count=len(self._storage)
            )
            
    def list_entries(self) -> List[str]:
        with self._lock:
            return list(self._storage.keys())
            
    def clear(self) -> None:
        with self._lock:
            self._storage.clear()
            self._sizes.clear()
            self._current_bytes = 0
            logger.info("Hot tier cleared.")
