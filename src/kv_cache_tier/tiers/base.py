"""
Abstract base class for storage tiers.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any

# We use typing.Any for CacheEntry to avoid circular imports.
# The actual CacheEntry class is defined in core.cache_metadata

@dataclass
class TierUsage:
    """Statistics about a storage tier's usage."""
    name: str
    current_bytes: int
    capacity_bytes: int  # 0 means unlimited
    entry_count: int
    
    @property
    def utilization(self) -> float:
        """Return the utilization fraction (0.0 to 1.0)."""
        if self.capacity_bytes <= 0:
            return 0.0
        return min(1.0, self.current_bytes / self.capacity_bytes)

class StorageTier(ABC):
    """Abstract interface for a storage tier (Hot, Warm, Cold)."""
    
    def __init__(self, name: str, capacity_bytes: int):
        self.name = name
        self.capacity_bytes = capacity_bytes
        self._lock = threading.RLock()
        
    def is_full(self, data_size: int = 0) -> bool:
        """Check if adding data_size bytes would exceed capacity."""
        if self.capacity_bytes <= 0:
            return False
            
        usage = self.usage()
        return (usage.current_bytes + data_size) > self.capacity_bytes

    @abstractmethod
    def put(self, key: str, data: bytes, metadata: Any) -> None:
        """Store data in the tier."""
        pass
        
    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """Retrieve data from the tier. Returns None if not found."""
        pass
        
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data from the tier. Returns True if deleted, False if not found."""
        pass
        
    @abstractmethod
    def contains(self, key: str) -> bool:
        """Check if the key exists in this tier."""
        pass
        
    @abstractmethod
    def usage(self) -> TierUsage:
        """Get current usage statistics for this tier."""
        pass
        
    @abstractmethod
    def list_entries(self) -> List[str]:
        """List all keys stored in this tier."""
        pass
        
    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from this tier."""
        pass
