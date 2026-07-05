"""
Abstract base class for eviction policies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

from ..utils.clock import Clock, SystemClock

# We import CacheEntry lazily or rely on Duck Typing in the methods.
# For type hinting, we use Any and import at runtime if needed.

class EvictionPolicy(ABC):
    """Abstract interface for an eviction policy.

    Policies must read time via self.clock, never time.time()/datetime.now(),
    so that trace-driven experiments can substitute a SimulatedClock.
    """

    # Stateless default; replaced per-instance via set_clock().
    clock: Clock = SystemClock()

    def set_clock(self, clock: Clock) -> None:
        self.clock = clock

    @abstractmethod
    def on_access(self, session_id: str, entry: Any) -> None:
        """Called when a cache entry is accessed/loaded."""
        pass
        
    @abstractmethod
    def on_insert(self, session_id: str, entry: Any) -> None:
        """Called when a new entry is added to the cache."""
        pass
        
    @abstractmethod
    def on_remove(self, session_id: str) -> None:
        """Called when an entry is permanently removed."""
        pass
        
    @abstractmethod
    def select_victim(self, entries: Dict[str, Any]) -> Optional[str]:
        """
        Select an entry to evict from the provided dict of entries.
        Returns the session_id to evict, or None if no suitable victim is found.
        """
        pass
        
    @abstractmethod
    def should_evict(self, session_id: str, entry: Any) -> bool:
        """
        Check if a specific entry should be evicted (e.g., due to TTL expiry)
        independent of capacity constraints.
        """
        pass
        
    @property
    @abstractmethod
    def policy_name(self) -> str:
        pass
