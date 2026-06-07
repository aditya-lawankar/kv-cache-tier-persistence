"""
Cache metadata and indexing.
"""

import json
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

@dataclass
class CacheEntry:
    """Metadata for a cached session."""
    session_id: str
    user_id: str
    model_config_hash: str
    created_at: float
    last_accessed: float
    token_count: int
    num_blocks: int
    tier: str  # "hot", "warm", "cold"
    size_bytes: int
    access_count: int = 1
    metadata: dict = field(default_factory=dict)

class CacheIndex:
    """Thread-safe index mapping session_id -> CacheEntry."""
    
    def __init__(self):
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
    def add(self, entry: CacheEntry) -> None:
        with self._lock:
            self._entries[entry.session_id] = entry
            
    def get(self, session_id: str) -> Optional[CacheEntry]:
        with self._lock:
            return self._entries.get(session_id)
            
    def remove(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._entries:
                del self._entries[session_id]
                return True
            return False
            
    def update(self, session_id: str, **kwargs) -> bool:
        with self._lock:
            entry = self._entries.get(session_id)
            if not entry:
                return False
            for k, v in kwargs.items():
                if hasattr(entry, k):
                    setattr(entry, k, v)
            return True
            
    def find_by_user(self, user_id: str) -> List[CacheEntry]:
        with self._lock:
            return [e for e in self._entries.values() if e.user_id == user_id]
            
    def find_by_tier(self, tier: str) -> List[CacheEntry]:
        with self._lock:
            return [e for e in self._entries.values() if e.tier == tier]
            
    def find_oldest(self, n: int, tier: Optional[str] = None) -> List[CacheEntry]:
        with self._lock:
            entries = list(self._entries.values())
            if tier:
                entries = [e for e in entries if e.tier == tier]
            entries.sort(key=lambda e: e.last_accessed)
            return entries[:n]
            
    def all_entries(self) -> Dict[str, CacheEntry]:
        """Return a copy of the internal dictionary."""
        with self._lock:
            return dict(self._entries)
            
    def save(self, path: str) -> None:
        """Persist index to a JSON file."""
        with self._lock:
            data = {sid: asdict(entry) for sid, entry in self._entries.items()}
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
                
    @classmethod
    def load(cls, path: str) -> 'CacheIndex':
        """Load index from a JSON file."""
        index = cls()
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            for sid, entry_dict in data.items():
                index._entries[sid] = CacheEntry(**entry_dict)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return index
        
    def stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            counts = {"hot": 0, "warm": 0, "cold": 0}
            sizes = {"hot": 0, "warm": 0, "cold": 0}
            
            for entry in self._entries.values():
                tier = entry.tier
                if tier in counts:
                    counts[tier] += 1
                    sizes[tier] += entry.size_bytes
                    
            return {
                "total_entries": len(self._entries),
                "tier_counts": counts,
                "tier_sizes_bytes": sizes
            }
