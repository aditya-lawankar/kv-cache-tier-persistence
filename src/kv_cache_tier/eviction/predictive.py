"""
Predictive eviction policy based on access patterns.
"""

import time
import math
import threading
from typing import Dict, Optional, Any, List

from .base import EvictionPolicy

class PredictiveEvictionPolicy(EvictionPolicy):
    """
    Predictive eviction policy.
    Predicts likelihood of session resumption based on:
    - User access frequency
    - Recency of access
    - Session length (longer sessions are more valuable to cache)
    """
    
    def __init__(self, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2, decay_half_life: int = 1800):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.decay_half_life = decay_half_life # seconds (e.g. 30 mins)
        
        # Tracks user access history: user_id -> [(session_id, timestamp, token_count)]
        self._user_history: Dict[str, List[tuple]] = {}
        self._lock = threading.Lock()
        
    @property
    def policy_name(self) -> str:
        return "predictive"
        
    def on_access(self, session_id: str, entry: Any) -> None:
        self._record_access(entry.user_id, session_id, entry.token_count)
            
    def on_insert(self, session_id: str, entry: Any) -> None:
        self._record_access(entry.user_id, session_id, entry.token_count)
            
    def on_remove(self, session_id: str) -> None:
        # We keep history even after eviction to learn user patterns
        pass
        
    def _record_access(self, user_id: str, session_id: str, token_count: int):
        with self._lock:
            if user_id not in self._user_history:
                self._user_history[user_id] = []
            self._user_history[user_id].append((session_id, time.time(), token_count))
            
            # Prune history (keep last 50 accesses)
            if len(self._user_history[user_id]) > 50:
                self._user_history[user_id] = self._user_history[user_id][-50:]
                
    def get_scores(self, entries: Dict[str, Any]) -> Dict[str, float]:
        """Calculate prediction scores for all provided entries."""
        if not entries:
            return {}
            
        scores = {}
        now = time.time()
        
        # Find global maxes for normalization
        max_access = 1
        max_tokens = 1
        
        for entry in entries.values():
            max_access = max(max_access, entry.access_count)
            max_tokens = max(max_tokens, entry.token_count)
            
        for session_id, entry in entries.items():
            # 1. Frequency (normalized)
            freq_score = entry.access_count / max_access
            
            # 2. Recency (exponential decay)
            time_since_access = now - entry.last_accessed
            recency_score = math.exp(-math.log(2) * time_since_access / self.decay_half_life)
            
            # 3. Session value (normalized token count)
            value_score = entry.token_count / max_tokens
            
            # Final weighted score
            scores[session_id] = (
                self.alpha * freq_score +
                self.beta * recency_score +
                self.gamma * value_score
            )
            
        return scores
        
    def select_victim(self, entries: Dict[str, Any]) -> Optional[str]:
        """Select the entry with the LOWEST prediction score to evict."""
        if not entries:
            return None
            
        scores = self.get_scores(entries)
        
        # Find session with minimum score
        min_score = float('inf')
        victim = None
        
        for session_id, score in scores.items():
            if score < min_score:
                min_score = score
                victim = session_id
                
        return victim
        
    def should_evict(self, session_id: str, entry: Any) -> bool:
        return False
