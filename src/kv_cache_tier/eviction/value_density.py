"""
V2 Value-Density Eviction Policy.

Fixes the structural flaw identified in V1 (binary classification):
eviction under capacity constraints is a Weighted Caching / Knapsack
problem, not a classification problem.

V1 eviction criterion:     min P(resume)
                            → ignores size → Knapsack failure

V2 eviction criterion:     min [P(resume) × recompute_cost] / size_bytes
                            → value density → Weighted Caching

Additionally introduces admission control: entries whose value density
falls below a threshold are rejected before entering the cache, preventing
cache pollution from low-value one-off sessions.

Theoretical grounding:
    Bélády's Algorithm (1966) is optimal for uniform-size items.
    For heterogeneous sizes, optimal eviction maximizes expected
    value per cached byte — exactly what this policy implements.
"""

import time
import math
import threading
import logging
from typing import Dict, Optional, Any, List

from .base import EvictionPolicy
from .features import SessionFeatures
from ..utils.cost_model import CostModel

logger = logging.getLogger(__name__)


class ValueDensityPolicy(EvictionPolicy):
    """
    V2: Size-aware eviction via expected value per cached byte.

    Score = [P(resume) × GPU_recompute_cost(tokens)] / size_bytes
          = Expected GPU savings per byte of cache consumed

    Evict the entry with the LOWEST score (worst value per byte).
    Admit an entry only if its score exceeds the admission threshold.
    """

    def __init__(
        self,
        predictor=None,
        cost_model: Optional[CostModel] = None,
        admission_threshold: float = 0.0,
    ):
        """
        Args:
            predictor: A trained ResumePredictor (LogisticPredictor or GBTPredictor).
            cost_model: CostModel instance for computing GPU recompute costs.
            admission_threshold: Minimum value density to admit an entry.
                                 0.0 = admit everything (no admission control).
        """
        self._predictor = predictor
        self._cost_model = cost_model or CostModel()
        self.admission_threshold = admission_threshold

        # Tracks user access history: user_id -> [(session_id, timestamp, token_count)]
        self._user_history: Dict[str, List[tuple]] = {}
        self._lock = threading.Lock()

    def set_predictor(self, predictor):
        """Inject a trained predictor (for experiments)."""
        self._predictor = predictor

    @property
    def policy_name(self) -> str:
        model_name = "unknown"
        if self._predictor:
            model_name = getattr(self._predictor, "model_name", "ml")
        suffix = "_ac" if self.admission_threshold > 0 else ""
        return f"value_density_{model_name}{suffix}"

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_access(self, session_id: str, entry: Any) -> None:
        self._record_access(entry.user_id, session_id, entry.token_count)

    def on_insert(self, session_id: str, entry: Any) -> None:
        self._record_access(entry.user_id, session_id, entry.token_count)

    def on_remove(self, session_id: str) -> None:
        pass  # Keep history for learning

    def _record_access(self, user_id: str, session_id: str, token_count: int):
        with self._lock:
            if user_id not in self._user_history:
                self._user_history[user_id] = []
            self._user_history[user_id].append((session_id, time.time(), token_count))
            if len(self._user_history[user_id]) > 50:
                self._user_history[user_id] = self._user_history[user_id][-50:]

    # ------------------------------------------------------------------
    # Core: Value Density Computation
    # ------------------------------------------------------------------

    def _build_features(self, entry: Any) -> SessionFeatures:
        """Build SessionFeatures from a CacheEntry, reusing V1 feature logic."""
        now = time.time()
        user_id = entry.user_id
        history = self._user_history.get(user_id, [])
        total_user_sessions = len(set(sid for sid, _, _ in history))
        total_user_resumes = max(0, len(history) - total_user_sessions)
        user_return_rate = total_user_resumes / max(total_user_sessions, 1)

        session_age_minutes = (now - entry.created_at) / 60.0
        time_since_last_minutes = (now - entry.last_accessed) / 60.0

        import datetime
        dt = datetime.datetime.now()
        hour_of_day = dt.hour
        day_of_week = dt.weekday()

        return SessionFeatures(
            session_age_minutes=session_age_minutes,
            token_count=entry.token_count,
            revisit_count=entry.access_count - 1,
            time_since_last_access_minutes=time_since_last_minutes,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            user_historical_return_rate=min(user_return_rate, 1.0),
            is_business_hours=1 if (9 <= hour_of_day < 17 and day_of_week < 5) else 0,
            avg_session_tokens=float(entry.token_count),
        )

    def _recompute_cost_seconds(self, token_count: int) -> float:
        """
        Estimate the GPU time (in seconds) to recompute this session
        from scratch if it were evicted and later needed.

        Uses the cost model's quadratic cold-start estimation.
        """
        return self._cost_model.compute_recompute_time(token_count)

    def value_density(self, entry: Any) -> float:
        """
        Compute the expected value per cached byte.

        Score = [P(resume) × recompute_cost_seconds] / size_bytes

        Higher score = more valuable to keep in cache.
        """
        if self._predictor is None:
            # Fallback: use recency as a proxy (graceful degradation)
            recency = math.exp(-0.001 * (time.time() - entry.last_accessed))
            recompute = self._recompute_cost_seconds(entry.token_count)
            return (recency * recompute) / max(entry.size_bytes, 1)

        features = self._build_features(entry)
        p_resume = self._predictor.predict_resume_probability(features)
        recompute_cost = self._recompute_cost_seconds(entry.token_count)
        expected_savings = p_resume * recompute_cost

        return expected_savings / max(entry.size_bytes, 1)

    # ------------------------------------------------------------------
    # Eviction decisions
    # ------------------------------------------------------------------

    def select_victim(self, entries: Dict[str, Any]) -> Optional[str]:
        """Evict the entry with the LOWEST expected value per byte."""
        if not entries:
            return None

        scores = {}
        for session_id, entry in entries.items():
            scores[session_id] = self.value_density(entry)

        return min(scores, key=scores.get)

    def should_evict(self, session_id: str, entry: Any) -> bool:
        """
        Admission control gate: returns True if an entry's value density
        is below the admission threshold, signaling it should not be
        retained (or should be proactively evicted during maintenance).
        """
        if self.admission_threshold <= 0:
            return False
        return self.value_density(entry) < self.admission_threshold

    def get_scores(self, entries: Dict[str, Any]) -> Dict[str, float]:
        """Return value density scores for all entries (for debugging/analysis)."""
        return {sid: self.value_density(e) for sid, e in entries.items()}
