"""
Predictive eviction policy based on access patterns.

Supports three strategy modes:
  - "heuristic": The original weighted formula (alpha*freq + beta*recency + gamma*value).
  - "logistic":  Uses a trained LogisticPredictor for P(resume).
  - "gbt":       Uses a trained GBTPredictor for P(resume).

When an ML model is loaded, select_victim() evicts the session with the
LOWEST predicted probability of being resumed.
"""

import time
import math
import threading
import logging
from typing import Dict, Optional, Any, List

from .base import EvictionPolicy
from .features import SessionFeatures

logger = logging.getLogger(__name__)


class PredictiveEvictionPolicy(EvictionPolicy):
    """
    Predictive eviction policy.

    In 'heuristic' mode, scores entries using a weighted formula:
        score = alpha * frequency + beta * recency + gamma * session_length

    In 'logistic' or 'gbt' mode, scores entries using a trained ML model
    that predicts P(session will be resumed).
    """

    def __init__(
        self,
        strategy: str = "heuristic",
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        decay_half_life: int = 1800,
        model_path: Optional[str] = None,
    ):
        self.strategy = strategy
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.decay_half_life = decay_half_life  # seconds (e.g. 30 mins)

        # Tracks user access history: user_id -> [(session_id, timestamp, token_count)]
        self._user_history: Dict[str, List[tuple]] = {}
        self._lock = threading.Lock()

        # ML model (loaded lazily or from path)
        self._predictor = None
        if strategy in ("logistic", "gbt") and model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load a trained ResumePredictor from disk."""
        from .predictors import ResumePredictor
        try:
            self._predictor = ResumePredictor.load(path)
            logger.info(f"Loaded {self._predictor.model_name} from {path}")
        except Exception as e:
            logger.warning(f"Failed to load model from {path}: {e}. Falling back to heuristic.")
            self.strategy = "heuristic"

    def set_predictor(self, predictor):
        """Directly inject a trained predictor (useful in tests and experiments)."""
        self._predictor = predictor
        self.strategy = predictor.model_name.split("_")[0]  # "logistic" or "gbt"

    @property
    def policy_name(self) -> str:
        return f"predictive_{self.strategy}"

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

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_scores(self, entries: Dict[str, Any]) -> Dict[str, float]:
        """Calculate prediction scores for all provided entries."""
        if not entries:
            return {}

        if self.strategy == "heuristic" or self._predictor is None:
            return self._heuristic_scores(entries)
        else:
            return self._ml_scores(entries)

    def _heuristic_scores(self, entries: Dict[str, Any]) -> Dict[str, float]:
        """Original weighted heuristic scoring."""
        scores = {}
        now = time.time()

        max_access = max((e.access_count for e in entries.values()), default=1)
        max_tokens = max((e.token_count for e in entries.values()), default=1)

        for session_id, entry in entries.items():
            freq_score = entry.access_count / max(max_access, 1)
            time_since_access = now - entry.last_accessed
            recency_score = math.exp(-math.log(2) * time_since_access / self.decay_half_life)
            value_score = entry.token_count / max(max_tokens, 1)

            scores[session_id] = (
                self.alpha * freq_score
                + self.beta * recency_score
                + self.gamma * value_score
            )
        return scores

    def _ml_scores(self, entries: Dict[str, Any]) -> Dict[str, float]:
        """Score entries using a trained ML predictor."""
        now = time.time()
        scores = {}

        for session_id, entry in entries.items():
            # Build SessionFeatures from the CacheEntry
            user_id = entry.user_id
            history = self._user_history.get(user_id, [])
            total_user_sessions = len(set(sid for sid, _, _ in history))
            total_user_resumes = max(0, len(history) - total_user_sessions)
            user_return_rate = total_user_resumes / max(total_user_sessions, 1)

            session_age_minutes = (now - entry.created_at) / 60.0
            time_since_last_minutes = (now - entry.last_accessed) / 60.0

            # Simulate hour/day from real clock
            import datetime
            dt = datetime.datetime.now()
            hour_of_day = dt.hour
            day_of_week = dt.weekday()

            feat = SessionFeatures(
                session_age_minutes=session_age_minutes,
                token_count=entry.token_count,
                revisit_count=entry.access_count - 1,  # first access is the creation
                time_since_last_access_minutes=time_since_last_minutes,
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                user_historical_return_rate=min(user_return_rate, 1.0),
                is_business_hours=1 if (9 <= hour_of_day < 17 and day_of_week < 5) else 0,
                avg_session_tokens=float(entry.token_count),  # approximate with current session
            )

            scores[session_id] = self._predictor.predict_resume_probability(feat)

        return scores

    # ------------------------------------------------------------------
    # Eviction decisions
    # ------------------------------------------------------------------

    def select_victim(self, entries: Dict[str, Any]) -> Optional[str]:
        """Select the entry with the LOWEST prediction score to evict."""
        if not entries:
            return None

        scores = self.get_scores(entries)
        return min(scores, key=scores.get)

    def should_evict(self, session_id: str, entry: Any) -> bool:
        return False
