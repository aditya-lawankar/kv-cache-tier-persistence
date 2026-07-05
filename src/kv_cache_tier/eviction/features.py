"""
Session feature engineering for ML-based eviction prediction.

Extracts structured features from raw session metadata and user history
to feed into learned eviction models (Logistic Regression, GBT).
"""

import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import numpy as np

from ..utils.clock import Clock, SystemClock


@dataclass
class SessionFeatures:
    """
    Feature vector for a single cache entry at a point in time.
    Each field is a numeric feature suitable for direct ingestion
    by scikit-learn estimators.
    """
    session_age_minutes: float             # How long since session was created
    token_count: int                       # Conversation length (proxy for context value)
    revisit_count: int                     # Number of times this user resumed this session
    time_since_last_access_minutes: float  # Minutes since the last interaction
    hour_of_day: int                       # 0-23, captures enterprise 9-5 patterns
    day_of_week: int                       # 0=Monday ... 6=Sunday
    user_historical_return_rate: float     # Per-user return probability estimate [0, 1]
    is_business_hours: int                 # 1 if hour_of_day in [9, 17] and weekday, else 0
    avg_session_tokens: float              # User's average token count across their sessions

    def to_array(self) -> np.ndarray:
        """Convert to a 1D numpy feature vector."""
        return np.array([
            self.session_age_minutes,
            self.token_count,
            self.revisit_count,
            self.time_since_last_access_minutes,
            self.hour_of_day,
            self.day_of_week,
            self.user_historical_return_rate,
            self.is_business_hours,
            self.avg_session_tokens,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "session_age_minutes",
            "token_count",
            "revisit_count",
            "time_since_last_access_minutes",
            "hour_of_day",
            "day_of_week",
            "user_historical_return_rate",
            "is_business_hours",
            "avg_session_tokens",
        ]

    def to_dict(self) -> dict:
        return asdict(self)


class FeatureExtractor:
    """
    Builds SessionFeatures from CacheEntry metadata plus per-user access
    history. Shared by all ML-backed eviction policies so that feature
    semantics cannot drift between them, and reads time exclusively from
    the injected Clock so that simulated experiments produce features on
    the same scale the models were trained on.
    """

    HISTORY_LIMIT = 50  # accesses retained per user

    def __init__(self, clock: Optional[Clock] = None):
        self.clock = clock or SystemClock()
        # user_id -> [(session_id, timestamp, token_count)]
        self._user_history: Dict[str, List[tuple]] = {}
        self._lock = threading.Lock()

    def set_clock(self, clock: Clock) -> None:
        self.clock = clock

    def record_access(self, user_id: str, session_id: str, token_count: int) -> None:
        with self._lock:
            history = self._user_history.setdefault(user_id, [])
            history.append((session_id, self.clock.now(), token_count))
            if len(history) > self.HISTORY_LIMIT:
                del history[: len(history) - self.HISTORY_LIMIT]

    def build(self, entry: Any) -> SessionFeatures:
        """Build the feature vector for a cache entry as of clock.now()."""
        now = self.clock.now()
        history = self._user_history.get(entry.user_id, [])
        total_user_sessions = len(set(sid for sid, _, _ in history))
        total_user_resumes = max(0, len(history) - total_user_sessions)
        user_return_rate = min(total_user_resumes / max(total_user_sessions, 1), 1.0)
        if history:
            avg_tokens = sum(tc for _, _, tc in history) / len(history)
        else:
            avg_tokens = float(entry.token_count)

        hour_of_day = self.clock.hour_of_day()
        day_of_week = self.clock.day_of_week()

        return SessionFeatures(
            session_age_minutes=(now - entry.created_at) / 60.0,
            token_count=entry.token_count,
            revisit_count=entry.access_count - 1,  # first access is the creation
            time_since_last_access_minutes=(now - entry.last_accessed) / 60.0,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            user_historical_return_rate=user_return_rate,
            is_business_hours=1 if (9 <= hour_of_day < 17 and day_of_week < 5) else 0,
            avg_session_tokens=avg_tokens,
        )
