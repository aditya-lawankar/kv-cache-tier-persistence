"""
Session feature engineering for ML-based eviction prediction.

Extracts structured features from raw session metadata and user history
to feed into learned eviction models (Logistic Regression, GBT).
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import numpy as np


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
