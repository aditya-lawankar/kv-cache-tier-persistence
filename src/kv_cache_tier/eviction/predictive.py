"""
Predictive eviction policy based on access patterns.

Supports three strategy modes:
  - "heuristic": The original weighted formula (alpha*freq + beta*recency + gamma*value).
  - "logistic":  Uses a trained LogisticPredictor for P(resume).
  - "gbt":       Uses a trained GBTPredictor for P(resume).

When an ML model is loaded, select_victim() evicts the session with the
LOWEST predicted probability of being resumed.

All time reads go through self.clock (see utils/clock.py) so that
trace-driven experiments evaluate the model on the same feature scales
it was trained on.
"""

import math
import logging
from typing import Dict, Optional, Any

from .base import EvictionPolicy
from .features import FeatureExtractor
from ..utils.clock import Clock

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
        clock: Optional[Clock] = None,
    ):
        self.strategy = strategy
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.decay_half_life = decay_half_life  # seconds (e.g. 30 mins)

        self._features = FeatureExtractor(clock)
        if clock is not None:
            self.clock = clock

        # ML model (loaded lazily or from path)
        self._predictor = None
        if strategy in ("logistic", "gbt") and model_path:
            self._load_model(model_path)

    def set_clock(self, clock: Clock) -> None:
        super().set_clock(clock)
        self._features.set_clock(clock)

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
        self._features.record_access(entry.user_id, session_id, entry.token_count)

    def on_insert(self, session_id: str, entry: Any) -> None:
        self._features.record_access(entry.user_id, session_id, entry.token_count)

    def on_remove(self, session_id: str) -> None:
        # We keep history even after eviction to learn user patterns
        pass

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
        now = self.clock.now()

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
        """Score entries using a trained ML predictor.

        Scores the whole candidate set in ONE predict_batch call: per-entry
        single-row sklearn predictions are ~100x slower in aggregate and were
        the dominant cost of ML-policy eviction sweeps."""
        ids = list(entries.keys())
        feats = [self._features.build(entries[sid]) for sid in ids]
        probs = self._predictor.predict_batch(feats)
        return dict(zip(ids, probs.tolist()))

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
