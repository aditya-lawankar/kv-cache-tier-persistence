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

Theoretical grounding: size/cost-aware caching in the GreedyDual-Size
(Cao & Irani, 1997) / weighted-caching (Young, 1994) tradition. See the
paper's failure analysis for why the *static* form of this objective
collapses under Little's Law dynamics.
"""

import math
import logging
from typing import Dict, Optional, Any

from .base import EvictionPolicy
from .features import FeatureExtractor
from ..utils.clock import Clock
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
        clock: Optional[Clock] = None,
    ):
        """
        Args:
            predictor: A trained ResumePredictor (LogisticPredictor or GBTPredictor).
            cost_model: CostModel instance for computing GPU recompute costs.
            admission_threshold: Minimum value density to admit an entry.
                                 0.0 = admit everything (no admission control).
            clock: Time source; defaults to wall-clock.
        """
        self._predictor = predictor
        self._cost_model = cost_model or CostModel()
        self.admission_threshold = admission_threshold

        self._features = FeatureExtractor(clock)
        if clock is not None:
            self.clock = clock

    def set_clock(self, clock: Clock) -> None:
        super().set_clock(clock)
        self._features.set_clock(clock)

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
        self._features.record_access(entry.user_id, session_id, entry.token_count)

    def on_insert(self, session_id: str, entry: Any) -> None:
        self._features.record_access(entry.user_id, session_id, entry.token_count)

    def on_remove(self, session_id: str) -> None:
        pass  # Keep history for learning

    # ------------------------------------------------------------------
    # Core: Value Density Computation
    # ------------------------------------------------------------------

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
            recency = math.exp(-0.001 * (self.clock.now() - entry.last_accessed))
            recompute = self._recompute_cost_seconds(entry.token_count)
            return (recency * recompute) / max(entry.size_bytes, 1)

        features = self._features.build(entry)
        p_resume = self._predictor.predict_resume_probability(features)
        recompute_cost = self._recompute_cost_seconds(entry.token_count)
        expected_savings = p_resume * recompute_cost

        return expected_savings / max(entry.size_bytes, 1)

    # ------------------------------------------------------------------
    # Eviction decisions
    # ------------------------------------------------------------------

    def _batch_scores(self, entries: Dict[str, Any]) -> Dict[str, float]:
        """Value density for a candidate set, using ONE predict_batch call
        (per-entry single-row sklearn predictions dominate eviction cost)."""
        ids = list(entries.keys())
        if self._predictor is None:
            return {sid: self.value_density(entries[sid]) for sid in ids}

        feats = [self._features.build(entries[sid]) for sid in ids]
        probs = self._predictor.predict_batch(feats)
        scores = {}
        for sid, p_resume in zip(ids, probs):
            entry = entries[sid]
            expected_savings = float(p_resume) * self._recompute_cost_seconds(entry.token_count)
            scores[sid] = expected_savings / max(entry.size_bytes, 1)
        return scores

    def select_victim(self, entries: Dict[str, Any]) -> Optional[str]:
        """Evict the entry with the LOWEST expected value per byte."""
        if not entries:
            return None

        scores = self._batch_scores(entries)
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
        return self._batch_scores(entries)
