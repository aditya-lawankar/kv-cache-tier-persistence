"""
V3 Space-Time Density Eviction Policy.

Completes the V1 -> V2 -> V3 arc:

  V1 (classification):  min P(resume)
      Fails: ignores size and recompute cost (Knapsack mismatch).

  V2 (value density):   min [P(resume) x recompute_cost] / size
      Fails: static packing ignores time. Under Little's Law (L = lambda x W),
      retaining massive sessions with unbounded sojourn time W collapses
      cache cardinality L and with it the hit rate.

  V3 (space-time density):
      min [P(resume) x recompute_cost] / (size x E[dt])

      Each entry is charged for the space-time volume it occupies:
      bytes x expected time until its next access. A huge session that
      will not be touched for hours must beat dozens of small sessions
      that will be touched in minutes — exactly the trade V2 got wrong.

Theoretical lineage: this is LHD's hit density objective
(Beckmann et al., NSDI 2018) — expected hits per unit of size x lifetime —
adapted to KV caching, where (a) hits are weighted by quadratic prefill
recompute cost instead of counted uniformly, and (b) P(resume) comes from
a learned session-level model instead of age-binned statistics.

E[dt] estimation: per-session exponential moving average of observed
inter-access gaps, blended with the current idle time. If an entry has
been idle LONGER than its typical gap, its expected wait grows rather
than shrinks (inspection paradox): we take max(EMA gap, current idle).
Sessions with no observed gap yet fall back to a configurable prior.
"""

import logging
from typing import Dict, Optional, Any

from .base import EvictionPolicy
from .features import FeatureExtractor
from ..utils.clock import Clock
from ..utils.cost_model import CostModel

logger = logging.getLogger(__name__)


class SpaceTimeDensityPolicy(EvictionPolicy):
    """
    V3: Evict the entry with the lowest expected value per byte-second.

    Score = [P(resume) x recompute_cost(tokens)] / (size_bytes x E[dt])

    Higher score = more valuable per unit of space-time occupied.
    """

    def __init__(
        self,
        predictor=None,
        cost_model: Optional[CostModel] = None,
        prior_gap_seconds: float = 2 * 3600.0,
        ema_alpha: float = 0.3,
        clock: Optional[Clock] = None,
    ):
        """
        Args:
            predictor: A trained ResumePredictor (LogisticPredictor or GBTPredictor).
            cost_model: CostModel instance for computing GPU recompute costs.
            prior_gap_seconds: E[dt] prior for sessions with no observed
                               inter-access gap yet (default 2h).
            ema_alpha: Smoothing factor for the inter-access gap EMA.
            clock: Time source; defaults to wall-clock.
        """
        self._predictor = predictor
        self._cost_model = cost_model or CostModel()
        self.prior_gap_seconds = prior_gap_seconds
        self.ema_alpha = ema_alpha

        self._features = FeatureExtractor(clock)
        if clock is not None:
            self.clock = clock

        # session_id -> EMA of observed inter-access gaps (seconds)
        self._gap_ema: Dict[str, float] = {}
        # session_id -> timestamp of most recent access we processed
        self._last_seen: Dict[str, float] = {}

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
        return f"space_time_{model_name}"

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_insert(self, session_id: str, entry: Any) -> None:
        self._features.record_access(entry.user_id, session_id, entry.token_count)
        self._last_seen[session_id] = self.clock.now()

    def on_access(self, session_id: str, entry: Any) -> None:
        self._features.record_access(entry.user_id, session_id, entry.token_count)
        now = self.clock.now()
        prev = self._last_seen.get(session_id)
        if prev is not None:
            gap = now - prev
            if gap > 0:
                ema = self._gap_ema.get(session_id)
                if ema is None:
                    self._gap_ema[session_id] = gap
                else:
                    self._gap_ema[session_id] = (
                        self.ema_alpha * gap + (1 - self.ema_alpha) * ema
                    )
        self._last_seen[session_id] = now

    def on_remove(self, session_id: str) -> None:
        # Keep the gap EMA: if the session is re-admitted later, its rhythm
        # is still the best estimate we have.
        pass

    # ------------------------------------------------------------------
    # Core: Space-Time Density
    # ------------------------------------------------------------------

    def expected_gap_seconds(self, session_id: str, entry: Any) -> float:
        """
        E[dt]: expected time this entry will sit idle before its next access.

        max(EMA of observed gaps, current idle time): an entry idle longer
        than its typical rhythm is expected to wait longer still, not less.
        """
        ema = self._gap_ema.get(session_id, self.prior_gap_seconds)
        idle = max(self.clock.now() - entry.last_accessed, 0.0)
        return max(ema, idle, 1.0)  # floor avoids division blow-ups

    def space_time_density(self, session_id: str, entry: Any,
                           p_resume: Optional[float] = None) -> float:
        """Expected GPU-seconds saved per byte-second of cache occupied."""
        if p_resume is None:
            if self._predictor is None:
                p_resume = 0.5  # uninformative prior
            else:
                p_resume = self._predictor.predict_resume_probability(
                    self._features.build(entry)
                )
        recompute = self._cost_model.compute_recompute_time(entry.token_count)
        gap = self.expected_gap_seconds(session_id, entry)
        return (p_resume * recompute) / (max(entry.size_bytes, 1) * gap)

    def _batch_scores(self, entries: Dict[str, Any]) -> Dict[str, float]:
        """Score a candidate set with ONE predict_batch call."""
        ids = list(entries.keys())
        if self._predictor is None:
            return {sid: self.space_time_density(sid, entries[sid]) for sid in ids}

        feats = [self._features.build(entries[sid]) for sid in ids]
        probs = self._predictor.predict_batch(feats)
        return {
            sid: self.space_time_density(sid, entries[sid], p_resume=float(p))
            for sid, p in zip(ids, probs)
        }

    # ------------------------------------------------------------------
    # Eviction decisions
    # ------------------------------------------------------------------

    def select_victim(self, entries: Dict[str, Any]) -> Optional[str]:
        """Evict the entry with the LOWEST space-time density."""
        if not entries:
            return None
        scores = self._batch_scores(entries)
        return min(scores, key=scores.get)

    def should_evict(self, session_id: str, entry: Any) -> bool:
        return False

    def get_scores(self, entries: Dict[str, Any]) -> Dict[str, float]:
        """Return space-time density scores (for debugging/analysis)."""
        return self._batch_scores(entries)
