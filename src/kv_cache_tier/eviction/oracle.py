"""
Oracle eviction policies: ground-truth ablations for the policy study.

These policies are handed the full future of the replayed trace. They exist
to answer one question: when a learned policy loses to LRU, is the bottleneck
the *predictor* (imperfect P(resume) / E[dt] estimates) or the *objective*
(what the policy chooses to maximize)? Running the same objectives with
perfect inputs removes prediction error from the comparison entirely.

Three variants:

  BeladyPolicy
      Evicts the entry whose next access lies farthest in the future
      (never-returning entries first). Belady's MIN is offline-optimal for
      uniform-size, uniform-cost caches, so its hit rate is the classical
      upper reference. With variable entry sizes it is no longer provably
      optimal, but it remains the standard oracle baseline.

  OracleClassifierPolicy
      V1's objective (evict min P(resume within the window)) with a perfect
      classifier: P is the ground-truth indicator. Ties -- many entries share
      P=0 or P=1 -- are broken toward keeping the entry whose next access is
      nearer, which is strictly more informed than any real classifier, so a
      loss here cannot be blamed on AUC.

  OracleSpaceTimePolicy
      V3's objective with perfect inputs: P(resume) is the ground-truth
      indicator of ANY future access, and E[dt] is the exact time until that
      access. This is a value-weighted greedy Belady: the best-informed
      member of the density family.

Oracles are legitimate only in trace replay, where future arrivals are
independent of cache behavior (the event list is generated open-loop and is
identical for every policy).
"""

import math
from collections import defaultdict
from typing import Dict, Optional, Any

import numpy as np

from .base import EvictionPolicy
from ..utils.cost_model import CostModel


class TraceOracle:
    """Ground-truth index of future accesses, built from a replay event list.

    Only 'resume' events count as future accesses: they are the moments a
    cached entry is loaded, i.e. the only events an eviction decision can
    win or lose.
    """

    def __init__(self, events):
        acc = defaultdict(list)
        for e in events:
            if e.action == "resume":
                acc[e.session_id].append(e.timestamp)
        self._resumes: Dict[str, np.ndarray] = {
            sid: np.asarray(sorted(ts), dtype=np.float64) for sid, ts in acc.items()
        }

    def next_resume_after(self, session_id: str, t: float) -> float:
        """Timestamp of the first resume strictly after t, or +inf if none.

        side='right' excludes a resume at exactly t: at decision time the
        current event is already being processed, so only the future counts.
        """
        arr = self._resumes.get(session_id)
        if arr is None:
            return math.inf
        i = int(np.searchsorted(arr, t, side="right"))
        return float(arr[i]) if i < len(arr) else math.inf


class BeladyPolicy(EvictionPolicy):
    """Evict the entry whose next access is farthest in the future."""

    def __init__(self, oracle: TraceOracle):
        self._oracle = oracle

    @property
    def policy_name(self) -> str:
        return "belady"

    def on_access(self, session_id: str, entry: Any) -> None:
        pass

    def on_insert(self, session_id: str, entry: Any) -> None:
        pass

    def on_remove(self, session_id: str) -> None:
        pass

    def select_victim(self, entries: Dict[str, Any]) -> Optional[str]:
        if not entries:
            return None
        now = self.clock.now()
        return max(entries, key=lambda sid: self._oracle.next_resume_after(sid, now))

    def should_evict(self, session_id: str, entry: Any) -> bool:
        return False


class OracleClassifierPolicy(EvictionPolicy):
    """V1's objective with a perfect classifier.

    Score = ground-truth indicator of a resume within `resume_window_seconds`,
    plus a < 1e-6 tie-break term that keeps nearer-returning entries. The
    victim is the minimum score, exactly as in V1.
    """

    def __init__(self, oracle: TraceOracle, resume_window_seconds: float = 3600.0):
        self._oracle = oracle
        self.resume_window_seconds = resume_window_seconds

    @property
    def policy_name(self) -> str:
        return "oracle_v1"

    def on_access(self, session_id: str, entry: Any) -> None:
        pass

    def on_insert(self, session_id: str, entry: Any) -> None:
        pass

    def on_remove(self, session_id: str) -> None:
        pass

    def get_scores(self, entries: Dict[str, Any]) -> Dict[str, float]:
        now = self.clock.now()
        scores = {}
        for sid in entries:
            dnext = self._oracle.next_resume_after(sid, now) - now
            p = 1.0 if dnext <= self.resume_window_seconds else 0.0
            # Tie-break (bounded < 1e-6 so p always dominates): among equal
            # p, keep the entry that returns sooner. 0 for never-returning.
            tie = 0.0 if math.isinf(dnext) else 1e-6 / (1.0 + dnext / 3600.0)
            scores[sid] = p + tie
        return scores

    def select_victim(self, entries: Dict[str, Any]) -> Optional[str]:
        if not entries:
            return None
        scores = self.get_scores(entries)
        return min(scores, key=scores.get)

    def should_evict(self, session_id: str, entry: Any) -> bool:
        return False


class OracleSpaceTimePolicy(EvictionPolicy):
    """V3's objective with perfect inputs.

    Score = recompute_cost / (size_bytes x exact_time_to_next_access) for
    entries that will return, 0 for entries that never will. Uses the same
    1-second gap floor as the estimated V3 policy.
    """

    def __init__(self, oracle: TraceOracle, cost_model: Optional[CostModel] = None):
        self._oracle = oracle
        self._cost_model = cost_model or CostModel()

    @property
    def policy_name(self) -> str:
        return "oracle_v3"

    def on_access(self, session_id: str, entry: Any) -> None:
        pass

    def on_insert(self, session_id: str, entry: Any) -> None:
        pass

    def on_remove(self, session_id: str) -> None:
        pass

    def get_scores(self, entries: Dict[str, Any]) -> Dict[str, float]:
        now = self.clock.now()
        scores = {}
        for sid, entry in entries.items():
            dnext = self._oracle.next_resume_after(sid, now) - now
            if math.isinf(dnext):
                scores[sid] = 0.0
            else:
                recompute = self._cost_model.compute_recompute_time(entry.token_count)
                scores[sid] = recompute / (max(entry.size_bytes, 1) * max(dnext, 1.0))
        return scores

    def select_victim(self, entries: Dict[str, Any]) -> Optional[str]:
        if not entries:
            return None
        scores = self.get_scores(entries)
        return min(scores, key=scores.get)

    def should_evict(self, session_id: str, entry: Any) -> bool:
        return False
