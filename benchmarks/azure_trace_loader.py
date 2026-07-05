"""
Azure LLM inference trace -> WorkloadEvent loader.

Converts the AzurePublicDataset one-week conversation trace (27.3M
requests, 2024-05-12 .. 2024-05-19 UTC; TIMESTAMP / ContextTokens /
GeneratedTokens) into the WorkloadEvent schema replayed by
benchmarks/experiment_runner.py.

What the trace provides directly (and what we replay verbatim):
  * request arrival timestamps -- real burstiness, diurnal and weekly
    load shape, placed on the trace-relative axis with t=0 = Monday
    00:00 UTC (2024-05-06), matching the SimulatedClock convention;
  * per-request context sizes (ContextTokens + GeneratedTokens) -- the
    real heavy-tailed distribution of KV-cache entry sizes.

What the trace cannot provide: session identity. The public dataset
carries no conversation identifiers (customer privacy), and linkage
cannot be reconstructed from context-token growth either: at production
density (~60 req/s over ~8,000 distinct context values) a greedy
growth-chain matcher links 87.8% of requests -- and links the SAME
87.8% on a permutation control with sizes shuffled across requests,
i.e. the matches are collisions, not conversations. We therefore impose
return behavior with the same persona machinery used by the synthetic
workloads (enterprise return-window parameters), so that differences
between the synthetic-enterprise and real-trace rows are attributable
to the arrival process and size distribution, which ARE real.

Replay design:
  * WINDOWS: ten 6-hour windows spread across the week (weekday /
    weekend, peak / trough). Each window is one replicate ("seed");
    all policies replay the identical event list per window, so policy
    deltas stay paired.
  * Thinning: every k-th request of a window, k chosen to hit
    TARGET_SESSIONS_PER_HOUR (200/h -- the synthetic enterprise rate,
    keeping capacity pressure comparable). Each selected request
    becomes a session start at its real timestamp with its real size.
  * Resumes: scheduled by WorkloadSimulator._schedule_resumes with the
    enterprise profile; on resume the context grows by two empirical
    draws from the window's GeneratedTokens distribution (the response
    plus an approximated user turn) instead of the synthetic
    log-normal increment.

Usage:
    python benchmarks/azure_trace_loader.py            # summary of all windows
    python benchmarks/azure_trace_loader.py --window 3 # one window's stats
"""

import csv
import os
import sys
import urllib.request
from typing import List, Optional, Tuple

import numpy as np

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.workload_simulator import WorkloadSimulator, WorkloadEvent

TRACE_URL = ("https://github.com/Azure/AzurePublicDataset/releases/download/"
             "dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv")
DATA_DIR = os.path.join(_project_root, "data")
CSV_PATH = os.path.join(DATA_DIR, "AzureLLMInferenceTrace_conv_1week.csv")
NPZ_PATH = os.path.join(DATA_DIR, "azure_conv_1week.npz")

# t=0 = Monday 2024-05-06 00:00 UTC; the trace starts Sunday 2024-05-12.
# Timestamps keep their true UTC hour-of-day / day-of-week so temporal
# features read the same calendar the requests actually arrived on.
BASE_DAY_OF_MONTH = 6

WINDOW_HOURS = 6.0
# Thinning target. Real Azure entries are much smaller than synthetic
# enterprise ones (median ~1,000 vs ~1,800 tokens), so matching the
# enterprise ARRIVAL rate (200/h) leaves the 500 MB cache unconstrained
# (LRU ~98% -- no policy differentiation, like the casual workload).
# 300/h restores sustained eviction pressure comparable to the synthetic
# enterprise regime (byte-LRU surrogate: ~85% mean hit rate across the
# ten windows vs ~79% for enterprise).
TARGET_SESSIONS_PER_HOUR = 300

_H = 3600.0
_D = 86400.0
# (label, window start in trace-relative seconds). Trace span:
# Sun 00:00 = 518400 .. Sun 00:00 = 1123200.
WINDOWS: List[Tuple[str, float]] = [
    ("sun_00", 6 * _D + 0 * _H),
    ("sun_12", 6 * _D + 12 * _H),
    ("mon_03", 7 * _D + 3 * _H),
    ("mon_15", 7 * _D + 15 * _H),
    ("tue_06", 8 * _D + 6 * _H),
    ("wed_09", 9 * _D + 9 * _H),
    ("wed_18", 9 * _D + 18 * _H),
    ("thu_12", 10 * _D + 12 * _H),
    ("fri_15", 11 * _D + 15 * _H),
    ("sat_06", 12 * _D + 6 * _H),
]


def ensure_trace_npz(npz_path: str = NPZ_PATH, csv_path: str = CSV_PATH) -> str:
    """Return the path to the parsed trace cache, building it on first use.

    Downloads the CSV (~1.1 GB) from the AzurePublicDataset release if
    missing, then parses it once into a compact npz (ts / ctx / gen).
    """
    if os.path.exists(npz_path):
        return npz_path

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    if not os.path.exists(csv_path):
        print(f"Downloading Azure LLM inference trace (~1.1 GB) to {csv_path} ...")
        urllib.request.urlretrieve(TRACE_URL, csv_path)

    print(f"Parsing {csv_path} (one-time; cached to {npz_path}) ...")
    ts_l, ctx_l, gen_l = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            s = row[0]  # "2024-05-12 00:00:00.001163+00:00"
            t = ((int(s[8:10]) - BASE_DAY_OF_MONTH) * _D
                 + int(s[11:13]) * _H
                 + int(s[14:16]) * 60.0
                 + float(s[17:s.index("+")]))
            ts_l.append(t)
            ctx_l.append(int(row[1]))
            gen_l.append(int(row[2]))

    ts = np.asarray(ts_l, dtype=np.float64)
    ctx = np.asarray(ctx_l, dtype=np.int32)
    gen = np.asarray(gen_l, dtype=np.int32)
    order = np.argsort(ts, kind="stable")
    ts, ctx, gen = ts[order], ctx[order], gen[order]
    np.savez_compressed(npz_path, ts=ts, ctx=ctx, gen=gen)
    print(f"Cached {len(ts)} requests.")
    return npz_path


# Decompressing the 27M-row npz takes several seconds; instantiating one
# AzureTraceWorkload per (policy, window) pair would repeat it 60 times.
_trace_cache: dict = {}


def _load_trace_arrays(npz_path: str):
    arrays = _trace_cache.get(npz_path)
    if arrays is None:
        data = np.load(npz_path)
        arrays = (data["ts"], data["ctx"], data["gen"])
        _trace_cache[npz_path] = arrays
    return arrays


class AzureTraceWorkload(WorkloadSimulator):
    """WorkloadEvent generator driven by one window of the Azure trace.

    Session starts replay real arrivals and real sizes; personas and
    resume scheduling are inherited from WorkloadSimulator so the return
    model is identical to the synthetic enterprise workload.
    """

    def __init__(self, window_idx: int, seed: int = 1000,
                 npz_path: Optional[str] = None,
                 target_sessions_per_hour: int = TARGET_SESSIONS_PER_HOUR,
                 window_hours: float = WINDOW_HOURS):
        super().__init__("enterprise", duration_days=window_hours / 24.0, seed=seed)
        if not 0 <= window_idx < len(WINDOWS):
            raise ValueError(f"window_idx must be in [0, {len(WINDOWS) - 1}]")
        self.window_idx = window_idx
        self.window_label, self.window_start = WINDOWS[window_idx]
        self.window_end = self.window_start + window_hours * _H

        ts, ctx, gen = _load_trace_arrays(npz_path or ensure_trace_npz())
        lo, hi = np.searchsorted(ts, self.window_start), np.searchsorted(ts, self.window_end)
        if hi <= lo:
            raise ValueError(f"window {self.window_label} contains no trace requests")
        self._ts = ts[lo:hi]
        self._sizes = ctx[lo:hi].astype(np.int64) + gen[lo:hi].astype(np.int64)
        self._gen = gen[lo:hi]

        target_sessions = int(target_sessions_per_hour * window_hours)
        self._stride = max(1, len(self._ts) // target_sessions)

    # Resume growth: two empirical draws from the window's generated-token
    # distribution (the model response plus an approximated user turn),
    # replacing the synthetic log-normal increment.
    def _sample_growth(self, persona) -> int:
        i = self.rng.randrange(len(self._gen))
        j = self.rng.randrange(len(self._gen))
        return max(16, int(self._gen[i]) + int(self._gen[j]))

    def generate(self) -> List[WorkloadEvent]:
        events: List[WorkloadEvent] = []
        for n, i in enumerate(range(0, len(self._ts), self._stride)):
            start_time = float(self._ts[i])
            tokens = max(16, min(int(self._sizes[i]), 128000))
            persona = self._pick_user(start_time)
            session_id = f"azure_{self.window_label}_{n}"

            events.append(WorkloadEvent(
                timestamp=start_time, user_id=persona.user_id,
                session_id=session_id, action="start", token_count=tokens,
            ))
            end_time = start_time + self.rng.uniform(5.0, 300.0)
            events.append(WorkloadEvent(
                timestamp=end_time, user_id=persona.user_id,
                session_id=session_id, action="end", token_count=tokens,
            ))
            self._schedule_resumes(events, end_time, persona, session_id,
                                   prev_tokens=tokens, resume_round=0)

        events.sort(key=lambda e: e.timestamp)
        # Same horizon convention as the synthetic generator: events
        # scheduled beyond the window are dropped.
        return [e for e in events if e.timestamp <= self.window_end]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Azure trace -> WorkloadEvent conversion stats")
    parser.add_argument("--window", type=int, default=None,
                        help="Window index 0-9 (default: summarize all)")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    ensure_trace_npz()
    indices = [args.window] if args.window is not None else range(len(WINDOWS))
    for w in indices:
        wl = AzureTraceWorkload(w, seed=args.seed + w)
        events = wl.generate()
        s = wl.summary(events)
        s["window"] = wl.window_label
        s["raw_requests_in_window"] = len(wl._ts)
        s["thinning_stride"] = wl._stride
        print(json.dumps(s, indent=2))
