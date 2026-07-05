"""Tests for the Azure trace -> WorkloadEvent loader.

The real trace is 1.1 GB and not present in CI, so these tests run the
loader against a small fabricated npz with the same structure: one week
of trace-relative timestamps (t=0 = Monday 00:00) with ContextTokens /
GeneratedTokens columns.
"""

import numpy as np
import pytest

from benchmarks.azure_trace_loader import AzureTraceWorkload, WINDOWS, WINDOW_HOURS


@pytest.fixture(scope="module")
def fake_npz(tmp_path_factory):
    """A synthetic week-long trace dense enough to populate every window."""
    rng = np.random.default_rng(0)
    start = WINDOWS[0][1]
    end = WINDOWS[-1][1] + WINDOW_HOURS * 3600.0
    n = 200_000
    ts = np.sort(rng.uniform(start, end, size=n))
    ctx = rng.integers(16, 8000, size=n).astype(np.int32)
    gen = rng.integers(1, 1500, size=n).astype(np.int32)
    path = tmp_path_factory.mktemp("azure") / "fake_trace.npz"
    np.savez_compressed(path, ts=ts, ctx=ctx, gen=gen)
    return str(path)


class TestSchema:
    def test_events_match_workload_event_schema(self, fake_npz):
        events = AzureTraceWorkload(0, seed=1, npz_path=fake_npz).generate()
        assert events
        for e in events[:200]:
            assert e.action in ("start", "end", "resume")
            assert isinstance(e.token_count, int) and e.token_count >= 16
            assert e.user_id and e.session_id

    def test_timestamps_sorted_and_within_window(self, fake_npz):
        wl = AzureTraceWorkload(2, seed=1, npz_path=fake_npz)
        events = wl.generate()
        stamps = [e.timestamp for e in events]
        assert stamps == sorted(stamps)
        assert stamps[0] >= wl.window_start
        assert stamps[-1] <= wl.window_end

    def test_starts_replay_real_arrivals_and_sizes(self, fake_npz):
        """Session starts must carry the trace's timestamps and sizes
        (ctx+gen), not synthetic draws."""
        wl = AzureTraceWorkload(0, seed=1, npz_path=fake_npz)
        data = np.load(fake_npz)
        ts, ctx, gen = data["ts"], data["ctx"], data["gen"]
        lo = np.searchsorted(ts, wl.window_start)
        hi = np.searchsorted(ts, wl.window_end)
        expected = {
            (float(ts[i]), int(ctx[i]) + int(gen[i]))
            for i in range(lo, hi, wl._stride)
        }
        starts = {(e.timestamp, e.token_count)
                  for e in wl.generate() if e.action == "start"}
        assert starts == expected


class TestReproducibility:
    def test_same_window_same_seed_same_trace(self, fake_npz):
        a = AzureTraceWorkload(1, seed=9, npz_path=fake_npz).generate()
        b = AzureTraceWorkload(1, seed=9, npz_path=fake_npz).generate()
        assert [(e.timestamp, e.session_id, e.action, e.token_count) for e in a] == \
               [(e.timestamp, e.session_id, e.action, e.token_count) for e in b]

    def test_different_windows_different_traces(self, fake_npz):
        a = AzureTraceWorkload(0, seed=9, npz_path=fake_npz).generate()
        b = AzureTraceWorkload(1, seed=9, npz_path=fake_npz).generate()
        assert [(e.timestamp, e.action) for e in a] != \
               [(e.timestamp, e.action) for e in b]


class TestResumes:
    def test_sessions_resume_and_grow_monotonically(self, fake_npz):
        events = AzureTraceWorkload(0, seed=1, npz_path=fake_npz).generate()
        resumes = [e for e in events if e.action == "resume"]
        assert resumes, "return model should schedule resumes within the window"
        last = {}
        for e in events:
            if e.action == "resume":
                assert e.token_count >= last.get(e.session_id, 0)
            if e.action in ("start", "resume"):
                last[e.session_id] = e.token_count

    def test_thinning_hits_target_rate(self, fake_npz):
        wl = AzureTraceWorkload(0, seed=1, npz_path=fake_npz,
                                target_sessions_per_hour=100)
        n_starts = sum(1 for e in wl.generate() if e.action == "start")
        assert abs(n_starts - 100 * WINDOW_HOURS) <= 0.05 * 100 * WINDOW_HOURS


class TestValidation:
    def test_rejects_bad_window_index(self, fake_npz):
        with pytest.raises(ValueError):
            AzureTraceWorkload(len(WINDOWS), npz_path=fake_npz)
