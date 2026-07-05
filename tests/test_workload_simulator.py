"""Tests for workload simulator realism properties."""

from collections import defaultdict

from benchmarks.workload_simulator import WorkloadSimulator, PROFILES


class TestReproducibility:
    def test_same_seed_same_trace(self):
        a = WorkloadSimulator("enterprise", duration_days=0.05, seed=7).generate()
        b = WorkloadSimulator("enterprise", duration_days=0.05, seed=7).generate()
        assert [(e.timestamp, e.session_id, e.action, e.token_count) for e in a] == \
               [(e.timestamp, e.session_id, e.action, e.token_count) for e in b]

    def test_different_seed_different_trace(self):
        a = WorkloadSimulator("enterprise", duration_days=0.05, seed=7).generate()
        b = WorkloadSimulator("enterprise", duration_days=0.05, seed=8).generate()
        assert [(e.timestamp, e.action) for e in a] != [(e.timestamp, e.action) for e in b]

    def test_instance_rng_isolated_from_global(self):
        """Trace generation must not be perturbed by other users of the
        global random module (regression test for global random.seed)."""
        import random
        sim_a = WorkloadSimulator("casual", duration_days=0.05, seed=3)
        random.seed(999)
        random.random()
        trace_a = sim_a.generate()

        sim_b = WorkloadSimulator("casual", duration_days=0.05, seed=3)
        random.seed(123)
        [random.random() for _ in range(50)]
        trace_b = sim_b.generate()

        assert [(e.timestamp, e.action) for e in trace_a] == \
               [(e.timestamp, e.action) for e in trace_b]


class TestSessionGrowth:
    def test_token_counts_grow_monotonically_within_session(self):
        """Conversations only grow: every resume must carry at least as many
        tokens as the previous interaction of the same session."""
        events = WorkloadSimulator("power_user", duration_days=1.0, seed=11).generate()
        last_tokens = {}
        violations = 0
        for e in events:
            if e.action in ("start", "resume"):
                if e.action == "resume" and e.session_id in last_tokens:
                    if e.token_count < last_tokens[e.session_id]:
                        violations += 1
                last_tokens[e.session_id] = e.token_count
        assert violations == 0


class TestPersonas:
    def test_users_have_stable_differentiated_return_behavior(self):
        """Per-user return rates must vary across users (personas), so the
        user_historical_return_rate feature carries learnable signal."""
        events = WorkloadSimulator("enterprise", duration_days=3.0, seed=5).generate()
        sessions_by_user = defaultdict(set)
        resumes_by_user = defaultdict(int)
        for e in events:
            if e.action == "start":
                sessions_by_user[e.user_id].add(e.session_id)
            elif e.action == "resume":
                resumes_by_user[e.user_id] += 1

        rates = [
            resumes_by_user[u] / len(s)
            for u, s in sessions_by_user.items()
            if len(s) >= 3  # users with enough sessions to estimate a rate
        ]
        assert len(rates) > 10
        # Differentiated personas -> wide spread of per-user return rates
        assert max(rates) - min(rates) > 0.5

    def test_resume_probability_decays(self):
        """Sessions must not resume forever: later resume rounds are rarer."""
        events = WorkloadSimulator("enterprise", duration_days=3.0, seed=5).generate()
        resume_counts = defaultdict(int)
        for e in events:
            if e.action == "resume":
                resume_counts[e.session_id] += 1

        n_sessions = len(set(e.session_id for e in events))
        once = sum(1 for c in resume_counts.values() if c >= 1)
        thrice = sum(1 for c in resume_counts.values() if c >= 3)
        assert once > 0
        # With decay, >=3 resumes should be much rarer than >=1
        assert thrice < once * 0.6
