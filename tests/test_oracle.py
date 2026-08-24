"""Tests for the ground-truth oracle eviction policies."""

import math
from dataclasses import dataclass

import pytest

from src.kv_cache_tier.eviction.oracle import (
    TraceOracle, BeladyPolicy, OracleClassifierPolicy, OracleSpaceTimePolicy,
)
from src.kv_cache_tier.utils.clock import SimulatedClock


@dataclass
class FakeEvent:
    timestamp: float
    session_id: str
    action: str


@dataclass
class FakeEntry:
    token_count: int = 100
    size_bytes: int = 1000


def _make_oracle():
    events = [
        FakeEvent(0.0, "a", "start"),
        FakeEvent(100.0, "a", "resume"),
        FakeEvent(5000.0, "a", "resume"),
        FakeEvent(0.0, "b", "start"),
        FakeEvent(200.0, "b", "resume"),
        FakeEvent(0.0, "c", "start"),  # never resumes
    ]
    return TraceOracle(events)


def test_next_resume_lookup():
    oracle = _make_oracle()
    assert oracle.next_resume_after("a", 0.0) == 100.0
    # A resume at exactly t is being processed, not the future
    assert oracle.next_resume_after("a", 100.0) == 5000.0
    assert oracle.next_resume_after("a", 5000.0) == math.inf
    assert oracle.next_resume_after("c", 0.0) == math.inf
    assert oracle.next_resume_after("unknown", 0.0) == math.inf


def test_belady_evicts_farthest_next_access():
    oracle = _make_oracle()
    clock = SimulatedClock()
    clock.set(50.0)
    policy = BeladyPolicy(oracle)
    policy.set_clock(clock)
    entries = {"a": FakeEntry(), "b": FakeEntry(), "c": FakeEntry()}
    # c never returns -> evicted first
    assert policy.select_victim(entries) == "c"
    del entries["c"]
    # a returns at 100, b at 200 -> b is farther
    assert policy.select_victim(entries) == "b"


def test_oracle_classifier_matches_v1_objective():
    oracle = _make_oracle()
    clock = SimulatedClock()
    clock.set(150.0)
    policy = OracleClassifierPolicy(oracle, resume_window_seconds=3600.0)
    policy.set_clock(clock)
    entries = {"a": FakeEntry(), "b": FakeEntry(), "c": FakeEntry()}
    scores = policy.get_scores(entries)
    # b resumes at 200 (within 1h) -> P=1; a resumes at 5000 (4850s > 3600) -> P=0
    assert scores["b"] > 1.0 - 1e-9
    assert scores["a"] < 1e-5
    assert scores["c"] == 0.0
    # Among P=0 entries, the tie-break keeps the sooner-returning one (a)
    assert scores["a"] > scores["c"]
    assert policy.select_victim(entries) == "c"


def test_oracle_space_time_scores():
    oracle = _make_oracle()
    clock = SimulatedClock()
    clock.set(50.0)
    policy = OracleSpaceTimePolicy(oracle)
    policy.set_clock(clock)
    small_soon = FakeEntry(token_count=100, size_bytes=1000)     # "a": returns in 50s
    large_late = FakeEntry(token_count=100, size_bytes=100000)   # "b": returns in 150s
    dead = FakeEntry()                                            # "c": never
    entries = {"a": small_soon, "b": large_late, "c": dead}
    scores = policy.get_scores(entries)
    assert scores["c"] == 0.0
    assert scores["a"] > scores["b"] > 0.0
    assert policy.select_victim(entries) == "c"
