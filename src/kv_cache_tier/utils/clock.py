"""
Clock abstraction for time-dependent components.

Every component that reasons about time (eviction policies, feature
extraction, the tiered manager's metadata timestamps) must read time
through a Clock rather than calling time.time() / datetime.now()
directly. This allows trace-driven experiments to replay events at a
simulated timescale: without it, a 6-hour trace replayed in seconds
collapses every temporal feature to ~0 and silently breaks any policy
that reasons about time (train/serve skew).

Time-of-day convention for simulated time: timestamp 0 is Monday 00:00.
This matches the convention used to derive hour_of_day / day_of_week
labels in the training pipeline (train_predictors.py), so features
computed at serving time are on the same scale as features computed at
training time.
"""

import time
import datetime
from abc import ABC, abstractmethod


class Clock(ABC):
    """Source of the current time for cache components."""

    @abstractmethod
    def now(self) -> float:
        """Current time in seconds (epoch for SystemClock, trace-relative for SimulatedClock)."""

    @abstractmethod
    def hour_of_day(self) -> int:
        """Current hour, 0-23."""

    @abstractmethod
    def day_of_week(self) -> int:
        """Current day, 0=Monday ... 6=Sunday."""


class SystemClock(Clock):
    """Wall-clock time, for production use."""

    def now(self) -> float:
        return time.time()

    def hour_of_day(self) -> int:
        return datetime.datetime.now().hour

    def day_of_week(self) -> int:
        return datetime.datetime.now().weekday()


class SimulatedClock(Clock):
    """
    Manually advanced clock for trace-driven simulation.

    Timestamps are trace-relative seconds with t=0 defined as Monday 00:00,
    matching the training pipeline's label convention.
    """

    def __init__(self, start: float = 0.0):
        self._now = start

    def now(self) -> float:
        return self._now

    def set(self, timestamp: float) -> None:
        """Jump the clock to an absolute trace timestamp (must not go backwards)."""
        if timestamp < self._now:
            raise ValueError(
                f"SimulatedClock cannot go backwards: {timestamp} < {self._now}"
            )
        self._now = timestamp

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("advance() requires a non-negative duration")
        self._now += seconds

    def hour_of_day(self) -> int:
        return int((self._now / 3600) % 24)

    def day_of_week(self) -> int:
        return int((self._now / 86400) % 7)
