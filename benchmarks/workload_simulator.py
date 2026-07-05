"""
Realistic Workload Simulation for KV Cache Benchmarks.

Uses Poisson processes for session arrivals and Log-normal distributions
for token counts. Users are persistent personas with stable individual
behavior (return propensity, verbosity, diurnal activity window) so that
per-user features like user_historical_return_rate carry real signal —
an i.i.d. user pool would make those features unlearnable by construction.

Timestamps are trace-relative seconds with t=0 = Monday 00:00 (the same
convention used by the training pipeline and SimulatedClock).
"""

import math
import random
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class WorkloadProfile:
    name: str
    # Session arrival: Poisson process
    arrival_rate_per_hour: float
    # Session length: Log-normal (heavy-tailed — matches real chat data)
    token_count_mean: float
    token_count_sigma: float
    # Return behavior: probability of resuming within each time window
    p_return_within_1h: float
    p_return_within_24h: float
    p_return_within_7d: float

PROFILES = {
    "casual": WorkloadProfile(
        name="Casual User",
        arrival_rate_per_hour=50,
        token_count_mean=512,
        token_count_sigma=0.8,
        p_return_within_1h=0.05,
        p_return_within_24h=0.15,
        p_return_within_7d=0.30,
    ),
    "enterprise": WorkloadProfile(
        name="Enterprise Agent",      # Support bots, coding assistants
        arrival_rate_per_hour=200,
        token_count_mean=2048,
        token_count_sigma=0.5,
        p_return_within_1h=0.60,
        p_return_within_24h=0.85,
        p_return_within_7d=0.95,
    ),
    "power_user": WorkloadProfile(
        name="Power User",            # Researchers, developers
        arrival_rate_per_hour=80,
        token_count_mean=8192,
        token_count_sigma=1.2,
        p_return_within_1h=0.35,
        p_return_within_24h=0.70,
        p_return_within_7d=0.90,
    ),
}

# Each successive resume of the same session multiplies the return
# probabilities by this factor: users progressively lose interest.
RESUME_DECAY = 0.8

# Fraction of the profile's mean token count added (in expectation) per
# resume. Conversations grow monotonically; they never shrink.
RESUME_GROWTH_FRACTION = 0.35


@dataclass
class UserPersona:
    """Stable per-user behavior. Drawn once per user, reused for all
    of that user's sessions, so per-user features are learnable."""
    user_id: str
    return_affinity: float   # multiplies profile return probabilities (clamped to [0, 0.98])
    token_scale: float       # multiplies profile token count mean
    active_start_hour: int   # start of the user's 10-hour daily activity window


@dataclass
class WorkloadEvent:
    timestamp: float
    user_id: str
    session_id: str
    action: str  # 'start', 'end', 'resume'
    token_count: int

class WorkloadSimulator:
    """Generates synthetic user traffic using statistically rigorous distributions."""

    def __init__(self, profile_name: str, duration_days: float = 1.0, seed: int = 42,
                 user_pool_size: int = 1000):
        if profile_name not in PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}")
        self.profile = PROFILES[profile_name]
        self.duration_seconds = duration_days * 24 * 3600
        # Instance RNG: never seed the global random module — other code
        # sharing the interpreter would perturb trace reproducibility.
        self.rng = random.Random(seed)
        self._personas: Dict[int, UserPersona] = {}
        self.user_pool_size = user_pool_size

    # ── Personas ────────────────────────────────────────────────

    def _get_persona(self, index: int) -> UserPersona:
        """Lazily create the persona for user pool slot `index`."""
        persona = self._personas.get(index)
        if persona is None:
            persona = UserPersona(
                user_id=f"user_{index}",
                # Log-normal around 1.0: some users almost never return,
                # some return nearly always.
                return_affinity=self.rng.lognormvariate(0.0, 0.6),
                token_scale=self.rng.lognormvariate(0.0, 0.4),
                active_start_hour=self.rng.randint(0, 23),
            )
            self._personas[index] = persona
        return persona

    def _is_active(self, persona: UserPersona, timestamp: float) -> bool:
        """Users are active in a 10-hour daily window."""
        hour = (timestamp / 3600) % 24
        offset = (hour - persona.active_start_hour) % 24
        return offset < 10

    def _pick_user(self, timestamp: float) -> UserPersona:
        """Pick an arrival's user, preferring personas active at this hour."""
        for _ in range(8):
            persona = self._get_persona(self.rng.randint(1, self.user_pool_size))
            if self._is_active(persona, timestamp):
                return persona
        return persona  # fall back to the last draw (rare off-hours arrival)

    # ── Trace generation ────────────────────────────────────────

    def generate(self) -> List[WorkloadEvent]:
        events = []
        current_time = 0.0

        # Mean time between arrivals in seconds
        mtba_seconds = 3600.0 / self.profile.arrival_rate_per_hour

        session_counter = 0

        while current_time < self.duration_seconds:
            # Poisson arrival: inter-arrival times are exponentially distributed
            inter_arrival = self.rng.expovariate(1.0 / mtba_seconds)
            current_time += inter_arrival

            if current_time >= self.duration_seconds:
                break

            session_counter += 1
            persona = self._pick_user(current_time)
            session_id = f"session_{session_counter}"

            self._generate_session_lifecycle(events, current_time, persona, session_id)

        # Since we append future resume events during lifecycle generation, we must sort
        events.sort(key=lambda x: x.timestamp)

        # Filter out events that fall beyond the simulation duration
        events = [e for e in events if e.timestamp <= self.duration_seconds]
        return events

    def _generate_session_lifecycle(self, events: List[WorkloadEvent], start_time: float,
                                     persona: UserPersona, session_id: str):
        # 1. Start event
        tokens = self._sample_tokens(persona)
        events.append(WorkloadEvent(
            timestamp=start_time,
            user_id=persona.user_id,
            session_id=session_id,
            action="start",
            token_count=tokens
        ))

        # 2. End event (session lasts a few seconds to a few minutes)
        session_duration = self.rng.uniform(5.0, 300.0)
        end_time = start_time + session_duration
        events.append(WorkloadEvent(
            timestamp=end_time,
            user_id=persona.user_id,
            session_id=session_id,
            action="end",
            token_count=tokens
        ))

        # 3. Simulate future resumes based on probability windows
        self._schedule_resumes(events, end_time, persona, session_id,
                               prev_tokens=tokens, resume_round=0)

    def _sample_tokens(self, persona: UserPersona) -> int:
        # Log-normal distribution scaled by the user's verbosity persona
        mean = self.profile.token_count_mean * persona.token_scale
        mu = math.log(mean) - (self.profile.token_count_sigma**2)/2
        tokens = int(self.rng.lognormvariate(mu, self.profile.token_count_sigma))
        return max(16, min(tokens, 128000)) # Bound it between 16 and 128k context window

    def _sample_growth(self, persona: UserPersona) -> int:
        """Tokens ADDED to an existing conversation on resume."""
        mean = self.profile.token_count_mean * persona.token_scale * RESUME_GROWTH_FRACTION
        mu = math.log(max(mean, 16)) - (self.profile.token_count_sigma**2)/2
        return max(16, int(self.rng.lognormvariate(mu, self.profile.token_count_sigma)))

    def _schedule_resumes(self, events: List[WorkloadEvent], last_end_time: float,
                          persona: UserPersona, session_id: str,
                          prev_tokens: int, resume_round: int):
        # Persona-adjusted, decayed return probabilities. Each resume round
        # multiplies by RESUME_DECAY so sessions don't resume forever.
        decay = RESUME_DECAY ** resume_round
        adj = persona.return_affinity * decay
        p_1h = min(self.profile.p_return_within_1h * adj, 0.98)
        p_24h = min(self.profile.p_return_within_24h * adj, 0.98)
        p_7d = min(self.profile.p_return_within_7d * adj, 0.98)

        p = self.rng.random()

        if p < p_1h:
            # Resumes within 1 hour
            delay = self.rng.uniform(60, 3600)
        elif p < p_24h:
            # Resumes within 24 hours
            delay = self.rng.uniform(3600, 24 * 3600)
        elif p < p_7d:
            # Resumes within 7 days
            delay = self.rng.uniform(24 * 3600, 7 * 24 * 3600)
        else:
            # Never resumes
            return

        resume_time = last_end_time + delay

        # Conversations grow monotonically: the resumed session's context is
        # everything cached so far plus the new turns. (Resampling a fresh
        # independent token count here would let a "hit" on a small cached
        # session be credited with the recompute cost of a huge unrelated one.)
        new_tokens = prev_tokens + self._sample_growth(persona)
        new_tokens = min(new_tokens, 128000)

        events.append(WorkloadEvent(
            timestamp=resume_time,
            user_id=persona.user_id,
            session_id=session_id,
            action="resume",
            token_count=new_tokens
        ))

        # New end time
        session_duration = self.rng.uniform(5.0, 300.0)
        new_end_time = resume_time + session_duration
        events.append(WorkloadEvent(
            timestamp=new_end_time,
            user_id=persona.user_id,
            session_id=session_id,
            action="end",
            token_count=new_tokens
        ))

        # Recursively see if they resume AGAIN, with decayed probability
        self._schedule_resumes(events, new_end_time, persona, session_id,
                               prev_tokens=new_tokens, resume_round=resume_round + 1)

    def summary(self, events: List[WorkloadEvent]) -> Dict[str, Any]:
        """Provide statistics about the generated trace."""
        unique_users = set()
        unique_sessions = set()
        resumes = 0
        total_tokens = 0

        for e in events:
            unique_users.add(e.user_id)
            unique_sessions.add(e.session_id)
            if e.action == 'resume':
                resumes += 1
            if e.action in ('start', 'resume'):
                total_tokens += e.token_count

        return {
            "profile": self.profile.name,
            "duration_hours": self.duration_seconds / 3600,
            "total_events": len(events),
            "unique_users": len(unique_users),
            "unique_sessions": len(unique_sessions),
            "total_resumes": resumes,
            "resume_rate": resumes / len(unique_sessions) if unique_sessions else 0,
            "avg_tokens_per_interaction": total_tokens / (len(unique_sessions) + resumes) if unique_sessions else 0
        }

    @staticmethod
    def save_trace(events: List[WorkloadEvent], path: str):
        with open(path, 'w') as f:
            json.dump([asdict(e) for e in events], f, indent=2)

    @staticmethod
    def load_trace(path: str) -> List[WorkloadEvent]:
        with open(path, 'r') as f:
            data = json.load(f)
            return [WorkloadEvent(**d) for d in data]

if __name__ == "__main__":
    # Quick test
    sim = WorkloadSimulator("enterprise", duration_days=0.5)
    events = sim.generate()
    print(json.dumps(sim.summary(events), indent=2))
