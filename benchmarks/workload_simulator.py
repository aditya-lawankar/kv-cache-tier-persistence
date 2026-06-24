"""
Realistic Workload Simulation for KV Cache Benchmarks
Uses Poisson processes for session arrivals and Log-normal distributions for token counts.
"""

import math
import random
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
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

@dataclass
class WorkloadEvent:
    timestamp: float
    user_id: str
    session_id: str
    action: str  # 'start', 'end', 'resume'
    token_count: int

class WorkloadSimulator:
    """Generates synthetic user traffic using statistically rigorous distributions."""
    
    def __init__(self, profile_name: str, duration_days: float = 1.0, seed: int = 42):
        if profile_name not in PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}")
        self.profile = PROFILES[profile_name]
        self.duration_seconds = duration_days * 24 * 3600
        random.seed(seed)
        
    def generate(self) -> List[WorkloadEvent]:
        events = []
        current_time = 0.0
        
        # Mean time between arrivals in seconds
        mtba_seconds = 3600.0 / self.profile.arrival_rate_per_hour
        
        session_counter = 0
        
        while current_time < self.duration_seconds:
            # Poisson arrival: inter-arrival times are exponentially distributed
            inter_arrival = random.expovariate(1.0 / mtba_seconds)
            current_time += inter_arrival
            
            if current_time >= self.duration_seconds:
                break
                
            session_counter += 1
            user_id = f"user_{random.randint(1, 1000)}" # Rough pool
            session_id = f"session_{session_counter}"
            
            self._generate_session_lifecycle(events, current_time, user_id, session_id)
            
        # Since we append future resume events during lifecycle generation, we must sort
        events.sort(key=lambda x: x.timestamp)
        
        # Filter out events that fall beyond the simulation duration
        events = [e for e in events if e.timestamp <= self.duration_seconds]
        return events

    def _generate_session_lifecycle(self, events: List[WorkloadEvent], start_time: float, user_id: str, session_id: str):
        # 1. Start event
        tokens = self._sample_tokens()
        events.append(WorkloadEvent(
            timestamp=start_time,
            user_id=user_id,
            session_id=session_id,
            action="start",
            token_count=tokens
        ))
        
        # 2. End event (session lasts a few seconds to a few minutes)
        session_duration = random.uniform(5.0, 300.0)
        end_time = start_time + session_duration
        events.append(WorkloadEvent(
            timestamp=end_time,
            user_id=user_id,
            session_id=session_id,
            action="end",
            token_count=tokens
        ))
        
        # 3. Simulate future resumes based on probability windows
        self._schedule_resumes(events, end_time, user_id, session_id)

    def _sample_tokens(self) -> int:
        # Log-normal distribution
        mu = math.log(self.profile.token_count_mean) - (self.profile.token_count_sigma**2)/2
        tokens = int(random.lognormvariate(mu, self.profile.token_count_sigma))
        return max(16, min(tokens, 128000)) # Bound it between 16 and 128k context window

    def _schedule_resumes(self, events: List[WorkloadEvent], last_end_time: float, user_id: str, session_id: str):
        # Roll for return probability
        p = random.random()
        
        if p < self.profile.p_return_within_1h:
            # Resumes within 1 hour
            delay = random.uniform(60, 3600)
        elif p < self.profile.p_return_within_24h:
            # Resumes within 24 hours
            delay = random.uniform(3600, 24 * 3600)
        elif p < self.profile.p_return_within_7d:
            # Resumes within 7 days
            delay = random.uniform(24 * 3600, 7 * 24 * 3600)
        else:
            # Never resumes
            return
            
        resume_time = last_end_time + delay
        
        # Session grows slightly on resume
        new_tokens = self._sample_tokens()
        
        events.append(WorkloadEvent(
            timestamp=resume_time,
            user_id=user_id,
            session_id=session_id,
            action="resume",
            token_count=new_tokens
        ))
        
        # New end time
        session_duration = random.uniform(5.0, 300.0)
        new_end_time = resume_time + session_duration
        events.append(WorkloadEvent(
            timestamp=new_end_time,
            user_id=user_id,
            session_id=session_id,
            action="end",
            token_count=new_tokens
        ))
        
        # Recursively see if they resume AGAIN after this new end time
        # We decay the probability slightly each time to prevent infinite loops
        self._schedule_resumes(events, new_end_time, user_id, session_id)

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
