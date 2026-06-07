"""
Synthetic workload generation for benchmarks.
"""

import math
import random
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class WorkloadConfig:
    num_users: int = 100
    mean_session_tokens: int = 512
    session_return_prob: float = 0.6
    return_time_decay: float = 1800  # seconds
    arrival_rate: float = 10.0       # users per minute
    duration_seconds: int = 3600     # 1 hour
    seed: int = 42

@dataclass
class WorkloadEvent:
    timestamp: float
    user_id: str
    session_id: str
    action: str  # 'start', 'end', 'resume'
    token_count: int

class WorkloadGenerator:
    """Generates synthetic user traffic for KV cache testing."""
    
    def __init__(self, config: WorkloadConfig):
        self.config = config
        random.seed(config.seed)
        
    def generate(self) -> List[WorkloadEvent]:
        events = []
        current_time = 0.0
        active_sessions = {}
        
        # Mean time between arrivals
        mtba = 60.0 / self.config.arrival_rate
        
        # User pool
        user_ids = [f"user_{i}" for i in range(self.config.num_users)]
        
        session_counter = 0
        
        while current_time < self.config.duration_seconds:
            # Next arrival
            # Exponential distribution for Poisson process
            inter_arrival = random.expovariate(1.0 / mtba)
            current_time += inter_arrival
            
            if current_time >= self.config.duration_seconds:
                break
                
            # Pick a user (some are more active than others, zipf distribution approx)
            user_idx = int(abs(random.gauss(0, self.config.num_users / 3)))
            user_idx = min(user_idx, self.config.num_users - 1)
            user_id = user_ids[user_idx]
            
            # Determine tokens for this session
            # Log-normal is realistic for session lengths
            mu = math.log(self.config.mean_session_tokens) - 0.5
            tokens = int(random.lognormvariate(mu, 1.0))
            tokens = max(16, min(tokens, 8192)) # Bound it
            
            # Check if this user has an old session to resume
            if user_id in active_sessions and random.random() < self.config.session_return_prob:
                session_id = active_sessions[user_id]
                action = 'resume'
            else:
                session_counter += 1
                session_id = f"session_{session_counter}"
                active_sessions[user_id] = session_id
                action = 'start'
                
            events.append(WorkloadEvent(
                timestamp=current_time,
                user_id=user_id,
                session_id=session_id,
                action=action,
                token_count=tokens
            ))
            
            # Session end happens shortly after
            end_time = current_time + random.uniform(5.0, 120.0)
            events.append(WorkloadEvent(
                timestamp=end_time,
                user_id=user_id,
                session_id=session_id,
                action='end',
                token_count=tokens
            ))
            
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)
        return events
        
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
