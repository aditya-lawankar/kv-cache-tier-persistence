"""
Session management.
"""

import time
import uuid
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

@dataclass
class SessionInfo:
    session_id: str
    user_id: str
    started_at: float
    last_active: float
    token_count: int
    is_active: bool

class SessionManager:
    """Tracks active user sessions."""
    
    def __init__(self, on_session_end: Optional[Callable[[str, SessionInfo], None]] = None):
        self._sessions: Dict[str, SessionInfo] = {}
        self._lock = threading.Lock()
        self.on_session_end = on_session_end
        
    def start_session(self, user_id: str, existing_session_id: Optional[str] = None) -> str:
        """Start a new session or resume an existing one."""
        with self._lock:
            session_id = existing_session_id or str(uuid.uuid4())
            now = time.time()
            
            if session_id in self._sessions:
                self._sessions[session_id].last_active = now
                self._sessions[session_id].is_active = True
            else:
                self._sessions[session_id] = SessionInfo(
                    session_id=session_id,
                    user_id=user_id,
                    started_at=now,
                    last_active=now,
                    token_count=0,
                    is_active=True
                )
            return session_id
            
    def end_session(self, session_id: str) -> None:
        """Mark a session as inactive and optionally trigger save."""
        info = None
        with self._lock:
            if session_id in self._sessions:
                info = self._sessions[session_id]
                info.is_active = False
                info.last_active = time.time()
                
        if info and self.on_session_end:
            self.on_session_end(session_id, info)
            
    def update_session(self, session_id: str, new_tokens: int) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].token_count += new_tokens
                self._sessions[session_id].last_active = time.time()
                
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        with self._lock:
            return self._sessions.get(session_id)
            
    def active_sessions(self) -> List[SessionInfo]:
        with self._lock:
            return [s for s in self._sessions.values() if s.is_active]
            
    def cleanup_stale(self, timeout_seconds: float) -> int:
        """End sessions that have been idle for too long."""
        now = time.time()
        stale_ids = []
        
        with self._lock:
            for sid, info in self._sessions.items():
                if info.is_active and (now - info.last_active) > timeout_seconds:
                    stale_ids.append(sid)
                    
        for sid in stale_ids:
            self.end_session(sid)
            
        return len(stale_ids)
