"""
Design document as code: Interceptor for vLLM integration.
"""

import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np

from ..core.tiered_manager import TieredCacheManager
from ..core.session import SessionManager

logger = logging.getLogger(__name__)

class KVCacheInterceptor:
    """
    Integration hook for vLLM.
    
    This class demonstrates how the TieredCacheManager would integrate with a real
    vLLM instance. It maps vLLM's internal events to our persistence system.
    
    Integration Points in vLLM:
    
    1. vllm/worker/cache_engine.py -> CacheEngine
       Hook into `swap_out` to capture blocks being evicted from GPU.
       Instead of just moving to CPU RAM, serialize and push to Warm/Cold tiers.
       
    2. vllm/v1/core/kv_cache_manager.py -> KVCacheManager
       Hook into `free` when a request completes. If it's a session end,
       save the KV cache for that sequence.
       
    3. vllm/v1/core/kv_cache_manager.py -> KVCacheManager.allocate_slots
       When allocating slots for a resuming session, check our `TieredCacheManager`
       first. If there's a cache hit, pre-fill the GPU blocks.
       
    4. vLLM KVTransferConfig (Connector API)
       This could be implemented as a vLLM `KVConnectorBase` plugin natively,
       similar to how LMCache works.
    """
    
    def __init__(self, manager: TieredCacheManager):
        self.manager = manager
        # Use our session manager to track request lifecycles
        self.sessions = SessionManager(on_session_end=self._on_session_idle)
        
    def on_request_start(self, session_id: str, user_id: str) -> Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]]:
        """
        Called when a user sends a new prompt.
        Checks if we have a saved KV cache for this session.
        
        If we return data, vLLM should load it into GPU blocks.
        """
        self.sessions.start_session(user_id, existing_session_id=session_id)
        
        # Check cache
        kv_data = self.manager.load(session_id)
        if kv_data is not None:
            logger.info(f"Cache hit for session {session_id}! Restoring to GPU.")
            # In real vLLM, this data would be copied to `cache_engine.gpu_cache`
        else:
            logger.info(f"Cache miss for session {session_id}. Cold start.")
            
        return kv_data
        
    def on_request_complete(self, session_id: str, kv_data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Called when generation finishes.
        In a real system, we'd copy the tensors from GPU to our system.
        """
        token_count = kv_data[0][0].shape[1] if kv_data else 0
        self.sessions.update_session(session_id, token_count)
        
        # We save immediately upon request completion for persistence
        user_id = self.sessions.get_session(session_id).user_id
        self.manager.save(session_id, user_id, kv_data)
        
    def intercept_swap_out(self, session_id: str, block_ids: list, kv_tensors: Any) -> None:
        """
        Alternative hook: Catch vLLM evicting blocks under memory pressure.
        """
        # In a real implementation, we'd convert the blocked layout to our layout
        pass
        
    def intercept_swap_in(self, session_id: str) -> Any:
        """
        Alternative hook: Supply blocks to vLLM when it swaps in.
        """
        pass
        
    def _on_session_idle(self, session_id: str, info: Any) -> None:
        """Called by SessionManager when a session times out."""
        # Session is cold, maybe demote it explicitly
        self.manager.demote(session_id)
