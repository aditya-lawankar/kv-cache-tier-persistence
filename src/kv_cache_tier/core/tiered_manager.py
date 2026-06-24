"""
Main orchestrator for tiered KV cache storage.
"""

import time
import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np

from ..config import SystemConfig
from .cache_metadata import CacheEntry, CacheIndex
from ..tiers import HotTier, WarmTier, ColdTier, StorageTier
from ..serialization import create_serializer, create_compressor, CompressedSerializer
from ..eviction import create_eviction_policy
from ..utils.metrics import metrics

logger = logging.getLogger(__name__)

class TieredCacheManager:
    """Orchestrates KV cache storage across hot, warm, and cold tiers."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # Initialize tiers
        self.hot_tier = HotTier(config.tiers.hot_capacity_mb * 1024 * 1024)
        self.warm_tier = WarmTier(
            config.tiers.warm_storage_path,
            config.tiers.warm_capacity_mb * 1024 * 1024
        )
        self.cold_tier = ColdTier.create(
            backend=config.tiers.cold_backend,
            storage_path=config.tiers.cold_storage_path,
            endpoint=config.tiers.minio_endpoint,
            access_key=config.tiers.minio_access_key,
            secret_key=config.tiers.minio_secret_key,
            bucket=config.tiers.minio_bucket,
            capacity_bytes=config.tiers.cold_capacity_mb * 1024 * 1024
        )
        
        # Initialize serialization
        base_serializer = create_serializer(config.serialization.format)
        compressor = create_compressor(config.serialization.compression, config.serialization.compression_level)
        self.serializer = CompressedSerializer(base_serializer, compressor)
        
        # Initialize metadata and eviction
        self.index = CacheIndex()
        self.eviction_policy = create_eviction_policy(
            config.eviction.policy,
            ttl_seconds={
                "hot": config.eviction.hot_ttl_seconds,
                "warm": config.eviction.warm_ttl_seconds,
                "cold": config.eviction.cold_ttl_seconds
            },
            alpha=config.eviction.predictive_alpha,
            beta=config.eviction.predictive_beta,
            gamma=config.eviction.predictive_gamma
        )
        
    def _get_tier(self, tier_name: str) -> StorageTier:
        if tier_name == "hot": return self.hot_tier
        if tier_name == "warm": return self.warm_tier
        if tier_name == "cold": return self.cold_tier
        raise ValueError(f"Unknown tier: {tier_name}")
        
    def save(self, session_id: str, user_id: str, kv_data: Dict[int, Tuple[np.ndarray, np.ndarray]], metadata: dict = None) -> None:
        """Save KV cache to the hot tier, evicting older entries if necessary."""
        with metrics.record_timer("manager.save", {"tier": "hot"}):
            # Serialize
            with metrics.record_timer("manager.serialize"):
                data = self.serializer.serialize(kv_data, self.config.model)
                
            data_size = len(data)
            
            # Ensure space in hot tier
            while self.hot_tier.is_full(data_size):
                evicted = self.evict_from_tier("hot")
                if not evicted:
                    # Hot tier couldn't free space — save directly to warm
                    self._ensure_space_in_tier("warm", data_size)
                    self._save_to_tier("warm", session_id, user_id, data, kv_data, metadata)
                    return
                    
            self._save_to_tier("hot", session_id, user_id, data, kv_data, metadata)

    def _ensure_space_in_tier(self, tier_name: str, data_size: int) -> None:
        """Cascade evictions through the tier hierarchy to make space."""
        tier = self._get_tier(tier_name)
        max_attempts = 50  # Safety valve
        attempts = 0
        while tier.is_full(data_size) and attempts < max_attempts:
            evicted = self.evict_from_tier(tier_name)
            if not evicted:
                # If this tier can't evict, the entry is too large or everything is stuck
                logger.warning(f"Cannot make space in {tier_name} for {data_size} bytes")
                break
            attempts += 1
            
    def _save_to_tier(self, tier_name: str, session_id: str, user_id: str, data: bytes, kv_data: dict, metadata: dict) -> None:
        tier = self._get_tier(tier_name)
        
        # Token count from layer 0 key tensor
        token_count = kv_data[0][0].shape[1] if kv_data else 0
        num_blocks = (token_count + self.config.model.block_size - 1) // self.config.model.block_size
        
        now = time.time()
        
        # Update or create entry
        entry = self.index.get(session_id)
        if entry:
            entry.tier = tier_name
            entry.last_accessed = now
            entry.size_bytes = len(data)
            entry.token_count = token_count
            entry.num_blocks = num_blocks
        else:
            entry = CacheEntry(
                session_id=session_id,
                user_id=user_id,
                model_config_hash=str(hash(str(self.config.model))), # Simple hash for now
                created_at=now,
                last_accessed=now,
                token_count=token_count,
                num_blocks=num_blocks,
                tier=tier_name,
                size_bytes=len(data),
                access_count=1,
                metadata=metadata or {}
            )
            self.index.add(entry)
            self.eviction_policy.on_insert(session_id, entry)
            
        tier.put(session_id, data, entry)
        logger.debug(f"Saved session {session_id} to {tier_name} tier ({len(data)} bytes)")

    def load(self, session_id: str) -> Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]]:
        """Load KV cache, searching tiers hot->warm->cold."""
        with metrics.record_timer("manager.load"):
            entry = self.index.get(session_id)
            if not entry:
                metrics.record("cache.miss", 1, {"reason": "not_found"})
                return None
                
            tier_name = entry.tier
            tier = self._get_tier(tier_name)
            
            data = tier.get(session_id)
            if not data:
                # Index out of sync with storage
                self.index.remove(session_id)
                metrics.record("cache.miss", 1, {"reason": "missing_data", "tier": tier_name})
                return None
                
            metrics.record("cache.hit", 1, {"tier": tier_name})
            
            # Update metadata
            entry.last_accessed = time.time()
            entry.access_count += 1
            self.eviction_policy.on_access(session_id, entry)
            
            # Deserialize
            with metrics.record_timer("manager.deserialize"):
                kv_data, _ = self.serializer.deserialize(data)
                
            # Promote if not in hot
            if tier_name != "hot":
                # Background promotion would be better, but we do it inline here
                self.promote(session_id, "hot")
                
            return kv_data
            
    def promote(self, session_id: str, target_tier: str) -> bool:
        """Move an entry to a hotter tier."""
        entry = self.index.get(session_id)
        if not entry or entry.tier == target_tier:
            return False
            
        with metrics.record_timer("manager.promote", {"from": entry.tier, "to": target_tier}):
            source_tier = self._get_tier(entry.tier)
            dest_tier = self._get_tier(target_tier)
            
            data = source_tier.get(session_id)
            if not data:
                return False
                
            # Ensure space in dest
            while dest_tier.is_full(len(data)):
                if not self.evict_from_tier(target_tier):
                    logger.warning(f"Could not promote to {target_tier}: tier full and eviction failed")
                    return False
                    
            dest_tier.put(session_id, data, entry)
            source_tier.delete(session_id)
            
            entry.tier = target_tier
            return True
            
    def demote(self, session_id: str) -> bool:
        """Move an entry to the next colder tier."""
        entry = self.index.get(session_id)
        if not entry:
            return False
            
        current_tier = entry.tier
        if current_tier == "hot":
            target_tier = "warm"
        elif current_tier == "warm":
            target_tier = "cold"
        else:
            # Cannot demote from cold
            return False
            
        with metrics.record_timer("manager.demote", {"from": current_tier, "to": target_tier}):
            source_tier = self._get_tier(current_tier)
            dest_tier = self._get_tier(target_tier)
            
            data = source_tier.get(session_id)
            if not data:
                return False
                
            # Ensure space in dest
            self._ensure_space_in_tier(target_tier, len(data))
            if dest_tier.is_full(len(data)):
                if target_tier == "cold":
                    logger.warning(f"Cold tier full, permanently deleting {session_id}")
                    source_tier.delete(session_id)
                    self.index.remove(session_id)
                    self.eviction_policy.on_remove(session_id)
                    return True
                return False
                    
            dest_tier.put(session_id, data, entry)
            source_tier.delete(session_id)
            
            entry.tier = target_tier
            return True
            
    def evict_from_tier(self, tier_name: str) -> bool:
        """Select a victim from the tier and demote it."""
        entries_in_tier = {e.session_id: e for e in self.index.all_entries().values() if e.tier == tier_name}
        
        if not entries_in_tier:
            return False
            
        victim_id = self.eviction_policy.select_victim(entries_in_tier)
        if not victim_id:
            return False
            
        if tier_name == "cold":
            # Demoting from cold means permanent deletion
            self.delete(victim_id)
            return True
        else:
            return self.demote(victim_id)
            
    def delete(self, session_id: str) -> bool:
        """Permanently delete from all tiers."""
        entry = self.index.get(session_id)
        if not entry:
            return False
            
        tier = self._get_tier(entry.tier)
        tier.delete(session_id)
        self.index.remove(session_id)
        self.eviction_policy.on_remove(session_id)
        return True
        
    def run_maintenance(self) -> None:
        """Periodic background task to clean up expired entries."""
        expired = []
        for session_id, entry in self.index.all_entries().items():
            if self.eviction_policy.should_evict(session_id, entry):
                expired.append(session_id)
                
        for sid in expired:
            self.demote(sid) # Demote instead of outright delete, eventually falls out of cold
            
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "index": self.index.stats(),
            "usage": {
                "hot": self.hot_tier.usage(),
                "warm": self.warm_tier.usage(),
                "cold": self.cold_tier.usage()
            },
            "metrics": metrics.get_all_stats()
        }
