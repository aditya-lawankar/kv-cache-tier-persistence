"""
Warm Tier: Filesystem-backed storage targeting NVMe SSD.
"""

import os
import glob
from typing import List, Optional, Any
import logging

from .base import StorageTier, TierUsage

logger = logging.getLogger(__name__)

class WarmTier(StorageTier):
    """
    Filesystem-backed storage targeting NVMe SSD.
    Files stored as: {storage_path}/{session_id}.cache
    """
    
    def __init__(self, storage_path: str, capacity_bytes: int = 64 * 1024**3):
        super().__init__("warm", capacity_bytes)
        self.storage_path = storage_path
        
        # Create directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
    def _get_filepath(self, key: str) -> str:
        return os.path.join(self.storage_path, f"{key}.cache")
        
    def put(self, key: str, data: bytes, metadata: Any) -> None:
        with self._lock:
            data_size = len(data)
            
            if self.is_full(data_size):
                raise MemoryError(f"Warm tier is full. Cannot add {data_size} bytes.")
                
            filepath = self._get_filepath(key)
            with open(filepath, 'wb') as f:
                f.write(data)
                
    def get(self, key: str) -> Optional[bytes]:
        filepath = self._get_filepath(key)
        
        with self._lock:
            if not os.path.exists(filepath):
                return None
                
            # For a real implementation, we could use mmap here for large files,
            # but for simplicity we just read into memory.
            try:
                with open(filepath, 'rb') as f:
                    return f.read()
            except IOError as e:
                logger.error(f"Failed to read from warm tier: {e}")
                return None
                
    def delete(self, key: str) -> bool:
        filepath = self._get_filepath(key)
        
        with self._lock:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    return True
                except OSError as e:
                    logger.error(f"Failed to delete from warm tier: {e}")
                    return False
            return False
            
    def contains(self, key: str) -> bool:
        filepath = self._get_filepath(key)
        with self._lock:
            return os.path.exists(filepath)
            
    def usage(self) -> TierUsage:
        with self._lock:
            current_bytes = 0
            count = 0
            
            # Simple dir walk to sum sizes
            for filepath in glob.glob(os.path.join(self.storage_path, "*.cache")):
                try:
                    current_bytes += os.path.getsize(filepath)
                    count += 1
                except OSError:
                    pass
                    
            return TierUsage(
                name=self.name,
                current_bytes=current_bytes,
                capacity_bytes=self.capacity_bytes,
                entry_count=count
            )
            
    def list_entries(self) -> List[str]:
        with self._lock:
            files = glob.glob(os.path.join(self.storage_path, "*.cache"))
            # Extract key from filename (remove .cache extension)
            return [os.path.basename(f)[:-6] for f in files]
            
    def clear(self) -> None:
        with self._lock:
            for filepath in glob.glob(os.path.join(self.storage_path, "*.cache")):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            logger.info("Warm tier cleared.")
