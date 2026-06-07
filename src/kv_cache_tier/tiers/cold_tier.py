"""
Cold Tier: Object storage (S3/MinIO) or local archive.
"""

import os
import glob
import logging
from typing import List, Optional, Any

try:
    import boto3
    from botocore.exceptions import ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    boto3 = None
    ClientError = None

from .base import StorageTier, TierUsage

logger = logging.getLogger(__name__)

class LocalColdBackend(StorageTier):
    """
    Local filesystem-based cold tier.
    Stores files with a .cold extension.
    """
    
    def __init__(self, storage_path: str, capacity_bytes: int = 0):
        super().__init__("cold", capacity_bytes)
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        
    def _get_filepath(self, key: str) -> str:
        return os.path.join(self.storage_path, f"{key}.cold")
        
    def put(self, key: str, data: bytes, metadata: Any) -> None:
        with self._lock:
            if self.is_full(len(data)):
                raise MemoryError("Cold tier (local) is full.")
                
            filepath = self._get_filepath(key)
            with open(filepath, 'wb') as f:
                f.write(data)
                
    def get(self, key: str) -> Optional[bytes]:
        filepath = self._get_filepath(key)
        with self._lock:
            if not os.path.exists(filepath):
                return None
            try:
                with open(filepath, 'rb') as f:
                    return f.read()
            except IOError as e:
                logger.error(f"Failed to read from local cold tier: {e}")
                return None
                
    def delete(self, key: str) -> bool:
        filepath = self._get_filepath(key)
        with self._lock:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    return True
                except OSError:
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
            for filepath in glob.glob(os.path.join(self.storage_path, "*.cold")):
                try:
                    current_bytes += os.path.getsize(filepath)
                    count += 1
                except OSError:
                    pass
                    
            return TierUsage(self.name, current_bytes, self.capacity_bytes, count)
            
    def list_entries(self) -> List[str]:
        with self._lock:
            files = glob.glob(os.path.join(self.storage_path, "*.cold"))
            return [os.path.basename(f)[:-5] for f in files]
            
    def clear(self) -> None:
        with self._lock:
            for filepath in glob.glob(os.path.join(self.storage_path, "*.cold")):
                try:
                    os.remove(filepath)
                except OSError:
                    pass


class MinioColdBackend(StorageTier):
    """
    S3/MinIO compatible object storage cold tier.
    """
    
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str, capacity_bytes: int = 0):
        super().__init__("cold", capacity_bytes)
        self.bucket = bucket
        
        if not HAS_BOTO3:
            raise ImportError("boto3 is required for MinIO cold backend")
            
        try:
            self.s3 = boto3.client('s3',
                                   endpoint_url=f"http://{endpoint}",
                                   aws_access_key_id=access_key,
                                   aws_secret_access_key=secret_key,
                                   region_name='us-east-1') # minio default
            
            # Create bucket if it doesn't exist
            try:
                self.s3.head_bucket(Bucket=bucket)
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    self.s3.create_bucket(Bucket=bucket)
                else:
                    raise
                    
            self.is_connected = True
        except Exception as e:
            logger.error(f"Failed to connect to MinIO: {e}")
            self.is_connected = False
            self.s3 = None
            
    def put(self, key: str, data: bytes, metadata: Any) -> None:
        if not self.is_connected:
            raise RuntimeError("MinIO not connected")
            
        try:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        except Exception as e:
            logger.error(f"S3 Put failed: {e}")
            raise
            
    def get(self, key: str) -> Optional[bytes]:
        if not self.is_connected:
            return None
            
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            logger.error(f"S3 Get failed: {e}")
            return None
            
    def delete(self, key: str) -> bool:
        if not self.is_connected:
            return False
            
        if self.contains(key):
            self.s3.delete_object(Bucket=self.bucket, Key=key)
            return True
        return False
        
    def contains(self, key: str) -> bool:
        if not self.is_connected:
            return False
            
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
            
    def usage(self) -> TierUsage:
        if not self.is_connected:
            return TierUsage(self.name, 0, self.capacity_bytes, 0)
            
        current_bytes = 0
        count = 0
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        current_bytes += obj['Size']
                        count += 1
        except Exception as e:
            logger.error(f"S3 Usage failed: {e}")
            
        return TierUsage(self.name, current_bytes, self.capacity_bytes, count)
        
    def list_entries(self) -> List[str]:
        if not self.is_connected:
            return []
            
        entries = []
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket):
                if 'Contents' in page:
                    entries.extend([obj['Key'] for obj in page['Contents']])
        except Exception:
            pass
        return entries
        
    def clear(self) -> None:
        if not self.is_connected:
            return
            
        try:
            entries = self.list_entries()
            if entries:
                objects = [{'Key': key} for key in entries]
                self.s3.delete_objects(Bucket=self.bucket, Delete={'Objects': objects})
        except Exception as e:
            logger.error(f"S3 Clear failed: {e}")


class ColdTier:
    """Factory for creating the appropriate cold tier backend."""
    
    @staticmethod
    def create(backend: str, **kwargs) -> StorageTier:
        if backend == "minio":
            try:
                tier = MinioColdBackend(
                    endpoint=kwargs.get("endpoint", "localhost:9000"),
                    access_key=kwargs.get("access_key", "minioadmin"),
                    secret_key=kwargs.get("secret_key", "minioadmin"),
                    bucket=kwargs.get("bucket", "kv-cache"),
                    capacity_bytes=kwargs.get("capacity_bytes", 0)
                )
                if tier.is_connected:
                    return tier
                else:
                    logger.warning("MinIO connection failed. Falling back to local backend.")
            except ImportError:
                logger.warning("boto3 not installed. Falling back to local backend.")
                
        # Default or fallback
        return LocalColdBackend(
            storage_path=kwargs.get("storage_path", "./data/cold"),
            capacity_bytes=kwargs.get("capacity_bytes", 0)
        )
