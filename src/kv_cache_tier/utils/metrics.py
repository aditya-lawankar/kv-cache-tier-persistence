"""
Performance metrics collection.
"""

import time
import threading
from typing import Dict, List, Any, Optional
import numpy as np

class Timer:
    """Context manager for timing operations."""
    def __init__(self, name: str, collector: 'MetricsCollector', tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.collector = collector
        self.tags = tags
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> 'Timer':
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        self.collector.record(self.name, self.elapsed, self.tags)


class MetricsCollector:
    """Thread-safe collector for performance metrics."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: Dict[str, List[float]] = {}
        
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a single metric value."""
        # For simplicity, we incorporate tags into the metric name if present
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            name = f"{name}[{tag_str}]"
            
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []
            self._metrics[name].append(value)
            
    def record_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> Timer:
        """Return a Timer context manager that records to this collector."""
        return Timer(name, self, tags)
        
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a specific metric."""
        with self._lock:
            values = self._metrics.get(name, [])
            
        if not values:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
            
        arr = np.array(values)
        return {
            "count": len(values),
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }
        
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all recorded metrics."""
        with self._lock:
            names = list(self._metrics.keys())
            
        return {name: self.get_stats(name) for name in names}
        
    def reset(self) -> None:
        """Clear all recorded metrics."""
        with self._lock:
            self._metrics.clear()
            
    def to_dict(self) -> Dict[str, Any]:
        """Export raw metrics as a dictionary."""
        with self._lock:
            # Return a copy to avoid concurrent modification issues
            return {k: list(v) for k, v in self._metrics.items()}

# Global singleton
metrics = MetricsCollector()
