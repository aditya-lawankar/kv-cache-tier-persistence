"""
Production-grade observability layer using Prometheus.
"""
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging
import threading

logger = logging.getLogger(__name__)

class PrometheusMetrics:
    """
    Exposes core cache operations and tier migrations to Prometheus.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PrometheusMetrics, cls).__new__(cls)
                cls._instance._init_metrics()
            return cls._instance

    def _init_metrics(self):
        self.cache_hits = Counter(
            'kv_cache_hits_total',
            'Cache hits by tier',
            ['tier', 'policy']
        )
        
        self.cache_misses = Counter(
            'kv_cache_misses_total',
            'Cache misses (cold starts)',
            ['policy']
        )
        
        self.tier_utilization = Gauge(
            'kv_cache_tier_utilization_bytes',
            'Bytes used per tier',
            ['tier']
        )
        
        self.migration_latency = Histogram(
            'kv_cache_migration_seconds',
            'Tier migration latency',
            ['direction'],
            buckets=[.001, .005, .01, .05, .1, .5, 1.0, 5.0]
        )
        
        self.promotion_total = Counter(
            'kv_cache_promotions_total',
            'Promotions by direction',
            ['from_tier', 'to_tier']
        )
        
        self.server_started = False
        self.server_port = 8000

    def start_server(self, port: int = 8000):
        """Start the Prometheus metrics endpoint in a background thread."""
        with self._lock:
            if not self.server_started:
                try:
                    start_http_server(port)
                    self.server_started = True
                    self.server_port = port
                    logger.info(f"Prometheus metrics exposed on port {port}")
                except Exception as e:
                    logger.error(f"Failed to start Prometheus server on port {port}: {e}")

    # Helper methods to record metrics safely
    def record_hit(self, tier: str, policy: str = "default"):
        self.cache_hits.labels(tier=tier, policy=policy).inc()

    def record_miss(self, policy: str = "default"):
        self.cache_misses.labels(policy=policy).inc()

    def update_utilization(self, tier: str, bytes_used: int):
        self.tier_utilization.labels(tier=tier).set(bytes_used)

    def record_migration(self, from_tier: str, to_tier: str, duration_seconds: float):
        direction = f"{from_tier}_to_{to_tier}"
        self.migration_latency.labels(direction=direction).observe(duration_seconds)
        self.promotion_total.labels(from_tier=from_tier, to_tier=to_tier).inc()

# Global singleton
prom_metrics = PrometheusMetrics()
