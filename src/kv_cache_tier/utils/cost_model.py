from dataclasses import dataclass
from typing import Dict

@dataclass
class CostModel:
    """
    Models the economic value of KV cache tier persistence.
    Translates cache hit rates into GPU compute hours and dollar savings.
    """
    gpu_cost_per_hour: float = 2.50         # A100 80GB on-demand rough $/hr
    tokens_per_second_with_cache: int = 80  # Decoding speed (fast)
    tokens_per_second_cold_start: int = 15  # Prefill speed (slow for large contexts)
    attention_quadratic_factor: float = 5e-5 # O(N^2) attention overhead factor

    def compute_recompute_time(self, token_count: int) -> float:
        """Calculate time to process token_count WITHOUT cache (full recompute). Includes O(N^2) attention."""
        linear_time = token_count / self.tokens_per_second_cold_start
        quadratic_time = self.attention_quadratic_factor * (token_count ** 2)
        return linear_time + quadratic_time

    def compute_savings(self, cache_hit_rate: float,
                       sessions_per_hour: int,
                       avg_token_count: int) -> Dict[str, float]:
        """
        Compute daily savings based on avoided cold starts.
        
        Args:
            cache_hit_rate: Fraction of sessions that resumed without recomputing
            sessions_per_hour: Total concurrent sessions cycling through the system
            avg_token_count: Average context size in tokens
            
        Returns:
            Dictionary with GPU hours saved, USD saved, and effective capacity increase.
        """
        # How many cold starts did we avoid per hour?
        cold_starts_avoided_per_hour = sessions_per_hour * cache_hit_rate
        
        # Time to process avg_token_count WITHOUT cache (full recompute)
        time_cold_seconds = self.compute_recompute_time(avg_token_count)
        
        # Time to process avg_token_count WITH cache (instant resume, just decoding)
        time_warm_seconds = avg_token_count / self.tokens_per_second_with_cache
        
        # GPU seconds saved per avoided cold start
        seconds_saved_per_session = time_cold_seconds - time_warm_seconds
        
        # Total GPU seconds saved per hour
        gpu_seconds_saved_per_hour = cold_starts_avoided_per_hour * seconds_saved_per_session
        
        # Convert to daily metrics
        gpu_hours_saved_per_day = (gpu_seconds_saved_per_hour * 24) / 3600.0
        cost_saved_per_day_usd = gpu_hours_saved_per_day * self.gpu_cost_per_hour
        
        return {
            "gpu_hours_saved_per_day": round(gpu_hours_saved_per_day, 2),
            "cost_saved_per_day_usd": round(cost_saved_per_day_usd, 2),
            "effective_capacity_increase_pct": round(cache_hit_rate * 100, 2),
            "cold_starts_avoided_per_day": int(cold_starts_avoided_per_hour * 24)
        }

if __name__ == "__main__":
    # Quick test of the cost model
    model = CostModel()
    # Assume 500 concurrent users generating ~2000 sessions per hour total, 60% hit rate, 4096 avg tokens
    savings = model.compute_savings(
        cache_hit_rate=0.60,
        sessions_per_hour=2000,
        avg_token_count=4096
    )
    print("Cost Model Projections:")
    for k, v in savings.items():
        print(f"  {k}: {v}")
