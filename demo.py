import time
from rich.console import Console

console = Console()

console.print("\n[bold cyan]Loading KV Cache Tier Persistence Demo...[/bold cyan]")
start = time.time()
from kv_cache_tier.config import SystemConfig, ModelConfig
from kv_cache_tier.core.tiered_manager import TieredCacheManager
from kv_cache_tier.utils.tensor_utils import generate_random_kv_cache
console.print(f"[dim]Imports took {time.time() - start:.2f}s[/dim]\n")

def run_demo():
    # Use a tiny configuration so it runs instantly
    config = SystemConfig.default()
    config.model = ModelConfig(num_layers=2, num_heads=2, head_dim=32, block_size=16)
    config.tiers.hot_capacity_mb = 10
    config.tiers.warm_capacity_mb = 100
    
    console.print("[bold]1. Initializing Tiered Cache Manager...[/bold]")
    manager = TieredCacheManager(config)
    
    console.print("[bold]2. Generating dummy 1024-token KV Cache (Tiny Model)[/bold]")
    kv_data = generate_random_kv_cache(config.model, 1024)
    
    session_id = "demo_session_123"
    
    console.print(f"\n[bold]3. Simulating Operations:[/bold]")
    
    # Save (Hot)
    t0 = time.perf_counter()
    manager.save(session_id, "user_1", kv_data)
    console.print(f"  [green]OK[/green] Saved to Hot Tier [dim]({(time.perf_counter()-t0)*1000:.2f} ms)[/dim]")
    
    # Load (Hot)
    t0 = time.perf_counter()
    manager.load(session_id)
    console.print(f"  [green]OK[/green] Loaded from Hot Tier [dim]({(time.perf_counter()-t0)*1000:.2f} ms)[/dim]")
    
    # Demote (Hot -> Warm)
    t0 = time.perf_counter()
    manager.demote(session_id)
    console.print(f"  [yellow]OK[/yellow] Demoted to Warm Tier (Disk) [dim]({(time.perf_counter()-t0)*1000:.2f} ms)[/dim]")
    
    # Load (Warm -> Hot)
    t0 = time.perf_counter()
    manager.load(session_id)
    console.print(f"  [cyan]OK[/cyan] Promoted back to Hot Tier [dim]({(time.perf_counter()-t0)*1000:.2f} ms)[/dim]")
    
    # Demote (Hot -> Warm -> Cold)
    manager.demote(session_id) # to warm
    t0 = time.perf_counter()
    manager.demote(session_id) # to cold
    console.print(f"  [blue]OK[/blue] Demoted to Cold Tier (Compressed Archive) [dim]({(time.perf_counter()-t0)*1000:.2f} ms)[/dim]")
    
    console.print("\n[bold green]Demo Completed Successfully![/bold green]")
    
if __name__ == "__main__":
    run_demo()
