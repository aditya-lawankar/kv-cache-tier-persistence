"""
Main benchmark runner.
"""

import argparse
import json
import logging
import os
from rich.console import Console
from rich.table import Table

from .latency_bench import LatencyBenchmark
from .throughput_bench import ThroughputBenchmark
from .hit_rate_bench import HitRateBenchmark

logging.basicConfig(level=logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="KV Cache Tier Benchmarks")
    parser.add_argument("--suite", choices=['quick', 'latency', 'throughput', 'hitrate', 'all'], default='quick')
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--iterations", type=int, default=0)
    args = parser.parse_args()

    console = Console()
    is_quick = args.suite == 'quick'

    console.print(f"[bold blue]Starting Benchmark Suite: {args.suite}[/bold blue]")
    if is_quick:
        console.print("[dim]Using small model config for fast results. Use --suite all for full-size benchmarks.[/dim]")

    all_results = {}

    if args.suite in ['quick', 'latency', 'all']:
        console.print("\n[yellow]Running Latency Benchmark...[/yellow]")
        iters = args.iterations or (5 if is_quick else 50)
        bench = LatencyBenchmark(iterations=iters, output_dir=args.output_dir, use_small_model=is_quick)
        all_results['latency'] = bench.run()
        console.print("[green]  OK Latency benchmark done[/green]")

    if args.suite in ['quick', 'throughput', 'all']:
        console.print("\n[yellow]Running Throughput Benchmark...[/yellow]")
        iters = args.iterations or (10 if is_quick else 100)
        bench = ThroughputBenchmark(iterations=iters, output_dir=args.output_dir, use_small_model=is_quick)
        all_results['throughput'] = bench.run()
        console.print("[green]  OK Throughput benchmark done[/green]")

    if args.suite in ['quick', 'hitrate', 'all']:
        console.print("\n[yellow]Running Hit Rate Benchmark...[/yellow]")
        num_users = 10 if is_quick else 50
        duration = 300 if is_quick else 1800
        bench = HitRateBenchmark(output_dir=args.output_dir, num_users=num_users, duration_seconds=duration)
        all_results['hit_rate'] = bench.run()
        console.print("[green]  OK Hit rate benchmark done[/green]")

    # Save composite results
    with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    console.print("\n")
    _print_summary(console, all_results)
    console.print(f"\n[bold green]Benchmarks Complete![/bold green]")
    console.print(f"Results and charts saved to: [cyan]{args.output_dir}[/cyan]")


def _print_summary(console: Console, results: dict):
    """Print a nice summary table of results."""
    if 'latency' in results:
        table = Table(title="Latency (ms)", show_header=True, header_style="bold cyan")
        table.add_column("Token Count")
        table.add_column("Save -> Hot", justify="right")
        table.add_column("Load <- Hot", justify="right")
        table.add_column("Demote -> Warm", justify="right")
        table.add_column("Promote <- Warm", justify="right")
        for size_label, ops in results['latency'].items():
            table.add_row(
                size_label,
                f"{ops['save_hot'] * 1000:.2f}",
                f"{ops['load_hot'] * 1000:.2f}",
                f"{ops['demote_hot_warm'] * 1000:.2f}",
                f"{ops['promote_warm_hot'] * 1000:.2f}",
            )
        console.print(table)

    if 'throughput' in results:
        table = Table(title="Throughput (ops/sec)", show_header=True, header_style="bold cyan")
        table.add_column("Format")
        table.add_column("1 Worker", justify="right")
        table.add_column("2 Workers", justify="right")
        table.add_column("4 Workers", justify="right")
        for fmt, workers in results['throughput'].items():
            table.add_row(
                fmt,
                f"{workers.get('1_workers', 0):.1f}",
                f"{workers.get('2_workers', 0):.1f}",
                f"{workers.get('4_workers', 0):.1f}",
            )
        console.print(table)

    if 'hit_rate' in results:
        table = Table(title="Cache Hit Rates", show_header=True, header_style="bold cyan")
        table.add_column("Eviction Policy")
        table.add_column("Hit Rate", justify="right")
        for policy, rate in results['hit_rate'].items():
            table.add_row(policy, f"{rate * 100:.1f}%")
        console.print(table)


if __name__ == "__main__":
    main()
