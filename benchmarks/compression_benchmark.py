"""
Compression benchmark on REAL TinyLlama KV cache bytes.

Measures what general-purpose codecs (LZ4, Zstandard, zlib) actually achieve
on serialized FP16 KV tensors -- the exact bytes the warm tier stores --
rather than assuming ratios from the literature. Synthetic simulator tensors
are useless here (np.zeros compresses to nothing), so this script runs a real
prefill and compresses the resulting cache.

For each algorithm/level we report:
  * compression ratio (raw / compressed)
  * median compress / decompress latency over N trials
  * effective one-way transfer time at the paper's modeled tier bandwidths
    (NVMe 2 GB/s, object storage 500 MB/s):
        write:   compress_ms + compressed_bytes / B
        restore: compressed_bytes / B + decompress_ms
    vs. raw_bytes / B uncompressed. Compression pays on a tier iff the
    effective time beats the uncompressed transfer.

Usage:
    python benchmarks/compression_benchmark.py [--tokens 512 1792] [--trials 5]

Output: benchmarks/results/compression_benchmark.json + stdout table.
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.kv_cache_tier.config import ModelConfig
from src.kv_cache_tier.serialization.raw_binary_serde import RawBinarySerializer
from benchmarks.tinyllama_integration import load_model, build_long_prompt, hf_kv_to_numpy


TIER_BANDWIDTHS = {"nvme_2GBps": 2e9, "s3_500MBps": 500e6}


def real_kv_bytes(target_tokens: int):
    """Run a real prefill and serialize the KV cache with the repo's serde."""
    import torch

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    cfg = model.config
    model_config = ModelConfig(
        num_layers=cfg.num_hidden_layers,
        num_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16,
        dtype="float16",
    )

    prompt = build_long_prompt(tokenizer, target_tokens=target_tokens)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    n_tokens = input_ids.shape[1]
    print(f"  Prefilling {n_tokens} tokens on {device}...")
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    kv_data = hf_kv_to_numpy(out.past_key_values)

    serde = RawBinarySerializer()
    raw = serde.serialize(kv_data, model_config)
    print(f"  Serialized KV cache: {len(raw) / 1024:.1f} KB ({n_tokens} tokens)")
    return n_tokens, bytes(raw)


def _codecs():
    """(label, compress_fn, decompress_fn) triples."""
    import lz4.frame
    import zlib
    import zstandard

    out = []
    for level in (0, 9):
        out.append((
            f"lz4-{level}",
            lambda b, lv=level: lz4.frame.compress(b, compression_level=lv),
            lz4.frame.decompress,
        ))
    for level in (1, 3, 9, 19):
        cctx = zstandard.ZstdCompressor(level=level)
        dctx = zstandard.ZstdDecompressor()
        out.append((f"zstd-{level}", cctx.compress, dctx.decompress))
    out.append((
        "zlib-6",
        lambda b: zlib.compress(b, 6),
        zlib.decompress,
    ))
    return out


def bench_codec(label, compress, decompress, raw: bytes, trials: int) -> dict:
    comp_times, decomp_times = [], []
    compressed = compress(raw)  # warm-up + correctness copy
    assert decompress(compressed) == raw, f"{label}: roundtrip mismatch"
    for _ in range(trials):
        t0 = time.perf_counter()
        compressed = compress(raw)
        comp_times.append((time.perf_counter() - t0) * 1000)
        t1 = time.perf_counter()
        _ = decompress(compressed)
        decomp_times.append((time.perf_counter() - t1) * 1000)

    c_ms = statistics.median(comp_times)
    d_ms = statistics.median(decomp_times)
    row = {
        "algorithm": label,
        "raw_kb": len(raw) / 1024,
        "compressed_kb": len(compressed) / 1024,
        "ratio": len(raw) / len(compressed),
        "compress_ms_median": round(c_ms, 2),
        "decompress_ms_median": round(d_ms, 2),
        "trials": trials,
    }
    for tier, bw in TIER_BANDWIDTHS.items():
        raw_ms = len(raw) / bw * 1000
        row[f"write_ms_{tier}"] = round(c_ms + len(compressed) / bw * 1000, 2)
        row[f"restore_ms_{tier}"] = round(len(compressed) / bw * 1000 + d_ms, 2)
        row[f"uncompressed_ms_{tier}"] = round(raw_ms, 2)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 1792],
                        help="Target prompt lengths (default: 512 1792)")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="benchmarks/results/compression_benchmark.json")
    args = parser.parse_args()

    all_results = []
    for target in args.tokens:
        print(f"\n=== Target {target} tokens ===")
        n_tokens, raw = real_kv_bytes(target)
        for label, comp, decomp in _codecs():
            row = bench_codec(label, comp, decomp, raw, args.trials)
            row["prompt_tokens"] = n_tokens
            all_results.append(row)
            print(f"  {label:<8} ratio={row['ratio']:.3f}x "
                  f"comp={row['compress_ms_median']:>8.1f}ms "
                  f"decomp={row['decompress_ms_median']:>6.1f}ms "
                  f"| restore S3: {row['restore_ms_s3_500MBps']:>6.1f}ms "
                  f"(raw {row['uncompressed_ms_s3_500MBps']:.1f}ms)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} rows to {args.output}")


if __name__ == "__main__":
    main()
