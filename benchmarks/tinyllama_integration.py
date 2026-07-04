"""
TinyLlama Live Integration: KV Cache Persistence Validation

Validates that our TieredCacheManager can:
1. Extract real KV caches from a HuggingFace Transformers model
2. Serialize them to our tiered storage (Hot → Warm → Cold)
3. Restore them accurately and resume generation seamlessly
4. Measure actual Time-to-First-Token (TTFT) improvement

This bridges the gap between our simulation-based evaluation and
real inference, validating the cost model against actual wall-clock times.

Usage:
    python benchmarks/tinyllama_integration.py
    python benchmarks/tinyllama_integration.py --max-tokens 512 --trials 3
"""

import os
import sys
import time
import shutil
import tempfile
import argparse
import json
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict

# Fix Windows cp1252 encoding for Unicode output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.kv_cache_tier.config import (
    SystemConfig, ModelConfig, TierConfig,
    EvictionConfig, SerializationConfig,
)
from src.kv_cache_tier.core.tiered_manager import TieredCacheManager


# ──────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────

@dataclass
class IntegrationResult:
    """Result of a single cache-restore trial."""
    prompt_tokens: int
    cold_start_ttft_ms: float      # TTFT without cache (full prefill)
    warm_start_ttft_ms: float      # TTFT with cache restored
    speedup_x: float               # cold / warm
    ttft_saved_ms: float           # cold - warm
    cache_size_bytes: int          # size of serialized KV cache
    save_latency_ms: float         # time to save cache to tier system
    load_latency_ms: float         # time to load cache from tier system
    semantic_match: bool           # whether restored generation matches
    cold_output: str               # text generated without cache
    warm_output: str               # text generated with cache


# ──────────────────────────────────────────────────────────────────
# KV cache conversion: HuggingFace ↔ our format
# ──────────────────────────────────────────────────────────────────

def hf_kv_to_numpy(
    past_key_values,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert HuggingFace past_key_values to our storage format.

    Handles both legacy tuple format and modern DynamicCache objects.

    HF format: DynamicCache with .key_cache / .value_cache lists
        key/value shape: (batch, num_heads, seq_len, head_dim)

    Our format: Dict[layer_idx, (key_np, value_np)]
        key/value shape: (num_heads, seq_len, head_dim)  — batch dim squeezed
    """
    kv_data = {}

    # Handle DynamicCache (transformers >= 4.36)
    if hasattr(past_key_values, 'key_cache'):
        num_layers = len(past_key_values.key_cache)
        for layer_idx in range(num_layers):
            k = past_key_values.key_cache[layer_idx]
            v = past_key_values.value_cache[layer_idx]
            k_np = k.squeeze(0).detach().cpu().to(torch.float16).numpy()
            v_np = v.squeeze(0).detach().cpu().to(torch.float16).numpy()
            kv_data[layer_idx] = (k_np, v_np)
    else:
        # Legacy tuple-of-tuples format
        for layer_idx, (k, v) in enumerate(past_key_values):
            k_np = k.squeeze(0).detach().cpu().to(torch.float16).numpy()
            v_np = v.squeeze(0).detach().cpu().to(torch.float16).numpy()
            kv_data[layer_idx] = (k_np, v_np)

    return kv_data


def numpy_to_hf_kv(
    kv_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    """
    Convert our storage format back to HuggingFace DynamicCache.

    Returns a DynamicCache object compatible with transformers 4.36+.
    Adds batch dimension back, converts to model dtype.
    """
    from transformers import DynamicCache

    cache = DynamicCache()
    for layer_idx in sorted(kv_data.keys()):
        k_np, v_np = kv_data[layer_idx]
        k = torch.from_numpy(k_np.astype(np.float32)).unsqueeze(0).to(device, dtype)
        v = torch.from_numpy(v_np.astype(np.float32)).unsqueeze(0).to(device, dtype)
        cache.update(k, v, layer_idx)

    return cache


# ──────────────────────────────────────────────────────────────────
# Core integration logic
# ──────────────────────────────────────────────────────────────────

def load_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Load TinyLlama model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading model: {model_name}")
    print(f"  Device: CPU (no CUDA available)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU requires float32
        device_map="cpu",
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    num_q_heads = model.config.num_attention_heads
    # GQA models have fewer KV heads than query heads
    num_kv_heads = getattr(model.config, 'num_key_value_heads', num_q_heads)
    head_dim = model.config.hidden_size // num_q_heads

    print(f"  Model config: {num_layers} layers, {num_q_heads} Q-heads, {num_kv_heads} KV-heads, head_dim={head_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model, tokenizer


def build_long_prompt(tokenizer, target_tokens: int = 512) -> str:
    """
    Build a multi-turn conversation prompt that reaches approximately
    `target_tokens` in length. Uses the TinyLlama chat template.
    """
    turns = [
        ("What is the KV cache in large language models?",
         "The KV cache stores the key and value tensors computed during "
         "self-attention for all previous tokens. Instead of recomputing "
         "attention over the entire sequence for each new token, the model "
         "caches the K and V projections and only computes the new token's "
         "query against all cached keys. This reduces the computational "
         "complexity of autoregressive generation from O(N^2) per step to "
         "O(N) per step, though it requires O(N) memory to store."),

        ("Why is KV cache eviction important for serving?",
         "In production LLM serving, multiple users share GPU VRAM. Each "
         "user's conversation maintains its own KV cache, consuming memory "
         "proportional to the context length. When VRAM is exhausted, the "
         "system must evict some caches. A naive LRU policy treats all "
         "caches equally, but sessions with long contexts are far more "
         "expensive to recompute from scratch due to the quadratic cost "
         "of the prefill phase. Smart eviction should prioritize keeping "
         "expensive-to-recompute sessions."),

        ("How does tiered storage help with this problem?",
         "Tiered storage extends the effective cache capacity beyond VRAM. "
         "Instead of discarding evicted KV caches entirely, the system "
         "serializes them to NVMe SSD (warm tier) or object storage like "
         "S3 (cold tier). When a user returns, their cache can be restored "
         "from the warm tier in milliseconds rather than recomputing from "
         "scratch, which could take seconds for long contexts. The trade-off "
         "is serialization and deserialization latency versus full "
         "recomputation cost."),

        ("What is the difference between hit rate and cache value?",
         "Hit rate counts how many requests find their data in cache, "
         "treating all hits as equally valuable. Cache value weights each "
         "hit by the cost it avoids. In KV caching, a hit on a 4096-token "
         "session saves approximately 16x more GPU compute than a hit on a "
         "512-token session due to quadratic attention scaling. A policy "
         "achieving 87 percent hit rate on mostly large sessions can deliver "
         "more GPU savings than one achieving 93 percent on mostly small "
         "sessions. This is why we argue the correct optimization target "
         "is expected cache value, not hit rate."),
    ]

    # Build chat messages
    messages = []
    for user_msg, assistant_msg in turns:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    # Tokenize to check length, repeat turns if needed
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    current_tokens = len(tokenizer.encode(text))

    # If we need more tokens, repeat the conversation
    repeat = 0
    while current_tokens < target_tokens and repeat < 10:
        for user_msg, assistant_msg in turns:
            messages.append({"role": "user", "content": f"Elaborating further on turn {repeat+1}: {user_msg}"})
            messages.append({"role": "assistant", "content": assistant_msg})
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        current_tokens = len(tokenizer.encode(text))
        repeat += 1

    # Add final user question for the model to answer
    messages.append({"role": "user", "content": "Given everything we discussed, what is the single most important insight about KV cache eviction?"})

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_single_trial(
    model,
    tokenizer,
    manager: TieredCacheManager,
    prompt: str,
    session_id: str,
    max_new_tokens: int = 50,
) -> IntegrationResult:
    """
    Run one complete trial:
      1. Cold-start: full prefill + generate (no cache)
      2. Save KV cache to TieredCacheManager
      3. Load KV cache from TieredCacheManager
      4. Warm-start: generate with restored cache (skip prefill)
      5. Compare outputs for semantic coherence
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_tokens = input_ids.shape[1]
    print(f"    Prompt: {prompt_tokens} tokens")

    # ── Step 1: Cold start (full prefill + generation) ──
    print(f"    [1/5] Cold-start prefill...", end="", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        cold_output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            return_dict_in_generate=True,
            output_hidden_states=False,
        )
    cold_ttft_total = (time.perf_counter() - t0) * 1000  # ms

    cold_text = tokenizer.decode(
        cold_output.sequences[0][prompt_tokens:], skip_special_tokens=True
    )
    print(f" done ({cold_ttft_total:.0f}ms)")

    # ── Step 2: Get KV cache from a prefill-only pass ──
    print(f"    [2/5] Extracting KV cache...", end="", flush=True)
    with torch.no_grad():
        prefill_output = model(input_ids, use_cache=True)
    past_kv = prefill_output.past_key_values

    # Convert HF KV cache to our numpy format
    kv_data = hf_kv_to_numpy(past_kv)

    # Calculate cache size
    cache_size = sum(k.nbytes + v.nbytes for k, v in kv_data.values())
    print(f" done ({cache_size / 1024:.1f} KB)")

    # ── Step 3: Save to TieredCacheManager ──
    print(f"    [3/5] Saving to tier system...", end="", flush=True)
    t0 = time.perf_counter()
    manager.save(
        session_id=session_id,
        user_id="tinyllama_demo",
        kv_data=kv_data,
        metadata={"prompt_tokens": prompt_tokens, "model": "TinyLlama-1.1B"},
    )
    save_latency = (time.perf_counter() - t0) * 1000
    print(f" done ({save_latency:.1f}ms)")

    # ── Step 4: Load from TieredCacheManager ──
    print(f"    [4/5] Restoring from tier system...", end="", flush=True)
    t0 = time.perf_counter()
    restored_kv_data = manager.load(session_id)
    load_latency = (time.perf_counter() - t0) * 1000

    assert restored_kv_data is not None, "Cache load failed — session not found!"
    print(f" done ({load_latency:.1f}ms)")

    # Convert back to HuggingFace format
    restored_past_kv = numpy_to_hf_kv(restored_kv_data, device, dtype)

    # ── Step 5: Warm start (generate with restored cache) ──
    print(f"    [5/5] Warm-start generation...", end="", flush=True)

    # Manual autoregressive generation with restored KV cache.
    # This avoids transformers' generate() cache_position issues and
    # gives us precise TTFT measurement for the first forward pass.
    next_token = input_ids[:, -1:]  # Start from last prompt token
    generated_ids = []
    warm_past_kv = restored_past_kv
    cache_seq_len = prompt_tokens - 1  # Cache covers tokens 0..N-2

    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Build cache_position for current token
            pos = torch.tensor([[cache_seq_len + step]], dtype=torch.long, device=device)

            outputs = model(
                input_ids=next_token,
                past_key_values=warm_past_kv,
                cache_position=pos.squeeze(0),
                use_cache=True,
            )

            if step == 0:
                warm_ttft = (time.perf_counter() - t0) * 1000  # First token latency

            warm_past_kv = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    warm_ttft_total = (time.perf_counter() - t0) * 1000  # Total generation time
    print(f" done ({warm_ttft_total:.0f}ms, TTFT={warm_ttft:.0f}ms)")

    warm_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ── Compare outputs ──
    semantic_match = cold_text.strip() == warm_text.strip()
    speedup = cold_ttft_total / max(warm_ttft_total, 0.01)
    ttft_saved = cold_ttft_total - warm_ttft_total

    return IntegrationResult(
        prompt_tokens=prompt_tokens,
        cold_start_ttft_ms=round(cold_ttft_total, 2),
        warm_start_ttft_ms=round(warm_ttft_total, 2),
        speedup_x=round(speedup, 2),
        ttft_saved_ms=round(ttft_saved, 2),
        cache_size_bytes=cache_size,
        save_latency_ms=round(save_latency, 2),
        load_latency_ms=round(load_latency, 2),
        semantic_match=semantic_match,
        cold_output=cold_text[:200],
        warm_output=warm_text[:200],
    )


# ──────────────────────────────────────────────────────────────────
# Main integration test
# ──────────────────────────────────────────────────────────────────

def run_integration(
    max_tokens: int = 512,
    num_trials: int = 3,
    max_new_tokens: int = 50,
    output_dir: str = "benchmarks/results",
):
    """Run the full TinyLlama integration test."""
    print()
    print("=" * 80)
    print("  TinyLlama KV Cache Persistence — Live Integration Test")
    print("=" * 80)

    # Load model
    model, tokenizer = load_model()
    model_config = model.config
    num_layers = model_config.num_hidden_layers
    num_q_heads = model_config.num_attention_heads
    # GQA: KV cache uses num_key_value_heads, not num_attention_heads
    num_kv_heads = getattr(model_config, 'num_key_value_heads', num_q_heads)
    head_dim = model_config.hidden_size // num_q_heads

    # Set up TieredCacheManager with real model dimensions
    tmp_dir = tempfile.mkdtemp(prefix="kvcache_tinyllama_")
    print(f"  Tier storage: {tmp_dir}")

    config = SystemConfig(
        model=ModelConfig(
            num_layers=num_layers,
            num_heads=num_kv_heads,  # Use KV heads, not Q heads (GQA)
            head_dim=head_dim,
            block_size=16,
            dtype="float16",
        ),
        tiers=TierConfig(
            hot_capacity_mb=500,     # 500 MB hot
            warm_capacity_mb=1000,   # 1 GB warm (NVMe)
            cold_capacity_mb=2000,   # 2 GB cold
            warm_storage_path=os.path.join(tmp_dir, "warm"),
            cold_storage_path=os.path.join(tmp_dir, "cold"),
            cold_backend="local",
        ),
        eviction=EvictionConfig(policy="lru"),
        serialization=SerializationConfig(
            format="raw_binary",
            compression="none",
            compression_level=1,
        ),
    )
    manager = TieredCacheManager(config)

    # Build prompt
    print(f"\n  Building {max_tokens}-token conversation prompt...")
    prompt = build_long_prompt(tokenizer, target_tokens=max_tokens)
    actual_tokens = len(tokenizer.encode(prompt))
    print(f"  Actual prompt length: {actual_tokens} tokens")

    # Run trials
    results: List[IntegrationResult] = []
    token_targets = [max_tokens]

    # Add a shorter prompt trial if max_tokens > 256
    if max_tokens > 256:
        token_targets = [256, max_tokens]

    for target in token_targets:
        prompt = build_long_prompt(tokenizer, target_tokens=target)

        for trial in range(num_trials):
            print(f"\n  ── Trial {trial + 1}/{num_trials} (target: {target} tokens) ──")
            session_id = f"tinyllama_trial_{target}t_{trial}"

            result = run_single_trial(
                model=model,
                tokenizer=tokenizer,
                manager=manager,
                prompt=prompt,
                session_id=session_id,
                max_new_tokens=max_new_tokens,
            )
            results.append(result)

            print(f"    Result: Cold={result.cold_start_ttft_ms:.0f}ms → "
                  f"Warm={result.warm_start_ttft_ms:.0f}ms "
                  f"({result.speedup_x:.2f}x speedup)")
            print(f"    Semantic match: {'✓' if result.semantic_match else '✗ MISMATCH'}")
            if not result.semantic_match:
                print(f"      Cold: {result.cold_output[:100]}...")
                print(f"      Warm: {result.warm_output[:100]}...")

    # ── Print summary table ──
    print()
    print("=" * 100)
    print("  RESULTS: TinyLlama KV Cache Persistence Integration")
    print("=" * 100)
    print(f"  {'Tokens':>8} {'Cold TTFT':>12} {'Warm TTFT':>12} {'Speedup':>10} "
          f"{'TTFT Saved':>12} {'Cache Size':>12} {'Save':>10} {'Load':>10} {'Match':>7}")
    print("  " + "-" * 96)

    for r in results:
        print(f"  {r.prompt_tokens:>8} {r.cold_start_ttft_ms:>10.0f}ms "
              f"{r.warm_start_ttft_ms:>10.0f}ms {r.speedup_x:>9.2f}x "
              f"{r.ttft_saved_ms:>10.0f}ms {r.cache_size_bytes/1024:>10.1f}KB "
              f"{r.save_latency_ms:>8.0f}ms {r.load_latency_ms:>8.0f}ms "
              f"{'  ✓' if r.semantic_match else '  ✗'}")

    print("=" * 100)

    # ── Summary statistics ──
    avg_cold = np.mean([r.cold_start_ttft_ms for r in results])
    avg_warm = np.mean([r.warm_start_ttft_ms for r in results])
    avg_speedup = np.mean([r.speedup_x for r in results])
    all_match = all(r.semantic_match for r in results)

    print(f"\n  Average Cold TTFT:  {avg_cold:.0f}ms")
    print(f"  Average Warm TTFT:  {avg_warm:.0f}ms")
    print(f"  Average Speedup:    {avg_speedup:.2f}x")
    print(f"  Semantic Coherence: {'ALL MATCH ✓' if all_match else 'MISMATCHES DETECTED ✗'}")

    # ── Cost model validation ──
    # Compare predicted recompute time vs actual measured cold-start TTFT
    from src.kv_cache_tier.utils.cost_model import CostModel
    cost_model = CostModel()

    print(f"\n  ── Cost Model Validation ──")
    for r in results:
        predicted_ms = cost_model.compute_recompute_time(r.prompt_tokens) * 1000
        actual_ms = r.cold_start_ttft_ms
        error_pct = abs(predicted_ms - actual_ms) / actual_ms * 100
        print(f"    {r.prompt_tokens} tokens: predicted={predicted_ms:.0f}ms "
              f"actual={actual_ms:.0f}ms error={error_pct:.1f}%")

    # ── Save results ──
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "tinyllama_integration_results.json")
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TinyLlama KV Cache Persistence Integration Test"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Target prompt length in tokens (default: 512)"
    )
    parser.add_argument(
        "--trials", type=int, default=3,
        help="Number of trials per token length (default: 3)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50,
        help="Tokens to generate per trial (default: 50)"
    )
    parser.add_argument(
        "--output", type=str, default="benchmarks/results",
        help="Output directory for results"
    )
    args = parser.parse_args()

    run_integration(
        max_tokens=args.max_tokens,
        num_trials=args.trials,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output,
    )
