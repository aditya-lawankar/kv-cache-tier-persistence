# Reproducing the results

Detailed instructions for every experiment in the papers. The main README covers installation
and the headline `make` targets. All commands run from the repository root after
`pip install -e ".[dev]"`.

## Training the eviction predictors

The learned (V1) policies need trained resumption predictors. The training pipeline generates workload traces, extracts session features, and trains two models (Logistic Regression + Gradient Boosted Trees):

```bash
# Train both models on 7-day simulated workloads (all three profiles)
python src/kv_cache_tier/eviction/train_predictors.py
```

This produces:
- `models/logistic_predictor.pkl` — Interpretable model with coefficients
- `models/gbt_predictor.pkl` — High-accuracy ensemble model
- `models/training_results.json` — Structured evaluation metrics

Expected output:
```
======================================================================
  MODEL COMPARISON - Session Resumption Prediction
======================================================================
  Model                 Train Acc   Test Acc  Train AUC   Test AUC
  -------------------- ---------- ---------- ---------- ----------
  Logistic Reg.            0.6425     0.6442     0.7017     0.7022
  Gradient Boosted         0.6727     0.6729     0.7333     0.7300
======================================================================
```

The workload simulator assigns each user a persistent *persona* (return
propensity, verbosity, diurnal activity window), so per-user features such
as `user_historical_return_rate` carry learnable signal — with an i.i.d.
user pool those features are noise by construction and AUC saturates near
0.56 (coin-flip).

## Benchmark suite

Run the suite (cross-platform Python commands):
```bash
# Run a quick check (uses small model config, completes in seconds)
python -m benchmarks.run_benchmarks --suite quick

# Run the full rigorous research suite
python -m benchmarks.run_benchmarks --suite all
```

*(Charts and structured empirical results are saved to `benchmarks/results/`)*

## Running the synthetic experiment matrix

Regenerates every number in the paper's main results table (6 policies × 3 workloads × 10 seeds):

```bash
make reproduce
# or, step by step:
python src/kv_cache_tier/eviction/train_predictors.py   # only if models/*.pkl are missing
python benchmarks/experiment_runner.py --duration 0.25 --seeds 10
python benchmarks/generate_figures.py
```

## Running the real-trace evaluation (Azure LLM inference)

Replays ten 6-hour windows of the one-week Azure LLM inference trace
([AzurePublicDataset](https://github.com/Azure/AzurePublicDataset), 27.3M production requests)
through all six policies. See `benchmarks/azure_trace_loader.py` for the conversion
semantics: real arrival timestamps and request sizes, modeled return behavior.

```bash
make reproduce-azure
# equivalently:
python benchmarks/experiment_runner.py --azure
```

The first run downloads the trace (~1.1 GB) into `data/` and parses it once into a
compressed cache (~1 minute); both are gitignored and reused afterwards. The full run
takes ~5 hours serially. To finish in under an hour, shard by policy across parallel
processes and merge:

```bash
for p in lru heuristic logistic_v1 value_density value_density_ac space_time; do
  python benchmarks/experiment_runner.py --azure --policies $p \
    --output benchmarks/results/azure_shards/$p &
done
wait
python benchmarks/merge_results.py --prefix azure \
  benchmarks/results/azure_shards/*/experiment_results_azure_raw.json
```

Either path prints the paired results table and writes the canonical
`benchmarks/results/experiment_results_azure_{raw,aggregate}.json`. To inspect a single
window's converted workload without running experiments:

```bash
python benchmarks/azure_trace_loader.py --window 3
```

## Running the oracle ablation (ground-truth policies)

**Why:** when a learned policy loses to LRU, is the bottleneck the *predictor* or the
*objective*? These policies replay the same traces with the trace's actual future:
`belady` (evict farthest next access — the classical hit-rate oracle), `oracle_v1`
(V1's objective with a perfect classifier), and `oracle_v3` (V3's objective with perfect
P(resume) and the exact next-access gap — a value-weighted greedy Belady). Legitimate
only in open-loop replay, where arrivals do not depend on cache behavior.
See `src/kv_cache_tier/eviction/oracle.py`.

```bash
make reproduce-oracle
# sharded across 5 processes (seeds 42-51, ~2 per shard), then merged:
for base in 42 44 46 48 50; do
  python benchmarks/experiment_runner.py --duration 0.25 --seeds 2 --seed-base $base \
    --workloads enterprise,power_user --policies lru,belady,oracle_v1,oracle_v3 \
    --output benchmarks/results/oracle_shards/shard_$base &
done
wait
python benchmarks/merge_results.py --prefix oracle \
  benchmarks/results/oracle_shards/*/experiment_results_v3_raw.json
```

## Running the capacity regime sweep

**Why:** any single-capacity study implicitly picks a winner. This sweeps total capacity
from 125 MB to 2 GB (same 10/30/60 tier split, enterprise workload, 5 paired seeds per
point) for the three deployable policies and the three oracles, producing the regime map
in Figure 5: LRU wins at moderate pressure, value-aware eviction wins under severe
scarcity, and even clairvoyant policies trade hit rate for value once capacity binds.

```bash
make reproduce-capacity
```

## Running the long-context GPU sweep (Colab)

**Why:** the T4 sweep in the paper is capped at 1,792 tokens by TinyLlama's 2,048-token
context window, only just past its predicted crossover of N* ~ 1,561. `benchmarks/context_sweep_gpu.ipynb`
repeats the measurement on a long-context model (Qwen2.5-0.5B by default, ungated) across
512 to 16,384 tokens, entirely above N*, so the measured curve can be compared against the
prediction over a decade of context length rather than at a single crossing.

Defaults to `meta-llama/Llama-3.2-1B` (GQA 8/32, 32 KB/token), whose geometry matches the
architecture discussion in the paper. That repository is gated: accept its license on
HuggingFace and add a read token as a Colab secret named `HF_TOKEN`. `Qwen/Qwen2.5-0.5B` is
an ungated fallback one line away in the config cell.

Upload the notebook to [Colab](https://colab.research.google.com/), set
**Runtime > Change runtime type > T4 GPU** before running, then **Runtime > Run all**.
It takes 15-25 minutes and downloads `context_sweep_results.json` at the end. The notebook
derives N* from the same equations as `benchmarks/breakeven_analysis.py`, so it is
self-contained and needs nothing from this repo.

Note that a T4 is compute capability 7.5 and cannot run PyTorch's flash SDPA backend, which
requires sm80+. Left alone, attention falls back to the math backend, which materializes the
N x N score matrix: that OOMs at 16k tokens and inflates prefill time progressively with
context length. The notebook pins a tiled backend to avoid it and prints the device
capability so the fallback is visible rather than silent.

## Verifying the paper's numbers

The paper's remaining error class is arithmetic hygiene: a figure quoted in prose that no
longer matches the table it came from after a re-run. Those survive proofreading because
each sentence is locally plausible, so they are checked mechanically instead.

```bash
python benchmarks/audit_paper_numbers.py       # full paper
python benchmarks/audit_workshop_numbers.py    # 4-page workshop paper
# or both:
make audit
```

They re-derive every quoted figure from the committed result JSONs, assert that retracted
claims stay absent from the text, and check for rendering damage that has broken silently
before. Non-zero exit on any discrepancy.

The workshop audit additionally checks every cell of the paper's one table, the values
plotted in its one figure, and that the body still ends by page 4 — so the page limit is
enforced by the build rather than remembered.

## Running the persona strength sweep

**Why:** the resumption signal in the synthetic workloads is generated, so the
predictor's AUC is partly a property of the generator. This sweeps the parameter that
controls how much users differ from one another, retrains the predictor at each setting
so the model is as good as that world permits, and re-runs the enterprise matrix. It
separates the conclusion that depends on that modeling choice (whether learned eviction
is worth building) from the one that does not (whether hit rate is the right target).

```bash
make reproduce-persona
```

Each sigma writes its own predictor, so the three settings can also be run in parallel
by passing `--predictor <path>` to the runner. Note that a shared
`models/logistic_predictor.pkl` is what the main study loads by default; give
concurrent runs their own copies rather than overwriting it.

## Running the compression benchmark (real KV bytes)

**Why:** the simulator's tensors are zero-filled, so compression claims must be measured
on real model output. This runs a real TinyLlama prefill, serializes the cache with the
repo's warm-tier format, and benchmarks LZ4/Zstandard/zlib on those exact bytes,
reporting ratios, codec latencies, and effective transfer times at the modeled tier
bandwidths.

```bash
make bench-compression
```

## Validating on a real model (TinyLlama-1.1B, CPU)

**Why:** the eviction study runs on simulated caches; this proves the persistence layer
works end-to-end on a real transformer — extract a KV cache from HuggingFace, round-trip
it through the tier system, resume generation, and verify the output is unchanged.

```bash
python benchmarks/tinyllama_integration.py --max-tokens 598 --trials 2
```

Downloads TinyLlama-1.1B (~2.2 GB) on first run; ~10 minutes on CPU. Reports, per trial:

- **Cold TTFT** — a timed *prefill-only* forward pass (the cost a cache hit avoids), and
  **warm TTFT** — tier load + tensor restore + one forward pass. These are reported
  separately from **end-to-end** times: comparing a cold prefill+decode total against a
  warm first-token latency inflates the apparent speedup ~2.5× (a mistake this benchmark
  originally made, now guarded against by construction).
- **Output equality** against the cold path, plus an **in-memory control** (resume from
  the never-serialized cache) that attributes any divergence to either the storage
  round-trip or the batched-prefill vs incremental-decode kernel paths.

## GPU validation (Colab, free T4)

**Why:** on CPU, prefill is so slow that persistence always wins; the honest question is
where restoration beats recomputation on real inference hardware. This produces the
measured points that test the break-even model's crossover prediction (N* ≈ 1,561 tokens
for TinyLlama).

1. Open [`benchmarks/gpu_validation.ipynb` in Colab](https://colab.research.google.com/github/aditya-lawankar/kv-cache-tier-persistence/blob/main/benchmarks/gpu_validation.ipynb)
2. `Runtime` → `Change runtime type` → **T4 GPU** → Save
3. `Runtime` → `Run all` (~15 min). Every cell asserts its own success; the last cell
   downloads `tinyllama_gpu_results.json` — commit it to `benchmarks/results/`.

Measured result (T4): TTFT speedup 0.62× at 168 tokens → 0.72× at 440 → **1.11× at
1,792**, bracketing the predicted crossover.

## Break-even analysis (no hardware needed)

**Why:** restoring a cache moves O(N) bytes while prefill costs O(N²) compute, so for
every (model, GPU, storage tier) there is a context length N* beyond which persistence
always wins on latency. This derives N* analytically and generates Figure 4:

```bash
python benchmarks/breakeven_analysis.py
```

## Building the papers

```bash
make arxiv            # full paper → arxiv_bundle.zip
make workshop         # 4-page workshop paper, then audit it (fails if it spills past page 4)
make workshop-bundle  # workshop paper → workshop_submission.zip
# or manually:
cd paper/latex    && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
cd paper/workshop && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

The two papers are independent build products: neither target touches the other's sources
or output. Every number and figure in both regenerates from the committed result JSONs —
if a claim in either paper cannot be traced to `benchmarks/results/*.json`, that is a bug.
