<div align="center">
  <h1>🔄 KV Cache Tier Persistence</h1>
  <p><b>A Research Prototype: Predictive Tiered Storage for LLM Inference</b></p>

  ![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
  ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
  ![Status: Research Prototype](https://img.shields.io/badge/Status-Research_Prototype-success.svg)

  *Every time a ChatGPT session ends, gigabytes of GPU compute are thrown in the trash. This project investigates how to catch them, store them, and serve them back intelligently.*
</div>

---

## 🔬 Research Question & Findings

**Primary Research Question:**  
*Can a learned eviction policy applied to a three-tier KV cache hierarchy significantly reduce LLM cold-start cost compared to LRU and TTL under realistic multi-session workloads?*

We began with the hypothesis that predictive eviction would beat LRU on **hit rate**. The data
falsified that hypothesis — and the diagnosis of *why* became the project's actual contribution:

1. **Binary classification eviction loses to LRU under capacity pressure.** Eviction is a
   Knapsack-style packing problem; predicting P(resume) while ignoring entry size retains
   large sessions that choke out dozens of small ones.
2. **Static value-density (Value/Byte) eviction fails even harder on heavy-tailed workloads.**
   A cache is a dynamic system governed by Little's Law: maximizing value per byte without
   bounding sojourn time collapses cache cardinality.
3. **Hit rate is the wrong metric for KV caching.** With O(N²) prefill recompute costs, a
   policy with a *lower* hit rate can deliver *higher* GPU savings by retaining
   expensive-to-recompute long-context sessions. The correct objective is **expected cache
   value**, and hits must be discounted by the restore cost of the tier they come from.

**Headline results** (500MB tiered cache, 6h simulated, 10 seeds, paired 95% CIs vs LRU
on identical traces):

| Policy | Enterprise hit% | Δ$/day vs LRU [CI] | Power-user hit% | Δ$/day vs LRU [CI] |
|---|---|---|---|---|
| LRU | **79.2** | — | **96.0** | — |
| Heuristic | 55.3 | −255 [−312, −199] | 84.2 | **−41 [−60, −22]** |
| Logistic V1 | 67.9 | −230 [−323, −138] | 83.9 | −214 [−276, −152] |
| Value Density V2 | 47.9 | −257 [−348, −167] | 71.9 | −202 [−252, −152] |
| Space-Time V3 | 51.2 | **−163 [−249, −77]** | 74.8 | **−168 [−215, −120]** |

Three takeaways: **LRU wins everything under honest evaluation** (an earlier buggy harness
showed the learned policy beating LRU — evaluation bugs preferentially flatter complex
policies); **hit rate and value decouple** (heuristic keeps 97.5% of LRU's value while
conceding 12 hit-rate points; V1 beats V3 on hits but loses to it on dollars); and
**lifetime normalization works** (V3 recovers +$94/day over static V2, CI [+77, +112]).

**Real-trace replication** (Azure LLM inference trace, one week of production arrivals,
ten 6-hour windows as paired replicates): every finding holds. LRU wins both metrics
(83.9% hit rate, all paired deltas significant), the hit-rate/value inversion is *stronger*
than on synthetic workloads (Logistic V1 beats Space-Time V3 by 18.9 hit points yet
delivers $105/day *less* value, CI [+77, +133] in V3's favor), and V3 recovers +$78/day
over static V2 (CI [+61, +95]). Replays real arrival timestamps and request sizes;
session return behavior is modeled (the public trace has no conversation identifiers —
see `benchmarks/azure_trace_loader.py` for the permutation-control evidence). Regenerate
via `make reproduce-azure` (downloads ~1.1 GB on first use).

All numbers regenerate via `make reproduce` — see `benchmarks/experiment_runner.py`,
`benchmarks/results/experiment_results_v3_aggregate.json`, and
`benchmarks/results/experiment_results_azure_aggregate.json`.

---

## 📄 Research Paper

**[Hit Rate Is Not Value: A Rigorous Evaluation of Learned and Value-Aware Eviction for
Tiered LLM KV-Cache Persistence](paper.pdf)** — 7-page USENIX-style paper covering the
system design, the V1→V2→V3 eviction-policy progression, the statistical methodology,
the break-even analysis, and the TinyLlama end-to-end validation. LaTeX sources in
[`paper/latex/`](paper/latex/); every number and figure regenerates from the committed
result JSONs (`make reproduce`).

---

## 🛑 The Problem

Modern LLM inference relies heavily on the **KV Cache** (Key-Value Attention Cache) to avoid recomputing previous tokens in a sequence. This cache lives in GPU VRAM, which is extremely fast but heavily constrained.

Consider a LLaMA-7B model:
- A single 512-token conversation produces **~256MB** of KV cache.
- When the user closes the tab, this 256MB is **discarded**.
- If the user returns 10 minutes later, the system experiences a **Cold Start**, recomputing all 256MB from scratch.
- At scale (e.g., 500 concurrent users), you are wasting **128GB of VRAM capacity** continuously.

## 💡 The Solution

This project introduces a **Three-Tier Storage Hierarchy** for KV caches, modeled exactly on how enterprise petabyte-scale storage systems operate. Instead of discarding the cache on session end, it is migrated to cheaper storage and reloaded on demand.

```text
┌─────────────────────────────────┐
│  HOT TIER  — GPU VRAM           │  ← Active conversations right now
│  ~80GB, ~microsecond latency    │    (Simulated in memory)
├─────────────────────────────────┤
│  WARM TIER — NVMe SSD / CPU RAM │  ← Recent sessions, may resume soon
│  ~1–4TB, ~millisecond latency   │    (Filesystem-backed)
├─────────────────────────────────┤
│  COLD TIER — Object Storage     │  ← Archived sessions, long-term
│  Unlimited, ~100ms latency      │    (MinIO/S3 or compressed local)
└─────────────────────────────────┘
```

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Client
        U["User Session"]
    end

    subgraph "Inference Hook"
        I["KVCacheInterceptor"]
    end

    subgraph TieredCacheManager
        E["Learned Eviction Policy<br/>Predictive Model"]
        S["Serializer + Compressor<br/>Raw Binary (CRC32) / Zstd"]

        H[("Hot Tier<br/>GPU VRAM")]
        W[("Warm Tier<br/>NVMe SSD")]
        C[("Cold Tier<br/>S3 / MinIO")]
    end

    U -->|"End Session"| I
    I -->|"save()"| S
    S -->|"Store"| H

    H -.->|"Predictive Evict"| W
    W -.->|"Predictive Evict"| C

    U -->|"Resume Session"| I
    I -->|"load()"| H
    H -->|"Miss"| W
    W -->|"Miss"| C
    C -->|"Found & Promote"| W
    W -->|"Found & Promote"| H
```

## 🧠 The Eviction Policy Progression (V1 → V2 → V3)

Traditional systems use LRU (Least Recently Used) for cache eviction. However, LLM user patterns are highly predictable. A user debugging code (Enterprise) has vastly different return patterns than someone generating a quick recipe (Casual).

The project evaluates a progression of eviction policies:

- **V1 (Logistic/GBT):** treats retention as **binary classification** — predict
  P(resume) within a time window, evict the least likely. *Fails under capacity
  pressure*: classification ignores entry size (the Knapsack mismatch).
- **V2 (Value Density):** maximizes expected GPU savings per cached byte,
  P(resume) × RecomputeCost(N) / Size(N), with an optional admission-control gate.
  *Fails on heavy-tailed workloads*: static packing ignores sojourn time
  (the Little's Law collapse).
- **V3 (Space-Time Density):** divides value density by expected sojourn
  time E[Δt] (estimated from an EMA of observed inter-access gaps), charging
  each entry for the space-time volume it occupies — an LHD-style objective
  adapted to quadratic recompute costs. Implemented in
  `src/kv_cache_tier/eviction/space_time.py`.

Features engineered for the P(resume) models:
- `session_age_minutes`: How long since the session was created
- `token_count`: Conversation length (proxy for context value)
- `revisit_count`: Number of times the user resumed this session
- `hour_of_day`: Time-of-day signal (capturing enterprise 9-5 behavior)
- `user_historical_return_rate`: Per-user return probability estimate

## 💸 Cost Modeling

Drawing from enterprise storage design constraints where cost-per-GB is a first-class citizen, this system includes a formal cost model. By converting cache hit rates directly into GPU-hours saved, we can quantify the dollar value of tier promotion vs. recomputation at scale (e.g. 500 concurrent users on A100 instances).

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/aditya-lawankar/kv-cache-tier-persistence.git
cd kv-cache-tier-persistence
pip install -e ".[dev]"
```

### Usage Example
```python
import numpy as np
from kv_cache_tier.config import SystemConfig
from kv_cache_tier.core.tiered_manager import TieredCacheManager
from kv_cache_tier.utils.tensor_utils import generate_random_kv_cache

# 1. Initialize configuration
config = SystemConfig.default()

# 2. Start the manager
manager = TieredCacheManager(config)

# 3. Simulate a session ending
session_id = "user123_chat_1"
dummy_kv_data = generate_random_kv_cache(config.model, token_count=512)

# Save cache (goes to Hot tier, evicts older to Warm/Cold if full)
manager.save(session_id, user_id="user123", kv_data=dummy_kv_data)

# 4. User returns 20 minutes later!
loaded_data = manager.load(session_id)
if loaded_data:
    print("✅ Cache Hit! Resumed instantly without GPU recompute.")
```

## 🧪 Unit Tests

The project includes 55 tests covering serialization round-trips (including CRC32 corruption detection), eviction logic, tier migrations, ML predictor training, the simulated clock, workload realism properties (monotonic session growth, persona differentiation, reproducibility), and the Azure real-trace loader:

```bash
# Run all tests
pytest tests/ -v

# Run only the ML predictor tests
pytest tests/test_predictors.py -v
```

## 🤖 Training the Predictive Models

The core research contribution is a learned eviction policy. The training pipeline generates workload traces, extracts session features, and trains two models (Logistic Regression + Gradient Boosted Trees):

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

## 📊 Benchmarks & Realistic Workloads

The project utilizes a custom `WorkloadSimulator` that models user arrivals via **Poisson processes** and session lengths via heavy-tailed **Log-normal distributions** — accurately mirroring real-world LLM inference server loads.

Run the suite (cross-platform Python commands):
```bash
# Run a quick check (uses small model config, completes in seconds)
python -m benchmarks.run_benchmarks --suite quick

# Run the full rigorous research suite
python -m benchmarks.run_benchmarks --suite all
```

*(Charts and structured empirical results are saved to `benchmarks/results/`)*

### Running the synthetic experiment matrix

Regenerates every number in the paper's main results table (6 policies × 3 workloads × 10 seeds):

```bash
make reproduce
# or, step by step:
python src/kv_cache_tier/eviction/train_predictors.py   # only if models/*.pkl are missing
python benchmarks/experiment_runner.py --duration 0.25 --seeds 10
python benchmarks/generate_figures.py
```

### Running the real-trace evaluation (Azure LLM inference)

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

### Validating on a real model (TinyLlama-1.1B, CPU)

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

### GPU validation (Colab, free T4)

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

### Break-even analysis (no hardware needed)

**Why:** restoring a cache moves O(N) bytes while prefill costs O(N²) compute, so for
every (model, GPU, storage tier) there is a context length N* beyond which persistence
always wins on latency. This derives N* analytically and generates Figure 4:

```bash
python benchmarks/breakeven_analysis.py
```

### Building the paper

```bash
make arxiv          # compile + package LaTeX sources into arxiv_bundle.zip
# or manually:
cd paper/latex && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

Every number and figure in the paper regenerates from the committed result JSONs —
if a claim in the paper cannot be traced to `benchmarks/results/*.json`, that is a bug.

## 📁 Project Structure

```
kv-cache-tier-persistence/
├── src/kv_cache_tier/
│   ├── core/           # Tier orchestrator, cache blocks
│   ├── eviction/       # Eviction policies
│   │   ├── lru.py / ttl.py     # Baselines
│   │   ├── predictive.py       # V1: heuristic + learned P(resume)
│   │   ├── value_density.py    # V2: value per byte (+ admission control)
│   │   ├── space_time.py       # V3: value per byte-second (LHD-style)
│   │   ├── features.py         # SessionFeatures + shared FeatureExtractor
│   │   ├── predictors.py       # LogisticPredictor, GBTPredictor
│   │   └── train_predictors.py # Training pipeline
│   ├── serialization/  # Raw binary (CRC32), safetensors, LZ4, Zstd
│   ├── tiers/          # Hot, Warm, Cold tier implementations
│   └── utils/          # Clock (simulated time), cost model, metrics
├── benchmarks/         # Simulator, experiment runner, Azure loader,
│                       # break-even analysis, TinyLlama + GPU validation
├── models/             # Trained ML model artifacts (.pkl)
├── tests/              # 55 Pytest tests
├── paper/              # LaTeX sources + figures (paper.pdf at repo root)
└── docs/               # Architecture documents
```

---
*MIT License. See LICENSE file for details.*
