.PHONY: install test test-cov bench-quick bench-full clean lint reproduce reproduce-arch reproduce-azure reproduce-oracle reproduce-capacity reproduce-persona bench-compression arxiv

# Build the arXiv submission bundle (LaTeX sources + precompiled .bbl + figures).
# Compile the paper first so paper.bbl is current.
arxiv:
	cd paper/latex && pdflatex -interaction=nonstopmode paper.tex && bibtex paper && pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex
	cd paper/latex && rm -f ../../arxiv_bundle.zip && zip -r ../../arxiv_bundle.zip paper.tex references.bib paper.bbl figures/

# Regenerate every number and figure in the paper from scratch:
# retrain predictors, run the full multi-seed experiment matrix,
# and rebuild all figures from the resulting aggregates.
reproduce:
	python src/kv_cache_tier/eviction/train_predictors.py
	python benchmarks/experiment_runner.py --duration 0.25 --seeds 10 --arch tinyllama
	python benchmarks/generate_figures.py

# Multi-architecture matrix. Value and restore are priced from a real model's
# geometry (see src/kv_cache_tier/utils/hardware.py), and whether a hit is
# worth anything is architecture-dependent, so the policy ranking is too.
# Policies that ignore the cost model in their DECISIONS (lru, heuristic,
# logistic_v1) need only be simulated once and can be re-priced for other
# architectures from their logged hit histograms; the value-aware policies
# change behavior with the cost model and must genuinely be re-run.
reproduce-arch:
	python benchmarks/experiment_runner.py --duration 0.25 --seeds 10 --arch tinyllama \
		--output benchmarks/results/arch_tinyllama
	python benchmarks/experiment_runner.py --duration 0.25 --seeds 10 --arch llama70b \
		--policies value_density,value_density_ac,space_time \
		--output benchmarks/results/arch_llama70b
	python benchmarks/rescore_results.py --arch llama70b \
		--output-raw benchmarks/results/arch_llama70b/shard_rescored/experiment_results_v3_raw.json \
		benchmarks/results/arch_tinyllama/*/experiment_results_v3_raw.json
	python benchmarks/make_paper_tables.py

# Real-trace evaluation: replays ten 6-hour windows of the Azure LLM
# inference trace through all six policies. Downloads ~1.1 GB from the
# AzurePublicDataset release on first use.
reproduce-azure:
	python benchmarks/experiment_runner.py --azure

# Oracle ablation: rerun the constrained workloads with ground-truth
# policies (Belady, V1 with a perfect classifier, V3 with perfect inputs).
# Answers whether the learned policies' losses stem from prediction error
# or from the objective itself. Writes to a separate results dir; merge
# with: python benchmarks/merge_results.py --prefix oracle <raw files>
reproduce-oracle:
	python benchmarks/experiment_runner.py --duration 0.25 --seeds 10 \
		--workloads enterprise,power_user \
		--policies lru,belady,oracle_v1,oracle_v3 \
		--output benchmarks/results/oracle

# Persona strength sweep: how much do the conclusions depend on how predictable
# we made the simulated world? Retrains the predictor at each dispersion setting
# so the model is as good as that world allows, then re-runs the enterprise
# matrix. Each sigma gets its own predictor file so the runs can go in parallel.
reproduce-persona:
	for sg in 0.0 0.3 1.0; do \
		d=benchmarks/results/persona_sweep/sigma_$$sg; \
		mkdir -p $$d/models; \
		python src/kv_cache_tier/eviction/train_predictors.py \
			--output-dir $$d/models --persona-sigma $$sg; \
		python benchmarks/experiment_runner.py --duration 0.25 --seeds 5 \
			--workloads enterprise --persona-sigma $$sg \
			--predictor $$d/models/logistic_predictor.pkl \
			--policies lru,logistic_v1,value_density,space_time \
			--output $$d; \
	done

# Compression benchmark: LZ4/Zstd/zlib on REAL TinyLlama KV cache bytes
# (runs a real prefill; requires the TinyLlama weights, ~2.2 GB on first use).
bench-compression:
	python benchmarks/compression_benchmark.py

# Capacity regime sweep: reruns the enterprise workload at 125MB-2GB
# (10/30/60 tier split) for {lru, logistic_v1, space_time, belady,
# oracle_v1, oracle_v3}, then regenerates Figure 5. ~2h serially; each
# capacity point can run as its own parallel process.
reproduce-capacity:
	for cap in 125 250 500 1000 2000; do \
		python benchmarks/experiment_runner.py --duration 0.25 --seeds 5 \
			--workloads enterprise --capacity-mb $$cap \
			--policies lru,logistic_v1,space_time,belady,oracle_v1,oracle_v3 \
			--output benchmarks/results/capacity_sweep/cap_$$cap; \
	done
	python benchmarks/generate_capacity_figure.py

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src/kv_cache_tier --cov-report=html

bench-quick:
	python -m benchmarks.run_benchmarks --suite quick

bench-full:
	python -m benchmarks.run_benchmarks --suite all

clean:
	rm -rf data/ benchmarks/results/ .pytest_cache htmlcov src/**/*.pyc __pycache__

lint:
	python -m py_compile src/kv_cache_tier/*.py
