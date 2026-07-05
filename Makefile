.PHONY: install test test-cov bench-quick bench-full clean lint reproduce arxiv

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
	python benchmarks/experiment_runner.py --duration 0.25 --seeds 10
	python benchmarks/generate_figures.py

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
