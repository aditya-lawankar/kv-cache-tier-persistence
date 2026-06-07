.PHONY: install test test-cov bench-quick bench-full clean lint

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
