.PHONY: env install run verify-env test lint check

env:
	bash scripts/create_env.sh

install: env

run:
	@test -n "$(CONFIG)" || (echo "Usage: make run CONFIG=/path/to/detect.yaml" >&2; exit 2)
	bash scripts/run_inference.sh infer detect --config "$(CONFIG)"

verify-env:
	.venv/bin/python scripts/check_environment.py --require-cuda

test:
	PYTHONPATH=src .venv/bin/python -m pytest

lint:
	.venv/bin/python -m ruff check .

check: lint test
