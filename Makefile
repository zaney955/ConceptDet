.PHONY: env install run verify-env test lint check prototype-grpo-smoke

env:
	bash scripts/create_env.sh

install: env

run:
	bash scripts/run_inference.sh

verify-env:
	.venv/bin/python scripts/check_environment.py --require-cuda

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/python -m ruff check .

check: lint test

prototype-grpo-smoke:
	@test -n "$(REFERENCE)" || (echo "usage: make prototype-grpo-smoke REFERENCE=/path/to/reference.jpg INIT_ADAPTER=/path/to/adapter"; exit 2)
	@test -n "$(INIT_ADAPTER)" || (echo "usage: make prototype-grpo-smoke REFERENCE=/path/to/reference.jpg INIT_ADAPTER=/path/to/adapter"; exit 2)
	PYTHONPATH=src CUDA_VISIBLE_DEVICES="$${GPU:-6}" .venv/bin/python scripts/prototype_grpo_smoke.py --run --reference "$(REFERENCE)" --init-adapter "$(INIT_ADAPTER)" --visible-gpu "$${GPU:-6}"
