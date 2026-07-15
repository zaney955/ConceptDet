.PHONY: env install run verify-env test lint check prototype-lora-smoke

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

prototype-lora-smoke:
	@test -n "$(REFERENCE)" || (echo "usage: make prototype-lora-smoke REFERENCE=/path/to/reference.jpg"; exit 2)
	CUDA_VISIBLE_DEVICES="$${GPU:-6}" .venv/bin/python scripts/prototype_lora_smoke.py --reference "$(REFERENCE)" --visible-gpu "$${GPU:-6}"
