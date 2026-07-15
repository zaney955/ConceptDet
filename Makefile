.PHONY: env install run data train-sft verify-env test lint check

env:
	bash scripts/create_env.sh

install: env

run:
	@test -n "$(CONFIG)" || (echo "Usage: make run CONFIG=/path/to/detect.yaml" >&2; exit 2)
	bash scripts/run_inference.sh infer detect --config "$(CONFIG)"

data:
	@test -n "$(CONFIG)" || (echo "Usage: make data CONFIG=/path/to/data-voc.yaml" >&2; exit 2)
	PYTHONPATH=src .venv/bin/python -m conceptdet data voc --config "$(CONFIG)"

train-sft:
	@test -n "$(CONFIG)" || (echo "Usage: make train-sft CONFIG=/path/to/train-sft.yaml" >&2; exit 2)
	PYTHONPATH=src .venv/bin/python -m conceptdet train sft --config "$(CONFIG)" --resume "$(or $(RESUME),none)"

verify-env:
	.venv/bin/python scripts/check_environment.py --require-cuda

test:
	PYTHONPATH=src .venv/bin/python -m pytest

lint:
	.venv/bin/python -m ruff check .

check: lint test
