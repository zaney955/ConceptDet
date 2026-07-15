.PHONY: env install run data train-sft train-grpo evaluate accept-cpu accept-pr accept-release accept-distributed verify-env test lint check

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

train-grpo:
	@test -n "$(CONFIG)" || (echo "Usage: make train-grpo CONFIG=/path/to/train-grpo.yaml" >&2; exit 2)
	PYTHONPATH=src .venv/bin/python -m conceptdet train grpo --config "$(CONFIG)" --resume "$(or $(RESUME),none)"

evaluate:
	@test -n "$(CONFIG)" || (echo "Usage: make evaluate CONFIG=/path/to/evaluate.yaml" >&2; exit 2)
	PYTHONPATH=src .venv/bin/python -m conceptdet evaluate --config "$(CONFIG)" --workers "$(or $(WORKERS),1)"

accept-cpu:
	@test -n "$(OUTPUT)" || (echo "Usage: make accept-cpu OUTPUT=/path/to/cpu_acceptance_report.json" >&2; exit 2)
	PYTHONPATH=src .venv/bin/python -m conceptdet accept cpu --output "$(OUTPUT)"

accept-pr accept-release accept-distributed:
	@test -n "$(EVIDENCE)" || (echo "Usage: make $@ EVIDENCE=/path/to/evidence OUTPUT=/path/to/report.json" >&2; exit 2)
	@test -n "$(OUTPUT)" || (echo "Usage: make $@ EVIDENCE=/path/to/evidence OUTPUT=/path/to/report.json" >&2; exit 2)
	PYTHONPATH=src .venv/bin/python -m conceptdet accept assemble --profile "$(@:accept-%=%)" --evidence-dir "$(EVIDENCE)" --output "$(OUTPUT)"

verify-env:
	.venv/bin/python scripts/check_environment.py --require-cuda

test:
	PYTHONPATH=src .venv/bin/python -m pytest

lint:
	.venv/bin/python -m ruff check .

check: lint test
