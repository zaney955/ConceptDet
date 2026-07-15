.PHONY: env install run verify-env test lint check prototype-reward

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

# PROTOTYPE — throwaway terminal explorer for the Detection Set reward decision.
prototype-reward:
	PYTHONPATH=src python3 scripts/prototype_detection_set_reward.py
