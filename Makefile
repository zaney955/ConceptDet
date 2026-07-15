.PHONY: env install run verify-env test lint check prototype-reference-boxes

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

prototype-reference-boxes:
	@test -n "$(IMAGE)" || (echo "usage: make prototype-reference-boxes IMAGE=/path/to/reference.jpg"; exit 2)
	.venv/bin/python scripts/prototype_reference_box_rendering.py --image "$(IMAGE)"
