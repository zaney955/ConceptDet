.PHONY: env install run verify-env test lint check prototype-token-budget prototype-token-profile

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

# PROTOTYPE — Qwen3-VL dynamic-resolution token explorer.
prototype-token-budget:
	PYTHONPATH=src /home/yzy/TogeeWork/Projects/ConceptDet-R1/.venv/bin/python scripts/prototype_token_budget.py \
		--reference /home/yzy/TogeeWork/Projects/ConceptDet-R1/inputs/ref/GX/17286d22__3e852fa4-5dcc-40c6-a00e-addb89753b63.jpg \
		--reference-box 1165,2911,1354,3230 \
		--reference-box 4064,3087,4208,3375 \
		--target /home/yzy/TogeeWork/Projects/ConceptDet-R1/inputs/non_ref/GX2/201eb088__9aaa4be4-9dbf-4832-a089-f64278a587b6.jpg

# PROTOTYPE — one optimizer step; override TOKENS, GPU, or ATTENTION as needed.
TOKENS ?= 640
GPU ?= 0
ATTENTION ?= flash_attention_2
prototype-token-profile:
	PYTHONPATH=src CUDA_VISIBLE_DEVICES=$(GPU) /home/yzy/TogeeWork/Projects/ConceptDet-R1/.venv/bin/python scripts/profile_token_budget.py \
		--tokens-per-image $(TOKENS) --gpu 0 --attention $(ATTENTION) \
		--reference /home/yzy/TogeeWork/Projects/ConceptDet-R1/inputs/ref/GX/17286d22__3e852fa4-5dcc-40c6-a00e-addb89753b63.jpg \
		--reference-box 1165,2911,1354,3230 \
		--reference-box 4064,3087,4208,3375 \
		--target /home/yzy/TogeeWork/Projects/ConceptDet-R1/inputs/non_ref/GX2/201eb088__9aaa4be4-9dbf-4832-a089-f64278a587b6.jpg
