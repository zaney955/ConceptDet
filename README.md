# ConceptDet-R1

ConceptDet-R1 is a bbox-only, reference-guided detector built on
`Qwen/Qwen3-VL-8B-Instruct`. Give it a Reference Image with one or more XYXY
Reference Boxes, a Target Image, and a short concept description. It returns
every matching Target Instance as a strict JSON Detection Set.

The runtime is a clean break from ConceptSeg: there is no SAM, mask conversion,
learnable query, connector, projection, fixed 600×600 input, or Qwen2.5
compatibility path. The model-visible preprocessing and output contract are
shared by inference and future bbox SFT/GRPO stages.

## Contract

Model output is a bare array of normalized integer XYXY boxes:

```json
[{"bbox_2d":[125,240,510,780]}]
```

`[]` means no match. Confidence scores, prose, Markdown fences, floats, unknown
keys, and degenerate boxes are rejected. Coordinates use the Target Image on a
0–1000 grid and are decoded back to its original pixel dimensions.

The production adapter pins Qwen3-VL to revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, accepts two independently resized
images with at most 640 visual tokens each, and loads a validated rank-16 PEFT
LoRA Artifact. See the full [engineering specification](docs/specs/qwen3vl-reference-detection.md).

## Environment

Python 3.13, PyTorch 2.13, Transformers 5.13, PEFT 0.19, and FlashAttention 2
are pinned in `requirements/`.

```bash
bash scripts/create_env.sh
.venv/bin/python scripts/check_environment.py --require-cuda
```

The environment is local to this repository. Model files remain in the normal
Hugging Face cache; adapters and generated outputs are ignored by Git.

## Wrap a PEFT adapter as an Artifact

ConceptDet validates the complete inference contract before allocating the 8B
model. An existing compatible PEFT adapter must first be wrapped atomically:

```bash
cp examples/artifact-init.yaml /tmp/artifact-init.yaml
# Edit source_adapter and output_dir.
.venv/bin/python -m conceptdet artifact init --config /tmp/artifact-init.yaml
.venv/bin/python -m conceptdet artifact inspect /path/to/artifact --json
```

An Artifact contains `adapter_model.safetensors`, `adapter_config.json`,
`conceptdet_contract.json`, and `training_summary.json`. It is treated as
immutable after publication.

## Single-image inference

Copy [examples/detect.yaml](examples/detect.yaml), set the Artifact and image
paths, then validate without loading Qwen:

```bash
.venv/bin/python -m conceptdet config validate --config /tmp/detect.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m conceptdet infer detect \
  --config /tmp/detect.yaml
```

Successful stdout contains only the strict Detection Set. Diagnostics and
output paths go to stderr. The configured JSON file preserves the raw
completion, normalized boxes, original-pixel boxes, image grids, prompt-token
count, and canonical configuration hash. `layout` may be `annotated` or
`triptych`.

The shell wrapper forwards the same hierarchical CLI:

```bash
bash scripts/run_inference.sh infer detect --config /tmp/detect.yaml
make run CONFIG=/tmp/detect.yaml
```

## Batch inference

[examples/batch.yaml](examples/batch.yaml) points to a JSONL manifest. Each row
has exactly these fields:

```json
{"id":"sample-001","reference_image":"reference.jpg","reference_boxes":[[100,120,220,280]],"target_image":"target.jpg","query":"the same component as the boxed example"}
```

Image paths in a manifest are relative to that manifest. Run:

```bash
.venv/bin/python -m conceptdet infer batch --config /tmp/batch.yaml
```

The output directory receives one PNG and JSON per record plus `results.jsonl`.
Existing images are skipped unless `overwrite: true`.

## Development

```bash
make check
# Equivalent:
.venv/bin/python -m ruff check .
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The application depends on the small `DetectionAdapter` interface. CPU tests
use its deterministic Fake Adapter; production uses `Qwen3VLAdapter` through
the same path. Current implementation tickets are tracked in
[GitHub epic #14](https://github.com/zaney955/ConceptDet/issues/14).
