# ConceptDet-R1

ConceptDet-R1 is a bbox-only, reference-guided detector built on
`Qwen/Qwen3-VL-8B-Instruct`. Give it a Reference Image with one or more XYXY
Reference Boxes, a Target Image, and a short concept description. It returns
every matching Target Instance as a strict JSON Detection Set.

The runtime is a clean break from ConceptSeg: there is no SAM, mask conversion,
learnable query, connector, projection, fixed 600×600 input, or Qwen2.5
compatibility path. The model-visible preprocessing and output contract are
shared by deterministic VOC conversion, bbox SFT, inference, and the future
GRPO stage.

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

## Compile VOC bbox data

Copy [examples/data-voc.yaml](examples/data-voc.yaml) and point each source at
its image and XML directories. `source_box_semantics` must explicitly describe
the XML coordinates; use `xyxy_half_open` for zero-based boundary coordinates
or `voc_inclusive` for standard one-based VOC coordinates.

```bash
.venv/bin/python -m conceptdet config validate --config /tmp/data-voc.yaml
.venv/bin/python -m conceptdet data voc --config /tmp/data-voc.yaml
```

The immutable compiled dataset contains `train.jsonl`, `validation.jsonl`,
`test.jsonl`, `audit.json`, and `dataset.json`. Conversion retains every Target
Instance for the selected Visual Concept, creates deterministic empty Detection
Sets, groups duplicate/related images before splitting, chooses Reference Images
only inside the Target Image split, and hashes every output. XML files are the
authoritative record set; images without XML are listed as orphans in the audit.
An XML without an image, a mismatched size, malformed XML, or an invalid bbox
fails with its source XML and object index.

## Bbox-native SFT

Copy [examples/train-sft.yaml](examples/train-sft.yaml), set the compiled
dataset, work, and Artifact paths, then run:

```bash
.venv/bin/python -m conceptdet config validate --config /tmp/train-sft.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m conceptdet train sft \
  --config /tmp/train-sft.yaml --resume none
```

SFT uses the official Qwen3-VL Transformers path, rank-16 text-all plus
multimodal-merger LoRA, frozen base/vision parameters, assistant-only labels,
no packing, and no truncation. Checkpoints atomically preserve adapter,
optimizer, scheduler, dataset/config fingerprints, and the exact schedule
position. Resume is explicit:

```bash
# Latest valid checkpoint in work_dir
.venv/bin/python -m conceptdet train sft --config /tmp/train-sft.yaml --resume auto

# Specific checkpoint
.venv/bin/python -m conceptdet train sft --config /tmp/train-sft.yaml \
  --resume /path/to/checkpoint-00000100
```

On completion, the stage saves and releases the training model, publishes an
immutable Artifact, reloads it, runs strict positive/negative generation, and
enforces the 44 GiB peak-reserved gate. A short lifecycle run proves mechanics;
it is not a quality-trained model. Use `max_steps: null` for the configured full
epoch schedule.

## Native bbox GRPO

GRPO starts from an immutable SFT Artifact, never from an SFT run checkpoint,
and therefore inherits adapter weights without optimizer, scheduler, or RNG
state. Install the optional profile and copy
[examples/train-grpo.yaml](examples/train-grpo.yaml):

```bash
.venv/bin/python -m pip install -e '.[grpo]'
.venv/bin/python -m conceptdet config validate --config /tmp/train-grpo.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m conceptdet train grpo \
  --config /tmp/train-grpo.yaml --resume none
```

The stage uses the stock TRL 1.5 `GRPOTrainer` and native Transformers
generation: `beta=0`, no reference model, no vLLM, exactly two generations,
and a 192-token completion budget. The Dataset Module lazily emits a raw
conversation, ordered `[Reference Image, Target Image]`, and normalized truth;
the same Qwen processor owns tokenization and forward recomputation. Every
record must satisfy prompt + 192 ≤ 1,536; truncation remains forbidden.

Reward is exactly 10% strict Detection Set format plus 90% soft Set-F1 from the
Evaluation Module. Invalid JSON receives zero; duplicate boxes, misses, and
false positives reduce soft Set-F1; correct empty/empty receives 1.0. A run is
published only if the callback sees positive and negative groups, at least one
group has nonzero advantage, LoRA parameters change, save/release/reload strict
generation succeeds, parent lineage is exact, and peak reserved memory remains
at most 44 GiB. `max_steps: null` selects the configured full epoch. GRPO resume
other than `none` remains fail-closed until #21 completes resume certification.

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

## Confidence-free evaluation

[examples/evaluate.yaml](examples/evaluate.yaml) evaluates saved raw model
completions against one split of an immutable Dataset Artifact and records the
exact Adapter Artifact lineage. Prediction JSONL contains exactly one row per
dataset record; line order has no meaning:

```json
{"id":"dataset-record-id","raw_completion":"[{\"bbox_2d\":[125,240,510,780]}]"}
```

Malformed completions remain legal evaluation inputs and reduce the
`strict_valid_rate`; missing, duplicate, or extra record IDs fail closed. Run:

```bash
.venv/bin/python -m conceptdet config validate --config /tmp/evaluate.yaml
.venv/bin/python -m conceptdet evaluate --config /tmp/evaluate.yaml --workers 4
```

The atomically published directory contains `report.json` and sorted
`records.jsonl`, protected by an evaluation fingerprint. The primary metric is
positive macro mean Set-F1 across IoU 0.50:0.05:0.95. The report also includes
all-example Set-F1, micro Precision/Recall/F1@0.5, positive soft Set-F1,
strict-valid rate, correct-empty rate, negative false-positive boxes per image,
and count, relative-area, Visual Concept, and Reference Image swap slices.
Matching is exact, one-to-one, and order-independent. ConceptDet does not report
equal-score pseudo-mAP because the v1 Detection Set has no confidence scores.

## Development

```bash
make check
# Equivalent:
.venv/bin/python -m ruff check .
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The application depends on the small `DetectionAdapter` interface. CPU tests
use its deterministic Fake Adapter; production uses `Qwen3VLAdapter` through
the same path. Dataset compilation and SFT similarly expose one small interface
each while hiding pairing, grouping, token masking, checkpoint, and publication
details. Current implementation tickets are tracked in
[GitHub epic #14](https://github.com/zaney955/ConceptDet/issues/14).
