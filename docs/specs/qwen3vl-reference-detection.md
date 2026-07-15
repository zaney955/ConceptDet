# Qwen3-VL Reference-Guided Detection Engineering Specification

Status: implementation-ready  
Decision source: [Wayfinder #1](https://github.com/zaney955/ConceptDet/issues/1),
with authoritative child decisions #2–#13.

## 1. Goal and non-goals

ConceptDet identifies every Target Instance in one Target Image that belongs to
the Visual Concept demonstrated by one or more Reference Boxes in one Reference
Image. Version 1 replaces the ConceptSeg/Qwen2.5 runtime with
`Qwen/Qwen3-VL-8B-Instruct` and establishes the same model-visible contract for
inference, bbox SFT, evaluation, and optional bbox GRPO.

Version 1 does not preserve old checkpoints, prompts, fixed 600×600 inputs,
SAM/mask losses, learnable queries, connectors, projections, custom Qwen model
forks, or custom trainers. It does not claim that a smoke-trained adapter is a
quality model.

## 2. Architecture and seams

```text
CLI Application
  ├── Config Module
  ├── Detection Application
  │     ├── Detection Protocol Module
  │     └── Qwen Adapter seam
  │           ├── Fake Adapter
  │           └── Qwen3-VL Adapter
  ├── Manifest/Data Module
  ├── Artifact Module
  ├── SFT Stage Module
  ├── Evaluation Module
  └── GRPO Stage Module
```

The external inference seam is a single `DetectionApplication.detect(request)`
interface. It accepts source images, Reference Boxes, and query semantics; it
returns a complete Detection Set plus prepared-input provenance. The Qwen
Adapter owns model-visible preprocessing and generation. The Detection Protocol
owns serialization and coordinate conversion. Neither the CLI nor tests call a
Transformers processor directly.

The Qwen Adapter seam is real: production uses Qwen3-VL while application tests
use the deterministic Fake Adapter. Trainer seams remain private to their stage
modules.

## 3. Public CLI

The stable command tree is:

```text
conceptdet infer detect --config FILE
conceptdet infer batch --config FILE
conceptdet config validate --config FILE
conceptdet config render --config FILE [--output FILE]
conceptdet artifact init --config FILE
conceptdet artifact inspect ARTIFACT [--json]
conceptdet data voc --config FILE
conceptdet train sft --config FILE [--resume none|auto|PATH]

conceptdet train grpo --config FILE [--resume none|auto|PATH]
conceptdet predict dataset --config FILE
conceptdet evaluate --config FILE
```

YAML is the only semantic configuration source. CLI overrides are limited to
operational output, device, offline, dry-run, resume, and log level. There is no
Python configuration module, generic `--set`, environment interpolation, or
upstream trainer-kwargs escape hatch.

Inference stdout contains only the strict JSON Detection Set. Diagnostics and
paths use stderr. Configuration, compatibility, or input errors exit 2;
execution errors exit 1; success exits 0.

## 4. Strict configuration

The Config Module parses safe YAML into a closed discriminated union. It rejects
duplicate, unknown, missing, mistyped, merge-key, and command-incompatible
fields before model allocation. Relative paths resolve against the YAML file.
Every accepted configuration is expanded into a canonical resolved payload and
SHA-256 hash.

Implemented v1 kinds:

- `infer.detect`
- `infer.batch`
- `artifact.init`
- `data.voc`
- `train.sft`
- `train.grpo`
- `predict.dataset`
- `evaluate`

The Data Module treats VOC XML as the authoritative record set, validates image
size and bbox semantics, retains all instances of the selected Visual Concept,
groups exact duplicates and related capture sequences before splitting, and
publishes byte-deterministic JSONL plus an audit and Dataset Artifact
fingerprint. Empty XML annotations are legal negative Target Images; orphan
images are audited but do not silently become training records.

The SFT Stage Module consumes only a validated Dataset Artifact. It uses
assistant-only labels, no packing or truncation, deterministic positive/negative
ordering, atomic checkpoints, and explicit `none`, `auto`, or exact-path resume.
Publication records dataset/config fingerprints and lifecycle provenance in the
Adapter Artifact.

The Prediction Module loads an immutable Adapter Artifact on one or more
Accelerate ranks, shards a compiled dataset split by rank, and atomically
publishes one ID-sorted raw-completion JSONL with exact coverage. It preserves
malformed completions for strict evaluation and reduces only the completion
budget when needed to honor the 1,536-token sequence contract.

The Evaluation Module consumes a validated Dataset Artifact, immutable Adapter
Artifact, and complete prediction JSONL containing `id` and the saved
`raw_completion`. It never loads the model. It exact-matches Detection Sets,
atomically publishes a fingerprinted `report.json` plus sorted per-record JSONL,
and fails on missing, duplicate, or extra prediction identities. Prediction row
order and the operational worker count cannot change report bytes.

The GRPO Stage Module initializes the exact LoRA weights from an immutable SFT
Artifact without inheriting optimizer state. It lazily emits raw conversations,
ordered images, normalized truth, and record metadata to stock TRL 1.5
`GRPOTrainer`. Native generation is fixed to `beta=0`, no reference model, two
generations, and 192 completion tokens. Reward is 10% strict format plus 90%
soft Set-F1 from the Evaluation Module. Publication requires a nonzero-advantage
group, changed adapter parameters, exact SFT lineage, strict save/reload
generation, and the 44 GiB lifecycle gate. Complete checkpoints support
fail-closed `none`, `auto`, and explicit-path resume modes.

Example detection config:

```yaml
schema_version: 1
kind: infer.detect
artifact: artifacts/sft-final
request:
  reference_image: images/reference.jpg
  reference_boxes:
    - [1165, 2911, 1354, 3230]
  target_image: images/target.jpg
  query: the same bolt as the boxed example
output:
  image: outputs/result.png
  json: outputs/result.json
  layout: annotated
runtime:
  device: cuda:0
  dtype: bfloat16
  attention: flash_attention_2
  max_new_tokens: 192
  local_files_only: true
```

Batch config replaces `request` with `manifest` and `output_dir`. Each JSONL row
contains `id`, `reference_image`, `reference_boxes`, `target_image`, and `query`.

Known obsolete keys (`model_path`, `input_size`, `max_pixels`, `packing`,
`mask*`, `sam*`, `learnable_query`, `connector`, `projection`,
`resume_from_checkpoint`, `use_vllm`, Qwen2.5/ConceptSeg model fields) receive a
clean-break diagnostic rather than a generic unknown-field error.

## 5. Detection Protocol

Model output is exactly a bare JSON array:

```json
[{"bbox_2d":[x1,y1,x2,y2]}]
```

An optional string `label` is accepted but ignored by matching. Other keys,
Markdown fences, prose, booleans, floats, out-of-range coordinates, and
degenerate boxes are invalid. `[]` is the exact no-match result. Array order has
no meaning.

Coordinates are integer normalized 0–1000 XYXY. Original truth uses half-open
pixel XYXY. Encoding uses nonnegative half-up rounding. Decoding scales
continuous endpoints to the Target Image, clamps them to image bounds, and
rejects degeneracy.

Evaluation is confidence-free. The primary future metric is positive macro
mean Set-F1 at IoU thresholds 0.50:0.05:0.95, not COCO mAP.

## 6. Qwen3-VL Adapter contract

The production Adapter loads:

- model and processor: `Qwen/Qwen3-VL-8B-Instruct`;
- immutable revision:
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`;
- one compatible PEFT Adapter Artifact;
- BF16 by default, FlashAttention 2 on supported CUDA devices;
- greedy generation for inference.

Input roles are ordered `[Reference Image, Target Image]`. Both images are EXIF
transposed, converted to RGB, and independently smart-resized with factor 32,
64–640 merged visual tokens, and 65,536–655,360 pixels. The Adapter draws
Reference Boxes after resize using:

- inner color `#ff2020`;
- inner width `clamp(round(shortest_side / 256), 2, 4)`;
- outward white halo `#ffffff`, width 2.

The Target Image remains undecorated. The processor receives already prepared
images with resize disabled. The chat template uses vision IDs, Picture 1 for
Reference and Picture 2 for Target, and explicitly forbids Markdown fences.
Total prompt/response length is at most 1,536; inference and GRPO completion
length is at most 192.

## 7. Adapter Artifact

An Artifact directory is immutable after publication and contains:

```text
adapter_model.safetensors
adapter_config.json
conceptdet_contract.json
training_summary.json
```

`conceptdet_contract.json` records exact base/processor identities and
revisions, ordered image roles, resize/rendering/prompt/output schemas, sequence
limits, and LoRA topology. `contract_fingerprint` is SHA-256 over canonical JSON
excluding the fingerprint field.

`training_summary.json` stores an `artifact_fingerprint` over the contract
fingerprint, adapter/config file hashes, stage, parent fingerprint, and init
provenance. This distinguishes trained adapters that implement the same
contract and makes SFT→GRPO lineage tamper-evident.

The default LoRA topology is rank 16, alpha 32, dropout 0.05, bias none,
text-all plus multimodal mergers, 260 target modules, target-list hash
`fdff350e33d483666eb85ead6d1dc062df8739f2f6d78dc20663bf49fa755402`,
and 44,793,856 trainable parameters.

Artifact validation completes before the 8B model is allocated. Missing or
unknown contracts, fingerprint mismatch, wrong base/revision, modified adapter
configuration, or old ConceptSeg/Qwen2.5 checkpoints fail with no force bypass.

`artifact init` wraps an existing PEFT adapter directory in a new immutable
ConceptDet Artifact. It copies weights/config, validates topology metadata, and
writes the contract and provenance atomically. Training stages later use the
same Artifact Module to publish their output.

## 8. Tracer bullet

The first implementation slice proves the public seams without the 8B model:

```text
strict YAML
  → DetectionRequest
  → DetectionApplication(FakeAdapter)
  → strict Detection Protocol
  → pixel Detection Set + output JSON/image
  → inspectable Artifact
```

Application tests inject a deterministic Fake Adapter through the same seam as
Qwen3-VL. They assert only public outcomes, never adapter internals.

The second slice replaces the Fake Adapter with Qwen3-VL and executes the same
Detection Application and CLI. Old fixed-size/prompt/parser/model paths are
deleted once replacement tests pass; they are not retained as compatibility
layers.

## 9. Dependency profiles

- default: inference, configuration, data/protocol, evaluation primitives;
- `train`: PEFT and bbox SFT dependencies, without TRL;
- `grpo`: training plus pinned TRL;
- `distributed`: optional DeepSpeed; Accelerate remains core;
- `flash-attn`: separately installable hardware-specific acceleration;
- `dev`: tests and lint.

Missing profiles fail during validation with an exact install command before
model allocation.

## 10. Verification and resource gates

Every PR runs CPU contract, Fake Adapter application, and two-process fake
tests. Release/manual gates use the real pinned 8B model:

- SFT lifecycle and SFT→native-GRPO lifecycle must each remain at or below
  44.0 GiB process-local CUDA peak reserved memory;
- real two-GPU DDP must pass before multi-GPU support is advertised;
- ZeRO, multi-node, 4–8 GPU, and full fine-tuning are optional certifications.

The inference slice must pass CPU tests, CLI Fake Adapter tests, Artifact
validation, and a real base+adapter positive/negative strict-generation smoke.
The SFT slice must additionally complete train/save/release/reload/strict-generate
with exactly 44,793,856 trainable parameters and no more than 44.0 GiB peak
reserved memory.

## 11. Implementation order

1. Config, Detection Protocol, Artifact, Fake Adapter tracer bullet.
2. Deterministic bbox manifest and offline VOC conversion.
3. Real Qwen3-VL Adapter and inference replacement.
4. Bbox SFT and one-GPU lifecycle.
5. Confidence-free evaluation and frozen reports.
6. SFT→native-GRPO.
7. Resume, two-process tests, and real DDP certification.

Each slice is independently reviewable and preserves the public interfaces
defined here.
