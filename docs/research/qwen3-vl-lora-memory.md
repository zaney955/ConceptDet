# Qwen3-VL LoRA topology and 48 GB memory profile

Status: decision for [Issue #3](https://github.com/zaney955/ConceptDet/issues/3)

Scope: `Qwen/Qwen3-VL-8B-Instruct`, bbox-only SFT followed by optional bbox GRPO, no SAM

Research date: 2026-07-15

## Decision

Use one LoRA adapter topology for both SFT and GRPO:

- adapt all seven linear projections in every text decoder block (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`);
- adapt `linear_fc1` and `linear_fc2` in the main vision merger and all three DeepStack mergers;
- freeze the patch embedder, positional embedding, all 27 vision blocks, token embeddings, norms, and `lm_head`;
- default to rank 16, alpha 32, dropout 0.05, no bias, BF16 base weights, PEFT's default FP32 adapter autocast, gradient checkpointing, `use_cache=False`, and FlashAttention 2 when available;
- save an unmerged PEFT adapter and a small run manifest. Pin both the model and processor to the same base revision;
- for the one-GPU GRPO smoke path, use native Transformers generation, `beta=0`, two generations, and no colocated vLLM.

Call this topology **text-all + multimodal-mergers**. It exposes 260 linear modules and has exactly **44,793,856 trainable LoRA parameters at rank 16** under the pinned architecture.

This is the initial production candidate, not a claim that it is empirically optimal. The implementation must also make **text-attention + multimodal-mergers** available as a lower-capacity ablation. A later experiment may add LoRA to only the final three vision blocks, but full-vision LoRA is outside the 48 GB smoke acceptance path until measured.

## What the official sources establish

### Model boundaries

The pinned model config declares a dense 36-layer text decoder with hidden size 4096, intermediate size 12288, 32 query heads, 8 key/value heads, and BF16 weights. Its vision config declares 27 blocks, hidden size 1152, intermediate size 4304, output size 4096, spatial merge size 2, and DeepStack outputs at vision layers 8, 16, and 24. The architecture is `Qwen3VLForConditionalGeneration` and the text embedding and LM head are not tied ([official config at revision `e0a319f`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/e0a319f4d147b3916275a053b0583ca82f351e90/config.json)).

The stable Transformers v4.57.1 implementation shows these concrete seams:

- each vision block has a fused `qkv`, attention `proj`, and MLP `linear_fc1` / `linear_fc2`; the patch embedder is a `Conv3d` named `proj` ([official source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L46-L75), [vision attention source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L168-L247));
- a merger contains `linear_fc1: 4608 -> 4608` and `linear_fc2: 4608 -> 4096` ([official source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L93-L105));
- the vision model has one final merger plus one merger for each of the three DeepStack indexes ([official source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L564-L604));
- each text block has `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` ([official source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L385-L472));
- the vision model returns both its final merged features and three DeepStack features, which the wrapper passes into the text model ([official source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L720-L751), [wrapper source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1034-L1080)).

This means “the projector” is not one layer. Freezing all four mergers would freeze every learned 1152/4608-to-4096 multimodal interface, including DeepStack. Conversely, targeting `linear_fc1`, `linear_fc2`, or `proj` by suffix alone would accidentally match vision-block MLPs or attention/patch modules.

Qwen's model card recommends FlashAttention 2 for acceleration and memory savings, especially for multi-image inputs, and demonstrates BF16 loading and `AutoProcessor`/`apply_chat_template` ([official model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct#quickstart)). The processor config defaults to a very large maximum pixel budget, so training must override it rather than inherit it silently ([official processor config](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/e0a319f4d147b3916275a053b0583ca82f351e90/preprocessor_config.json)).

### PEFT and TRL behavior

PEFT accepts explicit module names or a regular expression in `target_modules`; list entries match exact names or module-name suffixes. `"all-linear"` selects all linear/Conv1D modules except a pretrained model's output layer ([official `LoraConfig` reference](https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig)). Because Qwen3-VL reuses short names across vision blocks and mergers, `"all-linear"` and bare `linear_fc*` suffixes are unsuitable for this decision.

PEFT defaults LoRA to a no-op initialization, and documents rank, alpha, bias, dropout, and module targeting as the main adapter controls ([official LoRA guide](https://huggingface.co/docs/peft/main/developer_guides/lora)). PEFT also autocasts FP16/BF16 adapter weights to FP32 by default when loading a `PeftModel`, for stable training ([official `PeftModel.from_pretrained` source documentation](https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py#L2977-L3022)).

TRL supports passing either a `peft_config` or an already wrapped PEFT model. When the model is wrapped first, no second `peft_config` should be passed to the trainer ([official TRL PEFT integration](https://huggingface.co/docs/trl/peft_integration#applying-peft-to-the-model-directly)). TRL saves adapter-only checkpoints through `save_model` / `save_pretrained`, and PEFT defaults to safe serialization ([official TRL saving guide](https://huggingface.co/docs/trl/peft_integration#saving-and-loading-peft-models), [official PEFT source](https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py#L2602-L2628)).

`SFTTrainer` supports VLM preprocessing on the fly and enables gradient checkpointing by default, but its default maximum sequence length is 1024 and longer sequences are truncated; visual training therefore needs an explicit token/pixel budget rather than relying on defaults ([official SFT reference](https://huggingface.co/docs/trl/sft_trainer#trl.SFTConfig)).

`GRPOTrainer` accepts a `PeftModel`, VLM image lists, and a PEFT config. Its official VLM example currently lists Qwen2/2.5-VL as tested, not Qwen3-VL, so Qwen3-VL compatibility remains a prototype gate rather than a sourced guarantee ([official GRPO VLM section](https://huggingface.co/docs/trl/grpo_trainer#vision-language-model-vlm-training)). Its defaults include eight generations and a 256-token completion; the effective batch must be divisible by `num_generations` ([official GRPO config](https://huggingface.co/docs/trl/grpo_trainer#trl.GRPOConfig)). `beta=0` avoids loading a reference model and reduces memory ([official GRPO config](https://huggingface.co/docs/trl/grpo_trainer#parameters-that-control-the-training)).

## Why this topology

Reference-guided detection needs two kinds of adaptation:

1. translate frozen visual features from both images, including DeepStack features, into a task-specific multimodal representation;
2. compare the reference concept with target-image instances and serialize an unordered detection set.

The merger adapters address the first seam. Adapting all text attention and MLP projections gives the decoder capacity for cross-image comparison, multi-instance enumeration, empty-set rejection, and strict structured output. Freezing vision blocks preserves Qwen3-VL's pretrained perception while ensuring the heavy 27-block vision forward does not require a full backward graph. This is an architectural inference from the official module graph, not an official Qwen training prescription.

Do not train `lm_head`, token embeddings, or norms in the default path. The bbox schema uses existing tokens, so vocabulary expansion is unnecessary, and saving a full untied 151,936 by 4,096 LM head would defeat adapter-only checkpoints.

### Exact target discovery

Build the target list from `model.named_modules()` and pass the resulting complete names to PEFT. Accept only:

```text
model.language_model.layers.<0..35>.self_attn.{q_proj,k_proj,v_proj,o_proj}
model.language_model.layers.<0..35>.mlp.{gate_proj,up_proj,down_proj}
model.visual.merger.{linear_fc1,linear_fc2}
model.visual.deepstack_merger_list.<0..2>.{linear_fc1,linear_fc2}
```

Then assert all of the following before training:

- 260 target modules were found;
- every trainable parameter name contains the active LoRA adapter name;
- no trainable name contains `visual.blocks`, `patch_embed`, `embed_tokens`, or `lm_head`;
- `print_trainable_parameters()` reports 44,793,856 parameters for rank 16;
- the discovered target list and a hash of it are stored in the run manifest.

These assertions protect against both suffix collisions and future Transformers renames.

## Parameter and checkpoint budget

For a linear map `in -> out`, LoRA adds `r * (in + out)` parameters. Under the pinned config:

- text attention contributes `36 * r * (8192 + 5120 + 5120 + 8192) = 958,464r`;
- text MLP contributes `36 * r * (16,384 * 3) = 1,769,472r`;
- four mergers contribute `4 * r * (9,216 + 8,704) = 71,680r`;
- total is **2,799,616r**.

| Rank | Trainable parameters | FP32 adapter file upper estimate | FP32 parameter + gradient + two Adam moments |
|---:|---:|---:|---:|
| 8 | 22,396,928 | 0.083 GiB | 0.334 GiB |
| 16 | 44,793,856 | 0.167 GiB | 0.667 GiB |
| 32 | 89,587,712 | 0.334 GiB | 1.335 GiB |

The last column is a transparent 16-byte-per-trainable-parameter estimate: FP32 parameter, FP32 gradient, and two FP32 Adam moments. Actual fused-optimizer allocation must be measured. It excludes transient workspaces and allocator fragmentation.

The official checkpoint index reports **17,534,247,392 bytes**, or **16.33 GiB**, of base checkpoint tensors ([official checkpoint index](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/e0a319f4d147b3916275a053b0583ca82f351e90/model.safetensors.index.json)). Thus rank-16 static tensors are approximately 17.0 GiB before activations, logits, CUDA kernels, and allocator reserve. A naive single-GPU full fine-tune is not a 48 GB target: base gradients and two FP32 Adam moments alone would exceed the remaining memory before activations.

## One-48-GB smoke profile

### Required SFT profile

```yaml
dtype: bfloat16
attention: flash_attention_2  # SDPA is a supported but separately profiled fallback
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
gradient_checkpointing: true
use_cache: false
images_per_sample: 2       # one reference, one target
max_pixels_per_image: 262144  # 512 * 512 cap, not resize-to-square
max_total_sequence_tokens: 1024
packing: false
```

The smoke fixture must include one positive multi-instance example and one negative/empty-set example. Run at least one optimizer step, save the adapter, reload it over the pinned base, and generate a parseable detection set.

### Required GRPO profile

```yaml
base_adapter: <SFT adapter, loaded trainable>
dtype: bfloat16
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
num_generations: 2
max_completion_length: 192
beta: 0.0
use_vllm: false
gradient_checkpointing: true
use_cache: false
images_per_sample: 2
max_pixels_per_image: 262144
```

Two generations and accumulation 2 satisfy TRL's divisibility rule on one process. `beta=0` deliberately avoids a second reference model. Native generation avoids an additional colocated inference engine. A later multi-GPU configuration may use a dedicated vLLM server; TRL documents server mode as intended for separate inference GPUs and warns about GPU-memory allocation ([official vLLM modes](https://huggingface.co/docs/trl/grpo_trainer#option-2-server-mode)).

### Estimated envelope, not a guarantee

| Component | SFT estimate | GRPO estimate | Confidence |
|---|---:|---:|---|
| frozen BF16 checkpoint tensors | 16.33 GiB | 16.33 GiB | exact checkpoint bytes |
| rank-16 adapter + gradient + Adam state | 0.67 GiB | 0.67 GiB | formula; optimizer-dependent |
| CUDA context, kernels, allocator slack | 1–3 GiB | 1–3 GiB | must measure |
| capped vision forward, merger gradients | 1–4 GiB | 1–4 GiB | must measure |
| decoder activations, logits, workspaces | 5–12 GiB | 7–16 GiB | sequence/backend-dependent |
| generation KV/cache and rollout tensors | — | 2–6 GiB | completion/backend-dependent |
| **expected peak reserved** | **24–36 GiB** | **28–44 GiB** | prototype hypothesis |

The acceptance threshold is `max_memory_reserved <= 44 GiB` on the nominated 48 GB Ada GPU, leaving at least 4 GiB against the nominal capacity. If the device exposes less usable capacity, require 10% measured headroom instead.

PyTorch exposes `memory_allocated`, `memory_reserved`, their peak variants, `memory_summary`, and allocator snapshots ([official CUDA memory API](https://docs.pytorch.org/docs/stable/cuda#memory-management)). Its memory profiler only sees memory managed by the PyTorch allocator, so the smoke log must also record device-wide usage for CUDA/NCCL/library allocations ([official PyTorch memory profiler note](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html#identifying-non-pytorch-allocations)).

Record allocated and reserved memory at these boundaries after `torch.cuda.synchronize()`:

1. process start;
2. base model loaded;
3. adapter installed;
4. collated batch on device;
5. forward complete;
6. backward complete;
7. optimizer step complete;
8. GRPO generation complete;
9. adapter reload and inference complete.

Also log GPU model, usable capacity, PyTorch/CUDA/driver versions, Transformers/PEFT/TRL commits or versions, attention backend, exact image grids, prompt tokens, completion tokens, target modules, and trainable parameter count. Reset peak stats immediately before each measured SFT/GRPO step. Treat OOM or less than the required headroom as a failed profile, not as evidence for silently enabling QLoRA.

## SFT-to-GRPO checkpoint contract

1. Load model and processor from the same pinned base revision.
2. Discover and assert the exact topology, then wrap once with PEFT.
3. Train SFT and save an unmerged adapter with safe serialization.
4. Save a manifest containing base ID/revision, processor revision, dependency versions, dtype, attention backend, LoRA config, exact targets/hash, bbox schema version, and dataset revision.
5. For GRPO, rebuild the same base and load the SFT adapter with `is_trainable=True`; pass the already wrapped model to `GRPOTrainer` without a second `peft_config`.
6. Save the resulting GRPO adapter as a new artifact; never overwrite the SFT adapter.
7. Inference loads exactly one selected adapter. Merging is an optional deployment export, not the canonical training checkpoint.

This contract lets GRPO continue the SFT weights without nesting adapters or changing which parameters are trainable. It also permits SFT-only deployment.

## Alternatives and escalation gates

### Lower-capacity ablation: text-attention + multimodal-mergers

Target only text `q/k/v/o` plus the eight merger linears. At rank 16 this has **16,482,304** trainable parameters. Use it as an ablation, not the default, because bbox set generation and empty-set calibration may benefit from MLP adaptation.

### Visual-tail ablation

Add `qkv`, `proj`, `linear_fc1`, and `linear_fc2` in vision blocks 24–26. At rank 16 this adds only **855,552** parameters, but it creates a backward graph through the visual tail and therefore costs much more activation memory than its parameter count suggests. Admit it only after the default smoke profile passes and compare it on reference-consistency slices.

### Full-vision LoRA

All 27 vision blocks add about **7.70 million** rank-16 parameters, but require backward activations through the entire vision tower. Do not include this in the first 48 GB path. It needs a measured memory/quality ablation and may require multi-GPU sharding.

### QLoRA

PEFT supports `target_modules="all-linear"` for QLoRA-style coverage, and TRL documents 4-bit base loading, but quantization is not needed to fit the static rank-16 LoRA state and would add another compatibility dimension ([official PEFT guide](https://huggingface.co/docs/peft/main/developer_guides/lora#q-lora-style-training), [official TRL QLoRA guide](https://huggingface.co/docs/trl/peft_integration#qlora-quantized-low-rank-adaptation)). Keep it out of the first implementation; reconsider only if the measured BF16 smoke profile fails after token/pixel controls are validated.

## Prototype questions that remain open

The following cannot be answered honestly from documentation alone:

- whether the selected Transformers/TRL versions run Qwen3-VL multi-image SFT and GRPO without a processor/collator patch;
- actual peak reserved and device-wide memory on the nominated Ada GPU;
- FlashAttention 2 training compatibility for this exact Qwen3-VL release and image-grid mix;
- whether freezing vision blocks while adapting all four mergers produces the expected gradient boundary in the installed PEFT/Transformers versions;
- whether text MLP adapters improve detection-set mAP/recall enough to justify the additional 28.3 million rank-16 parameters over attention-only;
- whether adapting the final three vision blocks improves small-object/reference consistency enough to justify its activation cost;
- GRPO throughput and variance with only two generations, and whether formal training needs four or eight generations on multiple GPUs.

These become implementation smoke tests and controlled ablations, not more paper decisions.

## Newly surfaced follow-up work

1. Build a topology assertion/profiling utility that prints exact targets, trainable parameters, gradient boundaries, image grids, token counts, and memory checkpoints.
2. Run the rank-16 default and attention-only SFT profiles on the nominated 48 GB Ada GPU.
3. Add a Qwen3-VL two-image TRL GRPO compatibility smoke test with native generation, `beta=0`, and two generations.
4. After a baseline exists, compare default vs final-three-vision-block LoRA on reference-consistency, small-object, multi-instance, and negative-image slices.
