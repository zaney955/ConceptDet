# Qwen3-VL integration contract

Resolved question: how ConceptDet should integrate
`Qwen/Qwen3-VL-8B-Instruct` for reference-guided Detection Sets without
carrying forward the Qwen2.5-VL/ConceptSeg protocol.

## Decision

Adopt Qwen3-VL as a clean model-boundary replacement:

- Accept only checkpoints whose `model_type` is `qwen3_vl`; use
  `Qwen3VLForConditionalGeneration` with the checkpoint's `AutoProcessor`.
- Send the reference image first and the target image second in one user message,
  and call the processor's multimodal chat template with `add_vision_id=True`.
- Preserve aspect ratio and let the Qwen3-VL processor perform dynamic image
  resizing. Remove the legacy 600×600 target stretch and its inverse transform.
- Serialize the model answer as a JSON Detection Set using Qwen3-VL's native
  `bbox_2d` XYXY coordinates on a relative 0–1000 grid. `[]` is ConceptDet's
  canonical no-match answer.
- Use the same chat template, image order, visual-token budget, coordinate
  conversion, and JSON serialization in SFT, GRPO, evaluation, and inference.
- Do not retain the old `<think>/<rule>/<bbox>/<answer>` protocol or old
  ConceptSeg/Qwen2.5 checkpoints.

This follows Qwen3-VL's official multi-target grounding behavior instead of
emulating the predecessor checkpoint.

## Supported software and model API

Qwen's stated floor is `transformers>=4.57.0`; ConceptDet's existing
`transformers==5.13.1` lock already satisfies it. The 8B checkpoint declares
`model_type: "qwen3_vl"`, architecture
`Qwen3VLForConditionalGeneration`, and BF16 weights. The concrete model class
is preferable here because it makes checkpoint validation explicit; the
official `AutoModelForImageTextToText` entry point remains equivalent for this
checkpoint.

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
```

Sources: [Qwen3-VL minimum version and loading examples](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md#using--transformers-to-chat),
[8B config](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json),
[Transformers Qwen3-VL API](https://huggingface.co/docs/transformers/model_doc/qwen3_vl).

The official fine-tuning repository lists a known working stack of
PyTorch 2.6.0, Transformers 4.57.0.dev0, Accelerate 1.7.0, PEFT 0.17.1, and
FlashAttention 2.7.4.post1. Those are reference versions, not declared lower
bounds. ConceptDet should retain its newer locked runtime and prove it with the
planned smoke test rather than downgrade to the reference stack.
[Source](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md#requirements).

## Multimodal message and processor contract

Put both in-memory PIL images directly in the ordered message. Do not render a
text-only template and separately maintain an image list unless an external
serving backend requires that split.

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": reference_image},
        {"type": "image", "image": target_image},
        {"type": "text", "text": prompt},
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    add_vision_id=True,
    return_dict=True,
    return_tensors="pt",
)
```

`add_vision_id=True` gives the visual inputs stable `Picture 1` and `Picture 2`
labels. The prompt must state that Picture 1 is the red-boxed reference and all
reported coordinates belong to Picture 2. Pass every processor-produced model
input (`input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, and any
version-specific multimodal token-type field) to the model; do not rebuild
vision placeholders manually.

Sources: [official multi-image and vision-ID examples](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md#add-ids-for-multiple-visual-inputs),
[Transformers multimodal chat templates](https://huggingface.co/docs/transformers/chat_templating_multimodal),
[checkpoint chat template](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/chat_template.json).

`qwen-vl-utils` is unnecessary for ConceptDet's in-memory still images because
the Transformers processor loads and preprocesses them directly. If a future
serving adapter uses the utility, use at least 0.0.14, pass
`image_patch_size=16`, and then pass `do_resize=False` to the processor to avoid
double resizing.
[Source](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md#new-qwen-vl-utils-usage).

## Dynamic resolution and visual budget

Qwen3-VL uses a 16-pixel image patch and spatial merge size 2, so processed
height and width are rounded to multiples of 32 while the original aspect ratio
is preserved. The checkpoint's default image budget is 65,536–16,777,216
pixels, approximately 64–16,384 visual tokens per image. With two images, the
unbounded worst case is inappropriate for a 48 GB smoke-training target.

ConceptDet must therefore expose one explicit visual-token/pixel budget and use
it unchanged across training and inference. The exact budget is a hardware and
quality decision, not part of the upstream compatibility contract; the Qwen
documentation illustrates a practical 256–1,280 token range. Training
resolution materially affects quality, so it must be stored with each adapter
configuration.

Sources: [Qwen pixel control](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md#pixel-control-via-official-processor),
[8B processor defaults](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/preprocessor_config.json),
[official training note on resolution](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md#usage).

Consequences for current preprocessing:

- Keep EXIF correction and RGB conversion.
- Draw or crop the reference before processor input, but do not stretch the
  prepared reference or target to a square.
- Delete `input_size=600`, `ImageTransform`-based target restoration, and the
  assumption that output coordinates are model-input pixels.
- Preserve the target's original `(width, height)` solely for normalized-to-
  pixel conversion and final visualization.

## Grounding and output contract

Qwen3-VL changed its default grounding coordinates from Qwen2.5-VL's resized-
image absolute coordinates to a relative 0–1000 grid; its official notebook
explicitly says no resized width calculation is needed. The official
multi-target output is a JSON list whose items contain
`"bbox_2d": [x1, y1, x2, y2]` and a label. Categories that are absent are
omitted from the generated list.

```json
[
  {"bbox_2d": [74, 361, 324, 808], "label": "object"}
]
```

Source: [official Qwen3-VL 2D grounding cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb).

ConceptDet tightens that behavior into this model-boundary schema:

- the top level is always a JSON array, without prose or Markdown fencing;
- `bbox_2d` is integer XYXY on the inclusive 0–1000 reference grid;
- `label` is optional metadata and cannot determine whether a box is valid;
- the list represents an unordered Detection Set;
- no target is `[]`.

Convert only at the model boundary, using the EXIF-corrected target dimensions:

```text
model_x = round(pixel_x / target_width  * 1000)
model_y = round(pixel_y / target_height * 1000)

pixel_x = model_x / 1000 * target_width
pixel_y = model_y / 1000 * target_height
```

Clamp model coordinates to 0–1000 and pixels to image bounds, then reject
degenerate boxes. Do not involve `image_grid_thw` or the processor's resized
dimensions in this conversion.

## Generation versus SFT

| Concern | Inference/evaluation | SFT |
|---|---|---|
| Messages | system/user | system/user plus assistant Detection Set |
| Chat-template suffix | `add_generation_prompt=True` | `add_generation_prompt=False` |
| Model call | `generate(...)` | `forward(..., labels=labels)` |
| Labels | none | same shape as `input_ids`; non-answer tokens are `-100` |
| Cache/mode | `eval()`, cache enabled | `train()`, cache disabled |
| Output handling | trim the input token prefix, decode completion | optimize assistant-answer loss only |

The Qwen3-VL forward API accepts `labels` and ignores positions set to `-100`.
The checkpoint chat template does not contain a Jinja `{% generation %}` block,
so the training collator must not assume `return_assistant_tokens_mask=True`
will identify answer tokens. For this single-turn task, build the prompt and
full conversation separately, verify that prompt token IDs are a prefix of the
full sequence, and mask that prefix plus padding. This is more robust than
hard-coding assistant-token IDs.

Sources: [Qwen3-VL forward contract](https://huggingface.co/docs/transformers/model_doc/qwen3_vl#transformers.Qwen3VLForConditionalGeneration.forward),
[Transformers chat-template training guidance](https://huggingface.co/docs/transformers/chat_templating#model-training),
[official Qwen training entry point](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/train/train_qwen.py).

Do not enable generic sequence packing for the first smoke implementation.
Qwen's official packing path has model-specific multimodal RoPE preprocessing;
ordinary padded batches can use the standard processor/model forward path.
[Source](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune/qwenvl/data).

## Attention and decoding

The Transformers Qwen3-VL implementation declares SDPA and FlashAttention
support. Use SDPA as the dependency-free fallback and
`flash_attention_2` on compatible CUDA hardware. Qwen recommends FA2 for
multi-image workloads; it requires FP16 or BF16.

Sources: [Transformers Qwen3-VL implementation](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py),
[Qwen FlashAttention instructions](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md#flash-attention-2-to-speed-up-generation).

The checkpoint generation config samples by default (`temperature=0.7`,
`top_p=0.8`, `top_k=20`). ConceptDet evaluation and normal detection should
explicitly request greedy decoding for reproducible structured JSON; sampling
can remain an opt-in experiment rather than an inherited hidden default.
[Source](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/generation_config.json).

## Concrete migration delta

| Current Qwen2.5 backend | Qwen3-VL contract |
|---|---|
| `Qwen2_5_VLForConditionalGeneration` | `Qwen3VLForConditionalGeneration` |
| accepts `qwen2_vl` / `qwen2_5_vl` configs | accepts only `qwen3_vl` |
| local ConceptSeg checkpoint and ignored SAM-era keys | official base checkpoint plus ConceptDet adapter; no ignored SAM keys |
| placeholder image messages plus separate image list | PIL images embedded in ordered content and one-step multimodal chat template |
| Qwen2 processor `use_fast=False` workaround | checkpoint default fast image processor; remove workaround |
| both images stretched to 600×600 | aspect-preserving dynamic resolution with an explicit visual budget |
| pixel bbox in 600×600 model coordinates | `bbox_2d` on the target's relative 0–1000 grid |
| `<think>/<rule>/<bbox>/<answer>` parsing | strict JSON Detection Set, including `[]` |
| single-/multi-tag boxes treated as ordered text | unordered multi-instance set |

## Follow-on decisions exposed by this research

1. Select the shared reference/target visual-token budgets that fit the 48 GB
   smoke target without losing small-object detail; measure peak memory rather
   than inheriting the checkpoint's large defaults.
2. Define how reference-box stroke width scales before dynamic resizing, so the
   red mark remains visible without obscuring small reference objects.
3. Version and persist a preprocessing contract beside each adapter (processor
   revision, budgets, coordinate schema, image order, and prompt version) and
   reject incompatible adapter/runtime combinations.
