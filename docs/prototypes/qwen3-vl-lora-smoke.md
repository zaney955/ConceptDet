# Qwen3-VL single-GPU LoRA smoke prototype

This is a throwaway CUDA prototype for one question: can the agreed rank-16
`text-all + multimodal-mergers` topology complete a two-image positive/negative training step,
adapter save/reload, and strict Detection Set generation below 44 GiB peak reserved memory on
one approximately 48 GiB Ada GPU, and what changes with the attention-only baseline?

Run the interactive profiler with:

```bash
make prototype-lora-smoke REFERENCE=/path/to/reference.jpg GPU=6
```

The TUI exposes `[f]` full, `[a]` attention-only, `[c]` comparison, and `[q]` quit. Reports and
throwaway PEFT artifacts go under `outputs/lora-smoke/`.

## Measured setup

- model: `Qwen/Qwen3-VL-8B-Instruct`, revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`;
- GPU: NVIDIA GeForce RTX 4090, UUID `f5d26a68-6bc0-f825-dbf9-9c226543b382`,
  47.37 GiB visible capacity;
- Torch 2.13.0+cu130, Transformers 5.13.1, PEFT 0.19.1, Accelerate 1.14.0;
- BF16 base, FP32 LoRA adapter state, FlashAttention 2, gradient checkpointing, AdamW;
- two 832×736 images per example, 598 visual tokens each under the 640-token cap;
- one positive multi-instance example plus one negative `[]` example, accumulated into one
  optimizer step;
- adapter save, complete model release, base/adapter reload, greedy generation, and strict JSON
  Detection Set parsing.

The positive SFT sequence contained 1,348 tokens and the negative sequence contained 1,303. This
invalidates the earlier 1,024-token draft limit: the v1 smoke and production profiles require a
1,536 total-sequence limit, without truncation or generic packing.

## Results

| Measurement | Text-all + mergers | Attention-only + mergers | Full minus baseline |
|---|---:|---:|---:|
| Target modules | 260 | 152 | 108 |
| Trainable parameters | 44,793,856 | 16,482,304 | 28,311,552 |
| Saved adapter | 179,252,976 bytes | 65,974,608 bytes | 113,278,368 bytes |
| Peak reserved memory | 21.193 GiB | 20.572 GiB | 0.621 GiB |
| Peak device-used memory | 21.680 GiB | 21.059 GiB | 0.621 GiB |
| Mean loss before step | 0.5724 | 0.5724 | 0 |
| Mean loss after step | 0.4728 | 0.5098 | -0.0370 |
| Mean loss improvement | 0.0996 | 0.0626 | 0.0370 |
| Save/reload/generate/parse | pass | pass | — |

The selected topology has 22.807 GiB of headroom against the 44 GiB peak-reserved gate. Its
reloaded generation peak was 17.299 GiB and produced the valid unordered Detection Set:

```json
[{"bbox_2d":[750,607,775,670]},{"bbox_2d":[216,607,241,670]}]
```

## Verdict

Keep `text-all + multimodal-mergers` rank 16 as the default. It completes the required lifecycle
with ample memory headroom. Compared with attention-only, adapting the text MLPs costs only
0.621 GiB peak memory in this profile and shows a larger one-step loss decrease.

The two-example loss delta is an optimization sanity signal, not evidence of generalization or
quality improvement. Attention-only remains a formal evaluation ablation; mAP, negative false
positives, reference-swap consistency, and small-object slices decide whether the extra capacity
earns its place after real fine-tuning.
