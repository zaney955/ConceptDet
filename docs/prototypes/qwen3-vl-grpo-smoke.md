# Qwen3-VL two-image native GRPO smoke

Wayfinder question: can stock TRL GRPO train the selected full LoRA adapter on
ordered `[reference, target]` images with the strict Detection Set reward, on one
48 GiB RTX 4090, without SAM, vLLM, a reference model, or a custom Trainer fork?

Run from a normal checkout after installing the prototype's pinned TRL version:

```bash
.venv/bin/pip install -r requirements/prototype-grpo.txt
make prototype-grpo-smoke \
  REFERENCE=/absolute/path/to/reference.jpg \
  INIT_ADAPTER=/absolute/path/to/sft-adapter \
  GPU=6
```

The command writes its machine-readable evidence and throwaway adapter under
`outputs/grpo-smoke/`. The decision and measured result are recorded below after
the hardware run.

## Result

Accepted on 2026-07-15. Stock `trl.trainer.grpo_trainer.GRPOTrainer` completed
the ordered two-image run with `Qwen/Qwen3-VL-8B-Instruct` and the full LoRA
adapter selected by the preceding SFT smoke. No Trainer subclass, model fork,
SAM dependency, vLLM path, or reference model was used.

Measured configuration and evidence:

- PyTorch 2.13.0+cu130, Transformers 5.13.1, TRL 1.5.0, PEFT 0.19.1.
- One RTX 4090, BF16 + FlashAttention 2, 44,793,856 trainable LoRA parameters.
- `beta=0`, two native generations, generation batch size 2, gradient
  accumulation 2, and two optimizer steps (one positive and one negative
  prompt group).
- Both images were adapter-resized to 832x736 and produced ordered grids
  `[[1,46,52],[1,46,52]]`, preserving the 640-token-per-image contract.
- Default processor preprocessing and explicit `do_resize=False` produced
  identical shapes and pixel tensors (`max_abs_difference=0.0`), so no
  processor wrapper is required for already contract-sized images.
- The reward callback received four strict-JSON completions across positive and
  negative examples. Positive rewards were 0.1603 and 0.2571, producing nonzero
  advantages and a nonzero gradient; both negative completions were correct
  empty sets with reward 1.0.
- LoRA SHA-256 changed from `5fae5bf6...` to `7fd1a46e...`, proving an optimizer
  update, and the resulting 171 MiB adapter was saved successfully.
- Peak CUDA reserved memory was 18.36 GiB, comfortably below the 44 GiB gate.
  Native training runtime after model load was 11.1 seconds.

The smoke validates compatibility and the optimization path, not detection
quality. The initial adapter localized the positive reference imperfectly; the
real SFT/GRPO corpus and evaluation split remain responsible for improving that
quality.

## Integration seam

The training dataset should emit a raw conversational `prompt`, an ordered
`images=[reference, target]` list, normalized Detection Set ground truth, and
sample metadata. The ConceptDet adapter owns image resize and reference-box
rendering. The stock Qwen processor and stock GRPO Trainer own chat-template
application, multimodal tokenization, native generation, forward recomputation,
and reward grouping. The strict prompt must explicitly forbid Markdown fences;
otherwise-correct fenced JSON is deliberately rejected by the strict parser.
