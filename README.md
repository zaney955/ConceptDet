# ConceptDet-R1

ConceptDet-R1 是从 ConceptSeg-R1 的“视觉示例归纳 → 目标定位”思想重新设计的独立检测仓库。输入参考图、参考框、目标图和概念描述后，模型只生成目标 bbox；默认输出“参考图｜目标图｜bbox 结果图”三图拼接，也可只保存原始分辨率结果图。

本仓库不包含、也不依赖 SAM3，不生成 mask，不传递隐式 concept token，不执行任何分割后处理。

## 流程

```text
参考图 + 原图坐标参考框 ──> 参考提示图（红框） ─┐
                                                   ├─> ConceptSeg-R1 / Qwen2.5-VL
目标图 ───────────────────> 600×600 目标提示图 ──┘
                                                            │
                                                            ▼
                                                  <bbox>[x1,y1,x2,y2]</bbox>
                                                            │
                                                            ▼
                                        反算到目标原图坐标 + 原图画框 + JSON
```

与旧实现相比，这里有几个有意的设计变化：

- 检测链路不再实例化 SAM3、learnable query、connector 或 SAM projection。
- bbox 协议、坐标变换、模型后端、业务流水线和可视化彼此解耦。
- 默认输出与 ConceptSeg 一致的固定三栏拼接图，亦可配置为原分辨率单结果图。
- 每张输出图都有 JSON 结果，保留原图坐标、模型坐标、规则和原始生成文本。
- 单张与批处理共用同一套推理实现；批处理只加载一次模型。
- 输入校验和模型输出解析采用显式错误，不静默吞掉错误框。

## 独立环境安装

ConceptDet 使用仓库自己的标准 Python venv：
`/home/yzy/TogeeWork/Projects/ConceptDet-R1/.venv`。它不读取、不激活、也不修改
ConceptSeg 的 `.venv`，同样不依赖当前终端的 Conda `base` 环境。

一键创建环境：

```bash
cd /home/yzy/TogeeWork/Projects/ConceptDet-R1
bash scripts/create_env.sh
```

脚本使用 Python 3.13 创建 `.venv`，按 `requirements/` 中的锁定版本安装依赖，
然后单独编译与本机 CUDA、PyTorch 和 Ada GPU（计算能力 8.9）匹配的 FlashAttention。
当前锁定的主要运行库为：

| 库 | 版本 |
|---|---:|
| Python | 3.13 |
| PyTorch | 2.13.0 |
| TorchVision | 0.28.0 |
| Transformers | 5.13.1 |
| Accelerate | 1.14.0 |
| FlashAttention | 2.8.3.post1 |
| Pillow | 12.3.0 |
| Safetensors | 0.8.0 |
| SentencePiece | 0.2.2 |
| NumPy | 2.5.1 |

检查独立环境：

```bash
.venv/bin/python scripts/check_environment.py --require-cuda
```

不需要执行 `conda activate` 或引用 ConceptSeg 的解释器。

为尽量复现 ConceptSeg `bbox_only=True` 的生成路径，默认使用 FlashAttention 2。
运行环境需要安装 `flash-attn`；如果显式切换到 SDPA，确定性贪心解码仍可能因数值误差
在早期 token 处分叉，从而生成完全不同的 bbox。

兼容设置还包括：参考提示图使用 2px 红框、用户侧问题模板对齐 ConceptSeg、BF16、
`do_sample=False`。ConceptDet 自己的 system prompt 保持不变。升级底层库可能带来细微数值
变化，因此依赖锁定在 `requirements/runtime.txt` 和
`requirements/flash-attention.txt`，不会每次运行时静默漂移。

模型权重不复制进新仓库。可以直接使用现有 ConceptSeg-R1 checkpoint：

```text
/home/yzy/TogeeWork/Projects/ConceptSeg-R1/ConceptSeg-R1-7B
```

checkpoint 中属于分割链路的 `learnable_query.*`、`connector.*` 和 `proj_to_sam.*` 权重会被有控制地忽略；基础 Qwen2.5-VL 权重缺失或出现其他不兼容权重时，加载会直接报错。

## 单张检测

```bash
conceptdet detect \
  --model /home/yzy/TogeeWork/Projects/ConceptSeg-R1/ConceptSeg-R1-7B \
  --device cuda:0 \
  --reference /data/reference.jpg \
  --reference-box "1165,2911,1354,3230" \
  --reference-box "4064,3087,4208,3375" \
  --target /data/target.jpg \
  --query "the same bolt as the red-boxed examples" \
  --output outputs/target_detected.png
```

也可以在一个参数中用分号传多个参考框：

```bash
--reference-box "1165,2911,1354,3230;4064,3087,4208,3375"
```

参考框统一使用参考原图的 `XYXY` 像素坐标。`x2/y2` 是右/下边界。图片会先按 EXIF 方向旋转为视觉上的正向，因此参考框应对应旋转后的图像尺寸。

默认输出：

- `outputs/target_detected.png`：`参考图｜目标图｜bbox 结果图` 三栏拼接。
- `outputs/target_detected.json`：结构化结果。

默认 `--input-size 600` 时，三栏图尺寸为 `1800×600`：左栏是已绘制参考框的
参考提示图，中栏是不带预测框的目标图，右栏是带预测 bbox 的目标图。只保存原始
分辨率检测结果时使用：

```bash
--output-layout annotated
```

### 直接在脚本中修改参数

如果不想每次填写命令行参数，可以直接编辑：

```text
scripts/inference_config.py
```

只需修改文件顶部的 `CONFIG` 和 `TASKS`。`mode="tasks"` 时，`TASKS` 可以配置
一个或多个任务；每个 GPU worker 会复用自己加载的模型：

```python
CONFIG = {
    "mode": "tasks",
    "model_path": "../ConceptSeg-R1/ConceptSeg-R1-7B",
    "gpu_ids": [0, 1, 2, 3],
    "min_free_memory_gb": 24.0,
    "retry_model_load_failures": True,
    "dtype": "bfloat16",
    "attention": "flash_attention_2",
    "input_size": 600,
    "max_new_tokens": 768,
    "box_color": "red",
    "box_width": 2,
    "reference_box_width": 2,
    "output_layout": "triptych",
}

TASKS = [
    {
        "reference_path": "/data/reference.jpg",
        "reference_boxes": "100,120,220,280;300,150,410,290",
        "target_path": "/data/target.jpg",
        "query": "the same bolt as the red-boxed examples",
        "output_path": "outputs/result.png",
        "reference_crop_mode": "full",
        "reference_crop_context_scale": 4.0,
    },
]
```

然后直接运行：

```bash
bash scripts/run_inference.sh
```

脚本中的相对路径统一相对于 ConceptDet-R1 仓库根目录，而不是当前终端目录。
`run_inference.sh` 总是使用 ConceptDet 自己的 `.venv/bin/python`。

### 手动脚本多卡推理

在 `scripts/inference_config.py` 中设置：

```python
CONFIG = {
    "model_path": "../ConceptSeg-R1/ConceptSeg-R1-7B",
    "gpu_ids": [0, 1, 2, 3],
    "min_free_memory_gb": 24.0,
    "retry_model_load_failures": True,
    "dtype": "bfloat16",
    "attention": "flash_attention_2",
    # 其他参数保持不变
}
```

然后在 `TASKS` 中填写多个任务并正常运行：

```bash
bash scripts/run_inference.sh
```

执行策略如下：

- 每张指定 GPU 启动一个独立的 `spawn` 进程。
- 启动前检查每张卡的实时空闲显存，低于 `min_free_memory_gb` 的忙卡不会启动 worker。
- 每个进程只加载一次模型，并依次处理分配给该 GPU 的任务。
- 任务按轮询方式分配，例如 6 个任务和 `[0, 1, 2, 3]` 会分配为
  `cuda:0 → 任务 0/4`、`cuda:1 → 任务 1/5`、`cuda:2 → 任务 2`、
  `cuda:3 → 任务 3`。
- 任务数少于 GPU 数时，不会在空闲 GPU 上加载模型。
- 一个任务失败不会阻止同一 GPU 继续执行后续任务。
- 若显存状态在预检后发生变化并导致模型加载失败，相关任务会在存活 GPU 上重试一次。

每张 GPU 都需要容纳一份完整模型。`gpu_ids` 不可重复，且必须是当前机器存在的
CUDA 编号；脚本启动时会先校验。

`min_free_memory_gb=24.0` 是针对当前 7B BF16 checkpoint 的保守默认值。若确认模型
和生成过程在更低显存下稳定，可以适当调低；设为 `0` 表示不按空闲显存过滤。

### 扫描目录进行多图批量推理

这相当于原仓库 `inference_batch_examples.py` 的共享参考图批处理模式。编辑
`scripts/inference_config.py`：

```python
CONFIG = {
    "mode": "batch",
    "model_path": "../ConceptSeg-R1/ConceptSeg-R1-7B",
    "gpu_ids": [0, 1, 2, 3],
    "min_free_memory_gb": 24.0,
    "retry_model_load_failures": True,
    "dtype": "bfloat16",
    "attention": "flash_attention_2",
    "input_size": 600,
    "max_new_tokens": 768,
    "box_color": "red",
    "box_width": 2,
    "reference_box_width": 2,
    "output_layout": "triptych",
}

BATCH_CONFIG = {
    # 可同时填写目录和单张图片。
    "input_paths": [
        "/data/target_images",
        "/data/extra/example.jpg",
    ],
    # True 会扫描 target_images 下的所有子目录。
    "recursive": False,

    # 所有目标图片共享以下参考信息。
    "reference_path": "/data/reference.jpg",
    "reference_boxes": "1165,2911,1354,3230;4064,3087,4208,3375",
    "query": "the same bolt as the red-boxed examples",
    "reference_crop_mode": "full",
    "reference_crop_context_scale": 4.0,

    "output_dir": "outputs/batch",
    "skip_existing": True,
    "log_path": "outputs/batch/results.jsonl",
}
```

运行方式不变：

```bash
bash scripts/run_inference.sh
```

批处理行为：

- 支持 `.jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff`。
- 自动排除参考图，输入目录包含输出目录时也不会反复处理旧输出。
- 自动去除重复输入；多个输入目录出现同名图片时使用 `__2` 等后缀防止覆盖。
- `recursive=True` 时，输出目录保留输入图片的相对子目录结构。
- `skip_existing=True` 时跳过已有标注图，不会为全跳过的任务加载模型。
- 每张标注图旁边保存同名 JSON，批次汇总写入 `log_path` 指定的 JSONL。
- 活跃任务按照轮询方式分给 `gpu_ids` 中的 GPU。
- 忙卡会在分配前被剔除，任务自动重新分配到剩余 GPU。

启动日志会明确显示每张卡的决定：

```text
GPU preflight: cuda:3 free=31.20/47.37 GiB required>=24.00 GiB -> USE
GPU preflight: cuda:4 free=0.06/47.37 GiB required>=24.00 GiB -> SKIP busy
GPU preflight: cuda:5 free=0.02/47.37 GiB required>=24.00 GiB -> SKIP busy
Multi-GPU assignment: cuda:3=2 task(s), cuda:6=1 task(s), cuda:7=1 task(s)
```

JSON 示例：

```json
{
  "target_image": "/data/target.jpg",
  "query": "the same bolt as the red-boxed examples",
  "model_input_size": [600, 600],
  "output_layout": "triptych",
  "detections": [
    {
      "bbox_xyxy": [1012, 532, 1178, 806],
      "model_bbox_xyxy": [253, 133, 295, 202],
      "label": "bolt"
    }
  ],
  "rule": "...",
  "reasoning": "...",
  "raw_completion": "..."
}
```

### 小目标参考框

默认 `--reference-crop full` 保留完整参考图上下文。参考目标很小时可在缩放前围绕所有参考框裁剪：

```bash
--reference-crop crop --reference-context-scale 4
```

`reference-context-scale` 分别按参考框联合区域的宽和高扩展，且裁剪范围不会越过原图。

## 批处理

准备 JSONL，每行一个任务。相对路径以 manifest 所在目录为基准：

```json
{"id":"gx-001","reference":"ref.jpg","reference_boxes":[[1165,2911,1354,3230],[4064,3087,4208,3375]],"target":"targets/001.jpg","query":"the same bolt as the red-boxed examples"}
{"id":"gx-002","reference":"ref.jpg","reference_boxes":"1165,2911,1354,3230","target":"targets/002.jpg","query":"the same bolt as the red-boxed example","reference_crop":"crop"}
```

运行：

```bash
conceptdet batch \
  --model /home/yzy/TogeeWork/Projects/ConceptSeg-R1/ConceptSeg-R1-7B \
  --device cuda:0 \
  --manifest examples/tasks.jsonl \
  --output-dir outputs/batch
```

每个任务输出一张标注图和一个同名 JSON；汇总写入 `outputs/batch/results.jsonl`。默认跳过已有输出，使用 `--overwrite` 覆盖。

## 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--device` | `auto` | 自动选择 `cuda:0`，CUDA 不可用时使用 CPU |
| `--dtype` | `auto` | CUDA 优先 BF16，CPU 使用 FP32 |
| `--attention` | `flash_attention_2` | 与 ConceptSeg bbox-only 路径一致 |
| `--input-size` | `600` | checkpoint 提示图边长；一般不要修改 |
| `--max-new-tokens` | `768` | 最大生成 token 数 |
| `--box-color` | `red` | 输出框颜色 |
| `--box-width` | `2` | 输出框线宽 |
| `--reference-box-width` | `2` | 模型参考提示图的红框线宽；兼容模式不要修改 |
| `--output-layout` | `triptych` | 三图拼接；`annotated` 为原分辨率单结果图 |

## Python API

```python
from pathlib import Path

from conceptdet.model import TransformersBackend
from conceptdet.pipeline import DetectionPipeline, DetectionRequest
from conceptdet.types import parse_boxes

backend = TransformersBackend.load(
    "/path/to/ConceptSeg-R1-7B",
    device="cuda:0",
    dtype="bfloat16",
)
pipeline = DetectionPipeline(backend)
request = DetectionRequest(
    reference_path=Path("reference.jpg"),
    reference_boxes=parse_boxes("100,120,220,280"),
    target_path=Path("target.jpg"),
    query="the same component as the red-boxed example",
)
result = pipeline.run(request, output_path=Path("outputs/result.png"))
print(result.to_dict()["detections"])
```

## 开发与验证

```bash
bash scripts/create_env.sh
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
```

测试覆盖 bbox 解析、坐标双向变换、参考图裁剪与画框、模型输出协议、原图坐标
反算、三图拼接、输出图与 JSON 的端到端生成。模型权重加载和 GPU 推理由于需要约
17 GB 权重及 CUDA 环境，应在实际推理机执行一次 smoke test。

## 目录结构

```text
src/conceptdet/
├── cli.py             # 单张与 JSONL 批处理入口
├── batch.py           # 目录扫描、输入去重与输出路径规划
├── multi_gpu.py       # 每 GPU 一进程的多卡任务执行器
├── model.py           # 纯 Qwen2.5-VL 生成后端
├── pipeline.py        # 检测用例编排与结果模型
├── geometry.py        # 参考/目标图坐标变换
├── parsing.py         # <bbox> 输出协议解析
├── prompts.py         # 双图定位提示词
├── visualization.py   # 原始目标图绘制
├── types.py           # bbox 值对象与输入解析
└── errors.py          # 可预期错误类型
```

## 边界说明

- 当前 checkpoint 的训练协议返回单个最佳目标框；解析器允许多个 `<bbox>` 标签，但提示词明确要求单个最佳匹配。
- 本仓库只做推理，不包含 ConceptSeg-R1 的训练代码、评测集代码或分割指标。
- 不支持 mask、polygon、SAM prompt、SAM feature 或 segmentation output。

## License

Apache-2.0。
