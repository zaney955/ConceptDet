#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/ConceptSeg-R1-7B}"

conceptdet detect \
  --model "${MODEL_PATH}" \
  --reference "reference.jpg" \
  --reference-box "100,120,220,280" \
  --target "target.jpg" \
  --query "the same component as the red-boxed example" \
  --output "outputs/result.png"
