#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${REPO_ROOT}/scripts/run_inference.sh" infer detect \
  --config "${1:-${REPO_ROOT}/examples/detect.yaml}"
