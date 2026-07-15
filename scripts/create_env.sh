#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${CONCEPTDET_ENV_DIR:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.13}"
PYTHON="${ENV_DIR}/bin/python"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "Creating independent ConceptDet environment: ${ENV_DIR}"
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
else
  echo "Updating existing ConceptDet environment: ${ENV_DIR}"
fi

"${PYTHON}" -m pip install --upgrade pip setuptools wheel packaging ninja
"${PYTHON}" -m pip install --upgrade -r "${REPO_ROOT}/requirements/runtime.txt"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-8}"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA compiler not found: ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi

"${PYTHON}" -m pip install --upgrade --no-build-isolation \
  -r "${REPO_ROOT}/requirements/flash-attention.txt"
"${PYTHON}" -m pip install --editable "${REPO_ROOT}[dev]" --no-deps
"${PYTHON}" "${REPO_ROOT}/scripts/check_environment.py"

echo
echo "Environment ready. Run inference with:"
echo "  ${REPO_ROOT}/scripts/run_inference.sh infer detect --config FILE"
