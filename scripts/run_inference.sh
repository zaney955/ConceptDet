#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CONCEPTDET_ENV_DIR:-${REPO_ROOT}/.venv}/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ConceptDet environment is missing. Create it first:" >&2
  echo "  bash ${REPO_ROOT}/scripts/create_env.sh" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 infer detect --config FILE" >&2
  echo "       $0 infer batch --config FILE" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -m conceptdet "$@"
