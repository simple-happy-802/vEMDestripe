#!/bin/bash
# Read-only saved-final-output evaluation (no normalization) for vEMDestripe and available baselines.

#SBATCH -J vem_eval
#SBATCH -p a100
#SBATCH --qos=a100g2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=72:00:00
#SBATCH -o logs/evaluate_%j.out
#SBATCH -e logs/evaluate_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
TEST_DATA_ROOT=${TEST_DATA_ROOT:-${PROJECT_DIR}/test_data}
RESULT_ROOT=${RESULT_ROOT:-${PROJECT_DIR}/results/vemdestripe}
GT_ROOT=${GT_ROOT:-${PROJECT_DIR}/data/ground_truth}
EVAL_DIR=${EVAL_DIR:-${PROJECT_DIR}/evaluation_output/vemdestripe}
DATASETS=${DATASETS:-}
GPU_BATCH_SIZE=${GPU_BATCH_SIZE:-8}

export VEMDESTRIPE_TEST_DATA_ROOT="${TEST_DATA_ROOT}"
export VEMDESTRIPE_RESULT_ROOT="${RESULT_ROOT}"
export VEMDESTRIPE_GT_ROOT="${GT_ROOT}"

mkdir -p "${PROJECT_DIR}/logs" "${EVAL_DIR}"
cd "${PROJECT_DIR}"

CMD=(
    "${PYTHON_BIN}" -m evaluation.evaluate
    --output-dir "${EVAL_DIR}"
    --gpu-batch-size "${GPU_BATCH_SIZE}"
    --evaluation-window-mode max-hf-mad
    --display-mode raw
)
if [ -n "${DATASETS}" ]; then CMD+=(--datasets "${DATASETS}"); fi

printf '[RUN]'; printf ' %q' "${CMD[@]}"; printf '\n'
"${CMD[@]}"
echo "[DONE] Evaluation: ${EVAL_DIR}"

