#!/bin/bash
# Generate paired pseudo-clean/stripe-corrupted training data.

#SBATCH -J vem_data
#SBATCH -p a100
#SBATCH --qos=a100g2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH -o logs/generate_data_%j.out
#SBATCH -e logs/generate_data_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE_TIF_DIR=${SOURCE_TIF_DIR:-${PROJECT_DIR}/data/clean_tifs}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data/training}

IMAGE_SIZE=${IMAGE_SIZE:-1024}
VAL_RATIO=${VAL_RATIO:-0.10}
TEST_RATIO=${TEST_RATIO:-0.10}
REFERENCE_NUM=${REFERENCE_NUM:-50}
KEEP_CLEAN_RATIO=${KEEP_CLEAN_RATIO:-0.15}
SEED=${SEED:-42}
MAX_SAMPLES=${MAX_SAMPLES:-}

# Exact weights from the supplied oblique-boundary generator (M0 ... M9).
MODE_WEIGHTS=(0.22 0.09 0.11 0.07 0.13 0.09 0.09 0.14 0.06 0.00)

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

if [ ! -d "${SOURCE_TIF_DIR}" ]; then
    echo "[ERROR] Missing clean TIFF directory: ${SOURCE_TIF_DIR}" >&2
    exit 1
fi
FIRST_DATA_ENTRY=
if [ -d "${DATA_DIR}" ]; then
    FIRST_DATA_ENTRY=$(find "${DATA_DIR}" -mindepth 1 -print -quit)
fi
if [ -n "${FIRST_DATA_ENTRY}" ]; then
    echo "[ERROR] DATA_DIR must be empty for a new simulation: ${DATA_DIR}" >&2
    exit 1
fi
mkdir -p "${DATA_DIR}"

CMD=(
    "${PYTHON_BIN}" -m data_generation.generate
    --input_dir "${SOURCE_TIF_DIR}"
    --output_dir "${DATA_DIR}"
    --split all
    --image_size "${IMAGE_SIZE}"
    --format png
    --val_ratio "${VAL_RATIO}"
    --test_ratio "${TEST_RATIO}"
    --reference_num "${REFERENCE_NUM}"
    --keep_clean_ratio "${KEEP_CLEAN_RATIO}"
    --seed "${SEED}"
    --weights "${MODE_WEIGHTS[@]}"
)
if [ -n "${MAX_SAMPLES}" ]; then CMD+=(--max_samples "${MAX_SAMPLES}"); fi

printf '[RUN]'; printf ' %q' "${CMD[@]}"; printf '\n'
"${CMD[@]}"

test -f "${DATA_DIR}/dataset_info.json" || {
    echo "[ERROR] Missing dataset_info.json after generation" >&2
    exit 1
}
echo "[DONE] Training data: ${DATA_DIR}"
