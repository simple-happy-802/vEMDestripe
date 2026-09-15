#!/bin/bash
# Batch inference over direct children of test_data.

#SBATCH -J vem_infer
#SBATCH -p a100
#SBATCH --qos=a100g2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH -o logs/infer_%j.out
#SBATCH -e logs/infer_%j.err

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH=${MODEL_PATH:-${PROJECT_DIR}/checkpoints/vemdestripe_seed42/best.pth}
TEST_DATA_ROOT=${TEST_DATA_ROOT:-${PROJECT_DIR}/test_data}
REFERENCE_ROOT=${REFERENCE_ROOT:-${PROJECT_DIR}/references}
RESULT_ROOT=${RESULT_ROOT:-${PROJECT_DIR}/results/vemdestripe}
ONLY_DATASET=${ONLY_DATASET:-}

# Paper-checkpoint profile. See docs/PAPER_CODE_ALIGNMENT.md before changing.
NUM_REFERENCE=${NUM_REFERENCE:-16}
STRUCT_BLOCK_SIZE=${STRUCT_BLOCK_SIZE:-16}

mkdir -p "${PROJECT_DIR}/logs" "${RESULT_ROOT}"
cd "${PROJECT_DIR}"
test -f "${MODEL_PATH}" || { echo "[ERROR] Missing checkpoint: ${MODEL_PATH}" >&2; exit 1; }
test -d "${TEST_DATA_ROOT}" || { echo "[ERROR] Missing test root: ${TEST_DATA_ROOT}" >&2; exit 1; }

processed=0
count_images() {
    find "$1" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \) 2>/dev/null | wc -l
}

for dataset_dir in "${TEST_DATA_ROOT}"/*; do
    [ -d "${dataset_dir}" ] || continue
    dataset=$(basename "${dataset_dir}")
    if [ -n "${ONLY_DATASET}" ] && [ "${dataset}" != "${ONLY_DATASET}" ]; then continue; fi

    input_dir=
    for candidate in \
        "${dataset_dir}" \
        "${dataset_dir}/input" \
        "${dataset_dir}/striped" \
        "${dataset_dir}/test/striped" \
        "${dataset_dir}/test"; do
        if [ -d "${candidate}" ] && [ "$(count_images "${candidate}")" -gt 0 ]; then
            input_dir="${candidate}"
            break
        fi
    done
    if [ -z "${input_dir}" ]; then
        echo "[SKIP] ${dataset}: no PNG/TIFF input sequence"
        continue
    fi

    # Priority 1 is a manually screened deployment reference. The groundtruth
    # fallbacks reproduce the paired benchmark only and must not be used to
    # describe performance on a real volume without clean labels.
    reference_dir="${REFERENCE_ROOT}/${dataset}"
    if [ ! -d "${reference_dir}" ] && [ -d "${dataset_dir}/reference" ]; then
        reference_dir="${dataset_dir}/reference"
    fi
    if [ ! -d "${reference_dir}" ] && [ -d "${dataset_dir}/groundtruth" ]; then
        reference_dir="${dataset_dir}/groundtruth"
        echo "[PAIRED BENCHMARK] ${dataset}: using groundtruth as the clean reference pool"
    fi
    if [ ! -d "${reference_dir}" ] && [ -d "${dataset_dir}/test/clean" ]; then
        reference_dir="${dataset_dir}/test/clean"
        echo "[PAIRED BENCHMARK] ${dataset}: using test/clean as the clean reference pool"
    fi
    if [ ! -d "${reference_dir}" ]; then
        echo "[ERROR] Missing screened reference directory for ${dataset}" >&2
        echo "        Expected ${REFERENCE_ROOT}/${dataset} or ${dataset_dir}/reference" >&2
        exit 1
    fi

    output_dir="${RESULT_ROOT}/${dataset}"
    mkdir -p "${output_dir}"
    echo "[RUN] ${dataset}: ${input_dir} -> ${output_dir}"
    "${PYTHON_BIN}" "${PROJECT_DIR}/infer.py" \
        --input_dir "${input_dir}" \
        --output_dir "${output_dir}" \
        --model "${MODEL_PATH}" \
        --reference "${reference_dir}" \
        --num_reference "${NUM_REFERENCE}" \
        --reference_mode fused \
        --reference_preprocess block_mean \
        --struct_block_size "${STRUCT_BLOCK_SIZE}" \
        --reference_selection uniform \
        --reference_seed 42 \
        --image_size 1024 \
        --inference_size 1024 \
        --overlap_ratio 0.25 \
        --no_z_context --no_z_postprocess --no_preview \
        --output_format png --disable_progress_bar
    processed=$((processed + 1))
done

if [ "${processed}" -eq 0 ]; then
    echo "[ERROR] No dataset was processed" >&2
    exit 1
fi
echo "[DONE] ${processed} dataset(s): ${RESULT_ROOT}"
