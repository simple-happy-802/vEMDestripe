#!/bin/bash
# Train the fixed vEMDestripe method with two GPUs and single-node DDP.

#SBATCH -J vem_train
#SBATCH -p a100
#SBATCH --qos=a100g2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=72:00:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data/training}
RUN_NAME=${RUN_NAME:-vemdestripe_retrain_seed42}

CHECKPOINT_DIR=${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints/${RUN_NAME}}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${RUN_NAME}}
TB_DIR=${TB_DIR:-${PROJECT_DIR}/logs/tensorboard/${RUN_NAME}}

IMAGE_SIZE=${IMAGE_SIZE:-1024}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-2}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
EPOCHS=${EPOCHS:-200}
LR=${LR:-0.0002}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
SEED=${SEED:-42}
AUTO_RESUME=${AUTO_RESUME:-true}

mkdir -p "${PROJECT_DIR}/logs" "${CHECKPOINT_DIR}" "${OUTPUT_DIR}" "${TB_DIR}"
cd "${PROJECT_DIR}"
test -f "${DATA_DIR}/dataset_info.json" || {
    echo "[ERROR] Missing generated dataset: ${DATA_DIR}/dataset_info.json" >&2
    exit 1
}
FIRST_RAW_REFERENCE=$(find "${DATA_DIR}" -type f -path '*/reference/*_ref_*' -print -quit)
if [ -z "${FIRST_RAW_REFERENCE}" ]; then
    echo "[ERROR] No individual split references found under ${DATA_DIR}" >&2
    echo "        The 16x16 paper model must be trained from *_ref_* images, not the stored 32x32 fused PNG." >&2
    exit 1
fi

case "${AUTO_RESUME}" in
    true|false) ;;
    *) echo "[ERROR] AUTO_RESUME must be true or false" >&2; exit 2 ;;
esac

RESUME_PATH=
if [ "${AUTO_RESUME}" = true ] && [ -f "${CHECKPOINT_DIR}/latest.pth" ]; then
    RESUME_PATH="${CHECKPOINT_DIR}/latest.pth"
elif [ "${AUTO_RESUME}" = true ] && [ -f "${CHECKPOINT_DIR}/best.pth" ]; then
    RESUME_PATH="${CHECKPOINT_DIR}/best.pth"
elif [ "${AUTO_RESUME}" = false ] && { [ -f "${CHECKPOINT_DIR}/latest.pth" ] || [ -f "${CHECKPOINT_DIR}/best.pth" ]; }; then
    echo "[ERROR] Checkpoint already exists in ${CHECKPOINT_DIR}" >&2
    echo "        Set AUTO_RESUME=true or choose a new RUN_NAME." >&2
    exit 1
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.device_count()==2, torch.cuda.device_count()'
CMD=(
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=2
    "${PROJECT_DIR}/train.py"
    --data_dir "${DATA_DIR}"
    --image_size "${IMAGE_SIZE}"
    --batch_size "${BATCH_SIZE_PER_GPU}"
    --accumulation_steps "${ACCUMULATION_STEPS}"
    --num_workers "${NUM_WORKERS}"
    --epochs "${EPOCHS}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --seed "${SEED}"
    --save_freq 10
    --checkpoint_dir "${CHECKPOINT_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --log_dir "${TB_DIR}"
    --lambda_l1 1.0
    --lambda_edge 0.1
    --lambda_photo 0.3
    --lambda_ssim 0.1
    --photo_std_weight 0.5
    --local_window_sizes 32,64,128
    --use_fused_reference
    --reference_mode fused
    --reference_preprocess block_mean
    --reference_block_size 16
    --amp --disable_progress_bar
)
if [ -n "${RESUME_PATH}" ]; then CMD+=(--resume "${RESUME_PATH}"); fi
printf '[RUN]'; printf ' %q' "${CMD[@]}"; printf '\n'
"${CMD[@]}"

echo "[DONE] Checkpoint: ${CHECKPOINT_DIR}/best.pth"
