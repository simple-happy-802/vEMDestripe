#!/bin/bash
# Submit generation -> training -> inference -> evaluation with Slurm dependencies.

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
RUN_NAME=${RUN_NAME:-vemdestripe_retrain_seed42}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints/${RUN_NAME}}
# The dependent inference job must consume the checkpoint produced by this
# training job, not the pretrained paper checkpoint bundled with the release.
MODEL_PATH=${MODEL_PATH:-${CHECKPOINT_DIR}/best.pth}
export PROJECT_DIR RUN_NAME CHECKPOINT_DIR MODEL_PATH
cd "${PROJECT_DIR}"
mkdir -p logs

if [ "${SKIP_DATA:-false}" = true ]; then
    train_job=$(sbatch --parsable scripts/02_train.sbatch)
else
    data_job=$(sbatch --parsable scripts/01_generate_data.sbatch)
    train_job=$(sbatch --parsable --dependency="afterok:${data_job}" scripts/02_train.sbatch)
    echo "Data job:       ${data_job}"
fi
infer_job=$(sbatch --parsable --dependency="afterok:${train_job}" scripts/03_infer.sbatch)
eval_job=$(sbatch --parsable --dependency="afterok:${infer_job}" scripts/04_evaluate.sbatch)

echo "Training job:   ${train_job}"
echo "Inference job:  ${infer_job}"
echo "Evaluation job: ${eval_job}"
echo "Monitor with: squeue -u \"${USER}\""
