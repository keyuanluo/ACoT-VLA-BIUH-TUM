#!/usr/bin/env bash
#SBATCH -J acot-depth-dry
#SBATCH -p compute
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-nvidia-h100,gpu-nvidia-h200-3
#SBATCH --gres=gpu:nvidia:1
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -o /storage/home/yiduo/slurm-acot-depth-dry-%j.out
#SBATCH -e /storage/home/yiduo/slurm-acot-depth-dry-%j.err

set -euo pipefail

echo "[INFO] start: $(date --iso-8601=seconds)"
echo "[INFO] host: $(hostname)"
echo "[INFO] job_id: ${SLURM_JOB_ID:-N/A}"

source /storage/home/yiduo/miniforge3/etc/profile.d/conda.sh
conda activate acotvla

export LD_LIBRARY_PATH="/storage/home/yiduo/miniforge3/envs/acotvla/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/storage/home/yiduo/ACoT-VLA-depth/src:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

cd /storage/home/yiduo/ACoT-VLA-depth

PYTHON_BIN="/storage/home/yiduo/ACoT-VLA-depth/.venv/bin/python"
CONFIG_NAME="${CONFIG_NAME:-acot_icra_simulation_challenge_reasoning_to_action_clean_depth_pi05libero_continue_45000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
DO_TRAIN_STEP="${DO_TRAIN_STEP:-0}"

echo "[INFO] config_name=${CONFIG_NAME}"
echo "[INFO] per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
echo "[INFO] num_workers=${NUM_WORKERS}"
echo "[INFO] fsdp_devices=${FSDP_DEVICES}"
echo "[INFO] do_train_step=${DO_TRAIN_STEP}"

dry_run_cmd=(
  "${PYTHON_BIN}" scripts/competition/validate_acot_depth_dry_run.py
  --config-name "${CONFIG_NAME}"
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --fsdp-devices "${FSDP_DEVICES}"
  --init-state
)

if [ "${DO_TRAIN_STEP}" = "1" ]; then
  dry_run_cmd+=(--train-step)
fi

"${dry_run_cmd[@]}"

echo "[INFO] finished: $(date --iso-8601=seconds)"
