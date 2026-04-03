#!/usr/bin/env bash
# Resume the depth continuation run from the latest checkpoint in the same experiment directory.
# This script is safe to submit repeatedly; training will continue toward NUM_TRAIN_STEPS.
#SBATCH -J acot-depth-rsm
#SBATCH -p compute
#SBATCH -t 1-00:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-nvidia-h100
#SBATCH --exclude=gpu-nvidia-h200-3
#SBATCH --gres=gpu:nvidia:1
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -o /storage/home/yiduo/slurm-acot-depth-resume-%j.out
#SBATCH -e /storage/home/yiduo/slurm-acot-depth-resume-%j.err

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
EXP_NAME="${EXP_NAME:-depth_pi05libero_continue45000_h100x1_bs24_s50000}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/storage/nobackup/yiduo/acotvla_depth_checkpoints}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
VAL_SPLIT_RATIO="${VAL_SPLIT_RATIO:-0.0}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-1}"

LOSS_PLOT_PATH="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}/metrics/loss.png"
TRAIN_METRICS_PATH="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}/metrics/train_metrics.csv"

echo "[INFO] config_name=${CONFIG_NAME}"
echo "[INFO] exp_name=${EXP_NAME}"
echo "[INFO] checkpoint_base_dir=${CHECKPOINT_BASE_DIR}"
echo "[INFO] num_train_steps=${NUM_TRAIN_STEPS}"
echo "[INFO] batch_size=${BATCH_SIZE}"
echo "[INFO] num_workers=${NUM_WORKERS}"
echo "[INFO] log_interval=${LOG_INTERVAL}"
echo "[INFO] save_interval=${SAVE_INTERVAL}"
echo "[INFO] keep_period=${KEEP_PERIOD}"
echo "[INFO] fsdp_devices=${FSDP_DEVICES}"
echo "[INFO] val_split_ratio=${VAL_SPLIT_RATIO}"
echo "[INFO] wandb_enabled=${WANDB_ENABLED}"
echo "[INFO] overwrite=${OVERWRITE}"
echo "[INFO] resume=${RESUME}"
echo "[INFO] loss_plot=${LOSS_PLOT_PATH}"
echo "[INFO] train_metrics_csv=${TRAIN_METRICS_PATH}"

train_cmd=(
  "${PYTHON_BIN}" scripts/competition/train_depth_local.py
  --config-name "${CONFIG_NAME}"
  --exp-name "${EXP_NAME}"
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
  --num-train-steps "${NUM_TRAIN_STEPS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --log-interval "${LOG_INTERVAL}"
  --val-split-ratio "${VAL_SPLIT_RATIO}"
  --save-interval "${SAVE_INTERVAL}"
  --keep-period "${KEEP_PERIOD}"
  --fsdp-devices "${FSDP_DEVICES}"
)

if [ "${WANDB_ENABLED}" = "1" ]; then
  train_cmd+=(--wandb-enabled)
fi
if [ "${OVERWRITE}" = "1" ]; then
  train_cmd+=(--overwrite)
fi
if [ "${RESUME}" = "1" ]; then
  train_cmd+=(--resume)
fi

"${train_cmd[@]}"

echo "[INFO] finished: $(date --iso-8601=seconds)"
