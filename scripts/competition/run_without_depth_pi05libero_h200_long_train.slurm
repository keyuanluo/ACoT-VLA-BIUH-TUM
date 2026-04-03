#!/usr/bin/env bash
#SBATCH -J acot-wod-h200
#SBATCH -p compute
#SBATCH -t 1-00:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-nvidia-h100,gpu-nvidia-h200-3
#SBATCH --gres=gpu:nvidia:2
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -o /storage/home/yiduo/slurm-acot-wod-h200-%j.out
#SBATCH -e /storage/home/yiduo/slurm-acot-wod-h200-%j.err

set -euo pipefail

echo "[INFO] start: $(date --iso-8601=seconds)"
echo "[INFO] host: $(hostname)"
echo "[INFO] job_id: ${SLURM_JOB_ID:-N/A}"

source /storage/home/yiduo/miniforge3/etc/profile.d/conda.sh
conda activate acotvla

export LD_LIBRARY_PATH="/storage/home/yiduo/miniforge3/envs/acotvla/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

cd /storage/home/yiduo/ACoT-VLA

DATA_ROOT="/storage/nobackup/yiduo/agibot_challenge_2026/Reasoning2Action-Sim/dataset_without_depth"
ASSETS_ROOT="/storage/home/yiduo/ACoT-VLA/assets/competition"
PYTHON_BIN="/storage/home/yiduo/ACoT-VLA/.venv/bin/python"

TASKS=(
  clean_the_desktop_addition
  clean_the_desktop_part_1
  clean_the_desktop_part_2
  hold_pot
  open_door
  place_block_into_box
  pour_workpiece
  scoop_popcorn
  scoop_popcorn_part_2
  sorting_packages_part_1
  sorting_packages_part_2
)
WEIGHT_PATH="${WEIGHT_PATH:-/storage/nobackup/yiduo/openpi_cache/openpi-assets/checkpoints/pi05_libero/pi05_libero/params}"

for task in "${TASKS[@]}"; do
  task_dir="${DATA_ROOT}/${task}"
  echo "[INFO] preparing ${task_dir}"
  for part in data meta videos; do
    archive="${task_dir}/${part}.tar.gz.000"
    out_dir="${task_dir}/${part}"
    if [ -f "${archive}" ] && [ ! -d "${out_dir}" ]; then
      echo "[INFO] extracting ${archive}"
      tar -xzf "${archive}" -C "${task_dir}"
    fi
  done

  norm_stats="${ASSETS_ROOT}/${task}/norm_stats.json"
  if [ ! -f "${norm_stats}" ]; then
    echo "[INFO] computing norm stats for ${task}"
    "${PYTHON_BIN}" scripts/competition/compute_go2_acot_norm_stats.py       --dataset-root "${task_dir}"       --output-dir "${ASSETS_ROOT}/${task}"
  else
    echo "[INFO] reusing ${norm_stats}"
  fi
done

EXP_NAME="${EXP_NAME:-without_depth_pi05libero_h200x2_bs32_long}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-24}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_SPLIT_RATIO="${VAL_SPLIT_RATIO:-0.01}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-64}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
FSDP_DEVICES="${FSDP_DEVICES:-2}"
OVERWRITE="${OVERWRITE:-1}"
RESUME="${RESUME:-0}"

echo "[INFO] exp_name=${EXP_NAME}"
echo "[INFO] weight_path=${WEIGHT_PATH}"
echo "[INFO] num_train_steps=${NUM_TRAIN_STEPS}"
echo "[INFO] batch_size=${BATCH_SIZE}"
echo "[INFO] num_workers=${NUM_WORKERS}"
echo "[INFO] val_split_ratio=${VAL_SPLIT_RATIO}"
echo "[INFO] val_interval=${VAL_INTERVAL}"
echo "[INFO] val_num_batches=${VAL_NUM_BATCHES}"
echo "[INFO] save_interval=${SAVE_INTERVAL}"
echo "[INFO] keep_period=${KEEP_PERIOD}"
echo "[INFO] fsdp_devices=${FSDP_DEVICES}"
echo "[INFO] overwrite=${OVERWRITE}"
echo "[INFO] resume=${RESUME}"

train_cmd=(
  "${PYTHON_BIN}" scripts/competition/train_without_depth_local.py
  --exp-name "${EXP_NAME}"
  --weight-path "${WEIGHT_PATH}"
  --num-train-steps "${NUM_TRAIN_STEPS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --log-interval "${LOG_INTERVAL}"
  --val-split-ratio "${VAL_SPLIT_RATIO}"
  --val-interval "${VAL_INTERVAL}"
  --val-num-batches "${VAL_NUM_BATCHES}"
  --save-interval "${SAVE_INTERVAL}"
  --keep-period "${KEEP_PERIOD}"
  --fsdp-devices "${FSDP_DEVICES}"
)

if [ "${OVERWRITE}" = "1" ]; then
  train_cmd+=(--overwrite)
fi
if [ "${RESUME}" = "1" ]; then
  train_cmd+=(--resume)
fi

"${train_cmd[@]}"

echo "[INFO] finished: $(date --iso-8601=seconds)"
