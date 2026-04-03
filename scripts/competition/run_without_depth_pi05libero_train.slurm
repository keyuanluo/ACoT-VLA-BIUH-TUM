#!/usr/bin/env bash
#SBATCH -J acot-wod
#SBATCH -p compute
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:nvidia:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -o /storage/home/yiduo/slurm-acot-wod-%j.out
#SBATCH -e /storage/home/yiduo/slurm-acot-wod-%j.err

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

EXP_NAME="${EXP_NAME:-without_depth_pi05libero_pilot_job${SLURM_JOB_ID}}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

"${PYTHON_BIN}" scripts/competition/train_without_depth_local.py \
  --exp-name "${EXP_NAME}" \
  --weight-path "${WEIGHT_PATH}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --log-interval "${LOG_INTERVAL}" \
  --save-interval "${SAVE_INTERVAL}" \
  --overwrite

echo "[INFO] finished: $(date --iso-8601=seconds)"
