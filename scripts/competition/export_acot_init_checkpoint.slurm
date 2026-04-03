#!/bin/bash
#SBATCH -J acot-step0
#SBATCH -p compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /storage/home/yiduo/slurm-acot-step0-%j.out
#SBATCH -e /storage/home/yiduo/slurm-acot-step0-%j.err

set -euo pipefail

CONFIG_NAME=${CONFIG_NAME:?CONFIG_NAME is required}
EXP_NAME=${EXP_NAME:?EXP_NAME is required}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-/storage/nobackup/yiduo/acotvla_competition_smoke_checkpoints}
STEP=${STEP:-0}
OVERWRITE=${OVERWRITE:-false}

source /storage/home/yiduo/miniforge3/etc/profile.d/conda.sh
conda activate acotvla
cd /storage/home/yiduo/ACoT-VLA

export TF_NUM_INTRAOP_THREADS=8
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TOKENIZERS_PARALLELISM=false

ARGS=(
  --args.config-name "$CONFIG_NAME"
  --args.exp-name "$EXP_NAME"
  --args.checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
  --args.step "$STEP"
)

if [[ "$OVERWRITE" == "true" ]]; then
  ARGS+=(--args.overwrite)
fi

uv run python scripts/competition/export_acot_init_checkpoint.py "${ARGS[@]}"
