#!/bin/bash
#SBATCH -J r2adl
#SBATCH -p compute
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --export=NONE
#SBATCH -o /storage/home/yiduo/slurm-r2adl-%j.out
#SBATCH -e /storage/home/yiduo/slurm-r2adl-%j.err

set -euo pipefail

echo "[INFO] start: $(date --iso-8601=seconds)"
echo "[INFO] host: $(hostname)"
echo "[INFO] job_id: ${SLURM_JOB_ID:-N/A}"

source /storage/home/yiduo/miniforge3/etc/profile.d/conda.sh
conda activate acotvla

cd /storage/home/yiduo/ACoT-VLA
mkdir -p /storage/nobackup/yiduo/agibot_challenge_2026

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
MAX_WORKERS="${MAX_WORKERS:-1}"

echo "[INFO] target: /storage/nobackup/yiduo/agibot_challenge_2026"
echo "[INFO] checking free space..."
df -h /storage/nobackup | sed -n '1,2p'
echo "[INFO] HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "[INFO] max_workers=${MAX_WORKERS}"

# Re-running the same command resumes and fills missing files.
uv run hf download agibot-world/AgiBotWorldChallenge-2026 \
  --repo-type dataset \
  --include "Reasoning2Action-Sim/**" \
  --local-dir /storage/nobackup/yiduo/agibot_challenge_2026 \
  --max-workers "${MAX_WORKERS}"

echo "[INFO] finished: $(date --iso-8601=seconds)"
du -sh /storage/nobackup/yiduo/agibot_challenge_2026 || true
