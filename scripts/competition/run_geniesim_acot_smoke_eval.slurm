#!/usr/bin/env bash
#SBATCH -J geniesim-smoke
#SBATCH -p compute
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:nvidia:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -o /storage/home/yiduo/slurm-geniesim-smoke-%j.out
#SBATCH -e /storage/home/yiduo/slurm-geniesim-smoke-%j.err

set -euo pipefail

PORT="${PORT:-8999}"
SERVER_GPU="${SERVER_GPU:-1}"
SIM_GPU="${SIM_GPU:-0}"
POLICY_CONFIG="${POLICY_CONFIG:-acot_icra_simulation_challenge_reasoning_to_action_clean_pi05_libero_smoke}"
POLICY_DIR="${POLICY_DIR:-/storage/nobackup/yiduo/acotvla_competition_smoke_checkpoints/acot_icra_simulation_challenge_reasoning_to_action_clean_pi05_libero_smoke/clean_pi05_libero_step0/0}"
GENIESIM_REPO_HOST="${GENIESIM_REPO_HOST:-/storage/home/yiduo/embodied_ai/genie_sim}"
ISAAC_SIF="${ISAAC_SIF:-/storage/nobackup/yiduo/containers/genie_sim_open_source.sif}"
GENIE_CONFIG_IN="${GENIE_CONFIG_IN:-/geniesim/main/source/geniesim/config/icra_clean_the_desktop.yaml}"
EXP_NAME="${EXP_NAME:-geniesim_clean_pi05libero_step0_smoke_official}"
RESULT_ROOT="${RESULT_ROOT:-/storage/home/yiduo/ACoT-VLA/logs/${EXP_NAME}_job${SLURM_JOB_ID}}"
RESULTS_FILE="${RESULTS_FILE:-/storage/home/yiduo/ACoT-VLA/logs/geniesim_smoke_results.tsv}"
BENCHMARK_DIR="${BENCHMARK_DIR:-${GENIESIM_REPO_HOST}/output/benchmark}"
BENCHMARK_BACKUP="${GENIESIM_REPO_HOST}/output/benchmark_pre_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="${RESULT_ROOT}/policy_server.log"
SIM_LOG="${RESULT_ROOT}/geniesim.log"
SUMMARY_TXT="${RESULT_ROOT}/summary.txt"
SUMMARY_JSON="${RESULT_ROOT}/summary.json"

mkdir -p "${RESULT_ROOT}"
mkdir -p "$(dirname "${RESULTS_FILE}")"

if [ -d "${BENCHMARK_DIR}" ]; then
  mv "${BENCHMARK_DIR}" "${BENCHMARK_BACKUP}"
fi

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd /storage/home/yiduo/ACoT-VLA
source /storage/home/yiduo/miniforge3/etc/profile.d/conda.sh
conda activate acotvla

{
  echo "[INFO] Official Genie Sim repo: ${GENIESIM_REPO_HOST}"
  echo "[INFO] Official Genie Sim SIF: ${ISAAC_SIF}"
  echo "[INFO] Genie Sim config: ${GENIE_CONFIG_IN}"
} | tee -a "${SUMMARY_TXT}"

CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
  bash /storage/home/yiduo/ACoT-VLA/scripts/competition/server.sh "${SERVER_GPU}" "${PORT}" \
  policy:checkpoint \
  --policy.config "${POLICY_CONFIG}" \
  --policy.dir "${POLICY_DIR}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[INFO] Started policy server PID=${SERVER_PID} on port ${PORT}" | tee -a "${SUMMARY_TXT}"

ready=0
for _ in $(seq 1 90); do
  if (echo > /dev/tcp/127.0.0.1/${PORT}) >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done

if [ "${ready}" != "1" ]; then
  echo "[ERROR] Policy server did not become ready on port ${PORT}" | tee -a "${SUMMARY_TXT}"
  exit 1
fi

echo "[INFO] Policy server is ready" | tee -a "${SUMMARY_TXT}"

CUDA_VISIBLE_DEVICES="${SIM_GPU}" \
  ISAAC_SIF="${ISAAC_SIF}" \
  GENIESIM_REPO_HOST="${GENIESIM_REPO_HOST}" \
  GENIESIM_ISOLATE_CACHE="${GENIESIM_ISOLATE_CACHE:-1}" \
  bash /storage/home/yiduo/embodied_ai/run_geniesim_headless.sh \
  /geniesim/main/source/geniesim/app/app.py \
  --config "${GENIE_CONFIG_IN}" \
  --app.headless true \
  --benchmark.infer_host "localhost:${PORT}" \
  --benchmark.num_episode 1 \
  --benchmark.record false \
  --app.record_video false \
  --app.record_img false \
  >"${SIM_LOG}" 2>&1

BENCHMARK_DIR="${BENCHMARK_DIR}" python3 /storage/home/yiduo/embodied_ai/genie_sim/scripts/stat_average.py "${BENCHMARK_DIR}" >"${SUMMARY_TXT}" || true

BENCHMARK_DIR="${BENCHMARK_DIR}" python3 - <<'INNERPY' > "${SUMMARY_JSON}"
import glob, json, os
benchmark_dir = os.environ['BENCHMARK_DIR']
summary = {
    'benchmark_dir': benchmark_dir,
    'json_files': [],
    'average': None,
}
items = []
for path in sorted(glob.glob(os.path.join(benchmark_dir, '**', '*.json'), recursive=True)):
    summary['json_files'].append(path)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        continue
    stats = data.get('statistics', {})
    avg = stats.get('average')
    if isinstance(avg, (int, float)):
        items.append((path, float(avg)))
if items:
    summary['average'] = sum(v for _, v in items) / len(items)
    summary['per_file'] = [{'path': p, 'average': v} for p, v in items]
print(json.dumps(summary, indent=2))
INNERPY

RESULTS_FILE="${RESULTS_FILE}" \
SUMMARY_JSON="${SUMMARY_JSON}" \
RESULT_ROOT="${RESULT_ROOT}" \
SERVER_LOG="${SERVER_LOG}" \
SIM_LOG="${SIM_LOG}" \
POLICY_CONFIG="${POLICY_CONFIG}" \
POLICY_DIR="${POLICY_DIR}" \
EXP_NAME="${EXP_NAME}" \
BENCHMARK_DIR="${BENCHMARK_DIR}" \
python3 - <<'INNERPY'
import fcntl, json, os
results_file = os.environ['RESULTS_FILE']
summary_json = os.environ['SUMMARY_JSON']
result_root = os.environ['RESULT_ROOT']
server_log = os.environ['SERVER_LOG']
sim_log = os.environ['SIM_LOG']
policy_config = os.environ['POLICY_CONFIG']
policy_dir = os.environ['POLICY_DIR']
exp_name = os.environ['EXP_NAME']
slurm_job_id = os.environ.get('SLURM_JOB_ID', '')
node = os.environ.get('SLURMD_NODENAME', os.environ.get('HOSTNAME', ''))
benchmark_dir = os.environ['BENCHMARK_DIR']
avg = ''
try:
    with open(summary_json, 'r', encoding='utf-8') as f:
        avg_val = json.load(f).get('average')
    avg = '' if avg_val is None else str(avg_val)
except Exception:
    avg = ''
header = 'exp_name\tjob_id\tnode\tpolicy_config\tpolicy_dir\tbenchmark_dir\taverage\tresult_root\tserver_log\tsim_log\n'
line = f'{exp_name}\t{slurm_job_id}\t{node}\t{policy_config}\t{policy_dir}\t{benchmark_dir}\t{avg}\t{result_root}\t{server_log}\t{sim_log}\n'
os.makedirs(os.path.dirname(results_file), exist_ok=True)
with open(results_file, 'a+', encoding='utf-8') as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    f.seek(0, os.SEEK_SET)
    first = f.read(1)
    if not first:
        f.write(header)
    f.seek(0, os.SEEK_END)
    f.write(line)
    f.flush()
    os.fsync(f.fileno())
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
INNERPY

echo "[INFO] Smoke evaluation finished. Summary: ${SUMMARY_JSON}" | tee -a "${SUMMARY_TXT}"
