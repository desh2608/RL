#!/usr/bin/env bash
# One 8xA100 node: Qwen3.8-27B (TP2xDP4, analysis/qwen-teacher-20261005/serve_qwen.sh) + the BFCL-style multi-turn
# pipeline in scripts/bfcl_mt/: gen_tasks -> make_variants -> rollout -> to_sft. A smoke pass (OUT/smoke, few tasks,
# every stage) runs first; the full pass runs only if the smoke produced SFT records.
# Usage: sbatch --account A --partition P --time 04:00:00 scripts/bfcl_mt/pipeline.sbatch OUT_DIR N_TASKS [K] [START]
#SBATCH --job-name=nvc-bfcl-mt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --no-requeue
set -Eeuo pipefail
export TZ=UTC
P=${NVC_RL_ROOT:?export NVC_RL_ROOT=<your workspace, see voicechat/README.md>}
export MT_OUT=${1:?out dir} MT_N=${2:?n tasks} MT_K=${3:-1} MT_START=${4:-0}
export MT_END=$(( $(date +%s) + $(squeue -h -j "$SLURM_JOB_ID" -o %L </dev/null | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1); if(n>2) s+=3600*$(n-2); if(n>3) s+=86400*$(n-3); print s}') ))
image=$P/runtime/gym-v1/NemotronVC-Gym34-vllm0271-v1.sqsh
mkdir -p "$MT_OUT" "$P/logs/qwen"
printf '[MT_JOB] job=%s node=%s out=%s n=%s k=%s end=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$MT_OUT" "$MT_N" "$MT_K" "$(date -u -d @$MT_END +%H:%M:%SZ)"
srun -N1 -n1 --cpus-per-task=64 --gpus-per-task=8 --mem=0 --gpu-bind=none --cpu-bind=none --mpi=none \
 --no-container-mount-home --container-image="$image" \
 --container-mounts="/lustre:/lustre,/raid/scratch:/raid/scratch,$P:/workspace/debug" \
 --container-workdir=/workspace/debug bash -c '
  set -Eeuo pipefail
  P=${NVC_RL_ROOT:?export NVC_RL_ROOT=<your workspace, see voicechat/README.md>}; S=$P/scripts/bfcl_mt
  PY=/opt/nemotronvc/venvs/vllm027/bin/python; LQ=$P/logs/qwen/serve-bfclmt-$SLURM_JOB_ID.log
  $PY -c "import openai, mpmath; import sys; sys.path.insert(0, \"$S\"); import common" || { echo "python deps missing"; exit 2; }
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_MODEL_LEN=65536 MAX_NUM_SEQS=96 bash $P/analysis/qwen-teacher-20261005/serve_qwen.sh 2 4 18200 "$LQ" & qwen=$!
  ok=0
  for i in $(seq 1 180); do
    kill -0 $qwen 2>/dev/null || { echo "Qwen server died"; tail -40 "$LQ"; exit 1; }
    curl -sf http://127.0.0.1:18200/v1/models >/dev/null && { ok=1; break; }
    sleep 10
  done
  (( ok )) || { echo "server not ready in 30 min"; tail -40 "$LQ"; exit 1; }
  echo "[MT_JOB] server ready $(date -u +%H:%M:%SZ)"
  stage() {  # out n start gen_deadline roll_deadline
    local o=$1 n=$2 st=$3 gd=$4 rd=$5
    mkdir -p "$o"
    $PY -u $S/gen_tasks.py --out "$o" --n "$n" --start "$st" --workers 64 --deadline "$gd" 2>&1 | tee -a "$o/gen.log"
    $PY -u $S/make_variants.py --tasks "$o/tasks.jsonl" --out "$o" 2>&1 | tee -a "$o/variants.log"
    $PY -u $S/rollout.py --entries "$o/entries.jsonl" --answers "$o/answers.jsonl" --out "$o" --k "$MT_K" --workers 128 \
      --deadline "$rd" 2>&1 | tee -a "$o/rollout.log"
    $PY -u $S/to_sft.py --rollouts "$o/rollouts.jsonl" --out "$o/sft" 2>&1 | tee -a "$o/sft.log"
  }
  set +e
  now=$(date +%s)
  stage "$MT_OUT/smoke" 6 900000 $((now + 1200)) $((now + 2400))
  n=$(wc -l < "$MT_OUT/smoke/sft/perturn.jsonl" 2>/dev/null || echo 0)
  echo "[MT_JOB] smoke records=$n $(date -u +%H:%M:%SZ)"
  if (( n > 0 )); then
    now=$(date +%s); left=$(( MT_END - now - 900 ))
    stage "$MT_OUT" "$MT_N" "$MT_START" $(( now + left * 45 / 100 )) $(( MT_END - 1200 ))
  fi
  code=$?
  kill -TERM $qwen 2>/dev/null
  exit $code
 ' </dev/null
echo "[MT_JOB] done $(date -u +%H:%M:%SZ)"
