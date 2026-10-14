#!/usr/bin/env bash
# Stack B: tau2-telecom profile on the pinned eval Gym runtime (cb588bb1), against an already
# running vLLM (stack A). Run inside the allocation container via interactive.sh exec.
# Usage: launch_tau2_profile.sh --out DIR [--agents tau2_hf399user,tau2_qwen,tau2_gptoss] [--concurrency 32] [--deadline-epoch N]
set -Eeuo pipefail; export TZ=UTC
E=/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_eval; P=${NVC_RL_ROOT:?export NVC_RL_ROOT=<your workspace, see voicechat/README.md>}
RT=$E/.runtime/gym-cs; GYM=$RT/cb588bb1; PY=$RT/venv/bin/python
OUT=; AGENTS=tau2_hf399user,tau2_qwen,tau2_gptoss; CONC=32; DEADLINE=; MODEL_URL=
while (( $# )); do case "$1" in --out) OUT=$2; shift 2;; --agents) AGENTS=$2; shift 2;; --concurrency) CONC=$2; shift 2;; --deadline-epoch) DEADLINE=$2; shift 2;; --model-url) MODEL_URL=$2; shift 2;; *) echo "bad arg $1" >&2; exit 2;; esac; done
[[ -n "$OUT" ]] || { echo "--out required" >&2; exit 2; }
JOB=${SLURM_JOB_ID:?}; mkdir -p "$OUT/logs"; LOG=$OUT/logs/stackB-$JOB.log; exec > >(tee -a "$LOG") 2>&1
log(){ printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
export PATH="$RT/venv/bin:$PATH" UV_CACHE_DIR=$RT/uv-cache UV_PYTHON_INSTALL_DIR=$RT/python UV_OFFLINE=1
export RAY_ADDRESS=local UV_LINK_MODE=copy RAY_OVERRIDE_RESOURCES='{"CPU": 8, "GPU": 0, "memory": 17179869184, "object_store_memory": 1073741824}' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=3 NEMO_GYM_TAU2_BENCH_REF=60c2a0dbf974ea7533456a4706f837c3a6d14afc NEMO_GYM_TAU2_BENCH_DATA_REF=7245a8f6f12a6046e3443b120428dcb695fe1f6e
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled RAY_USAGE_STATS_ENABLED=0
NVIDIA_API_KEY=$(cat "$P/.secrets/nvidia-api-key"); export NVIDIA_API_KEY
local=/tmp/nvB-$JOB; mkdir -p "$local/t" "$local/r"; export TMPDIR=$local/t RAY_TMPDIR=$local/r
"$PY" -c "import nemo_gym; print('root venv ok', nemo_gym.__file__)" || { log "root venv import failed"; exit 3; }
"$GYM/responses_api_agents/tau2/.venv/bin/python" -c "import tau2, rank_bm25" || { log "tau2 component venv import failed"; exit 3; }
IFS=, read -r -a agents <<<"$AGENTS"
for agent in "${agents[@]}"; do
  d=$OUT/$agent; [[ -f "$d/config.json" ]] || { log "missing $d/config.json (run stackB_tau2_config.py)"; exit 2; }
  if [[ -n "$MODEL_URL" ]]; then "$PY" - "$d/config.json" "$MODEL_URL" <<'PYEOF'
import json,sys; p,u=sys.argv[1],sys.argv[2]; c=json.load(open(p)); c['policy_base_url']=u; json.dump(c,open(p,'w'),indent=1)
PYEOF
  fi
  curl --silent --fail --max-time 10 -H 'Authorization: Bearer dummy' "$("$PY" -c "import json,sys;print(json.load(open('$d/config.json'))['policy_base_url'])")/models" >/dev/null || { log "policy endpoint not reachable for $agent"; exit 4; }
  if [[ -n "$DEADLINE" ]] && (( DEADLINE - $(date +%s) < 1200 )); then log "SKIP $agent: <20 min left"; continue; fi
  budget=0; [[ -n "$DEADLINE" ]] && budget=$(( DEADLINE - $(date +%s) - 600 ))
  export NEMO_GYM_RUN_ID=profile-$JOB-$agent
  log "RUN $agent conc=$CONC budget=${budget}s"; start=$(date +%s)
  set +e
  if (( budget > 0 )); then timeout --signal=INT --kill-after=60 "$budget" gym eval run --config "$d/config.json" --split train --output "$d/rollouts.jsonl" --concurrency "$CONC" --resume > "$d/eval-run-$JOB.log" 2>&1
  else gym eval run --config "$d/config.json" --split train --output "$d/rollouts.jsonl" --concurrency "$CONC" --resume > "$d/eval-run-$JOB.log" 2>&1; fi
  code=$?; set -e; end=$(date +%s)
  n=$( [[ -f "$d/rollouts.jsonl" ]] && wc -l < "$d/rollouts.jsonl" || echo 0 ); nf=$( [[ -f "$d/rollouts_failures.jsonl" ]] && wc -l < "$d/rollouts_failures.jsonl" || echo 0 )
  printf '{"agent":"%s","seconds":%s,"exit_code":%s,"rollout_rows":%s,"failure_rows":%s}\n' "$agent" "$((end-start))" "$code" "$n" "$nf" >> "$OUT/timing.jsonl"
  log "DONE $agent exit=$code seconds=$((end-start)) rollouts=$n failures=$nf"; (( code == 0 )) || tail -20 "$d/eval-run-$JOB.log"
done
log "stackB finished"
