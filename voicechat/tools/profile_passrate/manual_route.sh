#!/usr/bin/env bash
# Run ONE extra route against the already-running stack A servers (inside the allocation container).
# Usage: manual_route.sh --out DIR --agent NAME --input rows.jsonl --k 8 --conc 64 --cap 8192 --vllm-port P --gym-head-port P --gym-low P --gym-high P --ray ADDR --deadline-epoch N
set -Eeuo pipefail; export TZ=UTC
P=${NVC_RL_ROOT:?export NVC_RL_ROOT=<your workspace, see voicechat/README.md>}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NEMO_RL_ROOT=/workspace/nemo-rl; GYM_ROOT=$NEMO_RL_ROOT/3rdparty/Gym-workspace/Gym; VENV_ROOT=${NVC_VENV_ROOT:-/opt/nemotronvc/gym-v1/venvs}
GYM_BIN=/opt/nemotronvc/venvs/nemo-gym/bin/gym; SERVICES_YAML=${NVC_PROFILE_SERVICES_YAML:-$HERE/gym_profile_services.yaml}
OUT=; AGENT=; INPUT=; K=8; CONC=64; CAP=8192; VP=; HP=; GL=; GH=; RAY=; DEADLINE=; RESUME=
while (( $# )); do case "$1" in --out) OUT=$2;; --agent) AGENT=$2;; --input) INPUT=$2;; --k) K=$2;; --conc) CONC=$2;; --cap) CAP=$2;; --vllm-port) VP=$2;; --gym-head-port) HP=$2;; --gym-low) GL=$2;; --gym-high) GH=$2;; --ray) RAY=$2;; --deadline-epoch) DEADLINE=$2;; --resume) RESUME=--resume; shift; continue;; *) echo "bad arg $1" >&2; exit 2;; esac; shift 2; done
JOB=${SLURM_JOB_ID:?}; mkdir -p "$OUT"; LOG=$OUT/manual-$JOB.log; exec > >(tee -a "$LOG") 2>&1
log(){ printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
SCRATCH=/raid/scratch/$USER/nvcrl-$JOB/profile; export HF_HOME=$SCRATCH/hf-home HF_MODULES_CACHE=$SCRATCH/hf-home/modules XDG_CACHE_HOME=$SCRATCH/xdg TMPDIR=/tmp/nvcp-$JOB
if [[ -f /opt/nemotronvc/runtime-env.sh ]]; then set +e; source /opt/nemotronvc/runtime-env.sh; set -Eeuo pipefail; fi
export PATH="/opt/nemotronvc/bin:/opt/nemotronvc/venvs/driver/bin:$PATH" PYTHONPATH="$HF_MODULES_CACHE:$NEMO_RL_ROOT:$GYM_ROOT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled RAY_USAGE_STATS_ENABLED=0 RAY_DEDUP_LOGS=0
export NEMO_GYM_EXTRA_ROOTS=$GYM_ROOT NEMO_GYM_VENV_DIR=$VENV_ROOT UV_OFFLINE=1 UV_CACHE_DIR=$SCRATCH/gym-cache/uv RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor
unset NVIDIA_API_KEY OPENAI_API_KEY HF_TOKEN
if [[ -n ${NVC_HUB_KEY_FILE:-} ]]; then IFS= read -r NVIDIA_API_KEY < "$NVC_HUB_KEY_FILE"; export NVIDIA_API_KEY; fi
COMMON=(--config "$SERVICES_YAML" --model hf399 --model-url "[\"http://127.0.0.1:$VP/v1\"]" --model-api-key dummy
  ++head_server.host=127.0.0.1 ++head_server.port="$HP" ++port_range_low="$GL" ++port_range_high="$GH" ++ray_head_node_address="$RAY"
  ++skip_venv_if_present=true ++uv_venv_dir="$VENV_ROOT" ++uv_cache_dir="$SCRATCH/gym-cache/uv" ++results_dir="$OUT/gym-results" ++cache_dir="$SCRATCH/gym-cache"
  ++nemo_gym_log_dir="$OUT/gym-servers" ++model_endpoint_readiness_timeout_seconds=600 ++allow_openai_version_skew=true)
budget=0; [[ -n "$DEADLINE" ]] && budget=$(( DEADLINE - $(date +%s) - 600 ))
n=$(wc -l < "$INPUT"); log "RUN $AGENT n=$n k=$K cap=$CAP conc=$CONC budget=${budget}s"; start=$(date +%s); cd "$GYM_ROOT"
set +e
if (( budget > 0 )); then timeout --signal=INT --kill-after=60 "$budget" "$GYM_BIN" eval run --no-serve "${COMMON[@]}" --agent "$AGENT" --input "$INPUT" --output "$OUT/rollouts.jsonl" --num-repeats "$K" --temperature 1 --top-p 1 --max-output-tokens "$CAP" --concurrency "$CONC" ++num_repeats_add_seed=true ++upload_rollouts_to_wandb=false $RESUME > "$OUT/eval-run-$JOB.log" 2>&1
else "$GYM_BIN" eval run --no-serve "${COMMON[@]}" --agent "$AGENT" --input "$INPUT" --output "$OUT/rollouts.jsonl" --num-repeats "$K" --temperature 1 --top-p 1 --max-output-tokens "$CAP" --concurrency "$CONC" ++num_repeats_add_seed=true ++upload_rollouts_to_wandb=false $RESUME > "$OUT/eval-run-$JOB.log" 2>&1; fi
code=$?; set -e; end=$(date +%s)
printf '{"route_key":"%s","agent":"%s","start_epoch":%s,"end_epoch":%s,"seconds":%s,"exit_code":%s,"n_prompts":%s,"k":%s,"expected_rollouts":%s,"rollout_rows":%s,"failure_rows":%s,"max_output_tokens":%s,"concurrency":%s}\n' "$(basename $OUT)" "$AGENT" "$start" "$end" "$((end-start))" "$code" "$n" "$K" "$((n*K))" "$( [[ -f $OUT/rollouts.jsonl ]] && wc -l < $OUT/rollouts.jsonl || echo 0)" "$( [[ -f $OUT/rollouts_failures.jsonl ]] && wc -l < $OUT/rollouts_failures.jsonl || echo 0)" "$CAP" "$CONC" >> "$OUT/timing.jsonl"
log "DONE $AGENT exit=$code seconds=$((end-start)) rollouts=$( [[ -f $OUT/rollouts.jsonl ]] && wc -l < $OUT/rollouts.jsonl || echo 0)"
