#!/usr/bin/env bash
# HF399 pass-rate profile on ONE 8xA100 node: stock `vllm serve` (TP1 DP8 EP8) + NeMo Gym servers
# + `gym eval run --no-serve` per route, then aggregate.py.
#
# Runs INSIDE the project container (runtime/gym-v1/NemotronVC-Gym34-vllm0271-v1.sqsh) on an
# allocation held by `bash scripts/interactive.sh alloc gpu`; start it with `interactive.sh exec`.
# Never calls `ray stop` / `pkill`; only signals the processes it started.
#
# Usage:
#   bash launch_profile.sh --samples <dir from sample_routes.py> --out <profile dir>
#        [--k 8] [--max-num-seqs 48] [--max-model-len 98304] [--gpu-mem-util 0.85]
#        [--routes key1,key2] [--deadline-epoch <unix seconds>] [--assumed-tps 4000]
#        [--resume] [--check-only] [--mamba-cache-dtype float32|float16]
set -Eeuo pipefail
export TZ=UTC

# ----------------------------------------------------------------------------- defaults
P=${NVC_RL_ROOT:?export NVC_RL_ROOT=<your workspace, see voicechat/README.md>}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NEMO_RL_ROOT=${NVC_NEMO_RL_ROOT:-/workspace/nemo-rl}          # interactive.sh mounts $P/src/nemo-rl here
[[ -d "$NEMO_RL_ROOT/nemo_rl" ]] || NEMO_RL_ROOT=$P/src/pipeline/nemo_rl
GYM_ROOT=${NVC_GYM_ROOT:-$NEMO_RL_ROOT/3rdparty/Gym-workspace/Gym}
MODEL_PATH=${NVC_MODEL_PATH:-$P/models/sftkd-s3-c8-lr1e-5-s399/huggingface}
MODEL_LABEL=${NVC_MODEL_LABEL:-hf399}
SERVICES_YAML=${NVC_PROFILE_SERVICES_YAML:-$HERE/gym_profile_services.yaml}
VLLM_BIN=/opt/nemotronvc/venvs/vllm027/bin/vllm                 # vLLM 0.27.1 (scripts/runtime_env.sh NEMO_RL_NODE_VLLM_PYTHON_EXECUTABLE)
VLLM_PY=/opt/nemotronvc/venvs/vllm027/bin/python
GYM_BIN=/opt/nemotronvc/venvs/nemo-gym/bin/gym                  # VERIFY: console script of the nemo-gym venv (pyproject [project.scripts] gym=nemo_gym.cli.main:main)
GYM_PY=/opt/nemotronvc/venvs/nemo-gym/bin/python                # scripts/runtime_env.sh NEMO_RL_NEMO_GYM_PYTHON_EXECUTABLE
RAY_BIN=/opt/nemotronvc/venvs/nemo-gym/bin/ray
DRIVER_PY=/opt/nemotronvc/venvs/driver/bin/python
VENV_ROOT=${NVC_VENV_ROOT:-/opt/nemotronvc/gym-v1/venvs}         # pre-baked per-server uv venvs; NVC_VENV_ROOT=$P/runtime/gym-v1/venvs-ext adds tau2/openai_model
REASONING_PARSER_PLUGIN=$NEMO_RL_ROOT/nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py

SAMPLES=; OUT=; K=8; MAX_NUM_SEQS=48; MAX_MODEL_LEN=98304; GPU_MEM_UTIL=0.85
ROUTES_FILTER=; DEADLINE_EPOCH=; ASSUMED_TPS=4000; RESUME=0; CHECK_ONLY=0; SERVICES_ONLY=0
MAMBA_CACHE_DTYPE=${NVC_PROFILE_MAMBA_CACHE_DTYPE:-float32}      # eval used float32; v8 training used float16 (+stochastic rounding)
VLLM_SEED=42
# How vLLM renders list-valued message content. 'auto' detects 'openai' for this chat template and prints the system
# message as a Python list repr; the eval passes 'string' (RL infra audit 2026-10-13, item 3). Default string.
CONTENT_FORMAT=${NVC_PROFILE_CONTENT_FORMAT:-string}

while (( $# )); do
  case "$1" in
    --samples) SAMPLES=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --k) K=$2; shift 2;;
    --max-num-seqs) MAX_NUM_SEQS=$2; shift 2;;
    --max-model-len) MAX_MODEL_LEN=$2; shift 2;;
    --gpu-mem-util) GPU_MEM_UTIL=$2; shift 2;;
    --routes) ROUTES_FILTER=$2; shift 2;;
    --deadline-epoch) DEADLINE_EPOCH=$2; shift 2;;
    --assumed-tps) ASSUMED_TPS=$2; shift 2;;
    --resume) RESUME=1; shift;;
    --check-only) CHECK_ONLY=1; shift;;
    --services-only) SERVICES_ONLY=1; shift;;
    --mamba-cache-dtype) MAMBA_CACHE_DTYPE=$2; shift 2;;
    --content-format) CONTENT_FORMAT=$2; shift 2;;
    -h|--help) sed -n '2,16p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n "$SAMPLES" && -n "$OUT" ]] || { echo "--samples and --out are required" >&2; exit 2; }
[[ -f "$SAMPLES/routes.json" ]] || { echo "missing $SAMPLES/routes.json (run sample_routes.py first)" >&2; exit 2; }

JOB=${SLURM_JOB_ID:?must run inside a Slurm allocation (interactive.sh exec JOB ...)}
mkdir -p "$OUT/logs" "$OUT/rollouts"
LOG=$OUT/logs/launch-$JOB.log
exec > >(tee -a "$LOG") 2>&1
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
log "launch_profile.sh job=$JOB host=$(hostname) samples=$SAMPLES out=$OUT k=$K max_num_seqs=$MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN"

# ----------------------------------------------------------------------------- preflight
fail=0
need_path() { [[ -e "$1" ]] || { log "MISSING: $1  ($2)"; fail=1; }; }
need_path "$MODEL_PATH/config.json" "student HF399"
need_path "$GYM_ROOT/nemo_gym/__init__.py" "Gym root (worktree, has responses_api_models/vllm_model + multichallenge etc.)"
need_path "$GYM_ROOT/responses_api_models/vllm_model/app.py" "vllm_model server source"
need_path "$SERVICES_YAML" "profile Gym config"
need_path "$VLLM_BIN" "vllm027 venv vllm CLI"
need_path "$GYM_PY" "nemo-gym venv python"
need_path "$RAY_BIN" "nemo-gym venv ray"
need_path "$DRIVER_PY" "driver venv python"
need_path "$REASONING_PARSER_PLUGIN" "nano_v3 reasoning parser plugin"
if [[ ! -x "$GYM_BIN" ]]; then
  log "WARN: $GYM_BIN not found; falling back to '$GYM_PY -c from nemo_gym.cli.main import main; main()'"
  GYM_CMD=("$GYM_PY" -c 'import sys; from nemo_gym.cli.main import main; main()')
else
  GYM_CMD=("$GYM_BIN")
fi
# Pre-baked venvs for every server the config starts (setup_command.py:104-109 -> <VENV_ROOT>/<type>/<impl>/.venv).
for rel in responses_api_models/vllm_model responses_api_agents/simple_agent responses_api_agents/tool_simulation_agent \
           conversational_tool_use/simulation resources_servers/single_step_tool_use_with_argument_comparison \
           resources_servers/instruction_following resources_servers/structured_outputs resources_servers/multichallenge \
           resources_servers/workplace_assistant resources_servers/conversational_tool_use_simulation; do
  need_path "$VENV_ROOT/$rel/.venv/bin/python" "pre-baked Gym venv (image must be rebuilt if missing)"
done
gpus=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
(( gpus == 8 )) || { log "expected 8 GPUs, nvidia-smi shows $gpus"; fail=1; }
(( fail == 0 )) || { log "preflight failed"; exit 3; }

# Route table -> TSV (key agent sample cap concurrency est_tokens n_prompts), priority order, optional filter.
route_tsv=$("$DRIVER_PY" - "$SAMPLES/routes.json" "$ROUTES_FILTER" <<'PY'
import json, sys
routes = json.load(open(sys.argv[1])); want = [k for k in sys.argv[2].split(",") if k]
for r in sorted(routes, key=lambda r: r["priority"]):
    if want and r["route_key"] not in want: continue
    print("\t".join(str(r[k]) for k in ("route_key","agent","sample_path","max_output_tokens","concurrency","est_output_tokens_per_rollout","n_prompts")))
PY
)
[[ -n "$route_tsv" || $SERVICES_ONLY == 1 ]] || { log "no routes selected"; exit 2; }
expected_servers=$("$DRIVER_PY" - "$SERVICES_YAML" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])); n = 0
for name, tree in cfg.items():
    if not isinstance(tree, dict) or len(tree) != 1: continue
    impl = next(iter(tree.values()))
    if isinstance(impl, dict) and len(impl) == 1 and isinstance(next(iter(impl.values())), dict) and "entrypoint" in next(iter(impl.values())): n += 1
print(n)
PY
)
log "routes selected:"; printf '%s\n' "$route_tsv" | awk -F'\t' '{printf "  %-26s agent=%s n=%s cap=%s conc=%s est_tok/rollout=%s\n",$1,$2,$7,$4,$5,$6}'
log "gym servers expected healthy: $expected_servers"

# ----------------------------------------------------------------------------- ports (eval pattern: 1500-port block per slot, bind-checked)
slot=$("$DRIVER_PY" - "$((JOB % 32))" <<'PY'
import socket, sys
start = int(sys.argv[1])
for offset in range(32):
    slot = (start + offset) % 32; base = 10000 + slot * 1500
    try:
        for port in range(base, base + 1000):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: s.bind(("127.0.0.1", port))
    except OSError: continue
    print(slot); break
else: raise SystemExit("no free 1000-port block")
PY
)
port_base=$((10000 + slot * 1500)); [[ -n "${NVC_PORT_BASE:-}" ]] && port_base=$NVC_PORT_BASE
VLLM_PORT=$port_base
RAY_PORT=$((port_base + 100)); RAY_OBJ_PORT=$((port_base + 101)); RAY_NM_PORT=$((port_base + 102)); RAY_CLIENT_PORT=$((port_base + 103))
RAY_DASH_PORT=$((port_base + 104)); RAY_DASH_AGENT_PORT=$((port_base + 105)); RAY_DASH_GRPC_PORT=$((port_base + 106))
RAY_RTENV_PORT=$((port_base + 107)); RAY_METRICS_PORT=$((port_base + 108)); RAY_WORKER_LOW=$((port_base + 200)); RAY_WORKER_HIGH=$((port_base + 399))
GYM_HEAD_PORT=$((port_base + 450)); GYM_PORT_LOW=$((port_base + 500)); GYM_PORT_HIGH=$((port_base + 999))
NODE_IP=$(getent ahostsv4 "$(hostname)" | awk 'NR==1{print $1}')
RAY_ADDRESS="$NODE_IP:$RAY_PORT"
MODEL_URL="http://127.0.0.1:$VLLM_PORT/v1"
MODEL_URLS="[\"$MODEL_URL\"]"
log "ports: slot=$slot vllm=$VLLM_PORT ray=$RAY_PORT gym_head=$GYM_HEAD_PORT gym=$GYM_PORT_LOW-$GYM_PORT_HIGH node_ip=$NODE_IP"

# ----------------------------------------------------------------------------- environment (mirrors scripts/runtime_env.sh + ray_role.sh for the pieces a stock vLLM needs)
SCRATCH=/raid/scratch/$USER/nvcrl-$JOB/profile           # removed by `interactive.sh release JOB`
mkdir -p "$SCRATCH"/{hf-home/modules,xdg,triton,inductor,cuda,vllm,gym-cache} /tmp/nvcp-$JOB
export HF_HOME=$SCRATCH/hf-home HF_MODULES_CACHE=$SCRATCH/hf-home/modules XDG_CACHE_HOME=$SCRATCH/xdg
export TRITON_CACHE_DIR=$SCRATCH/triton TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor CUDA_CACHE_PATH=$SCRATCH/cuda VLLM_CACHE_ROOT=$SCRATCH/vllm
export TMPDIR=/tmp/nvcp-$JOB
if [[ -f /opt/nemotronvc/runtime-env.sh ]]; then set +e; source /opt/nemotronvc/runtime-env.sh; set -Eeuo pipefail; fi
export PATH="/opt/nemotronvc/bin:/opt/nemotronvc/venvs/driver/bin:$PATH"
export PYTHONPATH="$HF_MODULES_CACHE:$NEMO_RL_ROOT:$GYM_ROOT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled RAY_USAGE_STATS_ENABLED=0 RAY_DEDUP_LOGS=0
export NEMO_GYM_EXTRA_ROOTS=$GYM_ROOT NEMO_GYM_VENV_DIR=$VENV_ROOT UV_OFFLINE=1 UV_CACHE_DIR=$SCRATCH/gym-cache/uv RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_MAX_CONNECTIONS=1
unset NVIDIA_API_KEY OPENAI_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
# Hub-backed Gym servers (sim_gptoss) resolve ${oc.env:NVIDIA_API_KEY}; read it from a file so the value never appears
# on a command line or in a log (NVC_HUB_KEY_FILE=$P/.secrets/nvidia-api-key).
if [[ -n ${NVC_HUB_KEY_FILE:-} ]]; then IFS= read -r NVIDIA_API_KEY < "$NVC_HUB_KEY_FILE"; export NVIDIA_API_KEY; fi

if (( CHECK_ONLY )); then
  log "check-only: template/tokenizer import probe with the vllm027 venv"
  "$VLLM_PY" - "$MODEL_PATH" <<'PY'
import sys, json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
msgs = [{"role": "user", "content": "ping"}]
s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True, truncate_history_thinking=False)
print("template ok, rendered suffix:", json.dumps(s[-40:]))
PY
  "$VLLM_PY" -c 'import vllm; print("vllm", vllm.__version__)'
  "${GYM_CMD[@]}" --version || true
  log "check-only passed"
  exit 0
fi

# ----------------------------------------------------------------------------- cleanup (only our own processes)
VLLM_PID=; RAY_PID=; GYM_PID=; status_code=0
cleanup() {
  status_code=${1:-$?}; trap - EXIT INT TERM
  log "cleanup: gym=$GYM_PID ray=$RAY_PID vllm=$VLLM_PID"
  if [[ -n "$GYM_PID" ]] && kill -0 "$GYM_PID" 2>/dev/null; then
    kill -INT "$GYM_PID" 2>/dev/null || true; for _ in $(seq 1 30); do kill -0 "$GYM_PID" 2>/dev/null || break; sleep 1; done
    kill -TERM "$GYM_PID" 2>/dev/null || true; for _ in $(seq 1 15); do kill -0 "$GYM_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$GYM_PID" 2>/dev/null || true
  fi
  [[ -z "$GYM_PID" ]] || wait "$GYM_PID" 2>/dev/null || true
  if [[ -n "$RAY_PID" ]] && kill -0 "$RAY_PID" 2>/dev/null; then
    kill -TERM "$RAY_PID" 2>/dev/null || true; for _ in $(seq 1 45); do kill -0 "$RAY_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$RAY_PID" 2>/dev/null || true
  fi
  [[ -z "$RAY_PID" ]] || wait "$RAY_PID" 2>/dev/null || true
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill -TERM "$VLLM_PID" 2>/dev/null || true; for _ in $(seq 1 90); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$VLLM_PID" 2>/dev/null || true
  fi
  [[ -z "$VLLM_PID" ]] || wait "$VLLM_PID" 2>/dev/null || true
  printf '{"job_id":"%s","exit_code":%s,"utc":"%s"}\n' "$JOB" "$status_code" "$(date -u +%FT%TZ)" > "$OUT/PROFILE_EXIT.json"
  log "exit $status_code"
  exit "$status_code"
}
trap 'cleanup $?' EXIT; trap 'cleanup 130' INT; trap 'cleanup 143' TERM

# ----------------------------------------------------------------------------- Ray head (Gym's RunHelper needs a Ray address)
RAY_TMP=/tmp/nvr-$JOB; mkdir -p "$RAY_TMP"
"$RAY_BIN" start --head --block --node-ip-address="$NODE_IP" --port="$RAY_PORT" \
  --object-manager-port="$RAY_OBJ_PORT" --node-manager-port="$RAY_NM_PORT" --ray-client-server-port="$RAY_CLIENT_PORT" \
  --dashboard-port="$RAY_DASH_PORT" --dashboard-agent-listen-port="$RAY_DASH_AGENT_PORT" --dashboard-agent-grpc-port="$RAY_DASH_GRPC_PORT" \
  --runtime-env-agent-port="$RAY_RTENV_PORT" --metrics-export-port="$RAY_METRICS_PORT" \
  --min-worker-port="$RAY_WORKER_LOW" --max-worker-port="$RAY_WORKER_HIGH" \
  --num-cpus="${SLURM_CPUS_PER_TASK:-64}" --num-gpus=0 --include-dashboard=false --no-monitor --disable-usage-stats \
  --temp-dir="$RAY_TMP" > "$OUT/logs/ray-$JOB.log" 2>&1 &
RAY_PID=$!
ray_ok=0
for _ in $(seq 1 120); do
  kill -0 "$RAY_PID" 2>/dev/null || break
  if "$RAY_BIN" status --address="$RAY_ADDRESS" >/dev/null 2>&1; then ray_ok=1; break; fi
  sleep 1
done
(( ray_ok )) || { log "Ray head failed to start; see $OUT/logs/ray-$JOB.log"; tail -40 "$OUT/logs/ray-$JOB.log"; exit 1; }
log "Ray ready at $RAY_ADDRESS"

# ----------------------------------------------------------------------------- vLLM serve: exact eval DP8/EP8 flags (NemotronVC_eval/eval/runtime/internal/run_domain_inside_container.sh:436-452)
VLLM_LOG=$OUT/logs/vllm-$JOB.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name "$MODEL_LABEL" \
  --generation-config vllm \
  --host 127.0.0.1 --port "$VLLM_PORT" --api-key dummy \
  --api-server-count 1 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 --data-parallel-size 8 \
  --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --moe-backend triton \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens 8192 \
  --mamba-ssm-cache-dtype "$MAMBA_CACHE_DTYPE" --mamba-cache-mode align \
  --enable-prefix-caching \
  --skip-mm-profiling --language-model-only --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nano_v3 \
  --reasoning-parser-plugin "$REASONING_PARSER_PLUGIN" \
  --chat-template-content-format "$CONTENT_FORMAT" \
  --enable-chunked-prefill --seed "$VLLM_SEED" \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}' \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
vllm_ok=0; deadline=$(( $(date +%s) + 1800 ))
while (( $(date +%s) < deadline )); do
  kill -0 "$VLLM_PID" 2>/dev/null || { log "vLLM died during load"; tail -80 "$VLLM_LOG"; exit 1; }
  if curl --silent --fail --connect-timeout 2 --max-time 5 --header 'Authorization: Bearer dummy' "$MODEL_URL/models" >/dev/null; then vllm_ok=1; break; fi
  sleep 5
done
(( vllm_ok )) || { log "vLLM not ready after 30 min"; tail -80 "$VLLM_LOG"; exit 1; }
grep -q 'data_parallel_size=8' "$VLLM_LOG" || log "WARN: could not prove data_parallel_size=8 from $VLLM_LOG (eval asserted this string)"
log "vLLM ready at $MODEL_URL (max_num_seqs=$MAX_NUM_SEQS x DP8 = $((MAX_NUM_SEQS * 8)) sequence slots)"
# Render check: a list-valued system message must come out as plain text (no Python list repr).
"$DRIVER_PY" - "$MODEL_URL" "$MODEL_LABEL" "$OUT/RENDER_CHECK.json" <<'PY' || { log "render check FAILED (see RENDER_CHECK.json)"; exit 1; }
import json, sys, urllib.request
url, label, out = sys.argv[1:4]
def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"})
    return json.load(urllib.request.urlopen(req, timeout=60))
base = url[:-3] if url.endswith("/v1") else url
msgs = [{"role": "system", "content": [{"type": "text", "text": "SYSLINE_ONE\nSYSLINE_TWO"}]}, {"role": "user", "content": [{"type": "text", "text": "hi"}]}]
tok = post("/tokenize", {"model": label, "messages": msgs, "add_generation_prompt": True})
prompt = post("/detokenize", {"model": label, "tokens": tok["tokens"]})["prompt"]
ok = "SYSLINE_ONE\nSYSLINE_TWO" in prompt and "[{" not in prompt
json.dump({"ok": ok, "prompt_head": prompt[:200]}, open(out, "w"), indent=1)
print("render check", "ok" if ok else "FAILED", json.dumps(prompt[:120]))
sys.exit(0 if ok else 1)
PY

# ----------------------------------------------------------------------------- Gym servers
GYM_COMMON=(
  --config "$SERVICES_YAML"
  --model "$MODEL_LABEL" --model-url "$MODEL_URLS" --model-api-key dummy
  ++head_server.host=127.0.0.1 ++head_server.port="$GYM_HEAD_PORT"
  ++port_range_low="$GYM_PORT_LOW" ++port_range_high="$GYM_PORT_HIGH"
  ++ray_head_node_address="$RAY_ADDRESS"
  ++skip_venv_if_present=true ++uv_venv_dir="$VENV_ROOT" ++uv_cache_dir="$SCRATCH/gym-cache/uv"
  ++results_dir="$OUT/gym-results" ++cache_dir="$SCRATCH/gym-cache"
  ++nemo_gym_log_dir="$OUT/logs/gym-servers"
  ++model_endpoint_readiness_timeout_seconds=600
  ++allow_openai_version_skew=true
)
GYM_LOG=$OUT/logs/gym-$JOB.log
cd "$GYM_ROOT"
"${GYM_CMD[@]}" env start "${GYM_COMMON[@]}" > "$GYM_LOG" 2>&1 &
GYM_PID=$!
gym_ok=0; deadline=$(( $(date +%s) + 900 ))
while (( $(date +%s) < deadline )); do
  kill -0 "$GYM_PID" 2>/dev/null || { log "gym env start exited early"; tail -120 "$GYM_LOG"; exit 1; }
  st=$(timeout 15s "${GYM_CMD[@]}" env status ++head_server.host=127.0.0.1 ++head_server.port="$GYM_HEAD_PORT" ++allow_openai_version_skew=true 2>/dev/null || true)
  if grep -q "${expected_servers} servers found (${expected_servers} healthy" <<<"$st"; then gym_ok=1; break; fi
  sleep 5
done
(( gym_ok )) || { log "Gym servers not all healthy after 15 min"; tail -120 "$GYM_LOG"; exit 1; }
log "Gym ready: $expected_servers servers healthy (head http://127.0.0.1:$GYM_HEAD_PORT)"
cp -f "$SAMPLES/routes.json" "$OUT/routes.json"
cp -f "$SERVICES_YAML" "$OUT/gym_profile_services.used.yaml"
printf '{"job_id":"%s","host":"%s","model":"%s","model_label":"%s","vllm_port":%s,"gym_head_port":%s,"ray_address":"%s","k":%s,"max_num_seqs":%s,"max_model_len":%s,"mamba_ssm_cache_dtype":"%s","chat_template_content_format":"%s","gpu_memory_utilization":%s,"vllm_seed":%s,"temperature":1.0,"top_p":1.0,"chat_template_kwargs":{"enable_thinking":true,"truncate_history_thinking":false}}\n' \
  "$JOB" "$(hostname)" "$MODEL_PATH" "$MODEL_LABEL" "$VLLM_PORT" "$GYM_HEAD_PORT" "$RAY_ADDRESS" "$K" "$MAX_NUM_SEQS" "$MAX_MODEL_LEN" "$MAMBA_CACHE_DTYPE" "$CONTENT_FORMAT" "$GPU_MEM_UTIL" "$VLLM_SEED" > "$OUT/RUNTIME.json"

if (( SERVICES_ONLY )); then log "services-only: holding vLLM/Ray/Gym until killed"; wait "$GYM_PID"; exit 0; fi
# ----------------------------------------------------------------------------- rollouts per route
capacity=$((MAX_NUM_SEQS * 8))
timing=$OUT/timing.jsonl; touch "$timing"
while IFS=$'\t' read -r key agent sample cap conc est_tok n_prompts; do
  [[ -n "$key" ]] || continue
  (( conc > capacity )) && conc=$capacity
  route_out=$OUT/rollouts/$key; mkdir -p "$route_out"
  rollouts=$route_out/rollouts.jsonl
  now=$(date +%s)
  est_sec=$(( n_prompts * K * est_tok / ASSUMED_TPS + 180 ))
  if [[ -n "$DEADLINE_EPOCH" ]]; then
    remaining=$(( DEADLINE_EPOCH - now ))
    if (( remaining - 900 < est_sec )); then
      log "SKIP $key: est ${est_sec}s (n=$n_prompts k=$K est_tok=$est_tok tps=$ASSUMED_TPS) > remaining ${remaining}s - 15min guard"
      printf '{"route_key":"%s","agent":"%s","skipped":true,"reason":"deadline","est_seconds":%s,"remaining_seconds":%s}\n' "$key" "$agent" "$est_sec" "$remaining" >> "$timing"
      continue
    fi
    budget=$(( remaining - 600 ))
  else
    budget=0
  fi
  extra=()
  (( RESUME )) && extra+=(--resume)
  log "RUN $key agent=$agent n=$n_prompts k=$K cap=$cap conc=$conc est=${est_sec}s -> $rollouts"
  start=$(date +%s)
  set +e
  if (( budget > 0 )); then
    timeout --signal=INT --kill-after=60 "$budget" \
    "${GYM_CMD[@]}" eval run --no-serve "${GYM_COMMON[@]}" \
      --agent "$agent" --input "$sample" --output "$rollouts" \
      --num-repeats "$K" --temperature 1 --top-p 1 --max-output-tokens "$cap" --concurrency "$conc" \
      ++num_repeats_add_seed=true ++upload_rollouts_to_wandb=false "${extra[@]}" > "$route_out/eval-run-$JOB.log" 2>&1
  else
    "${GYM_CMD[@]}" eval run --no-serve "${GYM_COMMON[@]}" \
      --agent "$agent" --input "$sample" --output "$rollouts" \
      --num-repeats "$K" --temperature 1 --top-p 1 --max-output-tokens "$cap" --concurrency "$conc" \
      ++num_repeats_add_seed=true ++upload_rollouts_to_wandb=false "${extra[@]}" > "$route_out/eval-run-$JOB.log" 2>&1
  fi
  code=$?
  set -e
  end=$(date +%s)
  done_rows=$( [[ -f "$rollouts" ]] && wc -l < "$rollouts" || echo 0 )
  fail_rows=$( [[ -f "${rollouts%.jsonl}_failures.jsonl" ]] && wc -l < "${rollouts%.jsonl}_failures.jsonl" || echo 0 )
  printf '{"route_key":"%s","agent":"%s","start_epoch":%s,"end_epoch":%s,"seconds":%s,"exit_code":%s,"n_prompts":%s,"k":%s,"expected_rollouts":%s,"rollout_rows":%s,"failure_rows":%s,"max_output_tokens":%s,"concurrency":%s}\n' \
    "$key" "$agent" "$start" "$end" "$((end - start))" "$code" "$n_prompts" "$K" "$((n_prompts * K))" "$done_rows" "$fail_rows" "$cap" "$conc" >> "$timing"
  log "DONE $key exit=$code seconds=$((end - start)) rollouts=$done_rows/$((n_prompts * K)) failures=$fail_rows"
  (( code == 0 )) || tail -30 "$route_out/eval-run-$JOB.log"
  # Poll Gym health between routes; a dead server would silently zero the next route.
  "${GYM_CMD[@]}" env status ++head_server.host=127.0.0.1 ++head_server.port="$GYM_HEAD_PORT" ++allow_openai_version_skew=true 2>/dev/null | grep -E 'servers found' || log "WARN: gym env status unavailable"
done <<<"$route_tsv"

# ----------------------------------------------------------------------------- aggregate (stdlib-only; also runnable on the login node)
"$DRIVER_PY" "$HERE/aggregate.py" --out "$OUT" || log "aggregate.py failed; run it on the login node later"
curl --silent --fail --connect-timeout 2 --max-time 15 "http://127.0.0.1:$VLLM_PORT/metrics" > "$OUT/logs/vllm-metrics-final.prom" 2>/dev/null || true
log "profile finished; summary at $OUT/SUMMARY.md"
