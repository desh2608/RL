#!/usr/bin/env bash
# Simulator-node role: exec'd by ray_role.sh on the last NVC_SIM_NODES ranks of the main srun
# (they are NOT Ray nodes). Runs one standalone vLLM OpenAI server (run/SIM_SPEC.json) inside the
# training container and publishes run/sim/<SLURM job id>/READY-<host>.json (job-bound: the receipt
# carries job_id + port and ray_role.sh / gym_idle_watchdog.py read only this job's directory, which
# gpu_job.sh creates together with HOSTS); NeMo Gym's `sim_gptoss` openai_model server reaches it
# through ${oc.env:NVC_SIM_BASE_URL} exported by gpu_job.sh. Only this script's own vLLM process
# group is ever signalled (no pkill, no global cleanup); env vars are never printed.
set -Eeuo pipefail
run=${NVC_DEBUG_RUN:?}
local_root=${NVC_DEBUG_LOCAL:?}
job=${NVC_SIM_ROLE_JOB:?}
[[ "$job" =~ ^[0-9]+$ ]] || { echo "sim_role.sh: NVC_SIM_ROLE_JOB must be a numeric Slurm job id" >&2; exit 2; }
host=${SLURMD_NODENAME:-$(hostname -s)}
spec="$run/SIM_SPEC.json"
[[ -s "$spec" ]] || { echo "sim_role.sh: missing $spec" >&2; exit 2; }
helper_python=/opt/nemotronvc/venvs/driver/bin/python
[[ -x "$helper_python" ]] || helper_python=$(command -v python3)
sim_dir="$run/sim/$job"
mkdir -p "$sim_dir"
ready_file="$sim_dir/READY-$host.json"
failed_file="$sim_dir/FAILED-$host.json"
log_file="$sim_dir/vllm-$host.log"

# SIM_SPEC.json -> shell variables (values are shell-quoted by python; extra_args becomes an array).
eval "$("$helper_python" - "$spec" <<'PY'
import json, shlex, sys
spec = json.load(open(sys.argv[1]))
for key in ('model_path', 'served_name', 'port', 'tp', 'dp', 'max_model_len', 'gpu_memory_utilization', 'max_num_seqs', 'python'):
    print(f'sim_{key}={shlex.quote(str(spec[key]))}')
extra = spec.get('extra_args') or []
assert isinstance(extra, list) and all(isinstance(x, str) for x in extra), 'extra_args must be a list of strings'
print('sim_extra_args=(' + ' '.join(shlex.quote(x) for x in extra) + ')')
PY
)"
[[ "$sim_port" =~ ^[0-9]+$ && -d "$sim_model_path" && -f "$sim_model_path/config.json" && -x "$sim_python" ]] || {
  echo "sim_role.sh: invalid SIM_SPEC (port/model_path/python)" >&2; exit 2; }

# Node-local caches, same layout as ray_role.sh (NVC_DEBUG_LOCAL/tmp is bind-mounted at /tmp).
export HF_HOME="$local_root/hf-home"
export HF_MODULES_CACHE="$HF_HOME/modules"
export XDG_CACHE_HOME="$local_root/cache/xdg"
export TRITON_CACHE_DIR="$local_root/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$local_root/cache/inductor"
export CUDA_CACHE_PATH="$local_root/cache/cuda"
export VLLM_CACHE_ROOT="$local_root/cache/vllm"
export TMPDIR=/tmp/work
export VLLM_LOGGING_LEVEL=INFO
# The vllm027 venv is self-contained; drop runtime_env.sh's nemo-rl/Megatron/Gym PYTHONPATH so nothing shadows it.
unset PYTHONPATH
mkdir -p "$HF_MODULES_CACHE" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$VLLM_CACHE_ROOT" "$TMPDIR"

write_receipt() {  # write_receipt FILE STATUS SECONDS [MODELS_JSON_FILE]
  "$helper_python" - "$1" "$2" "$3" "$host" "$sim_port" "${vllm_pid:-}" "$job" "$sim_served_name" "${4:-}" <<'PY'
import json, os, sys
path, status, seconds, host, port, pid, job, served_name, models_file = sys.argv[1:10]
models = None
if models_file and os.path.isfile(models_file):
    try:
        models = json.load(open(models_file))
    except ValueError:
        models = open(models_file).read()[:2000]
record = {'schema': 'nemotronvc.sim_role.v2', 'status': status, 'job_id': job, 'host': host, 'port': int(port),
          'served_name': served_name, 'pid': int(pid) if pid else None, 'seconds': int(seconds), 'models': models}
tmp = path + '.tmp'
with open(tmp, 'w') as f:
    json.dump(record, f, indent=2); f.write('\n')
os.replace(tmp, path)
PY
}

http_get() {  # http_get URL OUT_FILE -> 0 on HTTP 200
  "$helper_python" - "$1" "$2" <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as r:
        body = r.read()
        if r.status != 200:
            sys.exit(1)
    open(sys.argv[2], 'wb').write(body)
except Exception:
    sys.exit(1)
PY
}

vllm_pid=
cleanup() {
  code=$?
  trap - EXIT
  set +e
  # set -m below puts the server in its own process group (pgid == pid), so this reaches the
  # api_server and the engine-core children it spawned, and nothing else on the node.
  if [[ -n "$vllm_pid" ]] && kill -0 "$vllm_pid" 2>/dev/null; then
    kill -TERM -- "-$vllm_pid" 2>/dev/null || kill -TERM "$vllm_pid" 2>/dev/null
    for ((i = 0; i < 30; i++)); do kill -0 "$vllm_pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$vllm_pid" 2>/dev/null && { kill -KILL -- "-$vllm_pid" 2>/dev/null || kill -KILL "$vllm_pid" 2>/dev/null; }
    wait "$vllm_pid" 2>/dev/null
  fi
  printf '[SIM_ROLE_EXIT] host=%s code=%s\n' "$host" "$code"
  exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

printf '[SIM_ROLE_START] job=%s host=%s port=%s tp=%s dp=%s served=%s log=%s\n' "$job" "$host" "$sim_port" "$sim_tp" "$sim_dp" "$sim_served_name" "$log_file"
set -m
"$sim_python" -m vllm.entrypoints.openai.api_server \
  --model "$sim_model_path" --served-model-name "$sim_served_name" \
  --host 0.0.0.0 --port "$sim_port" \
  --tensor-parallel-size "$sim_tp" --data-parallel-size "$sim_dp" \
  --max-model-len "$sim_max_model_len" --gpu-memory-utilization "$sim_gpu_memory_utilization" \
  --max-num-seqs "$sim_max_num_seqs" "${sim_extra_args[@]}" > "$log_file" 2>&1 < /dev/null &
vllm_pid=$!
set +m

models_tmp="$TMPDIR/sim-models-$host.json"
started=$SECONDS
ready=false
for ((i = 0; i < 180; i++)); do
  if ! kill -0 "$vllm_pid" 2>/dev/null; then
    wait "$vllm_pid" || true
    write_receipt "$failed_file" "process_exited_during_startup" "$((SECONDS - started))"
    echo "sim_role.sh: vLLM exited before serving; see $log_file" >&2
    exit 1
  fi
  if http_get "http://127.0.0.1:$sim_port/v1/models" "$models_tmp"; then ready=true; break; fi
  sleep 5
done
if [[ "$ready" != true ]]; then
  write_receipt "$failed_file" "startup_timeout_900s" "$((SECONDS - started))"
  echo "sim_role.sh: vLLM did not serve /v1/models within 900 s; see $log_file" >&2
  exit 1
fi
write_receipt "$ready_file" "ready" "$((SECONDS - started))" "$models_tmp"
printf '[SIM_ROLE_READY] host=%s seconds=%s\n' "$host" "$((SECONDS - started))"

# Serve until the head publishes END_CODE; a dead server before that is a role failure
# (the main srun runs with --kill-on-bad-exit=1, so the allocation fails fast).
while [[ ! -s "$run/END_CODE" ]]; do
  if ! kill -0 "$vllm_pid" 2>/dev/null; then
    wait "$vllm_pid" || true
    write_receipt "$failed_file" "process_exited_while_serving" "$((SECONDS - started))"
    echo "sim_role.sh: vLLM died while serving; see $log_file" >&2
    exit 1
  fi
  sleep 5
done
exit "$(cat "$run/END_CODE")"
