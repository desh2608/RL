#!/usr/bin/env bash
set -Eeuo pipefail
rank=${SLURM_PROCID:?}
job=${SLURM_JOB_ID:?}
run=${NVC_DEBUG_RUN:?}
local_root=${NVC_DEBUG_LOCAL:?}
expected_nodes=${NVC_EXPECTED_NODES:?}
export NVC_EXPECTED_NODES
export HF_HOME="$local_root/hf-home"
export HF_MODULES_CACHE="$HF_HOME/modules"
# HF import writes a distributed checkpoint, so every rank needs shared storage.
export NRL_MEGATRON_CHECKPOINT_DIR="$run/init-mcore"
export HF_DATASETS_CACHE="$local_root/cache/datasets"
export XDG_CACHE_HOME="$local_root/cache/xdg"
export TRITON_CACHE_DIR="$local_root/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$local_root/cache/inductor"
export CUDA_CACHE_PATH="$local_root/cache/cuda"
export VLLM_CACHE_ROOT="$local_root/cache/vllm"
export TMPDIR=/tmp/work RAY_TMPDIR=/tmp/ray
mkdir -p "$HF_MODULES_CACHE" "$TMPDIR" "$RAY_TMPDIR" "$local_root/cache"
source "$run/scripts/runtime_env.sh"
if [[ ${NVC_INPUT_MODE:-} == gym ]]; then
 /opt/nemotronvc/venvs/driver/bin/python "$run/scripts/gym_sandbox_probe.py" "$run" "$job"
fi
# Clear inherited launcher context before Ray serializes the actor environment.
for variable in $(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print $1}'); do unset "$variable"; done
# Secrets handed over by gpu_job.sh (run/SECRETS_REQUIRED) are exposed only while the node-local
# Ray supervisor starts: Ray workers inherit the raylet environment, so the NemoGym actor and the
# Gym server subprocesses it spawns (setup_command.run_command uses environ.copy()) resolve
# ${oc.env:NVIDIA_API_KEY} from their own environment. The driver never carries the value:
# nemo_gym.py copies the driver os.environ into the actor runtime_env env_vars, which Ray's
# runtime_env agent writes to runtime_env_setup-*.log (archived into <run>/ray/<host>/).
nvc_secret_nvidia_api_key=${NVIDIA_API_KEY:-}
unset NVIDIA_API_KEY OPENAI_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
nvc_expose_ray_secrets() { [[ -z "$nvc_secret_nvidia_api_key" ]] || export NVIDIA_API_KEY="$nvc_secret_nvidia_api_key"; }
nvc_withdraw_ray_secrets() { unset NVIDIA_API_KEY; }
# Simulator nodes (run/SIM_NODES, exported by gpu_job.sh as NVC_SIM_NODES / NVC_RAY_NODES) are the
# last ranks of this srun and never join Ray: they serve a standalone vLLM for NeMo Gym's
# sim_gptoss server (NVC_SIM_BASE_URL / NVC_SIM_API_KEY stay in the environment, see sim_role.sh).
if (( rank >= ${NVC_RAY_NODES:-$expected_nodes} )); then
 export NVC_SIM_ROLE_RANK=$rank NVC_SIM_ROLE_JOB=$job
 exec bash "$run/scripts/sim_role.sh"
fi
ulimit -Sn 65535 || true
ray_pid=
cleanup() {
 code=$?
 trap - EXIT
 set +e
 # --block registers Ray's own shutdown hooks. Signal only this role's
 # supervisor; global `ray stop` can kill unrelated processes on the node.
 ray_stop_code=0
 if [[ -n "$ray_pid" ]]; then
   if kill -0 "$ray_pid" 2>/dev/null; then
     kill -TERM "$ray_pid"
     wait "$ray_pid"
     ray_stop_code=$?
     # Ray's SIGTERM hook exits 1 after cleaning its registered children.
     (( ray_stop_code == 1 || ray_stop_code == 143 )) && ray_stop_code=0
   else
     wait "$ray_pid"
     ray_stop_code=$?
   fi
 fi
 # Ray 2.56 appends /ray to RAY_TMPDIR. Inspect both layouts through the
 # job-local /tmp backing directory, never through an arbitrary symlink target.
 if [[ "$local_root" == "/raid/scratch/$USER/super35-draco-$job" ]]; then
   local_real=$(readlink -f "$local_root")
   for layout in direct nested; do
     ray_root="$local_root/tmp/ray"
     [[ "$layout" != nested ]] || ray_root="$ray_root/ray"
     [[ -L "$ray_root/session_latest" ]] || continue
     latest=$(readlink "$ray_root/session_latest")
     session_name=${latest##*/}
     [[ "$session_name" == session_* && "$session_name" != session_latest ]] || continue
     logs=$(readlink -f "$ray_root/$session_name/logs") || continue
     [[ -n "$local_real" && "$logs" == "$local_real/tmp/"* && -d "$logs" ]] || continue
     destination="$run/ray/$(hostname)/$layout"
     if mkdir -p "$destination" && cp -a "$logs/." "$destination/"; then
       printf '[RAY_LOGS_ARCHIVED] host=%s layout=%s session=%s\n' "$(hostname)" "$layout" "$session_name"
     else
       printf '[RAY_LOG_ARCHIVE_FAILED] host=%s layout=%s\n' "$(hostname)" "$layout" >&2
     fi
   done
 fi
 (( code != 0 )) || code=$ray_stop_code
 exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
ray=/opt/nemotronvc/venvs/driver/bin/ray
python=/opt/nemotronvc/venvs/driver/bin/python
"$python" - "$run" "$rank" "$job" <<'PY'
import hashlib, json, os, pathlib, socket, subprocess, sys
if int(sys.argv[2]) == 0:
    run=pathlib.Path(sys.argv[1])
    contract=json.loads((run/'RUN_CONTRACT.json').read_text())
    actual=sorted(str(p.resolve()) for p in pathlib.Path(contract['checkpoint_dir']).glob('step_*') if p.is_dir())
    expected=contract['expected_resume_checkpoint']
    assert actual==([] if expected is None else [str(pathlib.Path(expected).resolve())]), ('Recovery changed since freeze',actual,expected)
    # The allocation holds the lineage writer lock. Preserve the previous
    # allocation's stop request before exposing this run as active to monitors.
    from omegaconf import OmegaConf
    config=OmegaConf.load(run/'config.yaml')
    monitor=pathlib.Path(config.checkpointing.monitor_dir)
    previous_stop=monitor/'STOP_REQUESTED.json'
    if previous_stop.is_file():
        archived=run/'PREVIOUS_STOP_REQUESTED.json'
        assert not archived.exists(), 'Previous stop receipt already archived'
        previous_stop.replace(archived)
        print('[PREVIOUS_STOP_REQUEST_ARCHIVED] '+str(archived),flush=True)
    parent_manifest=run/'SMOKE_RESUME_PARENT.json'
    if parent_manifest.is_file():
        parent=json.loads(parent_manifest.read_text())
        checkpoint=pathlib.Path(parent['checkpoint']['path']).resolve(strict=True)
        assert str(checkpoint)==expected
        verified={}
        for name,digest in parent['checkpoint']['files_sha256'].items():
            path=checkpoint/name
            assert path.resolve().is_relative_to(checkpoint) and not path.is_symlink()
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, name
            verified[name]=digest
        receipt={'schema':'nemotronvc.smoke_resume_preload.v1','manifest_sha256':hashlib.sha256(parent_manifest.read_bytes()).hexdigest(),
                 'verified':True,'checkpoint_path':str(checkpoint),'verified_files':verified}
        temporary=run/'PARENT_CHECKPOINT_VERIFIED.json.tmp'
        temporary.write_text(json.dumps(receipt,indent=2)+'\n')
        temporary.rename(run/'PARENT_CHECKPOINT_VERIFIED.json')
keys = ('NCCL_ALGO','NCCL_PROTO','NCCL_SOCKET_IFNAME','NCCL_IB_HCA',
        'NCCL_IB_GID_INDEX','NCCL_IB_TC','NCCL_IB_SL','NCCL_IB_TIMEOUT',
        'NCCL_IB_QPS_PER_CONNECTION','NCCL_CROSS_NIC','UCX_NET_DEVICES',
        'UCX_TLS','UCX_IB_TRAFFIC_CLASS','GLOO_SOCKET_IFNAME')
record = {'job_id':sys.argv[3], 'rank':int(sys.argv[2]), 'host':socket.gethostname(),
          'network':{key:os.environ.get(key) for key in keys},
          'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv,noheader'], text=True).splitlines()}
(pathlib.Path(sys.argv[1])/f'site-rank-{sys.argv[2]}.json').write_text(json.dumps(record,indent=2)+'\n')
print('[DRACO_GPU_SITE] '+json.dumps(record),flush=True)
PY
ip=$(getent ahostsv4 "$(hostname)" | awk '$2=="STREAM"{print $1;exit}')
[[ -n "$ip" ]]
args=(--disable-usage-stats --node-ip-address="$ip" --num-cpus=56 --num-gpus=8
 --min-worker-port=0 --max-worker-port=0 --node-manager-port=0
 --object-manager-port=0 --runtime-env-agent-port=0
 --dashboard-agent-grpc-port=0 --dashboard-agent-listen-port=0
 --metrics-export-port=0 --resources='{"worker_units":8}'
 --object-store-memory=17179869184)
if (( rank == 0 )); then
 nvc_expose_ray_secrets
 client_port=$("$python" "$run/scripts/ray_ports.py" client --ip "$ip")
 "$ray" start --block --head --port=0 --ray-client-server-port="$client_port" --dashboard-port=0 --include-dashboard=false "${args[@]}" &
 ray_pid=$!
 "$python" "$run/scripts/ray_ports.py" ready --run "$run" --job-id "$job" --rank "$rank" --head-ip "$ip" --supervisor-pid "$ray_pid" --client-port "$client_port"
 nvc_withdraw_ray_secrets
 "$python" -u - <<'PY'
import os,ray,time
# Ray nodes only: simulator nodes (NVC_SIM_NODES) never join the cluster.
expected=int(os.environ.get("NVC_RAY_NODES") or os.environ["NVC_EXPECTED_NODES"])
ray.init(address='auto')
for i in range(450):
    alive=[n for n in ray.nodes() if n['Alive']]
    if len(alive)==expected and ray.cluster_resources().get('GPU',0)==expected*8:
        print(f'[SUPER35_RAY_READY] nodes={expected} gpus={expected*8}',flush=True)
        break
    time.sleep(2)
else:
    raise TimeoutError(f'Expected {expected} nodes and {expected*8} GPUs')
ray.shutdown()
PY
 # Simulator nodes must serve before the driver starts Gym (sim_gptoss resolves NVC_SIM_BASE_URL at
 # server start). Wait <= 900 s for READY-<host>.json from every host in run/sim/<job>/HOSTS (written by
 # gpu_job.sh for exactly this allocation); a receipt only counts when its job_id is this job and its
 # port is the SIM_SPEC port. Any FAILED-*.json, a foreign/mismatched receipt or the deadline ends the
 # run with END_CODE 75 (workers and sim ranks follow it). gym_idle_watchdog.py reads the same paths.
 if (( ${NVC_SIM_NODES:-0} > 0 )); then
   sim_dir="$run/sim/$job"
   [[ -s "$sim_dir/HOSTS" ]]
   sim_port=$("$python" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["port"]))' "$run/SIM_SPEC.json")
   [[ "$sim_port" =~ ^[0-9]+$ ]]
   sim_receipt_state() {  # sim_receipt_state READY_FILE -> ready | pending | mismatch
     "$python" - "$1" "$job" "$sim_port" <<'PY'
import json, os, sys
path, job, port = sys.argv[1:4]
if not os.path.isfile(path) or os.path.islink(path) or os.path.getsize(path) == 0:
    print('pending'); sys.exit(0)
try:
    receipt = json.load(open(path))
except ValueError:
    print('mismatch'); sys.exit(0)   # receipts are written atomically; a non-JSON file is foreign
ok = (isinstance(receipt, dict) and receipt.get('status') == 'ready'
      and str(receipt.get('job_id')) == job and receipt.get('port') == int(port))
print('ready' if ok else 'mismatch')
PY
   }
   sim_deadline=$((SECONDS + 900))
   while :; do
     sim_ready=true
     sim_mismatch=
     while IFS= read -r sim_host; do
       [[ -n "$sim_host" ]] || continue
       case "$(sim_receipt_state "$sim_dir/READY-$sim_host.json")" in
         ready) ;;
         pending) sim_ready=false;;
         *) sim_ready=false; sim_mismatch="$sim_dir/READY-$sim_host.json";;
       esac
     done < "$sim_dir/HOSTS"
     if [[ "$sim_ready" == true ]]; then
       printf '[NVC_SIM_READY] job=%s port=%s hosts=%s base_url_host=%s\n' "$job" "$sim_port" "$(tr '\n' ',' < "$sim_dir/HOSTS")" "${NVC_SIM_BASE_URL%%/v1}"
       break
     fi
     if [[ -n "$sim_mismatch" ]] || compgen -G "$sim_dir/FAILED-*.json" > /dev/null || (( SECONDS >= sim_deadline )); then
       echo "[NVC_SIM_FAILED] simulator node not ready (FAILED receipt, mismatched receipt '${sim_mismatch}' or 900 s deadline); see $sim_dir/" >&2
       printf '75\n' > "$run/END_CODE.tmp"
       mv "$run/END_CODE.tmp" "$run/END_CODE"
       exit 75
     fi
     sleep 5
   done
 fi
 set +e
 NVC_GYM_DRIVER_PHASE=train "$python" -u examples/run_grpo.py --config "$run/config.yaml" 2>&1 | tee "$run/driver.log"
 code=${PIPESTATUS[0]}
 if (( code == 0 )) && [[ ${NVC_INPUT_MODE:-} == gym ]]; then
   "$python" -m nemo_rl.environments.gym_shutdown --run "$run" --job "$job" --phase train
   code=$?
 fi
 # An intentional idle-budget stop is not a completed run or smoke restart.
 if (( code == 0 )) && [[ -s "$run/idle-guard/$job/STOPPED.json" ]]; then
   code=74
 fi
 # A smoke validates a real restart inside the same three-node allocation.
 # Formal jobs do not alter their max step or start a second driver here.
 if (( code == 0 )) && [[ $(cat "$run/KIND") == smoke ]]; then
   "$python" - "$run" <<'PY_SMOKE'
import json,pathlib,sys
run=pathlib.Path(sys.argv[1]); p=run/'checkpoints/step_3'
assert p.is_dir(),'Smoke did not complete step 3; do not turn an early stop into a resume success'
assert json.loads((p/'training_info.json').read_text())['current_step']==3
assert (run/'hf/step_3').is_dir(),'Smoke HF export missing'
PY_SMOKE
   gate=$?
   if (( gate == 0 )); then
     NVC_GYM_DRIVER_PHASE=train-resume "$python" -u examples/run_grpo.py --config "$run/config.yaml" \
       "grpo.max_num_steps=4" "logger.log_dir=$run/train-resume" 2>&1 | tee "$run/resume-driver.log"
     code=${PIPESTATUS[0]}
     if (( code == 0 )) && [[ ${NVC_INPUT_MODE:-} == gym ]]; then
       "$python" -m nemo_rl.environments.gym_shutdown --run "$run" --job "$job" --phase train-resume
       code=$?
     fi
     if (( code == 0 )) && [[ -s "$run/idle-guard/$job/STOPPED.json" ]]; then
       code=74
     fi
     if (( code == 0 )); then
       "$python" - "$run" <<'PY_RESUME'
import json,pathlib,sys
run=pathlib.Path(sys.argv[1]); p=run/'checkpoints/step_4'
assert p.is_dir(),'Resume exited without completing step 4'
info=json.loads((p/'training_info.json').read_text())
assert info['current_step']==4,info
assert len(list((run/'checkpoints').glob('step_*')))==1,'Single-MCore retention failed'
assert (run/'hf/step_3/EXPORT_COMPLETE.json').is_file(),'Atomic HF completion missing'
(run/'SMOKE_PASS.json').write_text(json.dumps({'resume_step':4,'hf_step':3,'single_mcore':True},indent=2)+'\n')
PY_RESUME
       code=$?
     fi
   else
     code=$gate
   fi
 fi
 set -e
 printf '%s\n' "$code" > "$run/END_CODE.tmp"
 mv "$run/END_CODE.tmp" "$run/END_CODE"
 exit "$code"
else
 for ((i=0;i<900;i++)); do [[ -s "$run/RAY_ADDRESS" ]] && break; sleep 1; done
 [[ -s "$run/RAY_ADDRESS" ]]
 head_address=$(cat "$run/RAY_ADDRESS")
 nvc_expose_ray_secrets
 "$ray" start --block --address="$head_address" "${args[@]}" &
 ray_pid=$!
 "$python" "$run/scripts/ray_ports.py" ready --run "$run" --job-id "$job" --rank "$rank" --head-ip "${head_address%:*}" --supervisor-pid "$ray_pid" --expected-address "$head_address"
 nvc_withdraw_ray_secrets
 while [[ ! -s "$run/END_CODE" ]]; do
   if ! kill -0 "$ray_pid" 2>/dev/null; then wait "$ray_pid"; exit 1; fi
   sleep 5
 done
 exit "$(cat "$run/END_CODE")"
fi
