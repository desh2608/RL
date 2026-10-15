#!/usr/bin/env bash
# Header defaults are for cs-oci-ord; submit_run.sh overrides account/partition/nodes/job-name on the command line.
#SBATCH --partition=polar4
#SBATCH --job-name=nvcrl-v1
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@90
set -Eeuo pipefail
export NVC_DEBUG_RUN=${1:?frozen run required}
export NVC_DEBUG_PROJECT=${NVC_RL_ROOT:?export NVC_RL_ROOT=<your workspace, see voicechat/README.md>}   # exported by submit_run.sh (sbatch exports the env)
export NVC_DEBUG_LOCAL="/raid/scratch/$USER/super35-draco-$SLURM_JOB_ID"
run=$NVC_DEBUG_RUN
export NVC_INPUT_MODE=$(cat "$run/INPUT_MODE" 2>/dev/null || printf 'legacy-v1')
if [[ "$NVC_INPUT_MODE" == gym ]]; then
 image=$(cat "$run/RUNTIME_IMAGE")
 sandbox_image=$(cat "$run/SANDBOX_IMAGE")
 [[ "$image" == /* && -f "$image" && "$sandbox_image" == /* && -f "$sandbox_image" ]]
else
 # Non-gym (OPD / text) modes: prefer a frozen RUNTIME_IMAGE, else the CS copy of the MOPD v4 image.
 image=$(cat "$run/RUNTIME_IMAGE" 2>/dev/null || printf '%s' "${NVC_LEGACY_RUNTIME_IMAGE:-/lustre/fsw/portfolios/llmservice/users/cchen1/containers/NemotronVC-RL-MOPD-vllm0271-node-dp8-v4.sqsh}")
 [[ "$image" == /* && -f "$image" ]] || { echo "Runtime image not found: $image" >&2; exit 2; }
fi
export NVC_EXPECTED_NODES=${SLURM_NNODES:?}
export NVC_RUN_KIND=$(cat "$run/KIND")
[[ $(cat "$run/NODES") == "$NVC_EXPECTED_NODES" ]]
# Optional simulator nodes (make_run.py --sim-nodes N writes run/SIM_NODES + run/SIM_SPEC.json):
# the LAST N hosts of the allocation run scripts/sim_role.sh (standalone vLLM) instead of joining
# Ray, so the Ray head stays the first host. NVC_SIM_BASE_URL / NVC_SIM_API_KEY are launcher env
# (not secrets): inherited by every srun, kept by ray_role.sh, and resolved by Gym's sim_gptoss
# server through ${oc.env:...}. Runs without SIM_NODES keep NVC_RAY_NODES == NVC_EXPECTED_NODES.
# Simulator receipts are job-bound: HOSTS and READY/FAILED-<host>.json live under run/sim/<job id>/,
# which must not pre-exist (a resubmitted allocation never reads another job's receipts).
export NVC_SIM_NODES=$(cat "$run/SIM_NODES" 2>/dev/null || printf 0)
[[ "$NVC_SIM_NODES" =~ ^[0-9]+$ ]]
export NVC_RAY_NODES=$((NVC_EXPECTED_NODES - NVC_SIM_NODES))
(( NVC_RAY_NODES >= 1 ))
if (( NVC_SIM_NODES > 0 )); then
 [[ -s "$run/SIM_SPEC.json" ]]
 sim_port=$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["port"]))' "$run/SIM_SPEC.json")
 [[ "$sim_port" =~ ^[0-9]+$ ]]
 mapfile -t sim_hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n "$NVC_SIM_NODES")
 (( ${#sim_hosts[@]} == NVC_SIM_NODES ))
 sim_job_dir="$run/sim/$SLURM_JOB_ID"
 [[ ! -e "$sim_job_dir" && ! -L "$sim_job_dir" ]] || { echo "Simulator receipt directory already exists: $sim_job_dir" >&2; exit 75; }
 mkdir -p "$sim_job_dir"
 printf '%s\n' "${sim_hosts[@]}" > "$sim_job_dir/HOSTS"
 # All simulator hosts, comma-separated: Gym's openai_model (patched) opens one client per URL and
 # sends each request to the least-loaded one, so N sim nodes scale the user/tool simulator N-fold.
 sim_urls=()
 for sim_host in "${sim_hosts[@]}"; do sim_urls+=("http://$sim_host:$sim_port/v1"); done
 export NVC_SIM_BASE_URL=$(IFS=,; printf '%s' "${sim_urls[*]}")
 export NVC_SIM_API_KEY=nvc-local-sim
 printf '[NVC_SIM_NODES] job=%s sim_nodes=%s ray_nodes=%s hosts=%s base_url=%s receipts=%s\n' "$SLURM_JOB_ID" "$NVC_SIM_NODES" "$NVC_RAY_NODES" "${sim_hosts[*]}" "$NVC_SIM_BASE_URL" "$sim_job_dir"
 unset sim_port sim_hosts sim_host sim_urls sim_job_dir
fi
# Secrets for Gym openai_model servers (run/SECRETS_REQUIRED, written by make_run.py from the
# ${oc.env:NAME} references in the frozen services yaml). Values are read from .secrets/ at
# allocation time, exported only (never echoed, never written), inherited by gym_sandbox_pair.py's
# main srun (the sandbox sidecar strips them) and scoped further by ray_role.sh. Runs without the
# file keep the historical behavior (no secret in the job environment).
if [[ -s "$run/SECRETS_REQUIRED" ]]; then
 while IFS= read -r secret_name; do
   [[ -n "$secret_name" ]] || continue
   case "$secret_name" in
     NVIDIA_API_KEY) secret_file="$NVC_DEBUG_PROJECT/.secrets/nvidia-api-key";;
     *) echo "Unknown secret requested by run: $secret_name" >&2; exit 2;;
   esac
   [[ -s "$secret_file" ]] || { echo "Secret file for $secret_name is missing or empty" >&2; exit 2; }
   IFS= read -r secret_value < "$secret_file" || true
   [[ -n "$secret_value" ]] || { echo "Secret $secret_name has an empty first line" >&2; exit 2; }
   export "$secret_name=$secret_value"
   unset secret_value secret_file
 done < "$run/SECRETS_REQUIRED"
 unset secret_name
fi
# Absolute allocation time includes container import and model initialization.
start_text=$(scontrol show job -o "$SLURM_JOB_ID" | tr ' ' '\n' | sed -n 's/^StartTime=//p')
export NVC_ALLOCATION_START_EPOCH=$(date -d "$start_text" +%s)
export NVC_CHECKPOINT_DEADLINE_EPOCH=$((NVC_ALLOCATION_START_EPOCH + 13200))
export NVC_ALLOCATION_END_EPOCH=$((NVC_ALLOCATION_START_EPOCH + 14400))
# nemo_rl's logger measures container/startup overhead from this epoch.
export NRL_JOB_START_EPOCH=$NVC_ALLOCATION_START_EPOCH
# One real writer per lineage even if an operator submits twice.
lineage=$(sed -n 's/^  checkpoint_dir: //p' "$run/config.yaml")
[[ -n "$lineage" ]]
mkdir -p "$lineage"
exec 9>"$lineage/.writer.lock"
flock -n 9 || { echo 'Another writer owns this lineage' >&2; exit 75; }
printf '%s\n' "$SLURM_JOB_ID" > "$lineage/ACTIVE_JOB_ID"
(cd "$run"; sha256sum -c inputs.sha256)
cleanup_init_mcore() (
 set -Eeuo pipefail
 trap 'echo "Initialization cache cleanup refused or failed; inspect the retained path" >&2' ERR
 project_real=$(readlink -f "${2:?project required}")
 [[ -d "${1:?run required}" && ! -L "$1" ]]
 run_real=$(readlink -f "$1")
 [[ $(dirname "$run_real") == "$project_real/runs" ]]
 [[ ${run_real##*/} =~ ^(smoke|train)-v1-[0-9]{8}T[0-9]{6}Z$ ]]
 [[ ${3:?job required} =~ ^[0-9]+$ ]]
 owner_uid=$(id -u)
 [[ $(stat -c %u "$run_real") == "$owner_uid" ]]
 cache="$run_real/init-mcore"
 receipt="$run_real/INIT_MCORE_CLEANUP.json"
 [[ ! -e "$receipt" && ! -L "$receipt" ]]
 existed=false
 if [[ -e "$cache" || -L "$cache" ]]; then
   [[ -d "$cache" && ! -L "$cache" && $(readlink -f "$cache") == "$cache" ]]
   invalid_entry=$(find "$cache" -xdev \( -type l -o ! -uid "$owner_uid" \) -print -quit)
   [[ -z "$invalid_entry" ]]
   existed=true
   rm -rf --one-file-system -- "$cache"
   [[ ! -e "$cache" && ! -L "$cache" ]]
 fi
 printf '{"job_id":"%s","path":"%s","main_exit_code":0,"node_cleanup_exit_code":0,"end_code":0,"cache_existed":%s,"cache_absent":true}\n' \
   "$3" "$cache" "$existed" > "$receipt"
)
pair_pid=
cleanup() {
 code=$?
 trap - EXIT
 set +e
 # Join only our owned sidecar/main supervisor before touching node scratch.
 if [[ -n "$pair_pid" ]]; then
   kill -TERM "$pair_pid" 2>/dev/null || true
   wait "$pair_pid"
   pair_code=$?
   (( code != 0 )) || code=$pair_code
   pair_pid=
 fi
 # A failed rank can make srun terminate other role shells before their EXIT
 # traps finish. Recover each node's Ray logs from the host-visible /tmp bind.
 srun --nodes="$NVC_EXPECTED_NODES" --ntasks="$NVC_EXPECTED_NODES" --ntasks-per-node=1 --cpus-per-task=1 --mpi=none \
  bash -c '
   set -eu
   [[ "$NVC_DEBUG_LOCAL" == "/raid/scratch/$USER/super35-draco-$SLURM_JOB_ID" ]]
   local_real=$(readlink -f "$NVC_DEBUG_LOCAL")
   copied=false
   # RAY_TMPDIR is a system-temp override in Ray 2.56, which appends /ray.
   # Recover either layout, resolving only sessions inside this job scratch.
   for layout in direct nested; do
     ray_root="$NVC_DEBUG_LOCAL/tmp/ray"
     [[ "$layout" != nested ]] || ray_root="$ray_root/ray"
     [[ -L "$ray_root/session_latest" ]] || continue
     latest=$(readlink "$ray_root/session_latest")
     session_name=${latest##*/}
     [[ "$session_name" == session_* && "$session_name" != session_latest ]] || continue
     logs=$(readlink -f "$ray_root/$session_name/logs") || continue
     [[ -n "$local_real" && "$logs" == "$local_real/tmp/"* && -d "$logs" ]] || continue
     destination="$NVC_DEBUG_RUN/ray/$(hostname)/$layout"
     mkdir -p "$destination"
     cp -a "$logs/." "$destination/"
     copied=true
   done
   if [[ -d "$NVC_DEBUG_LOCAL/sandbox-container/runtime" ]]; then
     sandbox_logs="$NVC_DEBUG_RUN/sandbox/$SLURM_JOB_ID/logs-$(hostname)"
     mkdir -p "$sandbox_logs"
     cp -a "$NVC_DEBUG_LOCAL/sandbox-container/runtime/." "$sandbox_logs/"
   fi
   rm -rf -- "$NVC_DEBUG_LOCAL"
   [[ ! -e "$NVC_DEBUG_LOCAL" ]]
   printf "{\"host\":\"%s\",\"ray_logs_copied\":%s,\"scratch_removed\":true}\n" \
     "$(hostname)" "$copied" > "$NVC_DEBUG_RUN/cleanup-$(hostname).json"
  '
 cleanup_code=$?
 # The main srun has joined every role's Ray-stop trap, then node cleanup has
 # completed. Failed/cancelled jobs keep this shared cache for diagnosis.
 init_cache_cleanup_code=0
 init_cache_cleanup_attempted=false
 if (( code == 0 && cleanup_code == 0 )) && [[ -f "$run/END_CODE" && $(cat "$run/END_CODE") == 0 ]]; then
   init_cache_cleanup_attempted=true
   cleanup_init_mcore "$run" "$NVC_DEBUG_PROJECT" "$SLURM_JOB_ID"
   init_cache_cleanup_code=$?
 fi
 printf '{"job_id":"%s","exit_code":%s,"cleanup_exit_code":%s,"init_cache_cleanup_attempted":%s,"init_cache_cleanup_exit_code":%s}\n' \
   "$SLURM_JOB_ID" "$code" "$cleanup_code" "$init_cache_cleanup_attempted" "$init_cache_cleanup_code" > "$run/exit.json"
 (( code != 0 )) || code=$cleanup_code
 (( code != 0 )) || code=$init_cache_cleanup_code
 exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
srun --nodes="$NVC_EXPECTED_NODES" --ntasks="$NVC_EXPECTED_NODES" --ntasks-per-node=1 --cpus-per-task=8 --mpi=none --kill-on-bad-exit=1 \
 bash -c 'set -eu; mkdir -p "$NVC_DEBUG_LOCAL/source" "$NVC_DEBUG_LOCAL/tmp" "$NVC_DEBUG_LOCAL/sandbox-tmp" "$NVC_DEBUG_LOCAL/sandbox-container"; cd "$NVC_DEBUG_RUN"; sha256sum -c source.sha256; tar -xzf source.tar.gz -C "$NVC_DEBUG_LOCAL/source"'
mounts="/lustre:/lustre,/raid/scratch:/raid/scratch,$NVC_DEBUG_LOCAL/source:/workspace/nemo-rl:ro,$NVC_DEBUG_LOCAL/tmp:/tmp"
# Bridge builds its Python-ABI-specific dataset helper at learner startup.
# Permit build products only in this node-local extracted subtree; the frozen
# run/source archive and the rest of the source mount remain read-only.
dataset_helpers=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets
[[ -f "$NVC_DEBUG_LOCAL/source/$dataset_helpers/Makefile" ]]
mounts+=",$NVC_DEBUG_LOCAL/source/$dataset_helpers:/workspace/nemo-rl/$dataset_helpers"
if [[ "$NVC_INPUT_MODE" == gym ]]; then
 head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
 python3 "$run/scripts/gym_sandbox_pair.py" --run "$run" --head "$head" \
  --image "$image" --sandbox-image "$sandbox_image" --mounts "$mounts" &
 pair_pid=$!
 wait "$pair_pid"
 pair_pid=
else
srun --nodes="$NVC_EXPECTED_NODES" --ntasks="$NVC_EXPECTED_NODES" --ntasks-per-node=1 --cpus-per-task=64 --gpus-per-task=8 \
 --gpu-bind=none --cpu-bind=none --mpi=none --kill-on-bad-exit=1 \
 --no-container-mount-home --container-image="$image" --container-mounts="$mounts" \
 --container-workdir=/workspace/nemo-rl --output="$run/rank-%t-%N.log" \
 bash "$run/scripts/ray_role.sh"
fi
[[ $(cat "$run/END_CODE") == 0 ]]
