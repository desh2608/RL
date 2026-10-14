#!/bin/bash
# Reusable CPU-only task runner (cs-oci-ord). Each task owns its script, logs and receipt.
# Usage: sbatch scripts/cpu_task.sbatch /abs/path/under/receipts/<task-dir>   (task-dir must contain task.sh)
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
set -Eeuo pipefail
export TZ=UTC
P=${NVC_RL_ROOT:?export NVC_RL_ROOT=<your workspace, see voicechat/README.md>}
D=$(readlink -f "${1:?task directory required}")
[[ "$D" == "$(readlink -f "$P")/receipts/"* && -f "$D/task.sh" ]]
[[ ! -e "$D/CPU_CLEANUP.json" ]]
printf '%s\n' "$SLURM_JOB_ID" > "$D/CPU_JOB_ID"
image=$P/runtime/gym-v1/NemotronVC-Gym34-vllm0271-v1.sqsh
[[ -f "$image" ]]
scratch="/raid/scratch/$USER/nvcrl-cpu-$SLURM_JOB_ID"
[[ ! -e "$scratch" && ! -L "$scratch" ]]
mkdir -p "$scratch"/{tmp,enroot-runtime,enroot-cache,enroot-data}
export ENROOT_RUNTIME_PATH="$scratch/enroot-runtime"
export ENROOT_CACHE_PATH="$scratch/enroot-cache"
export ENROOT_DATA_PATH="$scratch/enroot-data"
export ENROOT_TEMP_PATH="$scratch/tmp"
cleanup() {
 code=$?
 trap - EXIT
 set +e
 remaining=$(find "$ENROOT_DATA_PATH" -mindepth 1 -maxdepth 1 -type d -print)
 if [[ -n "$remaining" ]]; then
   printf '{"job_id":"%s","status":"needs_container_cleanup","main_exit":%s}\n' "$SLURM_JOB_ID" "$code" > "$D/CPU_CLEANUP.json"
   exit 90
 fi
 [[ "$scratch" == "/raid/scratch/$USER/nvcrl-cpu-$SLURM_JOB_ID" && ! -L "$scratch" && $(stat -c %u "$scratch") == $(id -u) ]] || exit 91
 rm -rf --one-file-system -- "$scratch"
 [[ ! -e "$scratch" ]] || exit 92
 printf '{"job_id":"%s","status":"complete","main_exit":%s,"anonymous_container_data_empty":true,"scratch_removed":true}\n' "$SLURM_JOB_ID" "$code" > "$D/CPU_CLEANUP.json"
 exit "$code"
}
trap cleanup EXIT
# Optional task-owned scheduler evidence runs on the host, not in the container.
if [[ -f "$D/host_preflight.sh" ]]; then bash "$D/host_preflight.sh" "$D"; fi
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --mpi=none --cpu-bind=none \
 --container-image="$image" --container-mounts="/lustre:/lustre,/raid/scratch:/raid/scratch,$P/src/nemo-rl:/workspace/nemo-rl,$P:/workspace/debug,$scratch/tmp:/tmp" \
 --no-container-mount-home --container-workdir=/workspace/nemo-rl \
 bash -c 'set -Eeuo pipefail; export HF_HOME=/tmp/hf HF_MODULES_CACHE=/tmp/hf/modules; mkdir -p "$HF_MODULES_CACHE"; source /workspace/debug/scripts/runtime_env.sh; bash "$1/task.sh" "$1"' bash "$D" </dev/null
