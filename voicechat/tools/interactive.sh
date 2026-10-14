#!/usr/bin/env bash
# Interactive allocation helper for cs-oci-ord.
#   interactive.sh alloc cpu|gpu|multi N     hold an allocation with a sleep batch job, print JOB
#   interactive.sh shell JOB [NODE]          pty shell inside the project container (first call imports the image)
#   interactive.sh exec  JOB 'bash -c body'  run one command inside the container, stdin from /dev/null
#   interactive.sh release JOB               remove node-local scratch and cancel exactly this job
#   interactive.sh status                    list this helper's allocations
# Env overrides: NVC_SLURM_ACCOUNT, NVC_PARTITION, NVC_RUNTIME_IMAGE, NVC_MOUNTS, NVC_CPUS, NVC_MEM, NVC_TIME.
set -Eeuo pipefail
export TZ=UTC
P=${NVC_RL_ROOT:-${NVC_PROJECT:?export NVC_RL_ROOT=<your workspace>}}
ACCOUNT=${NVC_SLURM_ACCOUNT_CPU:-${NVC_SLURM_ACCOUNT%% *}}; [[ -n "$ACCOUNT" ]] || { echo 'export NVC_SLURM_ACCOUNT_CPU=<slurm account>' >&2; exit 2; }
IMG=${NVC_RUNTIME_IMAGE:-$P/runtime/gym-v1/NemotronVC-Gym34-vllm0271-v1.sqsh}
MOUNTS=${NVC_MOUNTS:-/lustre:/lustre,/raid/scratch:/raid/scratch,$P/src/nemo-rl:/workspace/nemo-rl,$P:/workspace/debug}
LOGDIR=$P/logs/interactive
mkdir -p "$LOGDIR"

die() { echo "interactive.sh: $*" >&2; exit 2; }
job_field() { squeue -j "$1" -h -o "$2" 2>/dev/null | head -n1; }
require_running() {
  local state; state=$(job_field "$1" %T)
  [[ "$state" == RUNNING ]] || die "job $1 is not RUNNING (state='${state:-gone}')"
}
is_gpu_job() { [[ $(job_field "$1" %b) == *gpu* ]]; }
container_name() { printf 'nvcrl-%s' "$1"; }

wait_running() {
  local job=$1 i
  for ((i=0; i<90; i++)); do
    [[ $(job_field "$job" %T) == RUNNING ]] && { echo "[RUNNING] job=$job node=$(job_field "$job" %N)"; return 0; }
    sleep 10
  done
  echo "[PENDING] job=$job still waiting after 15 min; reason=$(job_field "$job" %R)"
}

alloc() {
  local kind=${1:?alloc cpu|gpu|multi N}; shift || true
  local time=${NVC_TIME:-04:00:00} secs job
  secs=$(( $(echo "$time" | awk -F: '{print $1*3600+$2*60+$3}') - 100 ))
  local wrap="echo \"[READY] job=\$SLURM_JOB_ID host=\$(hostname)\"; sleep $secs"
  case "$kind" in
    cpu)
      job=$(sbatch --parsable --account="$ACCOUNT" --partition="${NVC_PARTITION:-cpu_interactive}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${NVC_CPUS:-32}" --mem="${NVC_MEM:-64G}" --time="$time" --no-requeue \
        --job-name=nvcrl-cpu --output="$LOGDIR/cpu-%j.log" --wrap "$wrap");;
    gpu)
      job=$(sbatch --parsable --account="$ACCOUNT" --partition="${NVC_PARTITION:-interactive_singlenode}" --nodes=1 --ntasks-per-node=1 \
        --gres=gpu:8 --cpus-per-task="${NVC_CPUS:-64}" --mem="${NVC_MEM:-0}" --exclusive --time="$time" --no-requeue \
        --job-name=nvcrl-gpu --output="$LOGDIR/gpu-%j.log" --wrap "$wrap");;
    multi)
      local n=${1:?alloc multi N}
      job=$(sbatch --parsable --account="$ACCOUNT" --partition="${NVC_PARTITION:-polar4}" --nodes="$n" --ntasks-per-node=1 \
        --gres=gpu:8 --cpus-per-task="${NVC_CPUS:-64}" --mem=0 --exclusive --time="$time" --no-requeue \
        --job-name=nvcrl-multi --output="$LOGDIR/multi-%j.log" --wrap "$wrap");;
    *) die "unknown alloc kind '$kind'";;
  esac
  job=${job%%;*}
  echo "[SUBMITTED] job=$job kind=$kind account=$ACCOUNT"
  wait_running "$job"
  echo "$job"
}

# First import of the 82 GB image must never run twice concurrently for one container name.
ensure_container() {
  local job=$1 name; name=$(container_name "$job")
  local marker="$LOGDIR/container-$job.ready" lock="$LOGDIR/.container-$job.lock"
  [[ -f "$marker" ]] && return 0
  exec 7>"$lock"
  flock 7
  if [[ ! -f "$marker" ]]; then
    echo "[IMPORT] creating container $name from $IMG (several minutes on first use)" >&2
    srun --jobid="$job" --overlap -N1 -n1 --cpus-per-task=8 --mem=64G --mpi=none \
      --container-name="$name" --container-image="$IMG" --container-mounts="$MOUNTS" \
      --no-container-mount-home --container-workdir=/workspace/debug true </dev/null
    printf '%s\n' "$name" > "$marker"
  fi
  exec 7>&-
}

step_flags() {
  local job=$1
  if is_gpu_job "$job"; then
    printf -- '--gres=gpu:8 --gpu-bind=none --cpus-per-task=%s --mem=0' "${NVC_CPUS:-64}"
  else
    printf -- '--cpus-per-task=%s --mem=%s' "${NVC_CPUS:-16}" "${NVC_MEM:-32G}"
  fi
}

shell() {
  local job=${1:?shell JOB [NODE]}; local node=${2:-}
  require_running "$job"; ensure_container "$job"
  local -a w=(); [[ -n "$node" ]] && w=(-w "$node")
  # shellcheck disable=SC2046
  exec srun --jobid="$job" --overlap -N1 -n1 "${w[@]}" $(step_flags "$job") --mpi=none --cpu-bind=none \
    --container-name="$(container_name "$job")" --container-mounts="$MOUNTS" \
    --no-container-mount-home --container-workdir=/workspace/nemo-rl --pty bash --noprofile --norc
}

exec_cmd() {
  local job=${1:?exec JOB 'command'}; local body=${2:?command body}
  require_running "$job"; ensure_container "$job"
  # shellcheck disable=SC2046
  srun --jobid="$job" --overlap -N1 -n1 $(step_flags "$job") --mpi=none --cpu-bind=none \
    --container-name="$(container_name "$job")" --container-mounts="$MOUNTS" \
    --no-container-mount-home --container-workdir=/workspace/debug \
    bash -c "$body" </dev/null
}

release() {
  local job=${1:?release JOB}
  [[ "$job" =~ ^[0-9]+$ ]] || die "job id must be numeric"
  if [[ $(job_field "$job" %T) == RUNNING ]]; then
    srun --jobid="$job" --overlap -N1 -n1 --cpus-per-task=1 --mem=1G --mpi=none \
      bash -c 'd="/raid/scratch/$USER/nvcrl-'"$job"'"; [[ -d "$d" && ! -L "$d" ]] && rm -rf --one-file-system -- "$d"; echo "[SCRATCH] $d removed=$([[ ! -e "$d" ]] && echo true || echo false)"' </dev/null || true
  fi
  scancel "$job" && echo "[CANCELLED] job=$job"
  rm -f "$LOGDIR/container-$job.ready" "$LOGDIR/.container-$job.lock"
}

status() { squeue -u "$USER" -n nvcrl-cpu,nvcrl-gpu,nvcrl-multi -o "%.10i %.22P %.12j %.8T %.10M %.6D %R"; }

cmd=${1:-}; shift || true
case "$cmd" in
  alloc) alloc "$@";;
  shell) shell "$@";;
  exec) exec_cmd "$@";;
  release) release "$@";;
  status) status;;
  *) sed -n '2,9p' "$0"; exit 2;;
esac
