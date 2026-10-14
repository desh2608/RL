#!/usr/bin/env bash
# Gradient-sync GPU probe: runs scripts/audit/gradsync_check/gradsync_probe.py on the frozen source + image of a run.
# Usage: sbatch --nodes N [--account/--partition/--time] gradsync_job.sbatch FROZEN_RUN OUT_DIR MCORE_CACHE "MODE [MODE...]"
#SBATCH --job-name=nvc-gradsync
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --no-requeue
set -Eeuo pipefail
export TZ=UTC
export GS_RUN=${1:?frozen run}; export GS_OUT=${2:?out dir}; export GS_CACHE=${3:?mcore cache dir}; GS_MODES=${4:-normal}
export GS_PROBE=${NVC_RL_ROOT:?}/scripts/audit/gradsync_check/gradsync_probe.py
export GS_LOCAL="/raid/scratch/$USER/nvc-gradsync-$SLURM_JOB_ID"
image=$(cat "$GS_RUN/RUNTIME_IMAGE"); [[ -f "$image" ]]
mkdir -p "$GS_OUT" "$GS_CACHE"
export GS_HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
printf '[GRADSYNC] job=%s nodes=%s head=%s run=%s modes=%s image=%s\n' "$SLURM_JOB_ID" "$SLURM_NNODES" "$GS_HEAD" "$GS_RUN" "$GS_MODES" "$image"
cleanup() { srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --cpus-per-task=1 --mem=0 --mpi=none bash -c 'rm -rf -- "$GS_LOCAL"' || true; }
trap cleanup EXIT
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --cpus-per-task=8 --mem=0 --mpi=none --kill-on-bad-exit=1 \
 bash -c 'set -eu; mkdir -p "$GS_LOCAL/source" "$GS_LOCAL/tmp"; cd "$GS_RUN"; sha256sum -c source.sha256 >/dev/null; tar -xzf source.tar.gz -C "$GS_LOCAL/source"'
mounts="/lustre:/lustre,/raid/scratch:/raid/scratch,$GS_LOCAL/source:/workspace/nemo-rl:ro,$GS_LOCAL/tmp:/tmp"
i=0
for mode in $GS_MODES; do
 i=$((i+1))
 export GS_MODE=$mode GS_PORT=$((21000 + (SLURM_JOB_ID % 4000) + 11 * i))
 echo "[GRADSYNC] mode=$mode start $(date -u +%H:%M:%SZ) port=$GS_PORT"
 srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --cpus-per-task=64 --gpus-per-task=8 \
  --mem=0 --gpu-bind=none --cpu-bind=none --mpi=none --kill-on-bad-exit=1 \
  --no-container-mount-home --container-image="$image" --container-name="nvc-gradsync-$SLURM_JOB_ID" --container-mounts="$mounts" \
  --container-workdir=/workspace/nemo-rl --output="$GS_OUT/log-$mode-node%n-%N.log" \
  bash -c '
   set -Eeuo pipefail
   node_rank=$SLURM_NODEID; nnodes=$SLURM_NNODES
   export HF_HOME=/tmp/hf-home HF_MODULES_CACHE=/tmp/hf-home/modules TMPDIR=/tmp/work
   export XDG_CACHE_HOME=/tmp/cache/xdg TRITON_CACHE_DIR=/tmp/cache/triton TORCHINDUCTOR_CACHE_DIR=/tmp/cache/inductor CUDA_CACHE_PATH=/tmp/cache/cuda
   mkdir -p "$HF_MODULES_CACHE" "$TMPDIR" /tmp/cache
   source "$GS_RUN/scripts/runtime_env.sh"
   unset NVC_VERIFY_HDO_RESUME NVC_REQUIRE_OPTIMIZER_PROGRESS NVC_HDO_RESUME_RECEIPT_DIR
   export NRL_MEGATRON_CHECKPOINT_DIR="$GS_CACHE"
   for v in $(env | awk -F= "/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print \$1}"); do unset "$v"; done
   exec /opt/nemotronvc/venvs/mcore/bin/python -m torch.distributed.run --nnodes "$nnodes" --nproc-per-node 8 \
     --node-rank "$node_rank" --master-addr "$GS_HEAD" --master-port "$GS_PORT" \
     "$GS_PROBE" --config "$GS_RUN/config.yaml" --mode "$GS_MODE" --out "$GS_OUT/$GS_MODE"
  ' || echo "[GRADSYNC] mode=$mode FAILED rc=$?"
 echo "[GRADSYNC] mode=$mode end $(date -u +%H:%M:%SZ) DONE=$(test -f "$GS_OUT/$mode/DONE.json" && echo yes || echo no)"
done
