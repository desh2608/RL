#!/usr/bin/env bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -Eeuo pipefail
unset NVIDIA_API_KEY OPENAI_API_KEY INFERENCE_API_KEY
umask 0077
helper=${1:?frozen helper path}; shift
image=/lustre/fs11/portfolios/nemotron/projects/nemotron_speechprod_chat/kevinhu/containers/vllm-super-v0.27.1-evals.sqsh
task_local="/raid/scratch/$USER/opd-eval-prepare-$SLURM_JOB_ID"
mkdir -p "$task_local"/{tmp,cache,enroot/data,enroot/runtime,enroot/cache,enroot/tmp}
export ENROOT_DATA_PATH="$task_local/enroot/data" ENROOT_RUNTIME_PATH="$task_local/enroot/runtime"
export ENROOT_CACHE_PATH="$task_local/enroot/cache" ENROOT_TEMP_PATH="$task_local/enroot/tmp"
export TMPDIR="$task_local/tmp" HF_HOME="$task_local/cache/hf" HF_MODULES_CACHE="$task_local/cache/hf_modules"
export PYTHONPYCACHEPREFIX="$task_local/cache/pycache" XDG_CACHE_HOME="$task_local/cache/xdg"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
status=0
srun -N1 -n1 --cpus-per-task=16 --mem=64G --mpi=none --cpu-bind=none \
  --no-container-mount-home --container-image="$image" \
  --container-mounts=/lustre:/lustre,/raid/scratch:/raid/scratch \
  bash -lc 'exec python3 "$@"' bash "$helper" "$@" < /dev/null || status=$?
if (( status == 0 )); then
  case "$task_local" in /raid/scratch/"$USER"/opd-eval-prepare-"$SLURM_JOB_ID") rm -rf -- "$task_local";; *) exit 3;; esac
fi
exit "$status"
