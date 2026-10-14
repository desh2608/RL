#!/usr/bin/env bash
# Per-prompt pass rate of the RL start policy on candidate rows (pivot_data/clean_candidates.py output), K samples at
# temperature 1.0 with the same Gym verifiers as RL training. One 8xA100 node: stock `vllm serve` (TP1 DP8 EP8) with
# chat_template_content_format=string (system rendered as text, as in eval) and an fp32 SSM cache, checked by
# RENDER_CHECK.json before any route runs. Output: OUT_DIR/{per_prompt.jsonl,route_summary.json,SUMMARY.md,rollouts/}.
# Usage: NVC_RL_ROOT=... NVC_MODEL_PATH=<start policy HF dir> [NVC_MODEL_LABEL=name] [NVC_PROFILE_SERVICES_YAML=...] \
#        sbatch --account A --partition polar4 --time 04:00:00 scripts/profile_passrate/profile_candidates.sbatch \
#        OUT_DIR SAMPLES_DIR ROUTE_KEYS(comma) [K]
#SBATCH --job-name=nvc-profile
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --no-requeue
set -Eeuo pipefail
export TZ=UTC
P=${NVC_RL_ROOT:?export NVC_RL_ROOT=<your workspace, see voicechat/README.md>}
export NVC_MODEL_PATH=${NVC_MODEL_PATH:?export NVC_MODEL_PATH=<start policy HF dir (hf-aligned view)>}
export NVC_MODEL_LABEL=${NVC_MODEL_LABEL:-policy}
export NVC_PROFILE_SERVICES_YAML=${NVC_PROFILE_SERVICES_YAML:-$P/scripts/profile_passrate/gym_profile_pivot.yaml}
export PROF_OUT=${1:?out} PROF_SAMPLES=${2:?samples dir with routes.json} PROF_ROUTES=${3:?route keys} PROF_K=${4:-8} NVC_RL_ROOT=$P
export PROF_END=$(( $(date +%s) + $(squeue -h -j "$SLURM_JOB_ID" -o %L </dev/null | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1); if(n>2) s+=3600*$(n-2); if(n>3) s+=86400*$(n-3); print s}') - 900 ))
mkdir -p "$PROF_OUT"
srun -N1 -n1 --cpus-per-task=64 --gpus-per-task=8 --mem=0 --gpu-bind=none --cpu-bind=none --mpi=none \
 --no-container-mount-home --container-image="$P/runtime/gym-v1/NemotronVC-Gym34-vllm0271-v1.sqsh" \
 --container-mounts="/lustre:/lustre,/raid/scratch:/raid/scratch,$P/src/nemo-rl:/workspace/nemo-rl,$P:/workspace/debug" \
 --container-workdir=/workspace/debug bash -c '
  set -Eeuo pipefail
  bash $NVC_RL_ROOT/scripts/profile_passrate/launch_profile.sh --samples "$PROF_SAMPLES" --content-format string \
    --mamba-cache-dtype float32 --out "$PROF_OUT" --k "$PROF_K" --max-num-seqs 48 --max-model-len 65536 \
    --routes "$PROF_ROUTES" --deadline-epoch "$PROF_END"
 ' </dev/null
