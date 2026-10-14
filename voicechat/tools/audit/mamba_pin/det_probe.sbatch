#!/usr/bin/env bash
# One 8xA100 node: 8 fresh processes (one per GPU) run the per-rank Mamba SSD fwd+bwd (scripts/audit/mamba_pin/tune_ssd.py
# --probe) twice: free autotuning (current RL) and MAMBA_DETERMINISTIC=1 (mamba_ssm's own deterministic mode: one
# config per kernel at import + zero-initialised tile workspaces instead of atomics). Summary: distinct forward/grad
# hashes across GPUs, max relative differences, det-vs-free gradient difference (correctness), seconds per fwd+bwd.
# Usage: sbatch --account A --partition P --time 00:30:00 scripts/audit/mamba_pin/det_probe.sbatch OUT_DIR
#SBATCH --job-name=nvc-mamba-det
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --exclusive
set -Eeuo pipefail
P=${NVC_RL_ROOT:?export NVC_RL_ROOT=<your workspace, see voicechat/README.md>}
export PIN_OUT=${1:?out dir}
mkdir -p "$PIN_OUT"
srun -N1 -n1 --cpus-per-task=32 --gpus-per-task=8 --mem=0 --gpu-bind=none --cpu-bind=none --mpi=none \
 --no-container-mount-home --container-image="$P/runtime/gym-v1/NemotronVC-Gym34-vllm0271-v1.sqsh" \
 --container-mounts="/lustre:/lustre,/raid/scratch:/raid/scratch,$P:/workspace/debug" --container-workdir=/workspace/debug bash -c '
  set -Eeuo pipefail
  P=${NVC_RL_ROOT:?export NVC_RL_ROOT=<your workspace, see voicechat/README.md>}; PY=/opt/nemotronvc/venvs/mcore/bin/python; T=$P/scripts/audit/mamba_pin/tune_ssd.py
  for mode in free det; do
    for g in 0 1 2 3 4 5 6 7; do
      env_det=""; [ $mode = det ] && env_det="MAMBA_DETERMINISTIC=1"
      env $env_det TRITON_CACHE_DIR=/tmp/triton-$mode-$g-$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$g $PY $T --probe $PIN_OUT/probe_${mode}_gpu$g.json > $PIN_OUT/probe_${mode}_gpu$g.log 2>&1 &
    done
    wait
  done
  $PY - <<PYEOF
import json, glob, collections, torch
out = "$PIN_OUT"
rel = lambda a, b: max(((x.float() - y.float()).norm() / (y.float().norm() + 1e-12)).item() for x, y in zip(a, b))
summary, ts = {}, {}
for mode in ("free", "det"):
    files = sorted(glob.glob(out + "/probe_%s_gpu*.json" % mode))
    reps = [json.load(open(f)) for f in files]
    ts[mode] = [torch.load(f + ".pt") for f in files]
    summary[mode] = {"processes": len(reps), "deterministic_mode": reps[0]["deterministic_mode"],
                     "autotuners_with_several_configs": reps[0]["autotuners_with_several_configs"],
                     "distinct_forward_hashes": len({r["forward_sha256"] for r in reps}),
                     "distinct_grad_hashes": len({r["grads_sha256"] for r in reps}),
                     "forward_max_rel_diff_across_gpus": max(rel(t["fwd"], ts[mode][0]["fwd"]) for t in ts[mode]),
                     "grads_max_rel_diff_across_gpus": max(rel(t["grads"], ts[mode][0]["grads"]) for t in ts[mode]),
                     "kernels_with_different_choice": sorted(k for k in reps[0]["chosen"] if len({str(r["chosen"].get(k)) for r in reps}) > 1),
                     "seconds_per_fwd_bwd_median_over_gpus": sorted(r["seconds_per_fwd_bwd_median"] for r in reps)[len(reps) // 2]}
summary["det_vs_free_gpu0"] = {"forward_rel": rel(ts["det"][0]["fwd"], ts["free"][0]["fwd"]),
                               "grads_rel_per_tensor": [round(((x.float() - y.float()).norm() / (y.float().norm() + 1e-12)).item(), 8)
                                                        for x, y in zip(ts["det"][0]["grads"], ts["free"][0]["grads"])]}
json.dump(summary, open(out + "/PROBE_SUMMARY.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
PYEOF
 ' </dev/null
