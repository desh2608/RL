#!/usr/bin/env bash
# One idempotent pass of the post-SFT routine (login node): after the HF export + hf-aligned prepare of LINEAGE step STEP,
# keep only the last MCore checkpoint, launch bfcl_v3, and once its MODEL.json exists launch tau2 (staggered, user rule:
# test only tau2 + bfcl_v3). Prints one status word: FAILED / WAIT / BFCL_LAUNCHED / TAU2_LAUNCHED / DONE.
# Usage: auto_eval_step.sh LINEAGE STEP "JOBID,JOBID,..."   (job ids of the train/export/prepare chain)
set -Eeuo pipefail
export TZ=UTC
lin=${1:?lineage}; step=${2:?step}; jobs=${3:?job ids}
R=${NVC_RL_ROOT:?export NVC_RL_ROOT=<your workspace, see voicechat/README.md>}
E=${NVC_EVAL_ROOT:?export NVC_EVAL_ROOT=<your copy of the NemotronVC_eval harness>}
L=$R/experiments/$lin; label=$lin-step$step
bad=$(sacct -j "$jobs" -X -n -o JobID,State </dev/null | grep -E "FAIL|CANCEL|TIMEOUT|NODE_FAIL|OUT_OF_ME" || true)
[[ -z "$bad" ]] || { echo "FAILED $bad"; exit 0; }
[[ -f $L/hf/step_$step/EXPORT_COMPLETE.json ]] && ls $L/eval/step_$step/prepare-slurm-*.log >/dev/null 2>&1 \
  && grep -q EVAL_HF_CPU_VALIDATED $L/eval/step_$step/prepare-slurm-*.log || { echo WAIT; exit 0; }
# keep only the last MCore (user rule); checkpoints of other steps + init-mcore
if [[ -f $L/checkpoints/step_$step/training_info.json && ! -e $L/checkpoints/ACTIVE_JOB_ID ]]; then
  for d in $L/checkpoints/step_* $L/init-mcore; do
    [[ -e "$d" && "$d" != "$L/checkpoints/step_$step" ]] && rm -rf -- "$d"
  done
fi
cd $E
if [[ ! -e $E/artifacts/evaluations/$label-bfcl ]]; then
  ./bin/unified-external-eval run --site cs --model $label --model-path $L/eval/step_$step/hf-aligned \
    --tokenizer ${NVC_EVAL_TOKENIZER:-eval/text/tokenizers/student-kevin-nvc4-sft-18k-fix-mistral} --benchmarks bfcl_v3 --run-id $label-bfcl </dev/null 2>&1 | tail -3 | tr '\n' ' '
  echo; echo BFCL_LAUNCHED; exit 0
fi
[[ -f $E/artifacts/evaluations/$label-bfcl/MODEL.json ]] || { echo WAIT; exit 0; }
if [[ ! -e $E/artifacts/evaluations/$label-tau2 ]]; then
  ./bin/unified-external-eval run --site cs --model $label --model-path $L/eval/step_$step/hf-aligned \
    --tokenizer ${NVC_EVAL_TOKENIZER:-eval/text/tokenizers/student-kevin-nvc4-sft-18k-fix-mistral} --benchmarks tau2 --run-id $label-tau2 </dev/null 2>&1 | tail -3 | tr '\n' ' '
  echo; echo TAU2_LAUNCHED; exit 0
fi
echo DONE
