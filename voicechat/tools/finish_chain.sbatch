#!/usr/bin/env bash
# Cluster-side finisher for a Gym RL train run (no local watcher needed): wait for the train job; if the lineage has no
# hf/step_STEP yet, build a --resume run with scripts/make_run.py inside the training image (same recipe/services/
# bundle/smoke) and submit it, wait again; then make the hf-aligned eval view (prepare_eval_hf_aligned.py, --base =
# the RL student) and, if NVC_EVAL_ROOT is set, launch bfcl_v3 then tau2 through scripts/eval/auto_eval_step.sh.
# Log: $NVC_RL_ROOT/logs/finish-chain-<job>.log. Env: NVC_RL_ROOT, NVC_SLURM_ACCOUNT_CPU, optional NVC_EVAL_ROOT,
# NVC_EVAL_TOKENIZER, NVC_EVAL_TEXT_SOURCE (defaults: read-only copies in cchen1's NemotronVC_eval).
# Usage: sbatch --account A --partition cpu_interactive --time 20:00:00 scripts/finish_chain.sbatch \
#        TRAIN_JOB LINEAGE STEP RECIPE SERVICES BUNDLE SMOKE_RUN STUDENT_HF "NEW_STUDENT_REASON"
#SBATCH --job-name=nvc-finish-chain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
# Memory: cpu_interactive allows 88G per user in total. This job (20G) + the 64G prepare job it submits must fit, otherwise
# the prepare job pends with QOSMaxMemoryPerUser while this job waits for it (deadlock found 2026-10-13, was 64G + 64G).
set -uo pipefail
R=${NVC_RL_ROOT:?export NVC_RL_ROOT=<your workspace, see voicechat/README.md>}
ACCT=${NVC_SLURM_ACCOUNT_CPU:?export NVC_SLURM_ACCOUNT_CPU=<slurm account for CPU jobs>}
EVAL_TOK=${NVC_EVAL_TOKENIZER:-/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_eval/eval/text/tokenizers/student-kevin-nvc4-sft-18k-fix-mistral}
EVAL_TEXT_SOURCE=${NVC_EVAL_TEXT_SOURCE:-/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_eval/eval/results/external-unified/rlvr-v8-rules-p128-s28-20260928-v1-text/source}
TJ=$1 LIN=$2 STEP=$3 RECIPE=$4 SERVICES=$5 BUNDLE=$6 SMOKE=$7 STUDENT=$8 REASON=$9
LOG=$R/logs/finish-chain-$SLURM_JOB_ID.log
log() { echo "[$(TZ=America/Los_Angeles date '+%m-%d %H:%M') PT] $*" >> $LOG; }
IMG=$R/runtime/gym-v1/NemotronVC-Gym34-vllm0271-v1.sqsh
MOUNTS="/lustre:/lustre,/raid/scratch:/raid/scratch,$R/src/nemo-rl:/workspace/nemo-rl,$R:/workspace/debug"
keep_cluster=$SLURM_CLUSTER_NAME; MYJOB=$SLURM_JOB_ID
clean_env() { for v in $(env | grep -o '^SLURM_[A-Z_0-9]*' | sort -u); do unset "$v"; done; export SLURM_CLUSTER_NAME=$keep_cluster SLURM_CONF=/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf; }
wait_job() { [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo INVALID_JOB_ID; return; }; while squeue -h -j "$1" </dev/null 2>/dev/null | grep -q .; do sleep 600; done; sacct -j "$1" -X -n -o State </dev/null | head -1 | tr -d ' '; }
L=$R/experiments/$LIN
log "start: train=$TJ lineage=$LIN step=$STEP"
st=$(wait_job $TJ); log "train job $TJ ended: $st"
for round in 1 2; do
  [ -f $L/hf/step_$STEP/EXPORT_COMPLETE.json ] && break
  ck=$(ls -d $L/checkpoints/step_* 2>/dev/null | wc -l)
  [ "$ck" = 1 ] || { log "FAILED: expected exactly one MCore checkpoint, found $ck"; exit 1; }
  log "no hf/step_$STEP yet; building resume from $(ls -d $L/checkpoints/step_*)"
  out=$(srun --overlap -n1 -c8 --mem=18G --no-container-mount-home --container-image=$IMG --container-mounts=$MOUNTS --container-workdir=/workspace/debug \
    bash -c "export HF_MODULES_CACHE=/tmp/hfmod NVC_INPUT_MODE=gym; source /workspace/debug/scripts/runtime_env.sh; /opt/nemotronvc/venvs/driver/bin/python $R/scripts/make_run.py --kind train --resume --project $R --recipe $RECIPE --services $SERVICES --bundle $BUNDLE --smoke-run $SMOKE --new-student-reason '$REASON'" </dev/null 2>&1 | grep MAKE_RUN_OK)
  run=$(echo "$out" | sed 's/.*"run": "\([^"]*\)".*/\1/')
  [ -d "$run" ] || { log "FAILED: make_run resume: $out"; exit 1; }
  ( clean_env; bash $R/scripts/submit_run.sh $run </dev/null >> $LOG 2>&1 )
  # submit_run.sh records the job in $run/JOB_ID (SFT runs use SUBMISSIONS); an empty id used to look like a finished job
  # and triggered a second resume build (2026-10-14).
  rj=$(cat $run/JOB_ID 2>/dev/null || tail -1 $run/SUBMISSIONS 2>/dev/null | awk '{print $1}')
  [[ "$rj" =~ ^[0-9]+$ ]] || { log "FAILED: no job id for resume run $run"; exit 1; }
  log "resume run $run job $rj"
  st=$(wait_job $rj); log "resume job $rj ended: $st"
done
[ -f $L/hf/step_$STEP/EXPORT_COMPLETE.json ] || { log "FAILED: no hf/step_$STEP"; exit 1; }
mkdir -p $L/eval/step_$STEP
pj=$( clean_env; sbatch --parsable --account=$ACCT --partition=cpu_interactive --time=01:00:00 --mem=64G --cpus-per-task=16 \
  --output=$L/eval/step_$STEP/prepare-slurm-%j.log $R/scripts/prepare_eval_hf_aligned.sbatch $R/scripts/prepare_eval_hf_aligned.py \
  --source $L/hf/step_$STEP --base $STUDENT --output $L/eval/step_$STEP/hf-aligned \
  --tokenizer $EVAL_TOK \
  --text-source $EVAL_TEXT_SOURCE --step $STEP </dev/null )
log "prepare job $pj"; st=$(wait_job $pj); log "prepare ended: $st"
[[ -n "${NVC_EVAL_ROOT:-}" ]] || { log "NVC_EVAL_ROOT not set: eval skipped; hf-aligned view at $L/eval/step_$STEP/hf-aligned"; log done; exit 0; }
for i in $(seq 1 30); do
  s=$( clean_env; bash $R/scripts/eval/auto_eval_step.sh $LIN $STEP "$pj" 2>&1 | tail -1 ); log "eval: $s"
  case "$s" in *TAU2_LAUNCHED*|*DONE*|*FAILED*) break;; esac
  sleep 600
done
log "done"
