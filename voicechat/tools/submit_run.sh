#!/usr/bin/env bash
# Login-node scheduler control only. Estimate three accounts; submit exactly once.
set -Eeuo pipefail
run=${1:?usage: submit_run.sh FROZEN_RUN}
gym_mode=false
if [[ -f "$run/INPUT_MODE" && $(cat "$run/INPUT_MODE") == gym ]]; then
 [[ -s "$run/GYM_SUBMIT_GATES.sha256" ]]
 (cd "$run"; sha256sum --check --status GYM_SUBMIT_GATES.sha256)
 source "$run/scripts/gym_submit_guard.sh"
 gym_pre_submit "$run"
 gym_mode=true
fi
[[ -s "$run/inputs.sha256" && -s "$run/source.sha256" && ! -e "$run/JOB_ID" ]]
# Tiny CPU-freeze evidence only: never hash the multi-GB dataset on the login node.
[[ -s "$run/LAUNCHER_PREFLIGHT.json" && -s "$run/launcher.sha256" ]] || {
 echo 'Missing CPU launcher preflight; create a new freeze before submission' >&2; exit 1;
}
(cd "$run"; sha256sum --check --status launcher.sha256) || {
 echo 'Frozen launcher or preflight evidence changed; refusing GPU submission' >&2; exit 1;
}
kind=$(cat "$run/KIND")
objective=opd
[[ ! -f "$run/OBJECTIVE" ]] || objective=$(cat "$run/OBJECTIVE")
[[ "$objective" == opd || "$objective" == rlvr ]]
if [[ "$objective" == rlvr ]]; then
 [[ -s "$run/role-preflight.sha256" ]] || { echo 'Missing CPU reward-role preflight' >&2; exit 1; }
 (cd "$run"; sha256sum --check --status role-preflight.sha256) || exit 1
fi
nodes=$(cat "$run/NODES")
[[ "$nodes" =~ ^[0-9]+$ ]]
# The partition is chosen below from the --test-only estimates; nothing is hard-coded per kind here.
case "$kind" in smoke|smoke-resume|train) ;; *) exit 2;; esac
job_name="$objective-v1-$kind"
[[ "$gym_mode" != true ]] || job_name="rlvr-gym-$kind"
# Partition candidates: train -> polar4 then polar3; smoke -> "interactive grizzly polar4" when the
# allocation fits the interactive partition (MaxNodes=3), else "grizzly polar4 polar3" (a localsim smoke is
# 3 Ray nodes + 1 simulator node = 4). Override with NVC_SLURM_PARTITIONS="p1 p2".
# Account policy: every account in NVC_SLURM_ACCOUNT is estimated with --test-only and the one the scheduler
# would start first is used
# and an account this login has no association for is simply skipped (its estimate is rejected).
# Override with NVC_SLURM_ACCOUNT="acct1 acct2" if needed.
# NVC_SUBMIT_ESTIMATE_ONLY=1 stops after the --test-only estimates (no intent, no sbatch).
if [[ "$kind" == train ]]; then default_partitions="polar4 polar3"
elif (( nodes <= 3 )); then default_partitions="interactive grizzly polar4"
else default_partitions="grizzly polar4 polar3"; fi
read -r -a candidate_partitions <<< "${NVC_SLURM_PARTITIONS:-$default_partitions}"
read -r -a accounts <<< "${NVC_SLURM_ACCOUNT:?export NVC_SLURM_ACCOUNT=\"acct1 [acct2 ...]\" (each is estimated with --test-only, the earliest start wins)}"
best_account= best_partition= best_epoch=9223372036854775807
for account in "${accounts[@]}"; do
 for candidate_partition in "${candidate_partitions[@]}"; do
 receipt="$run/scheduler-test-$account-$candidate_partition.log"
 if sbatch --test-only --account="$account" --partition="$candidate_partition" --nodes="$nodes" \
      --job-name="$job_name" --output="$run/slurm-%j.log" \
      "$run/scripts/gpu_job.sh" "$run" > "$receipt" 2>&1; then
   start=$(sed -n 's/.* to start at \([^ ]*\).*/\1/p' "$receipt" | head -n1)
   epoch=9223372036854775806
   if [[ -n "$start" ]]; then
     epoch=$(date -d "$start" +%s 2>/dev/null || printf '%s' 9223372036854775806)
   fi
   if [[ -z "$best_account" ]] || (( epoch < best_epoch )); then
     best_account=$account; best_epoch=$epoch; best_partition=$candidate_partition
   fi
 fi
 cat "$receipt"
 done
done
partition=$best_partition
[[ -n "$best_account" ]] || { echo 'All scheduler estimates rejected; no GPU job submitted' >&2; exit 1; }
if [[ ${NVC_SUBMIT_ESTIMATE_ONLY:-0} == 1 ]]; then
 printf '[ESTIMATE_ONLY] account=%s partition=%s nodes=%s run=%s (no sbatch submitted)\n' "$best_account" "$partition" "$nodes" "$run"
 exit 0
fi
printf '%s\n' "$best_account" > "$run/ACCOUNT"
printf '%s\n' "$partition" > "$run/PARTITION"
# Lock the submission itself as well as the writer inside the allocation.
exec 8>"$run/.submission.lock"
flock -n 8
[[ ! -e "$run/JOB_ID" ]]
if [[ "$gym_mode" == true ]]; then
 gym_commit_submit_intent
fi
job=$(sbatch --parsable --account="$best_account" --partition="$partition" --nodes="$nodes" \
 --job-name="$job_name" --output="$run/slurm-%j.log" "$run/scripts/gpu_job.sh" "$run")
job=${job%%;*}
[[ "$job" =~ ^[0-9]+$ ]] || { echo 'Ambiguous sbatch response; reconcile preserved submission intent before retry' >&2; exit 76; }
if [[ "$gym_mode" == true ]]; then
 printf '%s\n' "$job" > "$run/JOB_ID.tmp"
 mv "$run/JOB_ID.tmp" "$run/JOB_ID"
 gym_record_submission
else
 printf '%s\n' "$job" > "$run/JOB_ID"
fi
printf '[SUBMITTED] job=%s account=%s partition=%s nodes=%s run=%s\n' "$job" "$best_account" "$partition" "$nodes" "$run"
squeue -j "$job" -o '%.18i %.9P %.26j %.25a %.8T %.10M %.6D %R'
