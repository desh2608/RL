#!/usr/bin/env bash
# Sourced only for the explicitly selected Gym backend by submit_run.sh.
# This owns metadata/lineage locking; the shared submitter owns every sbatch.
gym_pre_submit() {
run=${1:?usage: submit_gym_run.sh FROZEN_RUN}
[[ "$run" =~ ^/[A-Za-z0-9_./-]+$ && -d "$run" && ! -L "$run" ]]
run=$(readlink -f "$run")
[[ -s "$run/GYM_SUBMIT_GATES.sha256" && -s "$run/inputs.sha256" && -s "$run/source.sha256" ]]
(cd "$run"; sha256sum --check --status GYM_SUBMIT_GATES.sha256; sha256sum --check --status launcher.sha256)
[[ $(cat "$run/INPUT_MODE") == gym && $(cat "$run/OBJECTIVE") == rlvr ]]
lineage=$(cat "$run/GYM_LINEAGE_DIR")
project=$(cat "$run/GYM_PROJECT_DIR")
expected=$(cat "$run/GYM_EXPECTED_RESUME")
[[ "$lineage" =~ ^/[A-Za-z0-9_./-]+$ && "$project" =~ ^/[A-Za-z0-9_./-]+$ ]]
[[ $(dirname "$run") == "$project/runs" ]]
kind=$(cat "$run/KIND"); nodes=$(cat "$run/NODES")
[[ "$nodes" =~ ^[0-9]+$ ]]
# Partition/account are chosen by submit_run.sh from the scheduler estimates (smoke: interactive/grizzly/polar
# by node count, train: polar4/polar3); this guard only validates the kind and records the pick below.
case "$kind" in smoke|smoke-resume|train) ;; *) exit 2;; esac
mkdir -p "$lineage"
[[ ! -L "$lineage" ]]
exec 7>"$lineage/.continuation-submit.lock"
flock -n 7 || { echo 'A lineage submission transaction is already active' >&2; exit 75; }
exec 9>"$lineage/.writer.lock"
flock -n 9 || { echo 'An allocation is writing this lineage' >&2; exit 75; }
[[ ! -e "$run/JOB_ID" && ! -e "$run/SUBMISSION_INTENT.json" ]]
active=$(squeue -h -u "$USER" -o '%i')
gym_is_active() { [[ "$1" =~ ^[0-9]+$ ]] && [[ $'\n'"$active"$'\n' == *$'\n'"$1"$'\n'* ]]; }
gym_terminal_job() {
 local job=$1 output state
 [[ "$job" =~ ^[0-9]+$ ]] && ! gym_is_active "$job" || return 1
 output=$(sacct -X -n -P -j "$job" --format=JobIDRaw,State,ExitCode)
 state=$(printf '%s\n' "$output" | awk -F'|' -v id="$job" '$1==id {print $2}')
 case "$state" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE) return 0;; *) return 1;; esac
}
# Detect other Gym freezes submitted through this entry, including queued jobs
# that do not yet own the allocation-time writer lock.
for candidate in "$project"/runs/*; do
 [[ -d "$candidate" && -f "$candidate/GYM_LINEAGE_DIR" ]] || continue
 [[ $(cat "$candidate/GYM_LINEAGE_DIR") == "$lineage" ]] || continue
 if [[ -f "$candidate/JOB_ID" ]] && gym_is_active "$(cat "$candidate/JOB_ID")"; then
  echo "An existing pending/running writer owns this lineage: $candidate" >&2; exit 75
 fi
done
if [[ -f "$lineage/SUBMISSION_OWNER" ]]; then
 previous=$(cat "$lineage/SUBMISSION_OWNER")
 [[ "$previous" =~ ^/[A-Za-z0-9_./-]+$ && $(dirname "$previous") == "$project/runs" ]]
 [[ -f "$previous/JOB_ID" ]] || { echo 'Unreconciled prior sbatch intent; inspect Slurm before retrying' >&2; exit 76; }
 gym_terminal_job "$(cat "$previous/JOB_ID")" || { echo 'Prior writer is not proven terminal' >&2; exit 75; }
fi
if [[ -f "$run/GYM_PRIOR_JOB" ]]; then
 prior=$(cat "$run/GYM_PRIOR_JOB")
 prior_run=$(cat "$run/GYM_PRIOR_RUN")
 [[ -s "$run/CONTINUATION_ACCEPTANCE.json" && $(cat "$prior_run/JOB_ID") == "$prior" ]]
 gym_terminal_job "$prior" || { echo 'CPU-accepted prior writer is no longer terminal' >&2; exit 75; }
fi
if [[ "$kind" == train ]]; then
 [[ -s "$run/GYM_SMOKE_JOB" && -s "$run/GYM_SMOKE_ACCEPTANCE.json" ]]
 if [[ -f "$run/GYM_SMOKE_SITE" && $(cat "$run/GYM_SMOKE_SITE") == draco ]]; then
  for evidence in GYM_SMOKE_SITE GYM_HISTORICAL_SMOKE_TERMINAL.json GYM_HISTORICAL_SMOKE_TERMINAL_ACCEPTED; do
   [[ $(awk -v file="$evidence" '$2==file {n++} END {print n+0}' "$run/GYM_SUBMIT_GATES.sha256") == 1 ]]
  done
  # Historical Draco IDs are not looked up in the unrelated CS Slurm database.
  # The CPU gate verifies portable original cleanup evidence and seals these files.
  [[ $(cat "$run/GYM_SMOKE_JOB") == 13604692 ]]
  [[ -s "$run/GYM_HISTORICAL_SMOKE_TERMINAL.json" ]]
  [[ $(cat "$run/GYM_HISTORICAL_SMOKE_TERMINAL_ACCEPTED") == 'draco:13604692:COMPLETED:0:0' ]]
 elif [[ $(cat "$run/GYM_SMOKE_JOB") == override ]]; then
  # make_run.py --accept-unsmoked-reason: explicit operator override, recorded in GYM_SMOKE_ACCEPTANCE.json.
  grep -q '"status": "override"' "$run/GYM_SMOKE_ACCEPTANCE.json" || { echo 'Override marker without override acceptance' >&2; exit 75; }
  printf '[SMOKE_GATE_OVERRIDE] run=%s reason=%s\n' "$run" "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["reason"])' "$run/GYM_SMOKE_ACCEPTANCE.json")" >&2
 else
  gym_terminal_job "$(cat "$run/GYM_SMOKE_JOB")" || { echo 'Accepted smoke is not terminal' >&2; exit 75; }
 fi
fi
shopt -s nullglob
checkpoints=("$lineage"/step_*)
temporary=("$lineage"/tmp_step_*)
(( ${#temporary[@]} == 0 ))
if [[ "$expected" == NONE ]]; then
 (( ${#checkpoints[@]} == 0 )) && [[ ! -e "$lineage/ORIGIN_RUN" ]]
else
 (( ${#checkpoints[@]} == 1 )) && [[ "${checkpoints[0]}" == "$expected" && -d "$expected" ]]
fi
}

gym_commit_submit_intent() {
# Write intent durably BEFORE sbatch; absence of JOB_ID after this point is
# ambiguous and is never retried automatically. Runtime lock must be released
# before sbatch because an allocation can start immediately.
[[ "$partition" =~ ^[A-Za-z0-9_-]+$ && "$best_account" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Submitter chose no partition/account' >&2; exit 2; }
printf '{"schema":"gym-submit-intent.v1","run":"%s","lineage":"%s","account":"%s","partition":"%s","nodes":%s}\n' \
 "$run" "$lineage" "$best_account" "$partition" "$nodes" > "$run/SUBMISSION_INTENT.json.tmp"
mv "$run/SUBMISSION_INTENT.json.tmp" "$run/SUBMISSION_INTENT.json"
printf '%s\n' "$run" > "$lineage/SUBMISSION_OWNER.tmp"
mv "$lineage/SUBMISSION_OWNER.tmp" "$lineage/SUBMISSION_OWNER"
if [[ "$expected" == NONE ]]; then
 printf '%s\n' "$run" > "$lineage/ORIGIN_RUN.tmp"
 mv "$lineage/ORIGIN_RUN.tmp" "$lineage/ORIGIN_RUN"
fi
flock -u 9
exec 9>&-
}

gym_record_submission() {
printf '{"job_id":"%s","account":"%s","partition":"%s","nodes":%s,"run":"%s"}\n' \
 "$job" "$best_account" "$partition" "$nodes" "$run" > "$run/SUBMISSION.json"
}
