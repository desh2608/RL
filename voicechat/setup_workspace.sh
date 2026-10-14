#!/usr/bin/env bash
# Create an RL workspace on cs-oci around this checkout, in the layout scripts/make_run.py and the job scripts expect:
#   $WS/src/pipeline/nemo_rl  -> this repo        $WS/src/nemo-rl -> pipeline/nemo_rl
#   $WS/scripts               -> voicechat/tools  $WS/config/     <- copies of voicechat/configs (edit freely)
#   $WS/runtime/gym-v1        -> shared read-only container images + Gym venvs on cs-oci
#   $WS/models/sim            -> shared read-only user simulator (gpt-oss-120b) for online RL
#   $WS/{runs,experiments,data,logs,receipts,analysis,.secrets}
# and copy the Megatron-Bridge tree (third_party/fetch_third_party.sh). Idempotent; never touches the shared assets.
# Usage: bash voicechat/setup_workspace.sh /lustre/fsw/portfolios/<portfolio>/users/$USER/<workspace> [--ref-root DIR]
set -Eeuo pipefail
WS=${1:?workspace dir}; shift
REF=/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl     # read-only shared assets (images, venvs, simulator)
while (( $# )); do case "$1" in --ref-root) REF=$2; shift 2;; *) echo "unknown $1" >&2; exit 2;; esac; done
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
mkdir -p "$WS"; WS=$(cd "$WS" && pwd -P)
mkdir -p "$WS"/{src/pipeline,runs,experiments,data,logs,receipts,analysis,config,runtime,models,.secrets}
chmod 700 "$WS/.secrets"
if [[ "$REPO" != "$WS/src/pipeline/nemo_rl" ]]; then
  [[ -e "$WS/src/pipeline/nemo_rl" && ! -L "$WS/src/pipeline/nemo_rl" ]] && { echo "$WS/src/pipeline/nemo_rl exists and is not a link" >&2; exit 2; }
  ln -sfn "$REPO" "$WS/src/pipeline/nemo_rl"
fi
ln -sfn pipeline/nemo_rl "$WS/src/nemo-rl"
ln -sfn "$REPO/voicechat/tools" "$WS/scripts"
for f in "$REPO"/voicechat/configs/*; do [[ -e "$WS/config/$(basename "$f")" ]] || cp "$f" "$WS/config/"; done
for d in runtime/gym-v1 models/sim; do
  [[ -d "$REF/$d" ]] || { echo "shared asset missing: $REF/$d" >&2; exit 2; }
  [[ -e "$WS/$d" ]] || ln -s "$REF/$d" "$WS/$d"
done
bash "$REPO/voicechat/third_party/fetch_third_party.sh"
cat <<EOF

Workspace ready: $WS
Add to your shell (and to every sbatch environment; submit_run.sh / finish_chain.sbatch read them):
  export NVC_RL_ROOT=$WS
  export NVC_SLURM_ACCOUNT="<gpu account> [<another account>]"   # submit_run.sh picks the earliest --test-only start
  export NVC_SLURM_ACCOUNT_CPU=<cpu/interactive account>
  # optional: export NVC_EVAL_ROOT=<your copy of the NemotronVC_eval harness>   (finish_chain.sbatch launches bfcl_v3 + tau2)
Online RL with the hub simulator needs \$NVC_RL_ROOT/.secrets/nvidia-api-key (your own key, mode 600).
Next: voicechat/README.md -> docs/OFFLINE_PIVOT_RL.md or docs/ONLINE_RL.md.
EOF
