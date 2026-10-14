#!/usr/bin/env bash
# Populate the third-party trees that are NOT stored in git (size): 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
# (with its nested 3rdparty/Megatron-LM). Gym is vendored in the repo (3rdparty/Gym-workspace/Gym).
#   default        copy the exact tree the RL runs use from the shared cs-oci workspace (read-only source), without .git
#   --upstream     clone NVIDIA-NeMo/Megatron-Bridge, check out the base commit below and apply the patches in
#                  megatron-bridge-patches/ and megatron-lm-patches/ (needs network + the base commits to be public)
#   --large-files  also copy the Gym test/example files skipped by size (see SKIPPED_LARGE_FILES.txt)
# Idempotent: an existing non-empty Megatron-Bridge tree is left alone (delete it to refetch).
set -Eeuo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P); REPO=$(cd "$HERE/../.." && pwd -P)
REF_TREE=${NVC_REF_TREE:-/lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/cchen1/NemotronVC_rl/src/pipeline/nemo_rl}
MB_BASE=3961f399     # upstream NVIDIA-NeMo/Megatron-Bridge (branch super-v3.5-posttraining) under the 4 local patches
MLM_BASE=f2f0f7bfd   # nested Megatron-LM under the 3 local patches
MODE=copy; LARGE=0
for a in "$@"; do case "$a" in --upstream) MODE=upstream;; --large-files) LARGE=1;; *) echo "unknown $a" >&2; exit 2;; esac; done
DST=$REPO/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
mkdir -p "$DST"
if [[ -n "$(ls -A "$DST" 2>/dev/null)" ]]; then
  echo "Megatron-Bridge already present: $DST"
elif [[ $MODE == copy ]]; then
  echo "copying Megatron-Bridge from $REF_TREE (about 170 MB, a few minutes) ..."
  rsync -a --exclude=.git --exclude=__pycache__ "$REF_TREE/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/" "$DST/"
else
  git clone --filter=blob:none https://github.com/NVIDIA-NeMo/Megatron-Bridge.git "$DST"
  git -C "$DST" checkout "$MB_BASE"
  git -C "$DST" submodule update --init 3rdparty/Megatron-LM
  git -C "$DST/3rdparty/Megatron-LM" checkout "$MLM_BASE"
  git -C "$DST" am "$HERE"/megatron-bridge-patches/*.patch
  git -C "$DST/3rdparty/Megatron-LM" am "$HERE"/megatron-lm-patches/*.patch
fi
if (( LARGE )); then
  while read -r f; do [[ -n "$f" ]] && install -D -m 644 "$REF_TREE/$f" "$REPO/$f"; done < "$HERE/SKIPPED_LARGE_FILES.txt"
fi
echo "third_party ok"
