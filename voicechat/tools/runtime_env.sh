#!/usr/bin/env bash
set -Eeuo pipefail
source /opt/nemotronvc/runtime-env.sh
export NVC_CONTAINER_REPO_ROOT=/workspace/nemo-rl
export PATH="/opt/nemotronvc/bin:/opt/nemotronvc/venvs/driver/bin:$PATH"
export PYTHONPATH="${HF_MODULES_CACHE:?}:/workspace/nemo-rl:/workspace/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/workspace/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:/workspace/nemo-rl/3rdparty/Gym-workspace/Gym"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled RAY_USAGE_STATS_ENABLED=0 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_REQUIRE_PREBUILT_ENVS=1 NVC_SKIP_MEGATRON_DATASET_HELPERS=true
export NEMO_RL_BASE_PYTHON_EXECUTABLE=/opt/nemotronvc/venvs/driver/bin/python
export NEMO_RL_VLLM_PYTHON_EXECUTABLE=/opt/nemotronvc/venvs/vllm/bin/python
export NEMO_RL_MCORE_PYTHON_EXECUTABLE=/opt/nemotronvc/venvs/mcore/bin/python
if [[ ${NVC_INPUT_MODE:-} == gym ]]; then
 export NEMO_RL_NEMO_GYM_PYTHON_EXECUTABLE=/opt/nemotronvc/venvs/nemo-gym/bin/python
 # Gym venv root. A run-level override (env.nemo_gym.uv_venv_dir in the frozen config.yaml,
 # e.g. <project>/runtime/gym-v1/venvs-ext) wins over the image-baked root so that
 # NEMO_GYM_VENV_DIR, Gym's uv_venv_dir (nemo_gym.py setdefault) and skip_venv_if_present all
 # name one tree; with UV_OFFLINE=1 nothing is built at job start. The key is flat at 4 spaces
 # under env: nemo_gym: in the OmegaConf rendering that make_run.py / seal_gym_recipe.py write.
 nvc_gym_venv_dir=/opt/nemotronvc/gym-v1/venvs
 if [[ -n ${NVC_DEBUG_RUN:-} && -f "$NVC_DEBUG_RUN/config.yaml" ]]; then
  nvc_gym_venv_override=$(sed -n 's/^    uv_venv_dir: \(\/[A-Za-z0-9_.\/-]*\)$/\1/p' "$NVC_DEBUG_RUN/config.yaml" | head -n1)
  [[ -z "$nvc_gym_venv_override" ]] || nvc_gym_venv_dir=$nvc_gym_venv_override
  unset nvc_gym_venv_override
  [[ -d "$nvc_gym_venv_dir" ]] || { echo "Gym venv root missing in this container: $nvc_gym_venv_dir" >&2; exit 2; }
 fi
 export NEMO_GYM_VENV_DIR=$nvc_gym_venv_dir
 unset nvc_gym_venv_dir
 export NEMO_GYM_EXTRA_ROOTS="$NVC_CONTAINER_REPO_ROOT/3rdparty/Gym-workspace/Gym"
 export UV_OFFLINE=1
 export RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor
 export UV_CACHE_DIR="${TMPDIR:-/tmp}/gym-uv-cache"
 [[ -x "$NEMO_RL_NEMO_GYM_PYTHON_EXECUTABLE" ]]
else
 export NEMO_RL_NEMO_GYM_PYTHON_EXECUTABLE=/opt/nemotronvc/venvs/nemo-gym/bin/python
fi
export NEMO_RL_NODE_VLLM_PYTHON_EXECUTABLE=/opt/nemotronvc/venvs/vllm027/bin/python
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=10800
export NRL_REFIT_BUFFER_MEMORY_RATIO=0.005 NRL_REFIT_NUM_BUFFERS=1
export NVC_REQUIRE_OPTIMIZER_PROGRESS=1
export NVC_VERIFY_HDO_RESUME=1
if [[ -n ${NVC_DEBUG_RUN:-} ]]; then
 export NVC_HDO_RESUME_RECEIPT_DIR="$NVC_DEBUG_RUN/monitor/hdo-resume"
fi
if [[ ${NVC_RUN_KIND:-smoke} == smoke || ${NVC_RUN_KIND:-} == smoke-resume ]]; then
 export NRL_LOG_LEVEL=DEBUG NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2
else
 export NRL_LOG_LEVEL=INFO NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
# POSTMASK_K8_ADMISSION_BEGIN: observation is off for every historical profile.
# cs_metrics_runtime_enabled() returns False without GYM_CS_ADMISSION_INPUTS.json, so runs built by
# make_run.py (which never writes that file) skip importing the 1000-line gate module here.
unset NVC_POSTMASK_K8_METRICS
if [[ ${NVC_INPUT_MODE:-} == gym && -n ${NVC_DEBUG_RUN:-} && -f "$NVC_DEBUG_RUN/GYM_CS_ADMISSION_INPUTS.json" ]]; then
 nvc_postmask_enabled=$("$NEMO_RL_BASE_PYTHON_EXECUTABLE" - "$NVC_DEBUG_RUN" <<'PY'
import pathlib,sys
run=pathlib.Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0,str(run/'scripts'))
from seal_gym_gates import cs_metrics_runtime_enabled
print('1' if cs_metrics_runtime_enabled(run) else '0')
PY
 )
 if [[ "$nvc_postmask_enabled" == 1 ]]; then
  export NVC_POSTMASK_K8_METRICS=1
 fi
 unset nvc_postmask_enabled
fi
# POSTMASK_K8_ADMISSION_END
