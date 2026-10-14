#!/usr/bin/env bash
# Build / refresh runtime/gym-v1/venvs-ext: a WRITABLE NeMo Gym `uv_venv_dir` that mirrors the
# read-only pre-baked venv root inside the training image (/opt/nemotronvc/gym-v1/venvs) and adds
# the server venvs the image is missing (responses_api_models/openai_model, responses_api_agents/tau2).
#
# Gym maps a server dir <type>/<impl> to `<uv_venv_dir>/<type>/<impl>/.venv`
# (nemo_gym/cli/setup_command.py::get_venv_path) and, with `skip_venv_if_present: true`, only checks
# that `<venv>/bin/python` and `<venv>/bin/activate` exist before `source <venv>/bin/activate`.
# So venvs-ext holds:
#   * one symlink per pre-baked venv  -> /opt/nemotronvc/gym-v1/venvs/<type>/<impl>/.venv (container path,
#     dangling on the login node, resolves inside any job container built from the same image),
#   * real uv venvs for the missing servers, built the way Gym's setup_env_command builds them
#     (`uv venv --seed --allow-existing --python ... && uv pip install -r requirements.txt <head_server_deps>`),
#   * the tau2 data tree (responses_api_agents/tau2/tau2_data) pre-staged from the pinned eval runtime,
#     plus a gitignored symlink from the Gym worktree so ensure_tau2_data_dir() finds it without cloning.
#
# Run INSIDE the project container (network needed for pypi/github on first build):
#   bash scripts/interactive.sh alloc cpu            -> JOB
#   bash scripts/interactive.sh exec JOB 'bash /workspace/debug/scripts/build_gym_venvs_ext.sh'
#   bash scripts/interactive.sh release JOB
# Idempotent: re-running only re-creates missing symlinks, re-verifies existing venvs, and re-syncs data.
# Flags: --force (rebuild the two venvs), --verify-only (no writes except MANIFEST), --no-data (skip tau2 data).
set -Eeuo pipefail
export TZ=UTC

P=${NVC_RL_ROOT:-${NVC_PROJECT:?export NVC_RL_ROOT=<your workspace>}}
EXT=${NVC_GYM_VENVS_EXT:-$P/runtime/gym-v1/venvs-ext}
BAKED=${NVC_GYM_VENVS_BAKED:-/opt/nemotronvc/gym-v1/venvs}
GYM=${NVC_GYM_ROOT:-/workspace/nemo-rl/3rdparty/Gym-workspace/Gym}   # interactive.sh / gpu_job.sh mount src/nemo-rl here
UV=${NVC_UV:-/opt/nemotronvc/bin/uv}
GYM_PY=${NVC_GYM_PY:-/opt/nemotronvc/venvs/nemo-gym/bin/python}        # parent Gym process interpreter (runtime_env.sh), 3.13.13
# Interpreter for the NEW venvs: the same cpython the 28 pre-baked venvs use (their pyvenv.cfg `home`), 3.13.14.
# The worktree Gym declares requires-python>=3.13.14, so the parent's 3.13.13 cannot install `-e nemo-gym`;
# parent 3.13.13 vs child 3.13.14 is already the production situation (RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor).
VENV_PY=${NVC_GYM_VENV_PY:-/opt/nemotronvc/gym-v1/python/cpython-3.13.14-linux-x86_64-gnu/bin/python3.13}
TAU2_COMMIT=60c2a0dbf974ea7533456a4706f837c3a6d14afc                      # NemotronVC_eval/gym_cs/recipes/upstream.lock.json tau2_commit
TAU2_DATA_SRC=${NVC_TAU2_DATA_SRC:-/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_eval/.runtime/gym-cs/cb588bb1/responses_api_agents/tau2/tau2_data}
TAU2_SITE_FALLBACK=/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_eval/.runtime/gym-cs/cb588bb1/responses_api_agents/tau2/.venv/lib/python3.13/site-packages
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/nemotronvc-uv-cache}           # node-local; venvs themselves live on lustre
export UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
unset UV_NO_SYNC UV_PROJECT_ENVIRONMENT VIRTUAL_ENV || true

FORCE=0; VERIFY_ONLY=0; DO_DATA=1
while (( $# )); do
  case "$1" in
    --force) FORCE=1; shift;;
    --verify-only) VERIFY_ONLY=1; shift;;
    --no-data) DO_DATA=0; shift;;
    -h|--help) sed -n '2,24p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

# ----------------------------------------------------------------------------- preflight
[[ -d "$BAKED" ]] || die "pre-baked venv root $BAKED not found: run inside the project container"
[[ -x "$UV" ]] || die "uv not found at $UV"
[[ -x "$GYM_PY" ]] || die "Gym parent python not found at $GYM_PY"
[[ -x "$VENV_PY" ]] || die "venv interpreter not found at $VENV_PY (pre-baked venvs' pyvenv.cfg home)"
[[ -f "$GYM/pyproject.toml" && -d "$GYM/nemo_gym" ]] || die "Gym root $GYM missing pyproject.toml/nemo_gym"
mkdir -p "$EXT/requirements"
PARENT_PY_VER=$("$GYM_PY" -c 'import platform; print(platform.python_version())')
VENV_PY_VER=$("$VENV_PY" -c 'import platform; print(platform.python_version())')
RAY_VER=$("$GYM_PY" -c 'import ray; print(ray.__version__)')
OPENAI_PARENT=$("$GYM_PY" -c 'import openai; print(openai.__version__)')
GYM_COMMIT=$(git -C "$GYM" rev-parse HEAD 2>/dev/null || echo unknown)
log "ext=$EXT baked=$BAKED gym=$GYM ($GYM_COMMIT) uv=$("$UV" --version) parent_python=$PARENT_PY_VER venv_python=$VENV_PY_VER ray=$RAY_VER parent_openai=$OPENAI_PARENT"

# head_server_deps mirror (nemo_gym/global_config.py): ray[default]==<parent ray> always; openai==<parent openai>
# only when it satisfies nemo-gym's own constraint. Every NemotronVC_rl recipe sets allow_openai_version_skew: true
# (parent openai 2.25.0 > Gym cap <=2.7.2), so openai is NOT pinned and resolves from nemo-gym's constraint.
HEAD_DEPS=("ray[default]==$RAY_VER")

# ----------------------------------------------------------------------------- 1. mirror pre-baked venvs as symlinks
n_link=0; n_skip=0
while IFS= read -r venv; do
  rel=${venv#"$BAKED"/}                  # <type>/<impl>/.venv
  dst=$EXT/$rel
  if [[ -L "$dst" ]]; then
    [[ $(readlink "$dst") == "$venv" ]] && { n_skip=$((n_skip+1)); continue; }
    (( VERIFY_ONLY )) || ln -sfn "$venv" "$dst"; n_link=$((n_link+1)); continue
  fi
  if [[ -e "$dst" ]]; then log "WARN: $dst exists and is not a symlink; leaving it alone"; continue; fi
  (( VERIFY_ONLY )) || { mkdir -p "$(dirname "$dst")"; ln -s "$venv" "$dst"; }
  n_link=$((n_link+1))
done < <(find "$BAKED" -mindepth 3 -maxdepth 3 -type d -name .venv | sort)
log "symlinks: created/updated=$n_link already-correct=$n_skip"

# ----------------------------------------------------------------------------- 2. build the missing venvs
write_requirements() {
  cat > "$EXT/requirements/openai_model.txt" <<EOF
# Mirror of <Gym>/responses_api_models/openai_model/requirements.txt (Gym $GYM_COMMIT) with the relative
# editable path made absolute for the container mount used by every NemotronVC_rl job.
# Written/consumed by scripts/build_gym_venvs_ext.sh; do not hand-edit the venv.
-e nemo-gym[dev] @ $GYM
EOF
  cat > "$EXT/requirements/tau2.txt" <<EOF
# Mirror of <Gym>/responses_api_agents/tau2/requirements.txt (Gym $GYM_COMMIT) with:
#  * the relative editable path made absolute for the container mount ($GYM),
#  * tau2-bench pinned to the exact commit the eval workspace locked
#    (NemotronVC_eval/gym_cs/recipes/upstream.lock.json tau2_commit) instead of the moving
#    branch name bxyu/nemo_gym_stable (branch head == this commit on 2026-09-30).
# Written/consumed by scripts/build_gym_venvs_ext.sh; do not hand-edit the venv.
-e nemo-gym[dev] @ $GYM
tau2[knowledge] @ git+https://github.com/bxyu-nvidia/tau2-bench@$TAU2_COMMIT
# tau2 [knowledge] extra deps listed explicitly: the extra is not applied to the
# \`tau2[knowledge] @ git+URL\` direct reference, so bm25/bm25_grep miss rank-bm25.
rank-bm25>=0.2.2
openai>=1.0.0
audioop-lts>=0.2.2
EOF
}

venv_ok() {  # what Gym checks with skip_venv_if_present=true
  [[ -x "$1/bin/python" && -f "$1/bin/activate" ]]
}

build_venv() {  # build_venv <type>/<impl> <requirements file>
  local rel=$1 reqs=$2 venv=$EXT/$1/.venv
  local srv_dir=$GYM/$rel
  [[ -f "$srv_dir/requirements.txt" ]] || die "server dir $srv_dir has no requirements.txt"
  if venv_ok "$venv" && [[ -f "$venv/.nvc_build.json" ]] && (( ! FORCE )); then
    log "venv $rel present (built $(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["built_utc"])' "$venv/.nvc_build.json" 2>/dev/null || echo '?')); skipping build"
    return 0
  fi
  (( VERIFY_ONLY )) && die "venv $rel missing/incomplete and --verify-only given"
  [[ -L "$venv" ]] && die "$venv is a symlink; refusing to build over it"
  (( FORCE )) && [[ -d "$venv" ]] && { log "--force: removing $venv"; rm -rf --one-file-system -- "$venv"; }
  mkdir -p "$(dirname "$venv")"
  log "building $rel -> $venv"
  # Same shape as setup_env_command(): `uv venv --seed --allow-existing --python <ver> <venv>`; we hand uv the
  # explicit interpreter the pre-baked venvs were built with (see VENV_PY above).
  "$UV" venv --seed --allow-existing --python "$VENV_PY" "$venv"
  # `cd <server dir>` like Gym does so any relative paths in requirements resolve the same way.
  ( cd "$srv_dir" && "$UV" pip install --python "$venv/bin/python" -r "$reqs" "${HEAD_DEPS[@]}" )
  python3 - "$venv" "$rel" "$reqs" "$GYM_COMMIT" "$TAU2_COMMIT" "$VENV_PY_VER" "$RAY_VER" <<'PY'
import json, sys, time, hashlib, pathlib
venv, rel, reqs, gym, tau2, pyver, ray = sys.argv[1:]
pathlib.Path(venv, ".nvc_build.json").write_text(json.dumps({
    "server": rel, "requirements": reqs, "requirements_sha256": hashlib.sha256(open(reqs, "rb").read()).hexdigest(),
    "gym_commit": gym, "tau2_commit": tau2 if "tau2" in rel else None, "python": pyver, "ray": ray,
    "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "builder": "scripts/build_gym_venvs_ext.sh"}, indent=1) + "\n")
PY
}

(( VERIFY_ONLY )) || write_requirements
build_venv responses_api_models/openai_model "$EXT/requirements/openai_model.txt"
if ! build_venv responses_api_agents/tau2 "$EXT/requirements/tau2.txt"; then
  # Fallback documented in the task: copy tau2* site-packages from the pinned eval venv.
  log "WARN: pinned git install of tau2 failed; falling back to copying site-packages from $TAU2_SITE_FALLBACK"
  venv=$EXT/responses_api_agents/tau2/.venv
  cp -a "$TAU2_SITE_FALLBACK"/tau2 "$TAU2_SITE_FALLBACK"/tau2-*.dist-info "$venv/lib/python3.13/site-packages/"
  echo '{"fallback":"copied tau2 site-packages from eval venv cb588bb1"}' > "$venv/.nvc_build_fallback.json"
fi

# ----------------------------------------------------------------------------- 3. verification helpers (run in step 5)
verify() {
  local venv=$EXT/responses_api_models/openai_model/.venv
  venv_ok "$venv" || die "openai_model venv fails Gym skip check"
  "$venv/bin/python" -c 'import openai, nemo_gym, ray; print("openai_model venv: openai", openai.__version__, "nemo_gym", nemo_gym.__file__, "ray", ray.__version__)'
  ( cd "$GYM" && PYTHONPATH="$GYM/responses_api_models/openai_model:$GYM" "$venv/bin/python" -c 'import responses_api_models.openai_model.app; print("component import passed: responses_api_models/openai_model")' )
  venv=$EXT/responses_api_agents/tau2/.venv
  venv_ok "$venv" || die "tau2 venv fails Gym skip check"
  "$venv/bin/python" -c 'import tau2, rank_bm25, audioop, nemo_gym, ray, openai; print("tau2 venv: tau2", tau2.__file__, "openai", openai.__version__, "ray", ray.__version__)'
  "$venv/bin/python" -c "import importlib.metadata as m, json; c=json.loads(m.distribution('tau2').read_text('direct_url.json'))['vcs_info']['commit_id']; assert c=='$TAU2_COMMIT', c; print('tau2 commit pinned:', c)" \
    || log "WARN: tau2 direct_url commit check failed (fallback copy?)"
  ( cd "$GYM" && PYTHONPATH="$GYM/responses_api_agents/tau2:$GYM" "$venv/bin/python" -c 'import responses_api_agents.tau2.app; print("component import passed: responses_api_agents/tau2")' )
}

# ----------------------------------------------------------------------------- 4. tau2 data (ensure_tau2_data_dir target)
# responses_api_agents/tau2/app.py: DATA_DIR = Path(__file__).parent / "tau2_data" (no env override; app.py itself
# exports TAU2_DATA_DIR=DATA_DIR). The Gym tree is read-only in training jobs (gpu_job.sh mounts the frozen source
# copy at /workspace/nemo-rl:ro), so the data lives here and the Gym tree only carries a gitignored symlink
# (freeze_run.sh tars the worktree with plain `tar`, which stores the symlink as-is; /lustre is mounted in every job).
if (( DO_DATA )); then
  data=$EXT/responses_api_agents/tau2/tau2_data
  need=(tau2/domains/banking_knowledge/db.json tau2/domains/banking_knowledge/documents tau2/domains/banking_knowledge/prompts tau2/domains/banking_knowledge/tasks tau2/domains/telecom/tasks_full.json tau2/user_simulator)
  complete=1; for f in "${need[@]}"; do [[ -e "$data/$f" ]] || complete=0; done
  if (( ! complete )); then
    (( VERIFY_ONLY )) && die "tau2 data incomplete at $data and --verify-only given"
    [[ -d "$TAU2_DATA_SRC/tau2/domains/banking_knowledge" ]] || die "tau2 data source $TAU2_DATA_SRC lacks banking_knowledge"
    log "staging tau2 data from $TAU2_DATA_SRC -> $data"
    mkdir -p "$data"
    cp -a "$TAU2_DATA_SRC"/. "$data"/
    rm -rf --one-file-system -- "$data/tau2/results" 2>/dev/null || true   # _ignore_generated_tau2_data
    for f in "${need[@]}"; do [[ -e "$data/$f" ]] || die "tau2 data still missing $f after copy"; done
  fi
  log "tau2 data ok at $data ($(du -sh "$data" 2>/dev/null | cut -f1))"
  link=$GYM/responses_api_agents/tau2/tau2_data
  if [[ -L "$link" ]]; then
    [[ $(readlink "$link") == "$data" ]] && log "Gym tau2_data symlink already -> $data" || { (( VERIFY_ONLY )) || ln -sfn "$data" "$link"; log "Gym tau2_data symlink re-pointed -> $data"; }
  elif [[ -e "$link" ]]; then
    log "WARN: $link exists as a real path (not a symlink); leaving it alone"
  elif (( VERIFY_ONLY )); then
    log "WARN: $link missing (verify-only, not created)"
  elif ln -s "$data" "$link" 2>/dev/null; then
    log "created gitignored symlink $link -> $data"
  else
    log "WARN: cannot create $link (read-only Gym tree?) - tau2 app will try to git clone into it"
  fi
fi

# ----------------------------------------------------------------------------- 5. verify (after data so tau2 sees its data dir)
verify

# ----------------------------------------------------------------------------- 6. manifest
python3 - "$EXT" "$BAKED" "$GYM_COMMIT" "$TAU2_COMMIT" "$VENV_PY_VER" "$RAY_VER" <<'PY'
import json, os, sys, time
ext, baked, gym, tau2, pyver, ray = sys.argv[1:]
links, venvs, data = [], [], []
for root, dirs, files in os.walk(ext):
    for d in list(dirs):
        p = os.path.join(root, d)
        if d == ".venv":
            (links if os.path.islink(p) else venvs).append({"path": os.path.relpath(p, ext), "target": os.readlink(p) if os.path.islink(p) else None})
            dirs.remove(d)
        elif d == "tau2_data":
            data.append(os.path.relpath(p, ext)); dirs.remove(d)
json.dump({"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "builder": "scripts/build_gym_venvs_ext.sh",
           "uv_venv_dir": ext, "mirrors": baked, "gym_commit": gym, "tau2_commit": tau2, "python": pyver, "ray": ray,
           "symlinked_venvs": sorted(links, key=lambda x: x["path"]), "built_venvs": sorted(venvs, key=lambda x: x["path"]),
           "data_dirs": sorted(data)}, open(os.path.join(ext, "MANIFEST.json"), "w"), indent=1)
print(f"MANIFEST.json: {len(links)} symlinked venvs, {len(venvs)} built venvs, data={data}")
PY
log "done"
