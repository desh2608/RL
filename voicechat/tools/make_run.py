#!/usr/bin/env python3
"""Build one immutable Gym RLVR run directory from recipe + services yaml + data bundle.

This is the repository entry point that replaces the per-experiment freeze ceremony
(`freeze_run.sh gym` with RUNTIME_API_ACCEPTANCE / NVC_GYM_ACTOR_QUALIFICATION /
NVC_GYM_CPU_RAY_READY, the 900-line per-profile SHA tables in seal_gym_gates.py and the
receipts-only build_direct.py / build_continuation.py bypass builders). It writes exactly the
files that the unchanged consumers read:

  scripts/submit_run.sh + gym_submit_guard.sh  GYM_SUBMIT_GATES.sha256 launcher.sha256
                                               role-preflight.sha256 inputs.sha256 source.sha256
                                               LAUNCHER_PREFLIGHT.json KIND NODES OBJECTIVE
                                               INPUT_MODE GYM_LINEAGE_DIR GYM_PROJECT_DIR
                                               GYM_EXPECTED_RESUME [GYM_SMOKE_JOB
                                               GYM_SMOKE_ACCEPTANCE.json for train]
  scripts/gpu_job.sh                           INPUT_MODE RUNTIME_IMAGE SANDBOX_IMAGE KIND NODES
                                               config.yaml(checkpoint_dir) inputs.sha256
                                               source.sha256 source.tar.gz SECRETS_REQUIRED
                                               scripts/gym_sandbox_pair.py (+gym_idle_watchdog)
  scripts/ray_role.sh + runtime_env.sh         scripts/* RUN_CONTRACT.json config.yaml KIND
  examples/run_grpo.py                         config.yaml gym-configs/*.yaml data.*.jsonl
  [--sim-nodes N>0] gpu_job.sh + sim_role.sh   SIM_NODES SIM_SPEC.json (NODES = Ray nodes + N; the
                                               last N hosts serve a standalone vLLM simulator that
                                               the services yaml reaches via ${oc.env:NVC_SIM_BASE_URL})

Structural validation is `seal_gym_recipe.validate_config` (parameterized topology arithmetic).
Run inside the training container on a CPU allocation (driver venv has omegaconf + nemo_rl):

  bash scripts/interactive.sh exec JOB 'export HF_MODULES_CACHE=/tmp/hfmod NVC_INPUT_MODE=gym; \
    source /workspace/debug/scripts/runtime_env.sh; \
    /opt/nemotronvc/venvs/driver/bin/python /workspace/debug/scripts/make_run.py --kind smoke'

It never submits. Submission stays `scripts/submit_run.sh <run>` on the login node.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import math
import os
import time
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = HERE.parent
SECRET_SOURCES = {'NVIDIA_API_KEY': '.secrets/nvidia-api-key'}
# ${oc.env:NAME} references that gpu_job.sh exports itself (no secret file): only legal when the run
# carries simulator nodes (--sim-nodes >= 1). Never written to SECRETS_REQUIRED.
LAUNCHER_ENV_SOURCES = {'NVC_SIM_BASE_URL': 'gpu_job.sh (sim node)', 'NVC_SIM_API_KEY': 'gpu_job.sh (constant)'}
SIM_URL_ENV = 'NVC_SIM_BASE_URL'
# Zero-variance GRPO groups (all K rollouts same reward) are dropped before the optimizer step; the
# epoch budget must still supply every planned step (plus the async age slack) at this drop rate.
DEFAULT_MAX_ZERO_VARIANCE_DROP = 0.6
DEFAULT_SIM_PYTHON = '/opt/nemotronvc/venvs/vllm027/bin/python'  # vLLM 0.27.1; venvs/vllm is 0.25.1 (policy)
DEFAULT_SIM_EXTRA_ARGS = '--enable-prefix-caching --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss'
# Image digests were measured once by the CPU transfer acceptance of the last real run; they are
# inherited when path (resolved) and byte size still match instead of rehashing 82 GB per build.
TEMPLATE_RUN = 'runs/train-v1-20260928T023000Z'
RUNTIME_SCRIPTS = ('gpu_job.sh', 'ray_role.sh', 'sim_role.sh', 'runtime_env.sh', 'gym_submit_guard.sh')
RUNTIME_MODULES = ('ray_ports.py', 'test_ray_ports.py', 'gym_sandbox_supervisor.py', 'gym_sandbox_pair.py',
                   'gym_sandbox_probe.py', 'gym_idle_watchdog.py', 'seal_gym_recipe.py', 'seal_gym_gates.py',
                   'make_run.py')
TAR_EXCLUDES = ('.git', '__pycache__', '.venv', '.pytest_cache', '.ruff_cache', '.mypy_cache')
SERVER_TYPES = ('responses_api_models', 'responses_api_agents', 'resources_servers')


class BuildError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise BuildError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def write_text(path, value):
    Path(path).write_text(value if value.endswith('\n') else value + '\n')


def checked_path(value):
    require(isinstance(value, str) and re.fullmatch(r'/[A-Za-z0-9_./-]+', value),
            'Unsafe/non-absolute path for the shell submitter: ' + repr(value))
    return value


def utc_stamp():
    return _dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def set_dotted(config, key, value):
    parts = key.split('.')
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def get_dotted(config, key, default=None):
    node = config
    for part in key.split('.'):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def parse_override(text):
    require('=' in text, 'Override must be key=value: ' + text)
    key, raw = text.split('=', 1)
    import yaml
    return key.strip(), yaml.safe_load(raw)


def git(args, cwd):
    return subprocess.run(['git', *args], cwd=str(cwd), text=True, capture_output=True, check=True).stdout.strip()


def git_head_fallback(repo):
    """HEAD without a git binary: enough to name the commit, not enough to prove cleanliness."""
    dot_git = Path(repo) / '.git'
    if dot_git.is_file():
        line = dot_git.read_text().strip()
        require(line.startswith('gitdir: '), 'Unrecognized gitlink: ' + str(dot_git))
        target = line[len('gitdir: '):]
        dot_git = (Path(repo) / target).resolve() if not Path(target).is_absolute() else Path(target)
    head = (dot_git / 'HEAD').read_text().strip()
    if head.startswith('ref: '):
        ref = head[5:]
        candidate = dot_git / ref
        if not candidate.is_file() and (dot_git / 'commondir').is_file():
            candidate = (dot_git / (dot_git / 'commondir').read_text().strip()).resolve() / ref
        if candidate.is_file():
            return candidate.read_text().strip()
        packed = dot_git / 'packed-refs'
        if not packed.is_file() and (dot_git / 'commondir').is_file():
            packed = (dot_git / (dot_git / 'commondir').read_text().strip()).resolve() / 'packed-refs'
        for entry in packed.read_text().splitlines() if packed.is_file() else []:
            if entry.endswith(' ' + ref):
                return entry.split()[0]
        raise BuildError('Cannot resolve git ref without a git binary: ' + ref)
    return head


def source_facts(project, repo):
    """Commit pins plus the exact uncommitted state of the trained tree."""
    have_git = shutil.which('git') is not None
    repos = {
        'project_wrapper': project,
        'pipeline': project / 'src/pipeline',
        'nemo_rl_applied': repo,
        'bridge': repo / '3rdparty/Megatron-Bridge-workspace/Megatron-Bridge',
        'megatron_lm': repo / '3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM',
        'gym': repo / '3rdparty/Gym-workspace/Gym',
    }
    pins = {}
    for name, path in repos.items():
        try:
            pins[name] = git(['rev-parse', 'HEAD'], path) if have_git else git_head_fallback(path)
        except (subprocess.CalledProcessError, BuildError, OSError) as error:
            pins[name] = 'unknown:' + type(error).__name__
    status = {'git_binary': have_git, 'porcelain': {}, 'patch': ''}
    if have_git:
        for name in ('nemo_rl_applied', 'gym'):
            path = repos[name]
            porcelain = git(['status', '--porcelain', '--untracked-files=normal'], path)
            status['porcelain'][name] = porcelain.splitlines()
            diff = subprocess.run(['git', 'diff', '--binary'], cwd=str(path), text=True, capture_output=True, check=True).stdout
            if diff:
                status['patch'] += f'# ==== {name} ({path}) HEAD {pins[name]}\n{diff}'
    return pins, status


def load_recipe(recipe_path, project):
    try:
        from omegaconf import OmegaConf
        from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    except ImportError as error:  # pragma: no cover
        raise BuildError(
            'make_run.py needs the driver venv with omegaconf and nemo_rl importable. Run it inside the '
            'training container (scripts/interactive.sh exec JOB ...) after sourcing scripts/runtime_env.sh; '
            f'import failed: {error}')
    register_omegaconf_resolvers()
    return OmegaConf.to_container(load_config(str(recipe_path)), resolve=True)


def to_yaml(config):
    from omegaconf import OmegaConf
    return OmegaConf.to_yaml(OmegaConf.create(config), resolve=True)


def apply_smoke_defaults(config, applied):
    """3 nodes (1 rollout + 2 learner), 3 steps, 16 prompts; ray_role.sh strictly resumes to step 4."""
    k = config['grpo']['num_generations_per_prompt']
    defaults = {
        'cluster.num_nodes': 3,
        'policy.generation.colocated.resources.num_nodes': 1,
        'grpo.max_num_steps': 3,
        'grpo.num_prompts_per_step': 16,
        'policy.train_global_batch_size': 16 * k,
        'policy.generation_batch_size': 16 * k,
        'checkpointing.ft_save_period': 3,
        'checkpointing.save_period': 3,
        'checkpointing.hf_export_period': 3,
    }
    if config.get('data', {}).get('validation'):
        defaults['grpo.max_val_samples'] = 32
        defaults['grpo.val_batch_size'] = 32
    name = get_dotted(config, 'logger.wandb.name')
    if isinstance(name, str) and not name.endswith('-smoke'):
        defaults['logger.wandb.name'] = name + '-smoke'
    for key, value in defaults.items():
        set_dotted(config, key, value)
        applied[key] = value


GYM_VALIDATION_OFF = (('grpo.val_period', 0), ('grpo.val_at_start', False), ('grpo.val_at_end', False))


def apply_train_gym_validation_guard(config, applied, keep=False):
    """Force Gym validation off for train builds unless --keep-gym-validation.

    The Gym validation path (`validate()` -> `run_nemo_gym_rollout_sync` -> `_postprocess_single_nemo_gym_group`)
    groups val rollouts by the TRAINING stride (num_generations_per_prompt) while val uses
    val_num_generations_per_prompt, so it raises `Expected one _ng_task_index per prompt group`. At step 0 the
    error is caught ("Initial validation failed"), inside the training loop it is fatal: train job 35974201
    (2026-10-04) died at the first periodic validation (step 10) before its save/HF export. The v3 train builds
    avoided it only because they were built with --override grpo.val_at_start=false --override grpo.val_period=0.
    Returns the keys it changed (empty when nothing was on or the guard is disabled)."""
    if keep or not config.get('env', {}).get('should_use_nemo_gym'):
        return []
    grpo = config.get('grpo', {})
    if not (grpo.get('val_period') or grpo.get('val_at_start') or grpo.get('val_at_end')):
        return []
    changed = []
    for key, value in GYM_VALIDATION_OFF:
        if get_dotted(config, key) != value:
            set_dotted(config, key, value)
            applied[key] = value
            changed.append(key)
    applied['gym_validation_guard'] = ('train build forces Gym validation off (val_period/val_at_start/val_at_end); '
                                       'pass --keep-gym-validation to opt out')
    return changed


def scan_rows(path, services):
    """Count rows, collect routes and structural facts; every row must name a configured agent."""
    rows = 0
    routes = {}
    max_output = {}
    with Path(path).open('rb') as stream:
        for line in stream:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            ref = row.get('agent_ref') or {}
            name = ref.get('name')
            require(isinstance(name, str) and name, f'Row {rows} has no agent_ref.name')
            require(ref.get('type') == 'responses_api_agents', f'Row {rows} agent_ref.type is not responses_api_agents')
            require(name in services and 'responses_api_agents' in services[name],
                    f'Row {rows} routes to agent {name!r} which is not a responses_api_agents entry of the services yaml')
            params = row.get('responses_create_params') or {}
            require(isinstance(params.get('input'), list) and params['input'],
                    f'Row {rows} has no responses_create_params.input')
            routes[name] = routes.get(name, 0) + 1
            limit = params.get('max_output_tokens')
            max_output.setdefault(name, {})
            max_output[name][str(limit)] = max_output[name].get(str(limit), 0) + 1
    return rows, routes, max_output


def server_refs(node, found):
    if isinstance(node, dict):
        if set(node) == {'type', 'name'} and node['type'] in SERVER_TYPES:
            found.append((node['type'], node['name']))
        for value in node.values():
            server_refs(value, found)
    elif isinstance(node, list):
        for value in node:
            server_refs(value, found)


def service_closure(services, gym_root, venv_dir, skip_venv_if_present, allow_missing_venv):
    """Every referenced server exists, its Gym source dir exists, and its venv is present."""
    closure = {'servers': [], 'references': [], 'problems': []}
    refs = []
    server_refs(services, refs)
    for kind, name in sorted(set(refs)):
        ok = name in services and kind in services[name]
        closure['references'].append({'type': kind, 'name': name, 'resolved': ok})
        if not ok:
            closure['problems'].append(f'{kind}/{name} referenced but not defined at top level')
    for top, body in services.items():
        if not isinstance(body, dict) or len(body) != 1:
            continue
        server_type, impls = next(iter(body.items()))
        if server_type not in SERVER_TYPES or not isinstance(impls, dict) or len(impls) != 1:
            continue
        impl, spec = next(iter(impls.items()))
        if not isinstance(spec, dict) or 'entrypoint' not in spec:
            continue
        server_dir = gym_root / server_type / impl
        entrypoint = server_dir / spec['entrypoint']
        venv = Path(venv_dir, *Path(server_type, impl).parts[-2:], '.venv')
        venv_ok = (venv / 'bin/python').exists() and (venv / 'bin/activate').exists()
        record = {'name': top, 'type': server_type, 'impl': impl, 'server_dir': str(server_dir),
                  'entrypoint_exists': entrypoint.is_file(), 'venv': str(venv), 'venv_present': venv_ok,
                  'install_marker': (server_dir / 'requirements.txt').exists() or (server_dir / 'pyproject.toml').exists()}
        closure['servers'].append(record)
        if not record['entrypoint_exists']:
            closure['problems'].append(f'{top}: entrypoint missing {entrypoint}')
        if not record['install_marker']:
            closure['problems'].append(f'{top}: no requirements.txt/pyproject.toml in {server_dir}')
        if skip_venv_if_present and not venv_ok:
            message = f'{top}: venv not present at {venv} (UV_OFFLINE=1 would fail building it at job start)'
            if allow_missing_venv:
                closure.setdefault('warnings', []).append(message)
            else:
                closure['problems'].append(message)
    return closure


def tau2_data_check(services, gym_root):
    if 'tau2' not in {impl for body in services.values() if isinstance(body, dict)
                      for impls in body.values() if isinstance(impls, dict) for impl in impls}:
        return {'needed': False}
    data_dir = gym_root / 'responses_api_agents/tau2/tau2_data'
    marker = data_dir / 'tau2/domains/banking_knowledge/db.json'
    return {'needed': True, 'data_dir': str(data_dir), 'is_symlink': data_dir.is_symlink(),
            'target': os.readlink(data_dir) if data_dir.is_symlink() else None,
            'present': data_dir.is_dir(), 'banking_knowledge_present': marker.is_file()}


def inherit_image_digest(project, role, path):
    template = project / TEMPLATE_RUN / 'RUN_CONTRACT.json'
    if not template.is_file():
        return None, None
    item = json.loads(template.read_text()).get('runtime_images', {}).get(role)
    if not item:
        return None, None
    same = Path(item['path']).resolve() == path.resolve() and item['bytes'] == path.stat().st_size
    return (item['sha256'], str(template)) if same else (None, None)


def freeze_images(project, run, runtime_image, sandbox_image):
    images = {}
    for role, given in (('runtime', runtime_image), ('sandbox', sandbox_image)):
        path = Path(given).resolve(strict=True)
        require(path.is_file() and path.suffix == '.sqsh', 'Invalid ' + role + ' image: ' + str(path))
        digest, source = inherit_image_digest(project, role, path)
        images[role] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest,
                        'sha256_source': source or 'not_measured_by_this_build'}
        write_text(run / (role.upper() + '_IMAGE'), checked_path(str(path)))
    write_json(run / 'GYM_IMAGE_CONTRACT.json', images)
    return images


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
        return 'hardlink'
    except OSError:
        shutil.copy2(source, destination)
        return 'copy'


def manifest_lines(run, names, known):
    lines = []
    for name in sorted(set(names)):
        digest = known.get(name) or sha(run / name)
        lines.append(f'{digest}  {name}\n')
    return ''.join(lines)


def node_layout(policy_nodes, sim_nodes):
    """Slurm allocation = Ray nodes (cluster.num_nodes) + simulator nodes (never in Ray)."""
    require(type(policy_nodes) is int and policy_nodes >= 1, 'Ray node count must be a positive integer')
    require(type(sim_nodes) is int and sim_nodes >= 0, '--sim-nodes must be a non-negative integer')
    return {'ray_nodes': policy_nodes, 'sim_nodes': sim_nodes, 'total_nodes': policy_nodes + sim_nodes}


def classify_env_refs(services_text, sim_nodes):
    """Split ${oc.env:NAME} references into secrets (SECRET_SOURCES) and launcher env (LAUNCHER_ENV_SOURCES).

    Launcher env is only admitted with simulator nodes, and simulator nodes require a services yaml
    that actually points sim_gptoss at them; anything else is refused before the run dir is filled.
    """
    names = sorted(set(re.findall(r'\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}', services_text)))
    secret_names, launcher_names = [], []
    for name in names:
        if name in SECRET_SOURCES:
            secret_names.append(name)
        elif name in LAUNCHER_ENV_SOURCES:
            require(sim_nodes >= 1,
                    f'Services reference ${{oc.env:{name}}} (exported by {LAUNCHER_ENV_SOURCES[name]}) but this build has '
                    '--sim-nodes 0; use the hub services yaml or add --sim-nodes N --sim-model-path PATH')
            launcher_names.append(name)
        else:
            require(False, f'Services reference ${{oc.env:{name}}} but gpu_job.sh knows no source for it')
    require(sim_nodes == 0 or SIM_URL_ENV in launcher_names,
            f'--sim-nodes {sim_nodes} requested but the services yaml never references ${{oc.env:{SIM_URL_ENV}}}; '
            'sim_gptoss would still call the hub (use config/gym-services-agentic-v1-localsim.yaml)')
    return secret_names, launcher_names


def sim_spec_from_args(args, gpus_per_node=8):
    """Validated SIM_SPEC.json content for --sim-nodes >= 1, else None (no new files, no new keys)."""
    import shlex
    sim_nodes = int(getattr(args, 'sim_nodes', 0) or 0)
    require(sim_nodes >= 0, '--sim-nodes must be >= 0')
    if sim_nodes == 0:
        return None
    require(getattr(args, 'sim_model_path', None), '--sim-nodes >= 1 requires --sim-model-path PATH (HF dir with config.json)')
    model_path = Path(args.sim_model_path).resolve(strict=True)
    require(model_path.is_dir() and (model_path / 'config.json').is_file(),
            'Simulator model path must be a directory containing config.json: ' + str(model_path))
    require(re.fullmatch(r'[A-Za-z0-9_./:-]+', args.sim_served_name or ''), 'Unsafe --sim-served-name')
    require(type(args.sim_port) is int and 1024 <= args.sim_port <= 65535, '--sim-port must be in 1024..65535')
    require(type(args.sim_tp) is int and type(args.sim_dp) is int and args.sim_tp >= 1 and args.sim_dp >= 1 and
            args.sim_tp * args.sim_dp <= gpus_per_node,
            f'--sim-tp x --sim-dp must be within one node ({gpus_per_node} GPUs)')
    require(type(args.sim_max_model_len) is int and args.sim_max_model_len >= 1024, '--sim-max-model-len too small')
    require(0.1 <= float(args.sim_gpu_memory_utilization) <= 0.99, '--sim-gpu-memory-utilization must be in 0.1..0.99')
    require(type(args.sim_max_num_seqs) is int and args.sim_max_num_seqs >= 1, '--sim-max-num-seqs must be >= 1')
    python = checked_path(args.sim_python or DEFAULT_SIM_PYTHON)
    require(os.access(python, os.X_OK), '--sim-python is not an executable inside this container: ' + python)
    extra_args = shlex.split(args.sim_extra_args or '')
    require(all(re.fullmatch(r'[A-Za-z0-9_.,:/=@+-]+', x) for x in extra_args), 'Unsafe --sim-extra-args token')
    return {'schema': 'gym-sim-spec.v1', 'model_path': checked_path(str(model_path)), 'served_name': args.sim_served_name,
            'port': args.sim_port, 'tp': args.sim_tp, 'dp': args.sim_dp, 'max_model_len': args.sim_max_model_len,
            'gpu_memory_utilization': float(args.sim_gpu_memory_utilization), 'max_num_seqs': args.sim_max_num_seqs,
            'extra_args': extra_args, 'python': python, 'api_key': 'nvc-local-sim',
            'base_url_env': SIM_URL_ENV, 'role_script': 'scripts/sim_role.sh'}


def smoke_sim_warning(train_sim_nodes, smoke_contract, smoke_path):
    """A localsim train may be gated by a hub smoke; that is allowed but recorded and printed."""
    smoke_sim_nodes = int(smoke_contract.get('sim_nodes') or 0)
    if train_sim_nodes >= 1 and smoke_sim_nodes == 0:
        return (f'smoke {smoke_path} did not exercise the simulator node '
                f'(smoke sim_nodes={smoke_sim_nodes}, train sim_nodes={train_sim_nodes})')
    return None


def sim_service_bindings(services, served_name):
    """responses_api_models entries whose openai_base_url points at the simulator (${oc.env:NVC_SIM_BASE_URL}).

    Every such entry must request exactly --sim-served-name as openai_model, otherwise the simulator
    (vLLM --served-model-name) answers 404 for every request; the match is recorded in RUN_CONTRACT.
    """
    needle = '${oc.env:' + SIM_URL_ENV + '}'
    bindings = []
    for top, body in services.items():
        if not isinstance(body, dict):
            continue
        for impl, spec in (body.get('responses_api_models') or {}).items():
            if not isinstance(spec, dict) or needle not in str(spec.get('openai_base_url', '')):
                continue
            bindings.append({'server': top, 'impl': impl, 'openai_base_url': spec.get('openai_base_url'),
                             'openai_model': spec.get('openai_model'), 'served_name': served_name})
            require(spec.get('openai_model') == served_name,
                    f'Services entry {top}.responses_api_models.{impl} requests openai_model '
                    f'{spec.get("openai_model")!r} from the simulator but --sim-served-name is {served_name!r}')
    require(bindings, f'No responses_api_models entry uses openai_base_url ${{oc.env:{SIM_URL_ENV}}}')
    return bindings


def epoch_budget(rows, epochs, prompts, planned_steps, age_steps, max_zero_variance_drop):
    """rows x epochs must cover (planned_steps + age_steps) x prompts after dropping zero-variance groups."""
    require(type(epochs) is int and epochs > 0, 'grpo.max_num_epochs must be a positive integer')
    require(type(rows) is int and rows > 0 and type(prompts) is int and prompts > 0, 'Invalid rows/prompts')
    require(type(planned_steps) is int and planned_steps > 0 and type(age_steps) is int and age_steps >= 0,
            'Invalid planned steps / trajectory age')
    drop = float(max_zero_variance_drop)
    require(math.isfinite(drop) and 0.0 <= drop < 1.0, '--max-zero-variance-drop must be in [0, 1)')
    required = math.ceil((planned_steps + age_steps) * prompts / (1.0 - drop))
    available = rows * epochs
    budget = {'rows': rows, 'max_num_epochs': epochs, 'prompts_per_step': prompts, 'planned_steps': planned_steps,
              'max_trajectory_age_steps': age_steps, 'max_zero_variance_drop': drop,
              'required_prompt_rows': required, 'available_prompt_rows': available,
              'formula': 'rows * epochs >= (planned_steps + max_trajectory_age_steps) * prompts / (1 - max_zero_variance_drop)'}
    require(available >= required,
            f'Data/epoch budget ({rows} rows x {epochs} epochs = {available}) cannot supply '
            f'({planned_steps} steps + {age_steps} age) x {prompts} prompts / (1 - {drop}) = {required} prompt rows')
    return budget


def smoke_gate(run, smoke_run, our_source_sha, our_images, allow_source_drift, train_sim_nodes=0,
               our_services_sha=None, our_sim_spec=None, allow_services_drift=False):
    smoke = Path(smoke_run).resolve(strict=True)
    require(smoke.parent == run.parent, 'Smoke run must live in the same runs/ directory')
    job = (smoke / 'JOB_ID').read_text().strip()
    require(job.isdigit(), 'Smoke run has no numeric JOB_ID (was it submitted?)')
    end_code = (smoke / 'END_CODE').read_text().strip() if (smoke / 'END_CODE').is_file() else None
    require(end_code == '0', f'Smoke run END_CODE is {end_code!r}, need 0: {smoke}')
    smoke_pass = smoke / 'SMOKE_PASS.json'
    require(smoke_pass.is_file(), 'Smoke did not write SMOKE_PASS.json (strict 3->4 resume not proven)')
    smoke_source = (smoke / 'source.sha256').read_text().split()[0]
    source_match = smoke_source == our_source_sha
    require(source_match or allow_source_drift,
            'Smoke was built from a different source tree; rebuild the smoke or pass --allow-source-drift')
    smoke_contract = json.loads((smoke / 'RUN_CONTRACT.json').read_text())
    smoke_images = {k: v.get('path') for k, v in smoke_contract.get('runtime_images', {}).items()}
    # The smoke must have exercised the same frozen services yaml (and, for a localsim train, the same
    # simulator identity): a hub smoke proves nothing about sim_gptoss -> NVC_SIM_BASE_URL plumbing.
    smoke_services = smoke_contract.get('services_sha256')
    services_match = our_services_sha is None or smoke_services == our_services_sha
    sim_fields = ('served_name', 'model_path', 'port')
    sim_spec_match = None
    smoke_sim_spec = smoke_contract.get('sim_spec') or {}
    if our_sim_spec is not None:
        sim_spec_match = all(smoke_sim_spec.get(k) == our_sim_spec.get(k) for k in sim_fields)
    services_drift = not services_match or sim_spec_match is False
    require(not services_drift or allow_services_drift,
            'Smoke services/simulator differ from this train '
            f'(services_match={services_match}, sim_spec_match={sim_spec_match}); '
            'rebuild the smoke from the same services yaml or pass --allow-services-drift')
    proof = {'schema': 'gym-smoke-acceptance.make-run.v2', 'status': 'PASS', 'job_id': job,
             'smoke_run': str(smoke), 'end_code': int(end_code), 'smoke_pass': json.loads(smoke_pass.read_text()),
             'smoke_pass_sha256': sha(smoke_pass), 'smoke_source_sha256': smoke_source,
             'train_source_sha256': our_source_sha, 'source_match': source_match,
             'images_match': smoke_images == {k: v['path'] for k, v in our_images.items()},
             'smoke_services_sha256': smoke_services, 'train_services_sha256': our_services_sha,
             'services_match': services_match,
             'sim_spec_match': sim_spec_match,
             'sim_spec_compared': {k: {'smoke': smoke_sim_spec.get(k), 'train': our_sim_spec.get(k)} for k in sim_fields}
             if our_sim_spec is not None else None,
             'services_drift': services_drift, 'allow_services_drift': bool(allow_services_drift),
             'terminal_state_checked_at_submit_by': 'scripts/gym_submit_guard.sh (sacct)'}
    if services_drift:
        print('[MAKE_RUN_WARN] smoke services/simulator drift accepted by --allow-services-drift: '
              f'services_match={services_match} sim_spec_match={sim_spec_match}', file=sys.stderr)
    if train_sim_nodes >= 1:
        warning = smoke_sim_warning(train_sim_nodes, smoke_contract, smoke)
        proof['simulator_node'] = {'train_sim_nodes': train_sim_nodes,
                                   'smoke_sim_nodes': int(smoke_contract.get('sim_nodes') or 0),
                                   'warning': warning}
        if warning:
            print('[MAKE_RUN_WARN] ' + warning, file=sys.stderr)
    write_json(run / 'GYM_SMOKE_ACCEPTANCE.json', proof)
    write_text(run / 'GYM_SMOKE_JOB', job)
    return proof


def build(args):
    project = Path(args.project).resolve()
    repo = project / 'src/pipeline/nemo_rl'
    gym_root = repo / '3rdparty/Gym-workspace/Gym'
    runs_root = Path(args.runs_root).resolve() if args.runs_root else project / 'runs'
    require(runs_root.name == 'runs', 'Runs root must be a directory literally named runs (gym_submit_guard.sh contract)')
    runs_root.mkdir(parents=True, exist_ok=True)
    guard_project = runs_root.parent
    require(os.environ.get('SLURM_JOB_ID') or os.environ.get('NVC_MAKE_RUN_ALLOW_NO_SLURM'),
            'Run make_run.py inside a CPU allocation (login node is text/submit only); '
            'set NVC_MAKE_RUN_ALLOW_NO_SLURM=1 only for unit tests')
    require(args.kind in ('smoke', 'train'), 'kind must be smoke or train')
    if args.kind == 'train':
        require(args.smoke_run or args.no_smoke_gate or args.accept_unsmoked_reason,
                'A train run needs --smoke-run <runs/smoke-v1-...> (gym_submit_guard.sh requires an accepted smoke); '
                '--no-smoke-gate builds a run that submit_run.sh will refuse; '
                '--accept-unsmoked-reason "<why>" records an explicit, logged override instead')
        require(not (args.smoke_run and args.accept_unsmoked_reason),
                'Use either --smoke-run or --accept-unsmoked-reason, not both')

    sys.path.insert(0, str(HERE))
    import seal_gym_recipe as seal
    import yaml

    recipe = Path(args.recipe).resolve(strict=True)
    bundle = Path(args.bundle).resolve(strict=True)
    config = load_recipe(recipe, project)
    applied = {}
    if args.kind == 'smoke':
        apply_smoke_defaults(config, applied)
    if args.kind == 'train':
        apply_train_gym_validation_guard(config, applied, keep=args.keep_gym_validation)
    if args.nodes is not None:
        require(args.nodes >= 3, '--nodes must be >= 3 (two learner nodes plus rollout)')
        set_dotted(config, 'cluster.num_nodes', args.nodes)
        set_dotted(config, 'policy.generation.colocated.resources.num_nodes', args.nodes - 2)
        applied.update({'cluster.num_nodes': args.nodes,
                        'policy.generation.colocated.resources.num_nodes': args.nodes - 2})
    for text in args.override or []:
        key, value = parse_override(text)
        set_dotted(config, key, value)
        applied[key] = value
    if args.services:
        services_path = Path(args.services).resolve(strict=True)
        set_dotted(config, 'env.nemo_gym.config_paths', [str(services_path)])
        applied['env.nemo_gym.config_paths'] = [str(services_path)]
    # Simulator nodes are validated before any directory exists; None means the historical layout.
    sim_spec = sim_spec_from_args(args, get_dotted(config, 'cluster.gpus_per_node', 8))

    run = runs_root / f'{args.kind}-v1-{utc_stamp()}'
    require(not run.exists() and not run.is_symlink(), 'Run already exists: ' + str(run))
    run.mkdir()
    checked_path(str(run))
    checks = {}
    try:
        _assemble(args, project, guard_project, repo, gym_root, run, recipe, bundle, config, applied, checks, seal, yaml,
                  sim_spec)
    except BaseException:
        if not args.keep_partial:
            shutil.rmtree(run, ignore_errors=True)
        raise
    return run


def _assemble(args, project, guard_project, repo, gym_root, run, recipe, bundle, config, applied, checks, seal, yaml,
              sim_spec=None):
    sim_nodes = int(getattr(args, 'sim_nodes', 0) or 0)
    # ---- services yaml -> gym-configs/ (frozen copy, config_paths rewritten)
    gym_config = config['env']['nemo_gym']
    config_paths = gym_config.get('config_paths')
    require(isinstance(config_paths, list) and config_paths, 'env.nemo_gym.config_paths required')
    (run / 'gym-configs').mkdir()
    server_configs = []
    services = {}
    services_text = ''
    for index, value in enumerate(config_paths):
        source_config = Path(value)
        require(source_config.is_absolute(), 'Only absolute external Gym services configs are frozen: ' + value)
        source_config = source_config.resolve(strict=True)
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', source_config.name), 'Unsafe Gym config filename')
        destination = run / 'gym-configs' / f'{index:03d}-{source_config.name}'
        shutil.copy2(source_config, destination)
        digest = sha(destination)
        require(digest == sha(source_config), 'Services config changed while copying')
        config_paths[index] = str(destination)
        text = destination.read_text()
        services_text += text
        loaded = yaml.safe_load(text)
        require(isinstance(loaded, dict), 'Services config must be a mapping: ' + str(source_config))
        services.update(loaded)
        server_configs.append({'source_path': str(source_config), 'path': str(destination),
                               'scope': 'frozen_external_config', 'sha256': digest})
    write_json(run / 'GYM_SERVER_CONFIGS.json', {'files': server_configs})

    # ---- secrets: names only; values are read by gpu_job.sh at allocation time. Launcher env
    # (NVC_SIM_*) is exported by gpu_job.sh itself and is only legal with simulator nodes.
    secret_names, launcher_names = classify_env_refs(services_text, sim_nodes)
    secrets = []
    for name in secret_names:
        secret_file = project / SECRET_SOURCES[name]
        present = secret_file.is_file() and secret_file.stat().st_size > 1
        require(present or args.allow_missing_secret, f'Secret file for {name} missing/empty: {secret_file}')
        secrets.append({'name': name, 'source': SECRET_SOURCES[name], 'present_at_build': present})
    write_text(run / 'SECRETS_REQUIRED', '\n'.join(secret_names) if secret_names else '')
    checks['secrets'] = secrets
    if launcher_names:
        checks['launcher_env'] = [{'name': name, 'source': LAUNCHER_ENV_SOURCES[name]} for name in launcher_names]
    sim_bindings = sim_service_bindings(services, sim_spec['served_name']) if sim_spec is not None else None

    # ---- data bundle
    manifest_path = bundle / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('schema') == 'gym-data-v1', 'Bundle MANIFEST.json schema must be gym-data-v1')
    train_source = (bundle / manifest['data_path']).resolve(strict=True)
    entry, = config['data']['train']
    require(Path(entry['data_path']).resolve(strict=True) == train_source,
            f'Recipe train data {entry["data_path"]} is not the bundle data {train_source}')
    train_sha = sha(train_source)
    require(train_sha == manifest['data_sha256'], 'Bundle data SHA differs from MANIFEST.json')
    rows, routes, max_output = scan_rows(train_source, services)
    require(rows == manifest['rows'] and rows > 0, f'Row count {rows} differs from MANIFEST rows {manifest["rows"]}')
    require(entry.get('max_samples') in (None, rows), 'Dataset truncation differs from manifested rows')
    data_link = link_or_copy(train_source, run / 'data.train.jsonl')
    if data_link == 'copy':
        require(sha(run / 'data.train.jsonl') == train_sha, 'Frozen data copy changed')
    known = {'data.train.jsonl': train_sha}
    entry['data_path'] = str(run / 'data.train.jsonl')
    val = None
    validation = config['data'].get('validation')
    if validation:
        val_entry, = validation
        val_source = Path(val_entry['data_path']).resolve(strict=True)
        val_rows, val_routes, _ = scan_rows(val_source, services)
        val_sha = sha(val_source)
        for key in ('val_sha256', 'validation_sha256'):
            if key in manifest:
                require(manifest[key] == val_sha, 'Validation data SHA differs from MANIFEST ' + key)
        val_link = link_or_copy(val_source, run / 'data.val.jsonl')
        if val_link == 'copy':
            require(sha(run / 'data.val.jsonl') == val_sha, 'Frozen validation copy changed')
        known['data.val.jsonl'] = val_sha
        val_entry['data_path'] = str(run / 'data.val.jsonl')
        val = {'source': str(val_source), 'sha256': val_sha, 'rows': val_rows, 'routes': val_routes, 'link': val_link}
    shutil.copy2(manifest_path, run / 'DATA_MIX_MANIFEST.json')
    asset_ref = manifest.get('asset_manifest')
    assets_verified = []
    if asset_ref:
        asset_manifest_path = (bundle / asset_ref['path']).resolve(strict=True)
        require(sha(asset_manifest_path) == asset_ref['sha256'], 'Asset manifest SHA mismatch')
        shutil.copy2(asset_manifest_path, run / 'GYM_ASSET_MANIFEST.json')
        assets = json.loads(asset_manifest_path.read_text())
        entries = assets.get('files', assets.get('assets', []))
        for asset in entries:
            path = (asset_manifest_path.parent / asset['path']).resolve(strict=True)
            if not args.skip_asset_hash:
                require(sha(path) == asset['sha256'], 'Gym data asset SHA mismatch: ' + str(path))
            assets_verified.append({'path': str(path), 'sha256': asset['sha256'], 'bytes': path.stat().st_size})
    write_json(run / 'GYM_EXTERNAL_ASSETS.json', {
        'files': assets_verified, 'hashed_at_build': not args.skip_asset_hash,
        'note': 'Immutable external data assets; gpu_job.sh rehashes them through inputs.sha256'})
    checks['data'] = {'rows': rows, 'sha256': train_sha, 'routes': routes, 'link': data_link,
                      'max_output_tokens_by_route': max_output, 'validation': val,
                      'assets': len(assets_verified), 'assets_hashed': not args.skip_asset_hash}

    # ---- service closure (source dirs, venvs, tau2 data) against the tree that will be tarred
    venv_dir = gym_config.get('uv_venv_dir') or os.environ.get('NEMO_GYM_VENV_DIR') or '/opt/nemotronvc/gym-v1/venvs'
    closure = service_closure(services, gym_root, venv_dir, bool(gym_config.get('skip_venv_if_present')),
                              args.allow_missing_venv)
    closure['uv_venv_dir'] = venv_dir
    closure['tau2_data'] = tau2_data_check(services, gym_root)
    if closure['tau2_data'].get('needed') and not closure['tau2_data'].get('banking_knowledge_present'):
        message = ('tau2 agent configured but Gym/responses_api_agents/tau2/tau2_data lacks '
                   'tau2/domains/banking_knowledge/db.json; the server would git-clone on a compute node')
        if args.allow_missing_tau2_data:
            closure.setdefault('warnings', []).append(message)
        else:
            closure['problems'].append(message)
    write_json(run / 'GYM_SERVICE_CLOSURE.json', closure)
    require(not closure['problems'], 'Service closure problems:\n  ' + '\n  '.join(closure['problems']))
    checks['service_closure'] = {'servers': len(closure['servers']), 'warnings': closure.get('warnings', [])}

    # ---- student identity
    student = Path(config['policy']['model_name']).resolve(strict=True)
    require(Path(config['policy']['tokenizer']['name']).resolve(strict=True) == student, 'Tokenizer differs from student')
    require(Path(config['policy']['generation']['model_name']).resolve(strict=True) == student, 'Rollout model differs from student')
    index_sha = sha(student / 'model.safetensors.index.json')
    audit = project / 'receipts/v1-input-contract.json'
    student_audit = 'receipt_missing'
    if audit.is_file():
        accepted = json.loads(audit.read_text())
        require(accepted['status'] in ('PASS', 'PASS_WITH_WARNINGS'), 'Base student input audit not accepted')
        if student != Path(accepted['student']['path']).resolve(strict=True):
            # Another checkpoint of the audited architecture (e.g. an SFT of SFTKD399's family): allowed only with an
            # explicit reason; the architecture list must equal the audited one (topology checks then carry over).
            require(bool(args.new_student_reason),
                    'Student differs from accepted SFTKD399 (pass --new-student-reason to start from another checkpoint)')
            arch = json.loads((student / 'config.json').read_text()).get('architectures')
            require(arch == accepted['student']['architecture']['architectures'],
                    f'New student architectures {arch} differ from the audited {accepted["student"]["architecture"]["architectures"]}')
            shutil.copy2(audit, run / 'MODEL_INPUT_AUDIT.json')
            student_audit = 'new_student_same_architecture: ' + args.new_student_reason
        else:
            require(index_sha == accepted['student']['index_sha256'], 'Student index changed since acceptance')
            shutil.copy2(audit, run / 'MODEL_INPUT_AUDIT.json')
            student_audit = 'matched_receipt'
    checks['student'] = {'path': str(student), 'index_sha256': index_sha, 'audit': student_audit}

    # ---- images
    images = freeze_images(project, run, args.runtime_image or project / 'runtime/gym-v1/NemotronVC-Gym34-vllm0271-v1.sqsh',
                           args.sandbox_image or project / 'runtime/gym-v1/sandbox-x86_64-v1.sqsh')

    # ---- run-local paths and lineage
    config['logger']['log_dir'] = str(run / 'train')
    if 'nemo_gym_log_dir' in gym_config:
        gym_config['nemo_gym_log_dir'] = str(run / 'nemo_gym')
    if args.kind == 'smoke':
        for key, subdir in (('checkpoint_dir', 'checkpoints'), ('hf_export_dir', 'hf'), ('monitor_dir', 'monitor')):
            config['checkpointing'][key] = str(run / subdir)
    ckpt = Path(config['checkpointing']['checkpoint_dir']).resolve()
    config['checkpointing']['checkpoint_dir'] = checked_path(str(ckpt))
    require(not list(ckpt.glob('tmp_step_*')), 'Incomplete checkpoint publication exists in ' + str(ckpt))
    existing = sorted(p.resolve() for p in ckpt.glob('step_*') if p.is_dir())
    if args.resume:
        require(args.kind == 'train', 'Only train runs resume')
        require(len(existing) == 1, 'Resume requires exactly one complete step_* checkpoint')
        info = json.loads((existing[0] / 'training_info.json').read_text())
        require(existing[0].name == f"step_{info['current_step']}", 'Checkpoint name/step mismatch')
        require(0 < info['current_step'] < config['grpo']['max_num_steps'], 'Resume source outside target horizon')
    else:
        require(not existing, 'New lineage must not contain step_* checkpoints: ' + str(ckpt))
        require(not (ckpt / 'ORIGIN_RUN').exists(), 'New lineage already has a submitted origin: ' + str(ckpt))

    # ---- structural validation (parameterized topology arithmetic)
    batch = seal.validate_config(config, args.kind)
    epochs = config['grpo']['max_num_epochs']
    # Smoke resumes strictly to step 4 (ray_role.sh); a train's max_num_steps equals the scheduler horizon.
    needed = 4 if args.kind == 'smoke' else config['grpo']['max_num_steps']
    budget = epoch_budget(rows, epochs, batch['prompts'], needed, config['grpo']['async_grpo']['max_trajectory_age_steps'],
                          getattr(args, 'max_zero_variance_drop', DEFAULT_MAX_ZERO_VARIANCE_DROP))

    # ---- config.yaml (OmegaConf rendering: 2-space flat scalars for gpu_job.sh sed + idle watchdog)
    text = to_yaml(config)
    (run / 'config.yaml').write_text(text)
    lines = [line for line in text.splitlines() if line.startswith('  checkpoint_dir: ')]
    require(lines == ['  checkpoint_dir: ' + config['checkpointing']['checkpoint_dir']],
            'config.yaml must expose exactly one flat checkpoint_dir for gpu_job.sh')
    if gym_config.get('uv_venv_dir'):
        require(('    uv_venv_dir: ' + gym_config['uv_venv_dir']) in text.splitlines(),
                'config.yaml must expose env.nemo_gym.uv_venv_dir flat for runtime_env.sh')
    from omegaconf import OmegaConf
    require(OmegaConf.to_container(OmegaConf.load(run / 'config.yaml'), resolve=True) == config,
            'config.yaml did not round-trip')
    sys.path.insert(0, str(project / 'scripts'))
    import gym_idle_watchdog
    checks['idle_watchdog_policy'] = gym_idle_watchdog.frozen_policy(run)
    write_json(run / 'GYM_RECIPE_VALIDATION.json', {
        'schema': 'gym-recipe-validation.make-run.v1', 'kind': args.kind, 'recipe': str(recipe),
        'recipe_sha256': sha(recipe), 'overrides_applied': applied, 'batch': batch,
        'data_rows': rows, 'planned_updates': needed, 'max_num_epochs': epochs,
        'max_zero_variance_drop': budget['max_zero_variance_drop'], 'epoch_budget': budget,
        'validator': 'scripts/seal_gym_recipe.py::validate_config'})

    # ---- launcher scripts + their structural tests
    (run / 'scripts').mkdir()
    for name in RUNTIME_SCRIPTS + RUNTIME_MODULES:
        shutil.copy2(project / 'scripts' / name, run / 'scripts' / name)
    for name in RUNTIME_SCRIPTS:
        subprocess.run(['bash', '-n', str(run / 'scripts' / name)], check=True)
    python = sys.executable
    with (run / 'LAUNCHER_TESTS.log').open('w') as log:
        result = subprocess.run([python, str(run / 'scripts/test_ray_ports.py')], stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(run / 'scripts'))
    require(result.returncode == 0, 'Frozen launcher tests failed; see ' + str(run / 'LAUNCHER_TESTS.log'))
    subprocess.run([python, str(run / 'scripts/ray_ports.py'), 'check-launcher', str(run / 'scripts/ray_role.sh'),
                    '--receipt', str(run / 'LAUNCHER_PREFLIGHT.json')], check=True)
    launcher_files = [str(p.relative_to(run)) for p in sorted((run / 'scripts').glob('*')) if p.is_file()]
    launcher_files += ['LAUNCHER_TESTS.log', 'LAUNCHER_PREFLIGHT.json']
    (run / 'launcher.sha256').write_text(manifest_lines(run, launcher_files, {}))
    checks['launcher'] = {'tests': 'PASS', 'check_launcher': json.loads((run / 'LAUNCHER_PREFLIGHT.json').read_text())}

    # ---- source snapshot of the CURRENT worktree (+ Gym submodule), uncommitted state recorded
    pins, status = source_facts(project, repo)
    dirty = {name: entries for name, entries in status['porcelain'].items() if entries}
    if args.require_clean:
        require(status['git_binary'], '--require-clean needs a git binary inside the container')
        require(not dirty, 'Worktree is not clean: ' + json.dumps(dirty))
    excludes = [f'--exclude={pattern}' for pattern in TAR_EXCLUDES]
    subprocess.run(['tar', *excludes, '-czf', str(run / 'source.tar.gz'), '-C', str(repo), '.'], check=True)
    source_sha = sha(run / 'source.tar.gz')
    known['source.tar.gz'] = source_sha
    write_text(run / 'source.sha256', f'{source_sha}  source.tar.gz')
    if status['patch']:
        (run / 'source.patch').write_text(status['patch'])
    write_json(run / 'SOURCE_STATUS.json', {
        'git_binary_available': status['git_binary'], 'porcelain': status['porcelain'],
        'untracked_files_sha256': {
            f'{name}:{line[3:]}': sha(repo / line[3:]) if name == 'nemo_rl_applied' else sha(gym_root / line[3:])
            for name, entries in status['porcelain'].items() for line in entries
            if line.startswith('?? ') and (repo / line[3:] if name == 'nemo_rl_applied' else gym_root / line[3:]).is_file()},
        'tar_excludes': list(TAR_EXCLUDES)})
    receipt = [f'{name}={value}' for name, value in pins.items()]
    receipt += ['source_kind=worktree_snapshot_including_uncommitted_changes',
                f'worktree_dirty={"true" if dirty else "false"}',
                f'dirty_files={";".join(f"{n}:{e.strip()}" for n, es in dirty.items() for e in es)}',
                f'source_archive_sha256={source_sha}', 'builder=scripts/make_run.py',
                f'new_cpu_tests_run={"true" if args.run_repo_tests else "false"}']
    write_text(run / 'SOURCE_RECEIPT.txt', '\n'.join(receipt))
    checks['source'] = {'pins': pins, 'dirty': dirty, 'sha256': source_sha}

    # ---- contracts consumed by ray_role.sh (RUN_CONTRACT.json) and the lineage
    identity = seal.config_identity(config)
    lineage_contract = {'schema': 'gym-lineage.v1', 'student': str(student), 'data_sha256': train_sha,
                        'rows': rows, 'research_config_sha256': identity,
                        'gym_server_config_identity': [{'scope': x['scope'], 'sha256': x.get('sha256'),
                                                        'relative_path': None} for x in server_configs], **batch}
    contract = {'schema': 'gym-run.v1', 'builder': 'scripts/make_run.py', 'input_mode': 'gym', 'objective': 'rlvr',
                'kind': args.kind, 'resume_mode': 'resume' if args.resume else 'new',
                'nodes': batch['policy_nodes'], 'checkpoint_dir': str(ckpt),
                'expected_resume_checkpoint': str(existing[0]) if existing else None,
                'student': str(student), 'student_index_sha256': index_sha, 'data_sha256': train_sha, 'data_rows': rows,
                'val_data_sha256': val['sha256'] if val else None, 'val_data_rows': val['rows'] if val else None,
                'routes': routes, 'env_name': 'nemo_gym', 'max_input_seq_length': config['data'].get('max_input_seq_length'),
                'render_scope': 'structural only; Gym trajectory/token parity is proven by the GPU smoke',
                'runtime_images': images, 'uv_venv_dir': venv_dir, 'secrets_required': secret_names,
                'services_sha256': [x['sha256'] for x in server_configs],
                'smoke_resume_step': 4 if args.kind == 'smoke' else None, 'source_sha256': source_sha, **batch}
    # Slurm allocation = Ray nodes + simulator nodes. Without --sim-nodes the files and keys are
    # exactly the historical ones (NODES == cluster.num_nodes, no SIM_* files, no sim keys).
    layout = node_layout(batch['policy_nodes'], sim_nodes)
    if sim_spec is not None:
        require(layout['sim_nodes'] >= 1, 'sim_spec without simulator nodes')
        contract.update({'sim_nodes': layout['sim_nodes'], 'ray_nodes': layout['ray_nodes'], 'sim_spec': sim_spec,
                         'launcher_env': launcher_names, 'sim_service_bindings': sim_bindings})
        write_text(run / 'SIM_NODES', str(layout['sim_nodes']))
        write_json(run / 'SIM_SPEC.json', sim_spec)
        checks['simulator'] = {'sim_nodes': layout['sim_nodes'], 'ray_nodes': layout['ray_nodes'],
                               'total_nodes': layout['total_nodes'], 'model_path': sim_spec['model_path'],
                               'python': sim_spec['python'], 'port': sim_spec['port']}
    write_json(run / 'RUN_CONTRACT.json', contract)
    write_json(run / 'GYM_LINEAGE_CONTRACT.json', lineage_contract)
    write_json(run / 'INPUT_MANIFEST.json', {
        'schema': 'gym-inputs.v1', 'status': 'IDENTITY_PASS_RUNTIME_PENDING', 'data_sha256': train_sha, 'data_rows': rows,
        'manifest_sha256': sha(manifest_path), 'source_manifest_path': str(manifest_path),
        'recipe_sha256': sha(recipe), 'recipe_path': str(recipe), 'student': str(student), 'teacher_enabled': False})
    flat = {'INPUT_MODE': 'gym', 'OBJECTIVE': 'rlvr', 'KIND': args.kind, 'NODES': str(layout['total_nodes']),
            'GYM_LINEAGE_DIR': checked_path(str(ckpt)), 'GYM_PROJECT_DIR': checked_path(str(guard_project)),
            'GYM_EXPECTED_RESUME': str(existing[0]) if existing else 'NONE'}
    for name, value in flat.items():
        write_text(run / name, value)

    if args.kind == 'train' and args.smoke_run:
        checks['smoke_gate'] = smoke_gate(run, args.smoke_run, source_sha, images, args.allow_source_drift,
                                          train_sim_nodes=layout['sim_nodes'],
                                          our_services_sha=contract['services_sha256'], our_sim_spec=sim_spec,
                                          allow_services_drift=args.allow_services_drift)
    elif args.kind == 'train' and args.accept_unsmoked_reason:
        # Explicit, recorded override of the smoke gate. Every smoke run dir in runs/ is listed with
        # its END_CODE / SMOKE_PASS state so the acceptance carries the evidence the operator relied on.
        observed = []
        for smoke_dir in sorted(run.parent.glob('smoke-v1-*')):
            end_code_path = smoke_dir / 'END_CODE'
            observed.append({
                'run': smoke_dir.name,
                'job_id': (smoke_dir / 'JOB_ID').read_text().strip() if (smoke_dir / 'JOB_ID').is_file() else None,
                'end_code': end_code_path.read_text().strip() if end_code_path.is_file() else None,
                'smoke_pass': (smoke_dir / 'SMOKE_PASS.json').is_file(),
                'source_match': ((smoke_dir / 'source.sha256').read_text().split()[0] == source_sha)
                if (smoke_dir / 'source.sha256').is_file() else None,
            })
        proof = {'schema': 'gym-smoke-acceptance.v1', 'status': 'override',
                 'reason': args.accept_unsmoked_reason, 'our_source_sha256': source_sha,
                 'observed_smokes': observed, 'accepted_by': os.environ.get('USER'),
                 'accepted_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        write_json(run / 'GYM_SMOKE_ACCEPTANCE.json', proof)
        write_text(run / 'GYM_SMOKE_JOB', 'override')
        checks['smoke_gate'] = {'status': 'override', 'reason': args.accept_unsmoked_reason,
                                'observed_smokes': len(observed)}
        print(f"[MAKE_RUN_WARN] smoke gate overridden: {args.accept_unsmoked_reason}", flush=True)

    # ---- manifests: role-preflight (submit_run.sh rlvr gate), GYM_SUBMIT_GATES (small files), inputs (all)
    (run / 'role-preflight.sha256').write_text(
        manifest_lines(run, ['config.yaml', 'RUN_CONTRACT.json', 'GYM_RECIPE_VALIDATION.json'], {}))
    ready = {'schema': 'gym-make-run-ready.v1', 'status': 'PASS', 'cpu_job': os.environ.get('SLURM_JOB_ID'),
             'run': str(run), 'checkpoint_dir': str(ckpt), 'config_sha256': sha(run / 'config.yaml'),
             'data_sha256': train_sha, 'source_sha256': source_sha,
             'admission_mode': 'make_run_structural',
             'checks': checks,
             'not_checked_here': [
                 'Gym actor spin-up / real rollout (was gym_actor_runtime_gate on CPU Ray): proven by the GPU smoke',
                 'native scheduler replay against DCP metadata (preflight_gym_scheduler): new lineage starts fresh',
                 'full-row Gym render / tokenizer parity (preflight_gym_runtime): rows are structurally scanned only',
                 'inference-api reachability / quota for gpt-oss-120b: first observed in the smoke driver.log']}
    write_json(run / 'GYM_FROZEN_READY.json', ready)
    everything = [str(p.relative_to(run)) for p in sorted(run.rglob('*')) if p.is_file()]
    big = {'source.tar.gz', 'data.train.jsonl', 'data.val.jsonl'}
    gates = [n for n in everything if n not in big and n not in ('inputs.sha256', 'GYM_SUBMIT_GATES.sha256')]
    (run / 'GYM_SUBMIT_GATES.sha256').write_text(manifest_lines(run, gates, known))
    everything = [str(p.relative_to(run)) for p in sorted(run.rglob('*')) if p.is_file() and p.name != 'inputs.sha256']
    inputs = manifest_lines(run, everything, known)
    inputs += ''.join(f"{asset['sha256']}  {asset['path']}\n" for asset in assets_verified
                      if '\n' not in asset['path'] and asset['path'].startswith('/'))
    (run / 'inputs.sha256').write_text(inputs)
    for name in everything + ['inputs.sha256']:
        path = run / name
        if name not in big and not path.is_symlink():
            path.chmod(0o444)
    for name in ('gpu_job.sh', 'ray_role.sh', 'runtime_env.sh'):
        (run / 'scripts' / name).chmod(0o555)
    if run.parent == (project / 'runs').resolve():
        # Convenience pointer only for real runs; throwaway --runs-root builds must not clobber it.
        latest = project / 'logs' / f'latest-frozen-{args.kind}'
        latest.parent.mkdir(exist_ok=True)
        write_text(latest, str(run))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kind', required=True, choices=('smoke', 'train'))
    parser.add_argument('--recipe', default=str(DEFAULT_PROJECT / 'config/agentic-teacher-v1.yaml'))
    parser.add_argument('--services', default=None,
                        help='Override env.nemo_gym.config_paths with this single services yaml')
    parser.add_argument('--bundle', default=str(DEFAULT_PROJECT / 'data/agentic-teacher-v1'))
    parser.add_argument('--nodes', type=int, default=None, help='Ray (cluster) nodes; rollout nodes = nodes - 2')
    sim = parser.add_argument_group('simulator nodes', 'extra GPU nodes outside Ray running a standalone vLLM '
                                    '(scripts/sim_role.sh) for the sim_gptoss server; NODES = Ray nodes + sim nodes')
    sim.add_argument('--sim-nodes', type=int, default=0, help='number of simulator nodes (default 0 = historical layout)')
    sim.add_argument('--sim-model-path', default=None, help='HF model dir with config.json (required when --sim-nodes >= 1)')
    sim.add_argument('--sim-served-name', default='gpt-oss-120b', help='--served-model-name; must equal openai_model in the services yaml')
    sim.add_argument('--sim-port', type=int, default=18100)
    sim.add_argument('--sim-tp', type=int, default=1, help='--tensor-parallel-size (TP1 x DP8 measured best on A100, SERVE_TEST.md)')
    sim.add_argument('--sim-dp', type=int, default=8, help='--data-parallel-size (tp x dp <= GPUs per node)')
    sim.add_argument('--sim-max-model-len', type=int, default=32768)
    sim.add_argument('--sim-gpu-memory-utilization', type=float, default=0.90)
    sim.add_argument('--sim-max-num-seqs', type=int, default=128)
    sim.add_argument('--sim-extra-args', default=DEFAULT_SIM_EXTRA_ARGS, help='extra vllm api_server flags (shell-split)')
    sim.add_argument('--sim-python', default=DEFAULT_SIM_PYTHON, help='vLLM 0.27.1 interpreter inside the container')
    parser.add_argument('--override', action='append', metavar='KEY=YAML', help='dotted config override (repeatable)')
    parser.add_argument('--smoke-run', default=None, help='train only: accepted smoke run dir (END_CODE 0 + SMOKE_PASS.json)')
    parser.add_argument('--no-smoke-gate', action='store_true', help='train only: build without GYM_SMOKE_JOB (unsubmittable)')
    parser.add_argument('--accept-unsmoked-reason', default=None,
                        help='train only: submit without an accepted smoke; writes GYM_SMOKE_JOB=override and '
                             'GYM_SMOKE_ACCEPTANCE.json {status: override, reason, observed smokes}; gym_submit_guard.sh '
                             'prints [SMOKE_GATE_OVERRIDE] and skips the terminal-smoke check')
    parser.add_argument('--new-student-reason', default=None,
                        help='start from a checkpoint other than the audited SFTKD399 (same architecture); recorded in checks')
    parser.add_argument('--allow-source-drift', action='store_true', help='train only: accept a smoke from another source sha')
    parser.add_argument('--allow-services-drift', action='store_true',
                        help='train only: accept a smoke whose services yaml / simulator spec differ (recorded in GYM_SMOKE_ACCEPTANCE.json)')
    parser.add_argument('--max-zero-variance-drop', type=float, default=DEFAULT_MAX_ZERO_VARIANCE_DROP,
                        help='epoch budget: rows*epochs >= (steps + age) * prompts / (1 - drop); recorded in GYM_RECIPE_VALIDATION.json')
    parser.add_argument('--resume', action='store_true', help='train only: expect exactly one step_* in checkpoint_dir')
    parser.add_argument('--keep-gym-validation', action='store_true',
                        help='train only: keep grpo.val_period/val_at_start/val_at_end from the recipe (default: forced off, '
                             'because the Gym validation path groups by the training stride and crashes the loop)')
    parser.add_argument('--runtime-image', default=None)
    parser.add_argument('--sandbox-image', default=None)
    parser.add_argument('--project', default=str(DEFAULT_PROJECT))
    parser.add_argument('--runs-root', default=None, help='default <project>/runs; must be named runs')
    parser.add_argument('--require-clean', action='store_true', help='refuse an uncommitted nemo_rl/Gym worktree')
    parser.add_argument('--skip-asset-hash', action='store_true', help='trust ASSET_MANIFEST.json digests for the asset files')
    parser.add_argument('--allow-missing-venv', action='store_true', help='warn instead of fail when a server venv is absent')
    parser.add_argument('--allow-missing-tau2-data', action='store_true')
    parser.add_argument('--allow-missing-secret', action='store_true')
    parser.add_argument('--run-repo-tests', action='store_true', help='record that repo tests were run in this CPU job')
    parser.add_argument('--keep-partial', action='store_true', help='keep the run dir when the build fails')
    return parser


def main():
    args = build_parser().parse_args()
    try:
        run = build(args)
    except BuildError as error:
        print('[MAKE_RUN_REFUSED] ' + str(error), file=sys.stderr)
        return 1
    contract = json.loads((run / 'RUN_CONTRACT.json').read_text())
    summary = {k: contract[k] for k in ('kind', 'nodes', 'prompts', 'K', 'response_gbs', 'scheduler_horizon',
                                        'data_rows', 'secrets_required', 'source_sha256')}
    if contract.get('sim_nodes'):
        summary.update({'sim_nodes': contract['sim_nodes'], 'ray_nodes': contract['ray_nodes'],
                        'slurm_nodes': (run / 'NODES').read_text().strip(), 'launcher_env': contract.get('launcher_env')})
    summary['run'] = str(run)
    print('[SOURCE_FROZEN] ' + str(run))
    print('[MAKE_RUN_OK] ' + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
