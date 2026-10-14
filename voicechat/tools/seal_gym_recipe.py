#!/usr/bin/env python3
"""Freeze Gym inputs and a fresh/resumed lineage contract on a CPU allocation.

This is identity/topology validation, not a claim that Gym trajectories or model
serving work. The separate native Gym CPU gate and GPU smoke establish those.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def resolve_member(root, value):
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve(strict=True)


def checked_path(value):
    # These paths are also consumed by a Python-free login-node submitter.
    require(isinstance(value, str) and re.fullmatch(r'/[A-Za-z0-9_./-]+', value),
            'Unsafe/non-absolute path in submission contract')
    return value


def normalized_config(config):
    value = copy.deepcopy(config)
    value['logger']['log_dir'] = '<RUN>/train'
    value['data']['train'][0]['data_path'] = '<FROZEN_DATA>'
    # Runtime logs are allocation-local; they cannot change the objective.
    if 'nemo_gym_log_dir' in value.get('env', {}).get('nemo_gym', {}):
        value['env']['nemo_gym']['nemo_gym_log_dir'] = '<RUN>/nemo_gym'
    gym = value.get('env', {}).get('nemo_gym', {})
    if 'config_paths' in gym:
        gym['config_paths'] = ['<RUN>/gym-configs/' + Path(path).name
                               if '/gym-configs/' in path else path for path in gym['config_paths']]
    return value


def config_identity(config):
    return hashlib.sha256(json.dumps(normalized_config(config), sort_keys=True,
                                   separators=(',', ':')).encode()).hexdigest()


def resolved_recipe_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def freeze_images(api, run):
    """Bind CPU-verified, immutable image publications without rehashing 50GB."""
    images = {}
    for role in ('runtime', 'sandbox'):
        item = api.get(role + '_image')
        require(isinstance(item, dict), 'Missing accepted ' + role + ' image')
        path = Path(checked_path(item['path'])).resolve(strict=True)
        require(path.is_file() and path.suffix == '.sqsh', 'Invalid ' + role + ' image')
        digest = item['sha256']
        require(re.fullmatch(r'[0-9a-f]{64}', digest), 'Invalid image digest')
        proof = item['acceptance']
        receipt = Path(checked_path(proof['path'])).resolve(strict=True)
        require(receipt.stat().st_size <= 32 << 20 and sha(receipt) == proof['sha256'],
                'Image acceptance digest changed')
        accepted = json.loads(receipt.read_text())
        accepted_path = accepted.get('image', accepted.get('path'))
        require(accepted['status'] == 'PASS' and accepted_path is not None and
                Path(accepted_path).resolve() == path and accepted['sha256'] == digest and
                type(accepted['bytes']) is int and accepted['bytes'] == path.stat().st_size > 0,
                'Image acceptance identity/size mismatch')
        shutil.copy2(receipt, run / (role.upper() + '_IMAGE_ACCEPTANCE.json'))
        (run / (role.upper() + '_IMAGE')).write_text(str(path) + '\n')
        images[role] = {'path': str(path), 'sha256': digest, 'bytes': accepted['bytes'],
                        'acceptance_sha256': proof['sha256'],
                        'acceptance_source': str(receipt)}
    write_json(run / 'GYM_IMAGE_CONTRACT.json', images)
    return images


def validate_config(config, kind):
    require(kind in ('smoke', 'train'), 'Unknown run kind')
    grpo, policy = config['grpo'], config['policy']
    require(config['on_policy_distillation']['enabled'] is False, 'Gym RLVR must disable OPD')
    require(config['on_policy_distillation'].get('non_colocated_teachers', {}).get('enabled', False) is False,
            'Gym RLVR may not reserve OPD teacher ranks')
    require(grpo['adv_estimator']['name'] == 'grpo', 'Expected native reward estimator')
    require(not grpo.get('reward_training', {}).get('enabled', False),
            'Static terminal reward extension is incompatible with native Gym')
    require(config['env'].get('should_use_nemo_gym') is True, 'Gym routing not enabled')
    generation = policy['generation']
    require(generation['backend'] == 'vllm' and generation['vllm_cfg']['async_engine'] is True,
            'Expected asynchronous vLLM')
    require(generation['vllm_cfg']['expose_http_server'] is True, 'Gym requires HTTP serving')
    require(generation['node_engine_data_parallel_size'] == 8 and
            generation['vllm_cfg']['tensor_parallel_size'] == 1 and
            generation['vllm_cfg']['expert_parallel_size'] == 8,
            'Expected reviewed vLLM TP1/nodeDP8/EP8 topology')
    # Age 1 is the historical contract; age 2 is admitted for interactive envs
    # (docs/AGENTIC_TEACHER_PLAN_20260930.md section 3.9, off-policy corrected by TIS).
    require(grpo['async_grpo']['enabled'] and grpo['async_grpo']['max_trajectory_age_steps'] in (1, 2),
            'Expected reviewed async age contract (max_trajectory_age_steps 1 or 2)')
    prompts, k = grpo['num_prompts_per_step'], grpo['num_generations_per_prompt']
    require(type(prompts) is int and prompts > 0 and type(k) is int and k >= 2, 'Invalid prompt/K batch')
    require(grpo.get('batch_multiplier', 1) == 1, 'Unreviewed batch multiplier')
    require(policy['train_global_batch_size'] == policy['generation_batch_size'] == prompts * k,
            'GBS/generation batch must equal prompts times K')
    horizon = policy['megatron_cfg'].get('train_iters')
    require(type(horizon) is int and horizon > 0, 'Explicit fixed scheduler horizon required')
    scheduler = policy['megatron_cfg']['scheduler']
    require(not scheduler.get('use_checkpoint_opt_param_scheduler', False) and
            not scheduler.get('override_opt_param_scheduler', False), 'Scheduler mismatch overrides forbidden')
    require(type(scheduler.get('lr_decay_iters')) is int and scheduler['lr_decay_iters'] > 0,
            'Explicit LR decay horizon required')
    require(type(scheduler.get('lr_warmup_iters')) is int and 0 <= scheduler['lr_warmup_iters'] <= horizon,
            'Explicit valid warmup required')
    entry, = config['data']['train']
    require(entry['env_name'] == 'nemo_gym' and entry['processor'] == 'nemo_gym_data_processor',
            'Expected native Gym processor and environment')
    require(entry['dataset_name'].split('.')[-1] == 'NemoGymDataset', 'Expected native Gym dataset')
    checkpoint = config['checkpointing']
    require(checkpoint['enabled'] and checkpoint['save_optimizer'] and
            checkpoint['ft_keep_latest_k'] == 1 and checkpoint['keep_top_k'] == 0 and
            checkpoint.get('require_single_recovery_checkpoint') is True,
            'One complete optimizer checkpoint must remain resumable')
    require(checkpoint.get('load_replay_buffer') is False, 'Use native frontier regeneration')
    require(checkpoint['allocation_deadline_env'] == 'NVC_CHECKPOINT_DEADLINE_EPOCH' and
            checkpoint['deadline_step_reserve_seconds'] >= 1800, 'Missing allocation save reserve')
    require(config['cluster']['gpus_per_node'] == 8, 'Expected eight GPUs per node')
    rollout = generation['colocated']['resources']['num_nodes']
    gym_nodes = config['env']['nemo_gym'].get('num_gpu_nodes', 0)
    require(gym_nodes == 0, 'Additional Gym GPU reservation needs a separate reviewed contract')
    learner_nodes = config['cluster']['num_nodes'] - rollout
    require(learner_nodes in (2, 4), 'Expected two or four learner nodes')
    if kind == 'smoke':
        require(config['cluster']['num_nodes'] == 3 and rollout == 1, 'Gym smoke must cover two learner nodes')
        require(grpo['max_num_steps'] == 3 and horizon >= 4, 'Smoke must support native 3-to-4 strict restore')
        require(checkpoint['ft_save_period'] == checkpoint['save_period'] == checkpoint['hf_export_period'] == 3,
                'Smoke must publish checkpoint/HF3 before strict resume4')
    else:
        require(grpo['max_num_steps'] == horizon, 'New formal target and scheduler horizon must agree')
        # Parameterized topology arithmetic replaces the former (16,14,512,8)/(10,8,256,8)
        # whitelist (MAINTAINER_GUIDE section 15.3): any N-node run with N-2 rollout nodes
        # is admitted when the batch divides the learner data-parallel size and the
        # checkpoint/HF cadence is a positive schedule that fits inside the horizon.
        require(config['cluster']['num_nodes'] >= 3 and rollout >= 1, 'Formal topology needs >=1 rollout node')
        for name in ('ft_save_period', 'save_period', 'hf_export_period'):
            require(type(checkpoint.get(name)) is int and checkpoint[name] > 0,
                    'Checkpoint cadence must be a positive integer: ' + name)
        require(checkpoint['hf_export_period'] <= horizon, 'HF export cadence must fit inside the horizon')
    megatron = policy['megatron_cfg']
    model_parallel = 1
    for name in ('tensor_model_parallel_size', 'context_parallel_size', 'pipeline_model_parallel_size'):
        size = megatron.get(name, 1)
        require(type(size) is int and size >= 1, 'Invalid learner parallel size: ' + name)
        model_parallel *= size
    learner_gpus = learner_nodes * config['cluster']['gpus_per_node']
    require(learner_gpus % model_parallel == 0, 'Learner parallelism must tile the learner nodes')
    learner_dp = learner_gpus // model_parallel
    require((prompts * k) % learner_dp == 0, 'GBS must divide the learner data-parallel size')
    # Keys are unchanged so frozen GYM_LINEAGE_CONTRACT.json files stay comparable on resume.
    return {'prompts': prompts, 'K': k, 'response_gbs': prompts * k, 'scheduler_horizon': horizon,
            'policy_nodes': config['cluster']['num_nodes'], 'rollout_nodes': rollout, 'learner_nodes': learner_nodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('recipe', type=Path)
    parser.add_argument('kind', choices=('smoke', 'train'))
    parser.add_argument('resume_mode', choices=('new', 'resume'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--runtime-api-acceptance', type=Path, required=True)
    parser.add_argument('--continuation-acceptance', type=Path)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID'), 'CPU allocation required')
    from omegaconf import OmegaConf
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    register_omegaconf_resolvers()
    run = args.run.resolve(strict=True)
    config = OmegaConf.to_container(load_config(str(args.recipe)), resolve=True)
    batch = validate_config(config, args.kind)
    api = json.loads(args.runtime_api_acceptance.read_text())
    require(api['status'] == 'PASS' and api['recipe_sha256'] == sha(args.recipe) and
            api['resolved_recipe_sha256'] == resolved_recipe_digest(config) and
            api['free_api_verified'] is True,
            'Actual runtime/free-API acceptance absent or bound to another recipe')
    shutil.copy2(args.runtime_api_acceptance, run / 'RUNTIME_API_ACCEPTANCE.json')
    images = freeze_images(api, run)
    server_configs = []
    gym_config = config['env']['nemo_gym']
    config_paths = gym_config.get('config_paths', [])
    require(isinstance(config_paths, list) and config_paths, 'Native Gym config_paths required')
    for index, value in enumerate(config_paths):
        source_config = Path(value)
        if not source_config.is_absolute():
            require('..' not in source_config.parts, 'Unsafe relative Gym source config path')
            server_configs.append({'path': value, 'scope': 'pinned_gym_source'})
            continue
        source_config = source_config.resolve(strict=True)
        require(source_config.is_file() and source_config.stat().st_size <= 32 << 20,
                'External Gym config must be a bounded YAML file')
        digest = sha(source_config)
        require(api['gym_config_files_sha256'].get(str(source_config)) == digest,
                'API review did not bind actual external Gym services config')
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', source_config.name), 'Unsafe Gym config filename')
        directory = run / 'gym-configs'; directory.mkdir(exist_ok=True)
        destination = directory / f'{index:03d}-{source_config.name}'
        shutil.copy2(source_config, destination)
        require(sha(destination) == digest, 'Frozen Gym services config changed')
        config_paths[index] = str(destination)
        server_configs.append({'source_path': str(source_config), 'path': str(destination),
                               'scope': 'frozen_external_config', 'sha256': digest})
    write_json(run / 'GYM_SERVER_CONFIGS.json', {'files': server_configs})
    manifest_path = (args.manifest or args.recipe.parent / 'MANIFEST.json').resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    require(manifest['schema'] == 'gym-data-v1', 'Wrong Gym data manifest schema')
    source = resolve_member(manifest_path.parent, manifest['data_path'])
    entry = config['data']['train'][0]
    require(Path(entry['data_path']).resolve(strict=True) == source, 'Recipe differs from manifested data')
    require(sha(source) == manifest['data_sha256'], 'Data SHA mismatch')
    with source.open() as stream:
        rows = sum(1 for line in stream if line.strip())
    require(rows == manifest['rows'] and rows > 0, 'Data row count mismatch')
    require(entry.get('max_samples') in (None, rows), 'Dataset truncation differs from manifested rows')
    epochs = config['grpo']['max_num_epochs']
    needed_updates = 4 if args.kind == 'smoke' else batch['scheduler_horizon']
    require(type(epochs) is int and epochs > 0 and (rows // batch['prompts']) * epochs >= needed_updates,
            'Data/epoch budget cannot supply every planned optimizer update')
    asset_ref = manifest['asset_manifest']
    asset_manifest_path = resolve_member(manifest_path.parent, asset_ref['path'])
    require(sha(asset_manifest_path) == asset_ref['sha256'], 'Asset manifest SHA mismatch')
    shutil.copy2(asset_manifest_path, run / 'GYM_ASSET_MANIFEST.json')
    assets = json.loads(asset_manifest_path.read_text())
    # The data materializer defines these identities; model/service readiness is
    # deliberately a separate native gate, never inferred from materialization.
    entries = assets.get('files', assets.get('assets', []))
    require(isinstance(entries, list), 'Asset manifest must have a files/assets list')
    verified = []
    for asset in entries:
        path = resolve_member(asset_manifest_path.parent, asset['path'])
        require(sha(path) == asset['sha256'], 'Gym data asset SHA mismatch')
        verified.append({'path': str(path), 'sha256': asset['sha256'], 'bytes': path.stat().st_size})
    write_json(run / 'GYM_EXTERNAL_ASSETS.json', {'files': verified, 'note': 'Verified external immutable data assets; rehashed at GPU startup'})
    student = Path(config['policy']['model_name']).resolve(strict=True)
    project = run.parent.parent
    accepted_path = project / 'receipts/v1-input-contract.json'
    accepted = json.loads(accepted_path.read_text())
    require(accepted['status'] in ('PASS', 'PASS_WITH_WARNINGS'), 'Base student input audit not accepted')
    require(student == Path(accepted['student']['path']).resolve(strict=True), 'Student differs from accepted SFTKD399')
    require(sha(student / 'model.safetensors.index.json') == accepted['student']['index_sha256'], 'Student index changed')
    require(Path(config['policy']['tokenizer']['name']).resolve(strict=True) == student, 'Tokenizer source differs from student399')
    require(Path(config['policy']['generation']['model_name']).resolve(strict=True) == student, 'Rollout source differs from student399')
    shutil.copy2(accepted_path, run / 'MODEL_INPUT_AUDIT.json')
    shutil.copy2(source, run / 'data.train.jsonl')
    require(sha(run / 'data.train.jsonl') == manifest['data_sha256'], 'Frozen data copy changed')
    (run / 'data.train.jsonl').chmod(0o444)
    shutil.copy2(manifest_path, run / 'DATA_MIX_MANIFEST.json')
    config['data']['train'][0]['data_path'] = str(run / 'data.train.jsonl')
    config['logger']['log_dir'] = str(run / 'train')
    if 'nemo_gym_log_dir' in config['env']['nemo_gym']:
        config['env']['nemo_gym']['nemo_gym_log_dir'] = str(run / 'nemo_gym')
    if args.kind == 'smoke':
        require(args.resume_mode == 'new', 'Inline smoke owns its strict resume')
        for key, subdir in [('checkpoint_dir', 'checkpoints'), ('hf_export_dir', 'hf'), ('monitor_dir', 'monitor')]:
            config['checkpointing'][key] = str(run / subdir)
    ckpt = Path(config['checkpointing']['checkpoint_dir']).resolve()
    checked_path(str(ckpt)); checked_path(str(run)); checked_path(str(project))
    require(not list(ckpt.glob('tmp_step_*')), 'Incomplete checkpoint publication exists')
    existing = sorted(p.resolve() for p in ckpt.glob('step_*') if p.is_dir())
    identity = config_identity(config)
    lineage_contract = {'schema': 'gym-lineage.v1', 'student': str(student), 'data_sha256': manifest['data_sha256'],
                        'rows': rows, 'research_config_sha256': identity,
                        'gym_server_config_identity': [{'scope': x['scope'], 'sha256': x.get('sha256'),
                            'relative_path': x['path'] if x['scope'] == 'pinned_gym_source' else None}
                            for x in server_configs], **batch}
    if args.resume_mode == 'new':
        require(not existing, 'New lineage must not contain a resumable checkpoint')
        require(not (ckpt / 'ORIGIN_RUN').exists(), 'New lineage already has a submitted origin')
    else:
        require(len(existing) == 1, 'Resume requires exactly one complete checkpoint')
        origin = Path((ckpt / 'ORIGIN_RUN').read_text().strip()).resolve(strict=True)
        original = json.loads((origin / 'GYM_LINEAGE_CONTRACT.json').read_text())
        require(original == lineage_contract, 'Resume changes the original research/data/scheduler contract')
        info = json.loads((existing[0] / 'training_info.json').read_text())
        require(existing[0].name == f"step_{info['current_step']}" and
                info['consumed_samples'] == batch['prompts'] * info['current_step'], 'Resume step/frontier mismatch')
        require(0 < info['current_step'] <= batch['scheduler_horizon'], 'Resume source outside fixed target')
        require(args.continuation_acceptance is not None, 'Resume needs fresh prior-job/cleanup/length acceptance')
        prior = json.loads(args.continuation_acceptance.read_text())
        require(prior['status'] == 'PASS' and Path(prior['checkpoint']).resolve() == existing[0] and
                prior['training_info_sha256'] == sha(existing[0] / 'training_info.json') and
                prior['prior_job_terminal'] is True and prior['node_cleanup_complete'] is True and
                prior['init_cache_removed'] is True and prior['length_review']['status'] == 'PASS' and
                prior['length_review']['through_optimizer_step'] == info['current_step'],
                'Prior allocation/length/checkpoint acceptance incomplete or stale')
        shutil.copy2(args.continuation_acceptance, run / 'CONTINUATION_ACCEPTANCE.json')
        require(re.fullmatch(r'[0-9]+', str(prior['prior_job'])), 'Invalid prior Slurm job identity')
        previous_run = Path(checked_path(prior['prior_run'])).resolve(strict=True)
        require(previous_run.parent == run.parent and
                (previous_run / 'JOB_ID').read_text().strip() == str(prior['prior_job']),
                'Prior run/job ownership mismatch')
        (run / 'GYM_PRIOR_JOB').write_text(str(prior['prior_job']) + '\n')
        (run / 'GYM_PRIOR_RUN').write_text(str(previous_run) + '\n')
        pending = Path(config['checkpointing']['hf_export_dir']) / 'HF_EXPORT_PENDING.json'
        if pending.is_file():
            value = json.loads(pending.read_text())
            require(value == {'step': info['current_step'], 'source_model': config['policy']['model_name'],
                              'reason': 'allocation_deadline', 'recovery_checkpoint': str(existing[0])},
                    'Deferred HF receipt differs from exact current recovery/source')
            require(info['current_step'] % config['checkpointing']['hf_export_period'] == 0,
                    'Deferred HF must be a cadence step')
            shutil.copy2(pending, run / 'PENDING_HF_ACCEPTED.json')
        elif info['current_step'] == batch['scheduler_horizon']:
            raise ValueError('Target already reached with no deferred HF; no continuation needed')
        (run / 'ORIGIN_CONFIG').write_text(str(origin / 'config.yaml') + '\n')
    contract = {'schema': 'gym-run.v1', 'input_mode': 'gym', 'objective': 'rlvr', 'kind': args.kind,
                'resume_mode': args.resume_mode, 'nodes': batch['policy_nodes'], 'checkpoint_dir': str(ckpt),
                'expected_resume_checkpoint': str(existing[0]) if existing else None,
                'student': str(student), 'data_sha256': manifest['data_sha256'], 'data_rows': rows,
                'env_name': 'nemo_gym', 'max_input_seq_length': config['data']['max_input_seq_length'],
                'render_scope': 'initial_request_only; GPU Gym trajectory/token parity remains required',
                'runtime_images': images,
                'smoke_resume_step': 4 if args.kind == 'smoke' else None, **batch}
    (run / 'config.yaml').write_text(OmegaConf.to_yaml(OmegaConf.create(config), resolve=True))
    write_json(run / 'RUN_CONTRACT.json', contract)
    write_json(run / 'GYM_LINEAGE_CONTRACT.json', lineage_contract)
    write_json(run / 'INPUT_MANIFEST.json', {'schema': 'gym-inputs.v1', 'status': 'IDENTITY_PASS_RUNTIME_PENDING',
        'data_sha256': manifest['data_sha256'], 'data_rows': rows, 'manifest_sha256': sha(manifest_path),
        'source_manifest_path': str(manifest_path),
        'asset_manifest_sha256': sha(asset_manifest_path), 'recipe_sha256': sha(args.recipe),
        'student': str(student), 'teacher_enabled': False})
    for name, value in {'INPUT_MODE': 'gym', 'OBJECTIVE': 'rlvr', 'KIND': args.kind,
                        'NODES': str(batch['policy_nodes']), 'GYM_LINEAGE_DIR': str(ckpt),
                        'GYM_PROJECT_DIR': str(project),
                        'GYM_EXPECTED_RESUME': str(existing[0]) if existing else 'NONE'}.items():
        (run / name).write_text(value + '\n')
    print('[GYM_INPUT_IDENTITY_SEALED] ' + json.dumps(contract), flush=True)


if __name__ == '__main__':
    main()
