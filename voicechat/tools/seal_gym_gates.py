#!/usr/bin/env python3
"""Bind successful CPU-only Gym/runtime/scheduler evidence before any sbatch."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_smoke(proof, run):
    require(proof['status'] == 'PASS' and proof['optimizer_step'] == 4 and proof['hf_step'] == 3 and
            proof['hdo_numeric_gates'] == 32 and proof['single_mcore_verified'] is True and
            proof['gym_route_checks_passed'] is True and proof['http_token_parity_passed'] is True and
            proof['resources_released'] is True, 'Incomplete GPU Gym smoke acceptance')
    pins = dict(line.split('=', 1) for line in (run / 'SOURCE_RECEIPT.txt').read_text().splitlines())
    for name in ('nemo_rl_applied', 'bridge', 'megatron_lm', 'gym'):
        require(proof['source_pins'][name] == pins[name], 'GPU smoke source pin changed: ' + name)
    for name in ('gpu_job.sh', 'ray_role.sh', 'runtime_env.sh', 'ray_ports.py',
                 'gym_sandbox_supervisor.py', 'gym_sandbox_pair.py', 'gym_sandbox_probe.py'):
        require(proof['runtime_scripts_sha256'][name] == sha(run / 'scripts' / name),
                'GPU smoke runtime script changed: ' + name)
    contract = json.loads((run / 'RUN_CONTRACT.json').read_text())
    require(proof['runtime_images_sha256'] == {k: v['sha256'] for k, v in
            contract['runtime_images'].items()}, 'GPU smoke runtime images changed')
    require(proof['student'] == contract['student'] and proof['learner_nodes'] == 2,
            'GPU smoke model/learner identity mismatch')
    prior = Path(proof['smoke_run']).resolve(strict=True)
    require(prior.parent == run.parent and (prior / 'JOB_ID').read_text().strip() == str(proof['job_id']),
            'GPU smoke ownership mismatch')
    require(proof['smoke_pass_sha256'] == sha(prior / 'SMOKE_PASS.json'), 'Native smoke completion changed')



REDUCED_RECIPE_SHA = '7a95c5d3fa5635251b1322d4c5be5d2ddf1a1bf2a7b21095a526d9bb754812d9'
BALANCED_DATA_SHA = '1be75f2667215472de314197cbe044946cce0c28ea38294792e93156b8d7568d'
STRUCTURAL_SMOKE_SHA = 'b69785a105473cf145bcb1f921d9d6403da0d333a0ca7ae83654b41f71f2bba7'
BALANCED_ACCEPTANCE_SHA = '649b75c744437461ff8cdb4f6243e862feb391271945f3d2002ab824bca57d93'


def bound_json(item):
    require(isinstance(item, dict) and isinstance(item.get('path'), str) and
            isinstance(item.get('sha256'), str) and len(item['sha256']) == 64,
            'Unbound CPU delta evidence')
    path = Path(item['path']).resolve(strict=True)
    require(path.is_file() and sha(path) == item['sha256'], 'CPU delta evidence changed')
    return json.loads(path.read_text())


CS_FILTER_SCHEMA = 'gym-structural-smoke-cpu-delta.cs-genrm-filter.v1'
CS_PROJECT = os.environ.get('NVC_RL_ROOT', '/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl')  # legacy gates compare against this root
CS_PARENT_READY_SHA = '3509cf0159d135123f69d00a137a56656e4e96d351ddb4f4b95a5819e54cebee'
CS_RL_PIN = 'bad627002690c01b5f5cea16ffe0c08cbbad49ba'
CS_GRPO_SHA = 'a923b8a2a83c5e7616f5c24a08f5582e532347410b0479c95dfbc279636c87ba'
CS_RECIPE = 'config/rlvr-super-light-v3-nogenrm-8r2l-p256.yaml'
CS_SERVICES = 'config/gym-services-light-v3-nogenrm.yaml'
CS_RECIPE_SHA = '44e53ed7398090679e8a0a623c6a399ae698458d3f88631251750a4091f81aa6'
CS_SERVICES_SHA = 'e004ccede45dbae584f3f02e4213e9d8f4d4322f3fbf22acfe445d9de7ad0500'
CS_DATA_SHA = '29437931d23d6396922aedcd861d1432f481e13cfba61a65629efab900e6d1ec'
CS_PATH_MAP_SHA = '9d89af7e1b715df935d6d7dd5f833bdd587b1a7dd297eb551574d818cb862983'
CS_ROUTES = {
 'code_gen_simple_agent':288, 'conversational_tool_use_agent':1280,
 'instruction_following_simple_agent':2816, 'math_with_judge_simple_agent':288,
 'search_pivot_single_step_tool_use_with_argument_comparison_agent':514,
 'single_step_tool_use_with_argument_comparison_agent':2471,
 'structured_outputs_simple_agent':1280,
 'toolcall_schema_single_step_tool_use_with_argument_comparison_agent':343,
}


CS_NOCONV_PARENT_MANIFEST_SHA = '3ddc90991daaa53243808180ad0af6b599e19af2617e2b272f789e46b2f3bf33'
CS_NOCONV_FILTER_SCHEMA = 'gym-structural-smoke-cpu-delta.cs-conversation-filter.v1'
CS_NOCONV_RECIPE_SHA = '5b14071164b26a50e5314bc9de74b263c0269906cc78fc7b2ff6e15739adc01e'
CS_NOCONV_SERVICES_SHA = '276cd12bc3827eb766fb3c9ff844b53479f503a61281fcedfa07ffdbd78b5d74'
CS_NOCONV_DATA_SHA = 'c83a2ebbfc93d72c5ba9757ba385ce00c8de12beff080113d1b7e134225e8b08'
CS_NOCONV_ROUTES = {k:v for k,v in CS_ROUTES.items() if k != 'conversational_tool_use_agent'}


# Separate, exact infrastructure-only admission; historical cleanup proof is unchanged.
CS_REFIT_FILTER_SCHEMA = 'gym-structural-smoke-cpu-delta.cs-refit-pause.v1'
CS_REFIT_INPUT_SCHEMA = 'gym-cs-refit-pause-inputs.v1'
CS_REFIT_RL_PIN = 'e9eb4e9d5ee5f276a7b27a0f771ebdbe0127604b'
CS_REFIT_RECIPE_SHA = '7dcffd8e1d3457aa113e9a0c4a8138b3523e3d5dfbccb4c419db928120c91c79'
CS_REFIT_MANIFEST_SHA = '887cf2797ca8fe4bedc86a256572ae60e8818a1cfc07955d1b1029de024d2ba8'

# One fresh HF399 research arm. None identities are deliberately fail-closed
# until reviewed child bytes are committed and the exact identities are bound.
CS_LR1E6_INPUT_SCHEMA = 'gym-cs-lr1e6-postmask-inputs.v1'
CS_LR1E6_FILTER_SCHEMA = 'gym-structural-smoke-cpu-delta.cs-lr1e6-postmask.v1'
CS_LR1E6_RECIPE = 'config/rlvr-light-v5-lr1e6-8r2l-p256.yaml'
CS_LR1E6_LINEAGE = 'experiments/rlvr-light-v5-lr1e6-s399-p256'
CS_LR1E6_RECIPE_SHA = 'e90c361747c3257a8dab01569aa8ab3ae331194d3e003f9b6c3cef2d95d4cf04'
CS_METRICS_RL_PIN = '9a50d8538ceed83ab6806e361437b8b0a8d62eca'
CS_METRICS_MANIFEST_SHA = '250c9e967be80410cf8685743037f59125d679e32404ac2ebd5aec0e43e2cd92'
CS_METRICS_RUNTIME_ENV_SHA = '742e0050e5040327a5b94bc4a85ba74a010c65022bc2e3233e02a579c6734264'
CS_METRICS_TEST_COUNT = 14
CS_METRICS_FILES = ('nemo_rl/algorithms/grpo.py',
    'nemo_rl/algorithms/postmask_k8_metrics.py',
    'tests/unit/algorithms/test_postmask_k8_metrics.py')
CS_METRICS_TEST = CS_METRICS_FILES[2]
CS_METRICS_PYTHON = '/opt/nemotronvc/venvs/mcore/bin/python'
CS_METRICS_NATIVE_LOSS_RELATIVE = 'nemo_rl/algorithms/loss/loss_functions.py'
CS_METRICS_NATIVE_LOSS_SHA = 'e0687539c27bf6c89c2e5c75219d231874acc9e6bf63d69d8f9ecd099a452b74'


def cs_is_refit(schema):
    return schema in (CS_REFIT_INPUT_SCHEMA, CS_LR1E6_INPUT_SCHEMA)


def cs_validate_lr1e6_recipe(parent_cfg, candidate_cfg):
    """Require the entire admitted refit recipe plus exactly five fixed leaves."""
    import copy
    expected=copy.deepcopy(parent_cfg)
    for key in ('lr','min_lr'):
        require(expected['policy']['megatron_cfg']['optimizer'][key]==2e-6,
                'LR arm parent optimizer differs')
        expected['policy']['megatron_cfg']['optimizer'][key]=1e-6
    for key,leaf in [('checkpoint_dir','checkpoints'),('hf_export_dir','hf'),('monitor_dir','monitor')]:
        require(expected['checkpointing'][key]==CS_PROJECT+'/'+
                'experiments/rlvr-light-v4-nogenrm-noconv-refitpause-s399-p256/'+leaf,
                'LR arm parent lineage differs')
        expected['checkpointing'][key]=CS_PROJECT+'/'+CS_LR1E6_LINEAGE+'/'+leaf
    require(json.dumps(expected,sort_keys=True)==json.dumps(candidate_cfg,sort_keys=True),
            'LR arm requires exactly two LR leaves and three fresh lineage paths')


def cs_validate_metrics_source(x):
    """Bind one reviewed observational child without relabeling refit history."""
    require(x['schema']==CS_LR1E6_INPUT_SCHEMA,'Metrics source on unrelated profile')
    require(isinstance(CS_METRICS_RL_PIN,str) and re.fullmatch(r'[0-9a-f]{40}',CS_METRICS_RL_PIN),
            'Metrics child commit identity unresolved')
    require(isinstance(CS_METRICS_MANIFEST_SHA,str) and re.fullmatch(r'[0-9a-f]{64}',CS_METRICS_MANIFEST_SHA),
            'Metrics manifest identity unresolved')
    require(isinstance(CS_METRICS_RUNTIME_ENV_SHA,str) and re.fullmatch(r'[0-9a-f]{64}',CS_METRICS_RUNTIME_ENV_SHA),
            'Metrics runtime identity unresolved')
    proof=x['metrics_source']
    require(set(proof)=={'schema','base_rl_pin','rl_pin','source_manifest','runtime_env_sha256'} and
            proof['schema']=='cs-postmask-k8-source.v1' and proof['base_rl_pin']==CS_REFIT_RL_PIN and
            proof['rl_pin']==CS_METRICS_RL_PIN and proof['runtime_env_sha256']==CS_METRICS_RUNTIME_ENV_SHA,
            'Unreviewed metrics child/runtime identity')
    require(proof['source_manifest']['sha256']==CS_METRICS_MANIFEST_SHA,'Unreviewed metrics manifest')
    manifest=bound_json(proof['source_manifest'])
    require(manifest['base_rl_pin']==CS_REFIT_RL_PIN and manifest['default_enabled'] is False and
            manifest['enable_env']=='NVC_POSTMASK_K8_METRICS=1' and
            set(manifest['source_files'])==set(CS_METRICS_FILES),'Metrics overlay file/enablement set differs')
    for relative,row in manifest['source_files'].items():
        require(set(row)=={'original_sha256','candidate_sha256'} and
                isinstance(row['candidate_sha256'],str) and re.fullmatch(r'[0-9a-f]{64}',row['candidate_sha256']),
                'Metrics file hash binding malformed')
        require(row['original_sha256']==('0ac7a8aeda0a0c215bb7e060646cf047cffb470fa34bdb6f65e0614db18fb5ab'
                if relative=='nemo_rl/algorithms/grpo.py' else None),'Metrics base file differs')
    return manifest


def cs_validate_metrics_cpu(item,x,job):
    manifest=cs_validate_metrics_source(x); proof=bound_json(item)
    require(type(CS_METRICS_TEST_COUNT) is int and CS_METRICS_TEST_COUNT>0,
            'Metrics test count unresolved')
    require(proof['schema']=='cs-postmask-k8-cpu-acceptance.v1' and proof['status']=='PASS' and
            proof['job_id']==job and proof['rl_pin']==CS_METRICS_RL_PIN and
            proof['source_manifest']==x['metrics_source']['source_manifest'] and
            proof['interpreter']==CS_METRICS_PYTHON and proof['test_file']==CS_METRICS_TEST and
            proof['test_file_sha256']==manifest['source_files'][CS_METRICS_TEST]['candidate_sha256'] and
            proof['exit_code']==0 and proof['tests_run']==CS_METRICS_TEST_COUNT and
            proof['failures']==0 and proof['errors']==0 and proof['skipped']==0,
            'Actual metrics CPU tests missing or mismatched')
    require(proof['native_loss_sha256']==CS_METRICS_NATIVE_LOSS_SHA and
            Path(proof['native_loss_path']).resolve()==Path(CS_PROJECT+'/src/nemo-rl/'+CS_METRICS_NATIVE_LOSS_RELATIVE).resolve(),
            'Metrics parity did not import the pinned native loss')
    measured=bound_json(proof['raw_result'])
    require(set(measured)=={'tests_run','failures','errors','skipped','native_loss_path','native_loss_sha256'} and
            all(measured[key]==proof[key] for key in measured),'Metrics raw test result differs')
    log=proof['test_log']; require(sha(Path(log['path']))==log['sha256'],'Metrics CPU log changed')
    return proof



def cs_validate_metrics_archive(archive, expected):
    """Verify every current overlay and unchanged historical refit member."""
    names=archive.getnames()
    for relative,row in expected.items():
        name='./'+relative
        require(names.count(name)==1,'Missing/duplicate frozen metrics/refit member: '+relative)
        member=archive.getmember(name)
        require(member.isfile() and member.size<=32<<20,'Unsafe frozen metrics/refit member: '+relative)
        require(hashlib.sha256(archive.extractfile(member).read()).hexdigest()==row['candidate_sha256'],
                'Frozen archive accepted source changed: '+relative)


# Exact fresh v6 control-only admission. Pending constants deliberately fail closed.
CS_CODE_MATH_INPUT_SCHEMA = 'gym-cs-code8-math16-inputs.v1'
CS_CODE_MATH_DELTA_SCHEMA = 'gym-structural-smoke-cpu-delta.cs-code8-math16.v1'
CS_CODE_MATH_RECIPE = 'config/rlvr-light-v6-code8-math16-8r2l-p256.yaml'
CS_CODE_MATH_RECIPE_SHA = '527d3fb6a666a1c0c244217bca0970912933b27ad6ffde75c0fa9651aaf4a261'
CS_CODE_MATH_LINEAGE = 'experiments/rlvr-light-v6-code8-math16-s399-p256'
CS_CODE_MATH_DATA_DIR = 'data/gym-light-v6-code8-math16/materialized-v1'
CS_CODE_MATH_PARENT = 'runs/train-v1-20260926T070100Z'
CS_CODE_MATH_PARENT_CPU = '35696789'
CS_CODE_MATH_PARENT_READY_SHA = '1f9a4e59a3e3333ed1039fb85e1500c6395cec3ae6f3cc3f62a5952d99ebede2'
CS_CODE_MATH_PARENT_ROOT_SHA = '2b8b106b4bac6f5a117c46651dc3a8fa739e258a11f547b519d59b2661e75a4c'
CS_CODE_MATH_PARENT_INDEPENDENT_SHA = 'e2d1e8a578fb60ef5e9d8612990eb89d13e968a0fcc69a98756258111962db77'
CS_CODE_MATH_DATA_SHA = 'b47a3d04b69e81a2c04dfdd3ea17cf966317081e61b00fa38299bb60adda3829'
CS_CODE_MATH_MANIFEST_SHA = '46e5e3716284a3e72a24aef23e4d2ab78eddacacef90edd7871c4631db936831'
CS_CODE_MATH_DATA_ACCEPTANCE_SHA = '2c0765b37c229377bd26f69d5fa8cb3b786ff3572a6f85cd66b39dd6f1188265'
CS_CODE_MATH_MATERIALIZATION_ROOT_SHA = '344f9b9821ca6895187683a0cece0b2bcf04a483c0f0544e5263bbfa189fc791'
CS_CODE_MATH_MATERIALIZATION_INDEPENDENT_SHA = '933d2311772ce5284106d7efe8717274bd03c5dbc4707e49839d433a0658f79e'
CS_CODE_MATH_ROUTES = {
    'code_gen_simple_agent':640,'math_with_judge_simple_agent':1280,
    'instruction_following_simple_agent':1892,'structured_outputs_simple_agent':860,
    'single_step_tool_use_with_argument_comparison_agent':2471,
    'search_pivot_single_step_tool_use_with_argument_comparison_agent':514,
    'toolcall_schema_single_step_tool_use_with_argument_comparison_agent':343,
}


def cs_code_math_resolved():
    for name in ('DATA_SHA','MANIFEST_SHA','DATA_ACCEPTANCE_SHA','MATERIALIZATION_ROOT_SHA',
                 'MATERIALIZATION_INDEPENDENT_SHA'):
        require(isinstance(globals()['CS_CODE_MATH_'+name],str) and
                re.fullmatch(r'[0-9a-f]{64}',globals()['CS_CODE_MATH_'+name]),
                'V6 reviewed immutable identity unresolved: '+name)


def cs_code_math_recipe(parent, candidate):
    import copy
    expected=copy.deepcopy(parent)
    require(expected['policy']['megatron_cfg']['optimizer']['lr']==1e-6 and
            expected['policy']['megatron_cfg']['optimizer']['min_lr']==1e-6,
            'V6 parent is not accepted LR1e-6')
    for key,leaf in [('checkpoint_dir','checkpoints'),('hf_export_dir','hf'),('monitor_dir','monitor')]:
        require(expected['checkpointing'][key]==CS_PROJECT+'/'+CS_LR1E6_LINEAGE+'/'+leaf,
                'V6 parent lineage differs')
        expected['checkpointing'][key]=CS_PROJECT+'/'+CS_CODE_MATH_LINEAGE+'/'+leaf
    expected['data']['train'][0]['data_path']=CS_PROJECT+'/'+CS_CODE_MATH_DATA_DIR+'/train.jsonl'
    expected['env']['nemo_gym']['gym_data_asset_dir']=CS_PROJECT+'/'+CS_CODE_MATH_DATA_DIR+'/assets'
    require(candidate==expected,'V6 admits only data+asset root and three output path changes')


def cs_code_math_contract(contract):
    cs_code_math_resolved()
    expected={'kind':'train','resume_mode':'new','nodes':10,'rollout_nodes':8,'learner_nodes':2,
        'prompts':256,'K':8,'response_gbs':2048,'scheduler_horizon':20,'data_rows':8000,
        'data_sha256':CS_CODE_MATH_DATA_SHA,'expected_resume_checkpoint':None,'smoke_resume_step':None}
    for key,value in expected.items():
        require(type(contract.get(key)) is type(value) and contract[key]==value,'Wrong v6 fresh contract: '+key)
    require(Path(contract['checkpoint_dir']).resolve()==Path(CS_PROJECT+'/'+CS_CODE_MATH_LINEAGE+'/checkpoints').resolve(),
            'V6 checkpoint lineage differs')
    require(Path(contract['student']).resolve()==Path(CS_PROJECT+'/models/sftkd-s3-c8-lr1e-5-s399/huggingface').resolve(),
            'V6 is not fresh HF399')


def cs_code_math_inputs(item):
    cs_code_math_resolved();x=bound_json(item)
    require(x['schema']==CS_CODE_MATH_INPUT_SCHEMA and x['project']==CS_PROJECT,'Wrong v6 profile')
    p=Path(CS_PROJECT).resolve(strict=True);parent=p/CS_CODE_MATH_PARENT
    # These refer to accepted literal v5 receipts; historical jobs are never relabelled.
    require(x['parent_ready']['sha256']==CS_CODE_MATH_PARENT_READY_SHA and
            x['parent_root']['sha256']==CS_CODE_MATH_PARENT_ROOT_SHA and
            x['parent_independent']['sha256']==CS_CODE_MATH_PARENT_INDEPENDENT_SHA,'V6 parent proof hashes differ')
    parent_inputs=bound_json(x['parent_admission_inputs'])
    require(Path(x['parent_admission_inputs']['path']).resolve()==parent/'GYM_CS_ADMISSION_INPUTS.json',
            'V6 parent input path differs')
    require(parent_inputs['schema']==CS_LR1E6_INPUT_SCHEMA,'V6 parent is not accepted metrics profile')
    parent_ready=bound_json(x['parent_ready'])
    require(Path(x['parent_ready']['path']).resolve()==parent/'GYM_FROZEN_READY.json' and
            parent_ready['status']=='PASS' and parent_ready['cpu_job']==CS_CODE_MATH_PARENT_CPU,'V6 parent freeze differs')
    for name,digest in parent_ready['files'].items():require(sha(parent/name)==digest,'V6 inherited parent proof changed: '+name)
    require(x['parent_admission_inputs']['sha256']==parent_ready['files']['GYM_CS_ADMISSION_INPUTS.json'],'V6 parent inputs hash differs')
    root=bound_json(x['parent_root']);independent=bound_json(x['parent_independent'])
    require(root['status']=='PASS' and root['cpu_job_id']==CS_CODE_MATH_PARENT_CPU and
            Path(root['run']).resolve()==parent and independent['status']=='PASS','V6 parent not independently accepted')
    parent_recipe=cs_load_configuration(x['parent_recipe']['path']);recipe=cs_load_configuration(x['recipe']['path'])
    require(x['parent_recipe']['sha256']==CS_LR1E6_RECIPE_SHA and sha(Path(x['parent_recipe']['path']))==CS_LR1E6_RECIPE_SHA,
            'V6 parent recipe bytes differ')
    require(Path(x['recipe']['path']).resolve()==p/CS_CODE_MATH_RECIPE and
            sha(Path(x['recipe']['path']))==x['recipe']['sha256']==CS_CODE_MATH_RECIPE_SHA,'V6 recipe bytes/path differ')
    cs_code_math_recipe(parent_recipe,recipe)
    require(x['services']==parent_inputs['services'] and
            sha(Path(x['services']['path']))==CS_NOCONV_SERVICES_SHA,'V6 services changed')
    require(cs_closure(cs_load_configuration(x['services']['path']),dict(routes=CS_CODE_MATH_ROUTES,alias_count=16,delta_schema=CS_NOCONV_FILTER_SCHEMA))==
            bound_json(parent_inputs['service_closure'])['retained_aliases'],'V6 service closure differs')
    for key,digest in [('manifest',CS_CODE_MATH_MANIFEST_SHA),('data_acceptance',CS_CODE_MATH_DATA_ACCEPTANCE_SHA),
            ('materialization_root',CS_CODE_MATH_MATERIALIZATION_ROOT_SHA),
            ('materialization_independent',CS_CODE_MATH_MATERIALIZATION_INDEPENDENT_SHA)]:
        require(x[key]['sha256']==digest,'V6 unreviewed release reference: '+key);bound_json(x[key])
    manifest=bound_json(x['manifest']);data=bound_json(x['data_acceptance'])
    require(Path(x['manifest']['path']).resolve()==p/CS_CODE_MATH_DATA_DIR/'MANIFEST.json' and
            manifest['schema']=='gym-data-v1' and manifest['rows']==8000 and
            manifest['data_sha256']==CS_CODE_MATH_DATA_SHA and manifest['route_counts']==CS_CODE_MATH_ROUTES and
            manifest['source_provenance_schema']=='explicit-two-parent-subsets-v1' and
            set(manifest['parent_sources'])=={'pool','current'},
            'V6 actual materialized data differs')
    require(data['status']=='PASS' and data['data_sha256']==CS_CODE_MATH_DATA_SHA,'V6 data clearance missing')
    # Exact finalized root/independent bytes, not a guessed materializer schema,
    # are pinned above. Root must review actual row provenance/family isolation.
    source=bound_json(x['source_release'])
    require(source['status']=='PASS' and source['numerical_source_unchanged'] is True and
            source['parent_run']==str(parent) and source['rl_pin']==CS_METRICS_RL_PIN and
            source['changed_runtime_files']==['scripts/seal_gym_gates.py'],
            'V6 source release is not the exact data/control-only delta')
    require(source['original_gate_sha256']==sha(parent/'scripts/seal_gym_gates.py') and
            source['runtime_env_sha256']==CS_METRICS_RUNTIME_ENV_SHA and
            x['runtime_env_sha256']==CS_METRICS_RUNTIME_ENV_SHA,'V6 original observer/runtime differs')
    return x,parent,parent_inputs,parent_ready,source


def validate_cs_code_math_delta(delta_ref,smoke_ref,run):
    delta=bound_json(delta_ref)
    require(delta['schema']==CS_CODE_MATH_DELTA_SCHEMA and delta['status']=='PASS' and
            delta['exact_gpu_match'] is False and delta['admission_mode']=='structural_gpu_plus_verified_cpu_delta',
            'V6 delta is not explicit CPU qualification')
    x,parent,parent_inputs,parent_ready,source=cs_code_math_inputs(delta['inputs'])
    require(smoke_ref==parent_inputs['structural_smoke'],'V6 historical structural smoke changed')
    current=bound_json(delta['source']);request=bound_json(delta['request']);judge=bound_json(delta['remaining_judge'])
    job=os.environ['SLURM_JOB_ID'];contract=json.loads((run/'RUN_CONTRACT.json').read_text());cs_code_math_contract(contract)
    require(current['job_id']==job and current['status']=='PASS' and current['inputs_sha256']==delta['inputs']['sha256'] and
            current['request_sha256']==delta['request']['sha256'] and current['source_release']==x['source_release'] and
            current['numerical_source_unchanged'] is True,'V6 current source proof missing')
    pins=dict(line.split('=',1) for line in (run/'SOURCE_RECEIPT.txt').read_text().splitlines())
    inherited=dict(line.split('=',1) for line in (parent/'SOURCE_RECEIPT.txt').read_text().splitlines())
    require(current['source_pins']==pins and all(pins[k]==inherited[k] for k in inherited if k!='project_wrapper'),
            'V6 numerical repository pins changed')
    require(pins['project_wrapper']==source['wrapper_pin'] and pins['nemo_rl_applied']==CS_METRICS_RL_PIN,'V6 reviewed wrapper/RL differs')
    require(sha(run/'source.tar.gz')==source['source_archive_sha256']==(parent/'source.sha256').read_text().split()[0],
            'V6 numerical source archive differs from accepted v5')
    require(sha(run/'scripts/seal_gym_gates.py')==source['candidate_gate_sha256'],'V6 frozen gate differs')
    archive_reuse=json.loads((run/'GYM_SOURCE_ARCHIVE_REUSE.json').read_text())
    require(archive_reuse['status']=='PASS' and archive_reuse['job_id']==job and
            archive_reuse['run']==str(run.resolve()) and archive_reuse['parent_run']==str(parent) and
            archive_reuse['source_archive_sha256']==source['source_archive_sha256'] and
            archive_reuse['source_release']==x['source_release'] and archive_reuse['set_equality'] is True and
            archive_reuse['source_unmodified'] is True and archive_reuse['archive_same_inode'] is True and
            archive_reuse['helper_sha256']==source['private_producer_files']['source_archive_reuse.py'] and
            archive_reuse['private_freeze_driver_sha256']==source['private_producer_files']['freeze_private.sh'],
            'V6 current full source-tree/archive reuse proof missing')
    require(current['runtime_scripts_sha256']=={n:sha(run/n) for n in current['runtime_scripts_sha256']},'V6 runtime byte map differs')
    for path in (parent/'scripts').iterdir():
        if path.is_file() and path.name!='seal_gym_gates.py':
            require(sha(run/'scripts'/path.name)==sha(path),'V6 unreviewed launcher/runtime change: '+path.name)
    require({k:v['sha256'] for k,v in contract['runtime_images'].items()}==
            {k:v['sha256'] for k,v in json.loads((parent/'RUN_CONTRACT.json').read_text())['runtime_images'].items()},
            'V6 runtime images differ')
    require(request['schema']=='gym-cs-code8-math16-request.v1' and request['data_sha256']==CS_CODE_MATH_DATA_SHA and
            request['recipe_sha256']==CS_CODE_MATH_RECIPE_SHA and request['services_sha256']==CS_NOCONV_SERVICES_SHA and
            request['inputs']==delta['inputs'],'V6 current request differs')
    require(judge['status']=='PASS' and judge['job_id']==job and judge['services_sha256']==CS_NOCONV_SERVICES_SHA and
            judge['actual_api_calls_in_verification_job']==0 and judge['inherited_from']==
            {'path':str(parent/'GYM_CPU_DELTA_JUDGE.json'),'sha256':parent_ready['files']['GYM_CPU_DELTA_JUDGE.json']},
            'V6 unchanged judge protocol evidence missing')
    origin=json.loads((run/'INPUT_MANIFEST.json').read_text());require(origin['recipe_sha256']==CS_CODE_MATH_RECIPE_SHA,'V6 origin recipe differs')
    role=json.loads((run/'ROLE_RUNTIME_PREFLIGHT.json').read_text())
    require(role['slurm_job_id']==job and role.get('initial_render_evidence_mode','executed')=='executed' and
            not (run/'GYM_RENDER_REUSE.json').exists(),'V6 new dataset needs full current render')
    require(json.loads((run/'GYM_NATIVE_RENDER_PREFLIGHT.json').read_text())['rows']==8000 and
            json.loads((run/'GYM_RUNTIME_PROBE.json').read_text())['converted_rows']==8000,'V6 full render missing')
    metrics_source=json.loads((parent/'GYM_CPU_DELTA_SOURCE.json').read_text())
    metrics=cs_validate_metrics_cpu(metrics_source['metrics_cpu'],parent_inputs,CS_CODE_MATH_PARENT_CPU)
    require(current['inherited_metrics_cpu']==metrics_source['metrics_cpu'],'V6 inherited native observer proof differs')
    copies={'GYM_SMOKE_CPU_DELTA.json':delta_ref,'GYM_CPU_DELTA_REQUEST.json':delta['request'],
        'GYM_CPU_DELTA_SOURCE.json':delta['source'],'GYM_CPU_DELTA_JUDGE.json':delta['remaining_judge'],
        'GYM_CS_ADMISSION_INPUTS.json':delta['inputs'],'GYM_CODE_MATH_SOURCE_RELEASE.json':x['source_release'],
        'GYM_CODE_MATH_MATERIALIZATION_ROOT.json':x['materialization_root'],
        'GYM_CODE_MATH_MATERIALIZATION_INDEPENDENT.json':x['materialization_independent'],
        'GYM_CODE_MATH_PARENT_READY.json':x['parent_ready'],'GYM_CODE_MATH_PARENT_ROOT.json':x['parent_root'],
        'GYM_CODE_MATH_PARENT_INDEPENDENT.json':x['parent_independent'],
        'GYM_CODE_MATH_PARENT_INPUTS.json':x['parent_admission_inputs'],'GYM_CODE_MATH_PARENT_RECIPE.json':x['parent_recipe']}
    for name in ('GYM_HISTORICAL_SMOKE_TERMINAL.json','GYM_POSTMASK_SOURCE_MANIFEST.json','GYM_POSTMASK_METRICS_CPU.json',
                 'GYM_POSTMASK_METRICS_TESTS.log','GYM_POSTMASK_METRICS_TEST_RESULT.json'):
        copies[name]={'path':str(parent/name),'sha256':parent_ready['files'][name]}
    return copies


def cs_code_math_metrics_runtime_enabled(run,x):
    cs_code_math_resolved();require(x['schema']==CS_CODE_MATH_INPUT_SCHEMA,'Wrong v6 observer profile')
    contract=json.loads((run/'RUN_CONTRACT.json').read_text());cs_code_math_contract(contract)
    source=json.loads((run/'GYM_CPU_DELTA_SOURCE.json').read_text())
    release=json.loads((run/'GYM_CODE_MATH_SOURCE_RELEASE.json').read_text())
    pins=dict(line.split('=',1) for line in (run/'SOURCE_RECEIPT.txt').read_text().splitlines())
    require(source['status']=='PASS' and source['numerical_source_unchanged'] is True and
            release['status']=='PASS' and release['numerical_source_unchanged'] is True and
            pins['nemo_rl_applied']==CS_METRICS_RL_PIN and pins['project_wrapper']==release['wrapper_pin'],
            'V6 observer source identity differs')
    require(sha(run/'GYM_CODE_MATH_SOURCE_RELEASE.json')==x['source_release']['sha256'] and
            sha(run/'scripts/seal_gym_gates.py')==release['candidate_gate_sha256'] and
            sha(run/'scripts/runtime_env.sh')==CS_METRICS_RUNTIME_ENV_SHA,'V6 observer source differs')
    parent_inputs=json.loads((run/'GYM_CODE_MATH_PARENT_INPUTS.json').read_text())
    require(sha(run/'GYM_CODE_MATH_PARENT_INPUTS.json')==x['parent_admission_inputs']['sha256'],
            'V6 inherited observer inputs differ')
    proof=cs_validate_metrics_cpu(source['inherited_metrics_cpu'],parent_inputs,CS_CODE_MATH_PARENT_CPU)
    require(sha(run/'GYM_POSTMASK_METRICS_CPU.json')==source['inherited_metrics_cpu']['sha256'] and
            sha(run/'GYM_POSTMASK_METRICS_TESTS.log')==proof['test_log']['sha256'] and
            sha(run/'GYM_POSTMASK_METRICS_TEST_RESULT.json')==proof['raw_result']['sha256'],
            'V6 inherited native observer proof differs')
    require(sha(run/'GYM_POSTMASK_SOURCE_MANIFEST.json')==CS_METRICS_MANIFEST_SHA and
            json.loads((run/'INPUT_MANIFEST.json').read_text())['recipe_sha256']==CS_CODE_MATH_RECIPE_SHA,
            'V6 observer manifest/recipe differs')
    return True


def cs_metrics_runtime_enabled(run):
    """Enable only after this exact fresh arm has frozen source/test admission."""
    run=Path(run)
    path=run/'GYM_CS_ADMISSION_INPUTS.json'
    if not path.is_file():return False
    x=json.loads(path.read_text())
    if x['schema']==CS_CODE_MATH_INPUT_SCHEMA:return cs_code_math_metrics_runtime_enabled(run,x)
    if x['schema']!=CS_LR1E6_INPUT_SCHEMA:return False
    manifest=cs_validate_metrics_source(x)
    require(sha(run/'GYM_POSTMASK_SOURCE_MANIFEST.json')==CS_METRICS_MANIFEST_SHA and
            sha(run/'scripts/runtime_env.sh')==CS_METRICS_RUNTIME_ENV_SHA,'Frozen metrics runtime/source differs')
    source=json.loads((run/'GYM_CPU_DELTA_SOURCE.json').read_text())
    require(source.get('metrics_only_rl_diff_verified') is True and
            source.get('refit_boundary_only_rl_diff_verified') is False and
            source.get('cleanup_only_rl_diff_verified') is False and
            source.get('metrics_source')==x['metrics_source'],'Metrics runtime source scope differs')
    proof=cs_validate_metrics_cpu(source['metrics_cpu'],x,source['job_id'])
    require(sha(run/'GYM_POSTMASK_METRICS_CPU.json')==source['metrics_cpu']['sha256'] and
            sha(run/'GYM_POSTMASK_METRICS_TESTS.log')==proof['test_log']['sha256'] and
            sha(run/'GYM_POSTMASK_METRICS_TEST_RESULT.json')==proof['raw_result']['sha256'],
            'Frozen metrics CPU proof differs')
    contract=json.loads((run/'RUN_CONTRACT.json').read_text())
    origin=json.loads((run/'INPUT_MANIFEST.json').read_text())
    require(contract['kind']=='train' and contract['resume_mode']=='new' and
            Path(contract['checkpoint_dir']).resolve()==Path(CS_PROJECT+'/'+CS_LR1E6_LINEAGE+'/checkpoints').resolve() and
            origin['recipe_sha256']==CS_LR1E6_RECIPE_SHA,'Metrics runtime not the admitted fresh arm')
    pins=dict(line.split('=',1) for line in (run/'SOURCE_RECEIPT.txt').read_text().splitlines())
    require(pins['nemo_rl_applied']==CS_METRICS_RL_PIN,'Metrics runtime child differs')
    return True

def cs_filter_profile(schema):
    """Explicit reviewed contracts; callers cannot provide arbitrary deltas."""
    if schema == CS_LR1E6_INPUT_SCHEMA:
        profile=cs_filter_profile(CS_REFIT_INPUT_SCHEMA)
        profile.update(input_schema=schema,delta_schema=CS_LR1E6_FILTER_SCHEMA,
            recipe=CS_LR1E6_RECIPE,recipe_sha=CS_LR1E6_RECIPE_SHA,lineage=CS_LR1E6_LINEAGE)
        return profile
    if schema == CS_REFIT_INPUT_SCHEMA:
        profile=cs_filter_profile('gym-cs-conversation-filter-inputs.v1')
        profile.update(input_schema=schema,delta_schema=CS_REFIT_FILTER_SCHEMA,
            recipe='config/rlvr-light-v4-nogenrm-noconv-refitpause-8r2l-p256.yaml',
            recipe_sha=CS_REFIT_RECIPE_SHA,
            lineage='experiments/rlvr-light-v4-nogenrm-noconv-refitpause-s399-p256')
        return profile
    if schema == 'gym-cs-genrm-filter-inputs.v1':
        return dict(input_schema=schema, delta_schema=CS_FILTER_SCHEMA,
            recipe=CS_RECIPE, recipe_sha=CS_RECIPE_SHA, services=CS_SERVICES,
            services_sha=CS_SERVICES_SHA, data_sha=CS_DATA_SHA, rows=9280,
            lineage='experiments/rlvr-super-light-v3-nogenrm-s399-p256',
            data_dir='data/gym-light-v3-nogenrm/materialized-v1',
            routes=CS_ROUTES, alias_count=19, parent_recipe_sha=REDUCED_RECIPE_SHA,
            parent_services_sha='5b3dba9c09dc1e379736eea81cf75c0f70465827006536c92d9b6ac069010a1b')
    require(schema == 'gym-cs-conversation-filter-inputs.v1', 'Unknown CS filter profile')
    return dict(input_schema=schema, delta_schema=CS_NOCONV_FILTER_SCHEMA,
        recipe='config/rlvr-light-v4-nogenrm-noconv-8r2l-p256.yaml',
        recipe_sha=CS_NOCONV_RECIPE_SHA,
        services='config/gym-services-light-v4-nogenrm-noconv.yaml',
        services_sha=CS_NOCONV_SERVICES_SHA, data_sha=CS_NOCONV_DATA_SHA, rows=8000,
        lineage='experiments/rlvr-light-v4-nogenrm-noconv-s399-p256',
        data_dir='data/gym-light-v4-nogenrm-noconv/materialized-v1',
        routes=CS_NOCONV_ROUTES, alias_count=16,
        parent_recipe_sha=CS_RECIPE_SHA,parent_services_sha=CS_SERVICES_SHA)


def cs_validate_filter_data(data, profile, parent_ref=None):
    """Bind the exact ordered subset and, for v4, the accepted v3 parent chain."""
    noconv = profile['delta_schema'] in (CS_NOCONV_FILTER_SCHEMA,CS_REFIT_FILTER_SCHEMA,CS_LR1E6_FILTER_SCHEMA)
    require(data['status']=='PASS' and data['rows']==profile['rows'] and
            data['data_sha256']==profile['data_sha'] and
            data['route_counts']==data['kept_route_counts']==profile['routes'] and
            data['retained_bytes_identical'] is True and data['retained_order_unchanged'] is True,
            'Exact filtered row/byte/order proof missing')
    if noconv:
        require(parent_ref is not None, 'Conversation filter requires accepted v3 parent')
        parent=bound_json(parent_ref)
        cs_validate_filter_data(parent,cs_filter_profile('gym-cs-genrm-filter-inputs.v1'))
        require(data['schema']=='gym-conversation-filter.v1' and
                data['parent_data_sha256']==CS_DATA_SHA and data['parent_rows']==9280 and
                data['removed_routes']=={'conversational_tool_use_agent':1280} and data['removed_rows']==1280,
                'Unreviewed conversation filter')
    else:
        require(data['schema']=='gym-genrm-filter.v1' and data['parent_data_sha256']==BALANCED_DATA_SHA and
                data['parent_rows']==12288 and data['removed_routes']=={'genrm_simple_agent':3008} and
                data['removed_rows']==3008,'Exact 9280-row filter proof missing')
    proofs={key:bound_json(data[key]) for key in ('manifest','selection','data_acceptance')}
    if noconv:
        manifest=proofs['manifest'];selection=proofs['selection'];clearance=proofs['data_acceptance']
        require(data['parent_data_filter']==parent_ref and data['independent_second_pass_rows']==8000,
                'Filter parent identity or second-pass verification missing')
        require(parent['manifest']['sha256']==CS_NOCONV_PARENT_MANIFEST_SHA and
                manifest['parent_manifest']==parent['manifest'] and selection['parent']==parent['manifest'],
                'Parent manifest chain differs')
        require(manifest['schema']=='gym-data-v1' and manifest['rows']==manifest['output_rows']==8000 and
                manifest['data_sha256']==profile['data_sha'] and manifest['source_rows']==9280 and
                manifest['source_sha256']==CS_DATA_SHA and manifest['selection_sha256']==data['selection']['sha256'] and
                manifest['route_counts']==profile['routes'], 'Filtered manifest identity differs')
        require(clearance['status']=='PASS' and clearance['data_sha256']==profile['data_sha'] and
                clearance['rows']==clearance['selected_raw_rows_verified']==8000 and
                clearance['manifest_sha256']==data['manifest']['sha256'] and
                clearance['parent_acceptance']==parent['data_acceptance'] and
                clearance['matched_training_rows']==clearance['substantive_matched_training_rows']==0,
                'Subset data acceptance identity differs')


def cs_relocate(value, mapping):
    """Literal longest-prefix-first path replacement; no regex or schema changes."""
    if isinstance(value,str):
        for before,after in sorted(mapping.items(),key=lambda pair:len(pair[0]),reverse=True):
            value=value.replace(before,after)
        return value
    if isinstance(value,list): return [cs_relocate(x,mapping) for x in value]
    if isinstance(value,dict): return {k:cs_relocate(v,mapping) for k,v in value.items()}
    return value


def cs_closure(services, profile=None):
    profile=profile or cs_filter_profile('gym-cs-genrm-filter-inputs.v1')
    todo=list(profile['routes']); seen=set()
    def visit(value):
        if isinstance(value,dict):
            if value.get('type') in ('responses_api_models','responses_api_agents','resources_servers') and 'name' in value:
                todo.append(value['name'])
            for child in value.values(): visit(child)
        elif isinstance(value,list):
            for child in value: visit(child)
    while todo:
        name=todo.pop()
        require(isinstance(name,str) and name in services,'Missing retained service dependency')
        if name not in seen: seen.add(name); visit(services[name])
    require(len(seen)==profile['alias_count'] and not any('genrm' in x for x in seen),'Unexpected retained service closure')
    if profile['delta_schema'] in (CS_NOCONV_FILTER_SCHEMA,CS_REFIT_FILTER_SCHEMA,CS_LR1E6_FILTER_SCHEMA):
        require(not any('conversational' in x or x=='simulator_model' for x in seen),'Conversation dependency still reachable')
    return sorted(seen)


def cs_validate_configuration(old_cfg,new_cfg,old_services,new_services,mapping,profile=None):
    profile=profile or cs_filter_profile('gym-cs-genrm-filter-inputs.v1')
    expected=cs_relocate(old_cfg,mapping)
    lineage=CS_PROJECT+'/'+profile['lineage']
    for key,leaf in [('checkpoint_dir','checkpoints'),('hf_export_dir','hf'),('monitor_dir','monitor')]:
        expected['checkpointing'][key]=lineage+'/'+leaf
    expected['data']['train'][0]['data_path']=CS_PROJECT+'/'+profile['data_dir']+'/train.jsonl'
    expected['env']['use_genrm_compare']=False
    expected['env']['genrm_agent_names']=[]
    expected['env']['nemo_gym']['config_paths']=[CS_PROJECT+'/'+profile['services']]
    expected['env']['nemo_gym']['gym_data_asset_dir']=CS_PROJECT+'/'+profile['data_dir']+'/assets'
    if profile['delta_schema'] in (CS_NOCONV_FILTER_SCHEMA,CS_REFIT_FILTER_SCHEMA,CS_LR1E6_FILTER_SCHEMA):
        # The strict native renderer forbids thresholds for absent domains.
        # Remove only the omitted conversation route's original override.
        caps=expected['checkpointing']['length_monitor']['cap_hit_stop_fraction_by_domain']
        require(caps.get('conversational_tool_use_agent')==0.05,'Original conversation cap override changed')
        del caps['conversational_tool_use_agent']
    if profile['delta_schema']==CS_LR1E6_FILTER_SCHEMA:
        expected['policy']['megatron_cfg']['optimizer'].update(lr=1e-6,min_lr=1e-6)
    require(json.dumps(expected,sort_keys=True)==json.dumps(new_cfg,sort_keys=True),'Unreviewed research recipe change')
    aliases=cs_closure(old_services,profile)
    expected_services=cs_relocate({name:old_services[name] for name in aliases},mapping)
    require(json.dumps(expected_services,sort_keys=True)==json.dumps(new_services,sort_keys=True),'Retained reward/service settings changed')
    return aliases


def cs_catalog_ref(original,catalog):
    """Relocate a proof file, never rewrite its historical content or digest."""
    item=catalog[original['path']]
    require(item['sha256']==original['sha256'] and sha(Path(item['path']))==original['sha256'],'Portable evidence digest differs')
    return dict(path=str(Path(item['path']).resolve(strict=True)),sha256=item['sha256'])


def cs_load_configuration(path):
    """Read JSON with JSON numeric semantics; otherwise accept ordinary YAML."""
    text = Path(path).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(text)



def cs_validate_refit_pause(x):
    """Bind accepted CPU imports and native DP pause proof without upgrading scope."""
    require(cs_is_refit(x['schema']),'Refit delta requires its exact profile')
    proof=x['refit_pause']
    require(proof['schema']=='cs-refit-pause-source.v1' and proof['base_rl_pin']==CS_RL_PIN and
            proof['rl_pin']==CS_REFIT_RL_PIN,'Unexpected refit source identity')
    manifest=bound_json(proof['source_manifest'])
    require(proof['source_manifest']['sha256']==CS_REFIT_MANIFEST_SHA and len(manifest)==5,'Refit five-file manifest differs')
    cpu=bound_json(proof['cpu_root']);stage=bound_json(proof['cpu_source_stage'])
    require(cpu['status']=='ACCEPTED_CPU_NATIVE_IMPORT_AND_AST_ONLY' and cpu['job_id']=='35629146' and
            cpu['records_all_completed_0']==4 and cpu['actual_ast_cases']==34 and
            cpu['sha256']['SOURCE_STAGE_ACCEPTANCE.json']==proof['cpu_source_stage']['sha256'] and
            cpu['formal_gpu_restart_qualified'] is False and cpu['cuda_nccl_proof'] is False,
            'Native CPU refit import evidence differs')
    require(cpu['cleanup']['status']=='complete' and cpu['cleanup']['main_exit']==0 and
            cpu['cleanup']['anonymous_container_data_empty'] is True and cpu['cleanup']['scratch_removed'] is True,
            'Refit CPU resources not released')
    require(stage['status']=='PASS' and stage['base_pin']==CS_RL_PIN and
            {r['path']:r['sha256'] for r in stage['files']}=={p:r['candidate_sha256'] for p,r in manifest.items()},
            'CPU refit files differ')
    gpu=bound_json(proof['gpu_root']);actual=bound_json(proof['gpu_probe'])
    phase_path=Path(proof['gpu_phases']['path']).resolve(strict=True)
    require(sha(phase_path)==proof['gpu_phases']['sha256']==gpu['phases_sha256'],'GPU phase evidence differs')
    require(gpu['status']=='ACCEPTED_GPU_NATIVE_DP_PAUSE_AND_RESOURCE_CLOSEOUT' and gpu['job_id']=='35631875' and
            gpu['slurm_all_four_records']=='COMPLETED 0:0' and gpu['source_manifest_sha256']==CS_REFIT_MANIFEST_SHA and
            gpu['gpu_probe_sha256']==proof['gpu_probe']['sha256'] and gpu['weight_transfer_tested'] is False and
            gpu['formal_training_qualified'] is False and gpu['historical_unique_cause_proven'] is False,
            'GPU native pause evidence missing or scope inflated')
    require(actual['status']=='PASS_PAUSE_WIRING_AND_NATIVE_DP_RENDEZVOUS' and actual['job_id']=='35631875' and
            actual['source']['manifest_sha256']==CS_REFIT_MANIFEST_SHA and actual['cores']==8 and actual['waves']==2 and
            actual['requests']==64 and actual['weight_transfer_tested'] is False and actual['formal_training_qualified'] is False,
            'Actual GPU probe identity/scope differs')
    require(len(gpu['waves'])==2 and all(w['admitted']==32 and w['native_cores']==8 and
            w['live_emitted_prefix_survivors']>0 and w['native_completed_prefix_preserved']==32 for w in gpu['waves']),
            'GPU probe lacks live request preservation')
    require(gpu['gpu_cleanup']['status']=='complete' and gpu['gpu_cleanup']['main_exit']==0 and
            all(gpu['gpu_cleanup'][key] is True for key in ('gpu_compute_processes_empty','anonymous_container_data_empty','scratch_removed')) and
            gpu['task_cleanup']['all_consumers_done'] is True and gpu['task_cleanup']['all_utility_tasks_done'] is True and
            gpu['ray_remaining_descendants']==[], 'GPU pause resources not released')
    return manifest


def cs_validate_inputs(inputs_ref):
    x=bound_json(inputs_ref)
    profile=cs_filter_profile(x['schema'])
    if cs_is_refit(x['schema']):cs_validate_refit_pause(x)
    else:require('refit_pause' not in x,'Refit proof on an unrelated profile')
    require(x['project']==CS_PROJECT,'Wrong CS admission inputs')
    p=Path(CS_PROJECT).resolve(strict=True)
    parent_ref=x['parent_frozen_ready']; parent=bound_json(parent_ref)
    require(parent_ref['sha256']==CS_PARENT_READY_SHA and parent['status']=='PASS','Unaccepted portable parent')
    prior=Path(parent_ref['path']).parent
    for name,digest in parent['files'].items():
        require(sha(prior/name)==digest,'Portable parent file changed: '+name)
    smoke_ref=x['structural_smoke']; smoke=bound_json(smoke_ref)
    require(smoke_ref['sha256']==STRUCTURAL_SMOKE_SHA and str(smoke['job_id'])=='13604692' and
            smoke['status']=='PASS' and smoke['resources_released'] is True and smoke['optimizer_step']==4 and
            smoke['hf_step']==3 and smoke['hdo_numeric_gates']==32 and smoke['single_mcore_verified'] is True and
            smoke['gym_route_checks_passed'] is True and smoke['http_token_parity_passed'] is True,
            'Historical structural GPU proof incomplete')
    sr=Path(x['portable_smoke_dir']).resolve(strict=True)
    require((sr/'JOB_ID').read_text().strip()=='13604692' and sha(sr/'SMOKE_PASS.json')==smoke['smoke_pass_sha256'],
            'Historical GPU job/run evidence differs')
    old_smoke_pins=dict(line.split('=',1) for line in (sr/'SOURCE_RECEIPT.txt').read_text().splitlines())
    for key in ('nemo_rl_applied','bridge','megatron_lm','gym'):
        require(old_smoke_pins[key]==smoke['source_pins'][key],'Historical GPU source differs')
    for name,digest in smoke['runtime_scripts_sha256'].items():
        require(sha(sr/'scripts'/name)==digest,'Historical GPU runtime changed')
    mapping=x['path_map']
    require(isinstance(mapping,dict) and mapping and all(isinstance(k,str) and isinstance(v,str) and k.startswith('/lustre/') and v.startswith('/lustre/') for k,v in mapping.items()),'Invalid exact site path map')
    require(hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()==CS_PATH_MAP_SHA,'Unreviewed site path replacement')
    require(cs_relocate('/lustre/fsw/portfolios/nemotron/users/cchen1/NemotronVC_super35_draco',mapping)==CS_PROJECT,
            'Project relocation missing')
    old_cfg=bound_json(x['parent_recipe']); old_services=bound_json(x['parent_services'])
    require(x['parent_recipe']['sha256']==profile['parent_recipe_sha'] and x['parent_services']['sha256']==profile['parent_services_sha'],'Unaccepted old recipe/services')
    # Both existing .yaml files are JSON; permit normal YAML for new user-owned copies.
    new_cfg=cs_load_configuration(x['recipe']['path']); new_services=cs_load_configuration(x['services']['path'])
    require(sha(Path(x['recipe']['path']))==x['recipe']['sha256']==profile['recipe_sha'] and
            sha(Path(x['services']['path']))==x['services']['sha256']==profile['services_sha'],'New config bytes differ')
    require(Path(x['recipe']['path']).resolve()==p/profile['recipe'] and Path(x['services']['path']).resolve()==p/profile['services'],'New recipe/services path differs')
    aliases=cs_validate_configuration(old_cfg,new_cfg,old_services,new_services,mapping,profile)
    if x['schema']==CS_LR1E6_INPUT_SCHEMA:
        parent_refit=bound_json(x['parent_refit_recipe'])
        require(x['parent_refit_recipe']['sha256']==CS_REFIT_RECIPE_SHA and
                Path(x['parent_refit_recipe']['path']).resolve()==p/'config/rlvr-light-v4-nogenrm-noconv-refitpause-8r2l-p256.yaml',
                'LR arm lacks exact parent refit recipe')
        cs_validate_lr1e6_recipe(parent_refit,new_cfg)
        cs_validate_metrics_source(x)
    else:
        require('metrics_source' not in x and 'parent_refit_recipe' not in x,
                'LR/metrics proof on unrelated profile')
    data=bound_json(x['data_filter']); closure=bound_json(x['service_closure'])
    cs_validate_filter_data(data,profile,x.get('parent_data_filter'))
    require(closure['schema']=='gym-service-closure.v1' and closure['status']=='PASS' and
            closure['services']['sha256']==x['services']['sha256'] and closure['retained_aliases']==aliases and
            closure['retained_alias_count']==profile['alias_count'] and closure['genrm_reachable'] is False and
            closure['retained_config_identical_except_site_paths'] is True and
            all(cs_relocate(k,mapping)==v for k,v in closure['path_map'].items()) and
            closure['math_should_use_judge'] is True, 'Current retained service closure proof missing')
    if profile['delta_schema'] in (CS_NOCONV_FILTER_SCHEMA,CS_REFIT_FILTER_SCHEMA,CS_LR1E6_FILTER_SCHEMA):
        require(closure['conversation_reachable'] is False and closure['conversation_llm_judge_retained'] is False and
                closure['removed_aliases']==['conversational_tool_use_agent','conversational_tool_use_simulation','simulator_model'] and
                closure['parent_services']['sha256']==x['parent_services']['sha256'] and
                bound_json(closure['parent_services'])==old_services and closure['path_map']=={},
                'Conversation closure delta differs')
    else:
        require(closure['conversation_llm_judge_retained'] is True,'Retained conversation judge changed')
    cleanup=bound_json(x['cleanup_root']); cd=Path(x['cleanup_root']['path']).parent
    require(cleanup['status']=='PASS' and str(cleanup['job_id'])=='13632910' and cleanup['gpu_training_qualified'] is False,'Cleanup regression not accepted')
    for name,digest in cleanup['source_hashes'].items(): require(sha(cd/name)==digest,'Cleanup regression receipt changed')
    ast=json.loads((cd/'CLEANUP_AST.json').read_text()); runtime=json.loads((cd/'CLEANUP_RUNTIME.json').read_text())
    released=json.loads((cd/'CPU_CLEANUP.json').read_text()); owned=json.loads((cd/'OWNED_RAY_ROUND_CLEANUP.json').read_text())
    require(ast['status']=='PASS' and ast['source_sha256']==CS_GRPO_SHA and ast['tests_run']==8 and runtime['status']=='PASS' and
            len(runtime['cases'])==4 and all(c['status']=='PASS' for c in runtime['cases']) and owned['remaining']==[] and
            released['main_exit']==0 and released['anonymous_container_data_empty'] is True and released['scratch_removed'] is True,
            'Cleanup source/lifecycle evidence incomplete')
    catalog=bound_json(x['portable_catalog'])
    require(isinstance(catalog,dict),'Portable catalog must map original path to copied immutable reference')
    if 'files' in catalog:
        require(catalog.get('missing')==[],'Portable proof transfer has missing files')
        base=Path(x['portable_catalog']['path']).resolve(strict=True).parent
        catalog={original:{'path':str((base/item['path']).resolve(strict=True)),'sha256':item['sha256']}
                 for original,item in catalog['files'].items()}
    return x,prior,smoke,data,aliases,catalog


def validate_cs_filter_delta(delta_ref,smoke_ref,run):
    import tarfile
    delta=bound_json(delta_ref)
    require(delta['schema'] in (CS_FILTER_SCHEMA,CS_NOCONV_FILTER_SCHEMA,CS_REFIT_FILTER_SCHEMA,CS_LR1E6_FILTER_SCHEMA) and delta['status']=='PASS' and delta['exact_gpu_match'] is False and
            delta['admission_mode']=='structural_gpu_plus_verified_cpu_delta','Invalid CS filtered admission')
    x,prior,smoke,data,aliases,catalog=cs_validate_inputs(delta['inputs'])
    profile=cs_filter_profile(x['schema'])
    require(delta['schema']==profile['delta_schema'],'CS data/delta profile mismatch')
    require(x['structural_smoke']==smoke_ref,'Different historical smoke passed to freeze')
    request_ref=delta['request']; request=bound_json(request_ref); source_ref=delta['source']; source=bound_json(source_ref)
    pins=dict(line.split('=',1) for line in (run/'SOURCE_RECEIPT.txt').read_text().splitlines())
    require(pins==x['expected_pins']==request['source_pins'],'CS source pins differ')
    old_pins=dict(line.split('=',1) for line in (prior/'SOURCE_RECEIPT.txt').read_text().splitlines())
    refit=cs_is_refit(x['schema'])
    metrics=x['schema']==CS_LR1E6_INPUT_SCHEMA
    target_rl=CS_METRICS_RL_PIN if metrics else (CS_REFIT_RL_PIN if refit else CS_RL_PIN)
    require(pins['nemo_rl_applied']==target_rl and all(pins[k]==old_pins[k] for k in ('gym','bridge','megatron_lm','pipeline')),'Unreviewed research source delta')
    require(source['status']=='PASS' and source['job_id']==os.environ['SLURM_JOB_ID'] and
            source['request_sha256']==request_ref['sha256'] and source['inputs_sha256']==delta['inputs']['sha256'] and
            source['source_pins']==pins and all(source[k] is True for k in ('clean_source_verified','runtime_script_relocation_verified')) and
            source.get('metrics_only_rl_diff_verified' if metrics else ('refit_boundary_only_rl_diff_verified' if refit else 'cleanup_only_rl_diff_verified')) is True,
            'Current CPU source/site proof missing')
    if refit:
        require(source.get('cleanup_only_rl_diff_verified') is False and source.get('refit_pause')==x['refit_pause'] and
                request.get('refit_pause')==x['refit_pause'],'Refit source proof lost or mislabeled')
    if metrics:
        require(source.get('refit_boundary_only_rl_diff_verified') is False and
                source.get('metrics_source')==request.get('metrics_source')==x['metrics_source'],
                'Metrics child mislabeled as historical refit-only source')
        metrics_cpu=cs_validate_metrics_cpu(source['metrics_cpu'],x,os.environ['SLURM_JOB_ID'])
        require(source['runtime_scripts_sha256']['scripts/runtime_env.sh']==CS_METRICS_RUNTIME_ENV_SHA,
                'Metrics enablement script not verified')
    require(request['recipe_sha256']==x['recipe']['sha256'] and request['data_sha256']==data['data_sha256'] and
            request['services_sha256']==x['services']['sha256'],'Current requested inputs differ')
    contract=json.loads((run/'RUN_CONTRACT.json').read_text()); origin=json.loads((run/'INPUT_MANIFEST.json').read_text())
    expected={'kind':'train','resume_mode':'new','nodes':10,'rollout_nodes':8,'learner_nodes':2,'prompts':256,'K':8,
        'response_gbs':2048,'scheduler_horizon':20,'data_rows':profile['rows'],'data_sha256':data['data_sha256'],
        'expected_resume_checkpoint':None,'smoke_resume_step':None}
    for key,value in expected.items(): require(type(contract.get(key)) is type(value) and contract[key]==value,'Wrong filtered run contract: '+key)
    require(origin['recipe_sha256']==x['recipe']['sha256'] and
            Path(contract['student']).resolve()==Path(CS_PROJECT+'/models/sftkd-s3-c8-lr1e-5-s399/huggingface').resolve() and
            {k:v['sha256'] for k,v in contract['runtime_images'].items()}==smoke['runtime_images_sha256'],
            'Student/recipe/runtime image identity changed')
    require(source['runtime_scripts_sha256']=={name:sha(run/name) for name in source['runtime_scripts_sha256']},'Frozen CS runtime differs from CPU verified bytes')
    with tarfile.open(run/'source.tar.gz','r:gz') as archive:
        frozen_files=cs_validate_refit_pause(x) if refit else {'nemo_rl/algorithms/grpo.py':{'candidate_sha256':CS_GRPO_SHA}}
        if metrics:
            frozen_files=dict(frozen_files,**cs_validate_metrics_source(x)['source_files'])
        if metrics:
            cs_validate_metrics_archive(archive,frozen_files)
        else:
            for relative,row in frozen_files.items():
                require(hashlib.sha256(archive.extractfile('./'+relative).read()).hexdigest()==row['candidate_sha256'],
                        'Frozen archive accepted source changed: '+relative)
    services=json.loads((run/'GYM_SERVER_CONFIGS.json').read_text())['files']
    require(len(services)==1 and services[0]['sha256']==x['services']['sha256'] and sha(Path(services[0]['path']))==x['services']['sha256'],'Frozen retained services differ')
    # Submit must use this literal historical terminal proof, never CS sacct for a Draco numeric job ID.
    terminal_ref=delta['historical_terminal']; terminal=bound_json(terminal_ref)
    require(terminal=={'schema':'historical-remote-gpu-smoke.v1','status':'PASS','cluster':'draco','job_id':'13604692',
        'smoke_acceptance':smoke_ref,'resources_released':True,'local_scheduler_job_identity':False},'Historical terminal identity misrepresented')
    judge=bound_json(delta['remaining_judge'])
    parent_api=json.loads((prior/'RUNTIME_API_ACCEPTANCE.json').read_text())
    require(judge['status']=='PASS' and judge['request_sha256']==request_ref['sha256'] and judge['services_sha256']==x['services']['sha256'] and
            judge['retained_aliases']==aliases and judge['retained_configs_unchanged_except_site_paths'] is True and
            judge['actual_api_calls_in_verification_job']==0 and
            judge['historical_qualification']==cs_catalog_ref(parent_api['free_api_qualification'],catalog),
            'Remaining API qualification identity or scope differs')
    copies={'GYM_SMOKE_CPU_DELTA.json':delta_ref,'GYM_CPU_DELTA_REQUEST.json':request_ref,
        'GYM_CPU_DELTA_SOURCE.json':source_ref,'GYM_CPU_DELTA_JUDGE.json':delta['remaining_judge'],
        'GYM_CS_ADMISSION_INPUTS.json':delta['inputs'],'GYM_FILTER_ACCEPTANCE.json':x['data_filter'],
        'GYM_SERVICE_CLOSURE_ACCEPTANCE.json':x['service_closure'],'GYM_HISTORICAL_SMOKE_TERMINAL.json':terminal_ref}
    if refit:
        for key,name in [('source_manifest','GYM_REFIT_SOURCE_MANIFEST.json'),('cpu_root','GYM_REFIT_CPU_ROOT.json'),('cpu_source_stage','GYM_REFIT_CPU_SOURCE_STAGE.json'),('gpu_root','GYM_REFIT_GPU_ROOT.json'),('gpu_probe','GYM_REFIT_GPU_PROBE.json'),('gpu_phases','GYM_REFIT_GPU_PHASES.jsonl')]:
            copies[name]=x['refit_pause'][key]
    if metrics:
        copies['GYM_LR1E6_PARENT_RECIPE.json']=x['parent_refit_recipe']
        copies['GYM_POSTMASK_SOURCE_MANIFEST.json']=x['metrics_source']['source_manifest']
        copies['GYM_POSTMASK_METRICS_CPU.json']=source['metrics_cpu']
        copies['GYM_POSTMASK_METRICS_TESTS.log']=metrics_cpu['test_log']
        copies['GYM_POSTMASK_METRICS_TEST_RESULT.json']=metrics_cpu['raw_result']
    if profile['delta_schema'] in (CS_NOCONV_FILTER_SCHEMA,CS_REFIT_FILTER_SCHEMA,CS_LR1E6_FILTER_SCHEMA):
        copies['GYM_PARENT_FILTER_ACCEPTANCE.json']=x['parent_data_filter']
    return copies


def validate_cpu_smoke_delta(delta_ref, smoke_ref, run):
    """Keep the historical GPU proof literal; require a separate exact CPU delta.

    The CPU producer must verify Git/file/config diffs and execute the current
    judge/stop tests. This consumer binds those real receipts; it makes no new
    GPU, new-data trajectory, or future unchanged-judge GPU-success claim.
    """
    delta = bound_json(delta_ref)
    if delta.get('schema')==CS_CODE_MATH_DELTA_SCHEMA:
        return validate_cs_code_math_delta(delta_ref,smoke_ref,run)
    if delta.get('schema') in (CS_FILTER_SCHEMA, CS_NOCONV_FILTER_SCHEMA, CS_REFIT_FILTER_SCHEMA, CS_LR1E6_FILTER_SCHEMA):
        return validate_cs_filter_delta(delta_ref, smoke_ref, run)
    require(delta.get('schema') == 'gym-structural-smoke-cpu-delta.v1' and
            delta.get('status') == 'PASS' and delta.get('exact_gpu_match') is False and
            delta.get('admission_mode') == 'structural_gpu_plus_verified_cpu_delta',
            'CPU delta is not an explicit qualified admission')
    require(smoke_ref['sha256'] == STRUCTURAL_SMOKE_SHA and
            delta['structural_smoke']['sha256'] == smoke_ref['sha256'] and
            Path(delta['structural_smoke']['path']).resolve(strict=True) == Path(smoke_ref['path']).resolve(strict=True), 'Historical GPU proof differs')
    smoke = bound_json(smoke_ref)
    require(str(smoke['job_id']) == '13604692', 'Unreviewed structural GPU ancestor')
    validate_smoke(smoke, Path(smoke['smoke_run']).resolve(strict=True))
    request_ref = delta['request']; request = bound_json(request_ref)
    require(request['schema'] == 'gym-8plus2-cpu-delta-request.v1' and
            request['recipe_sha256'] == REDUCED_RECIPE_SHA and
            request['base_formal_recipe_sha256'] == '647f19d4541722658e34bac6419d664f3fa7d26c9cbd08263b4523644cd5248b' and
            request['data_sha256'] == BALANCED_DATA_SHA and request['data_rows'] == 12288,
            'Unreviewed recipe/data delta')
    contract = json.loads((run/'RUN_CONTRACT.json').read_text())
    expected = {'kind':'train','resume_mode':'new','nodes':10,'rollout_nodes':8,'learner_nodes':2,
                'prompts':256,'K':8,'response_gbs':2048,'scheduler_horizon':20,'data_rows':12288,
                'data_sha256':BALANCED_DATA_SHA,'expected_resume_checkpoint':None,'smoke_resume_step':None}
    for key, value in expected.items():
        require(type(contract.get(key)) is type(value) and contract[key] == value,
                'CPU delta target differs: '+key)
    origin = json.loads((run/'INPUT_MANIFEST.json').read_text())
    require(origin['recipe_sha256'] == REDUCED_RECIPE_SHA, 'CPU delta master differs')
    pins = dict(line.split('=',1) for line in (run/'SOURCE_RECEIPT.txt').read_text().splitlines())
    require(request['source_pins'] == pins, 'CPU delta target source differs')
    require(contract['student'] == smoke['student'] and
            {k:v['sha256'] for k,v in contract['runtime_images'].items()} == smoke['runtime_images_sha256'],
            'CPU delta may not change student or images')
    wrapper_changes = request['wrapper_changes']
    require(isinstance(wrapper_changes, list) and wrapper_changes, 'Exact approved wrapper changes required')
    changes = {x['path']:x for x in wrapper_changes}
    require(len(changes) == len(wrapper_changes), 'Duplicate wrapper delta path')
    project = Path((run/'GYM_PROJECT_DIR').read_text().strip()).resolve(strict=True)
    for rel, change in changes.items():
        require(set(change) == {'path','before_sha256','after_sha256'} and
                rel.startswith(('scripts/','config/')) and not Path(rel).is_absolute() and '..' not in Path(rel).parts,
                'Unreviewed wrapper path')
        require(isinstance(change['after_sha256'],str) and len(change['after_sha256']) == 64 and
                sha(project/rel) == change['after_sha256'], 'Current wrapper bytes differ: '+rel)
        if rel.startswith('scripts/') and (run/rel).exists():
            require(sha(run/rel) == change['after_sha256'], 'Frozen wrapper bytes differ: '+rel)
    for name, digest in smoke['runtime_scripts_sha256'].items():
        change = changes.get('scripts/'+name)
        if change is None:
            require(sha(run/'scripts'/name) == digest, 'Structural runtime script changed: '+name)
        else:
            require(change['before_sha256'] == digest and sha(run/'scripts'/name) == change['after_sha256'],
                    'Runtime delta is not the approved before/after pair: '+name)
    for name in request['required_new_frozen_helpers']:
        require(name in changes and changes[name]['before_sha256'] is None and
                not (Path(smoke['smoke_run'])/name).exists() and
                (run/name).is_file() and sha(run/name) == changes[name]['after_sha256'],
                'New owned helper missing from frozen scripts')
    services = json.loads((run/'GYM_SERVER_CONFIGS.json').read_text())['files']
    require(len(services) == 1 and services[0]['sha256'] == request['services_sha256'] and
            sha(Path(services[0]['path'])) == request['services_sha256'], 'Current services differ')
    require(request['source_changes'] and request['services_changes']==[] and request['services_sha256']=='5b3dba9c09dc1e379736eea81cf75c0f70465827006536c92d9b6ac069010a1b', 'Only unchanged original services are admitted')
    for change in request['source_changes']:
        require(set(change) == {'repository','path','before_sha256','after_sha256'} and
                change['repository'] in ('rl','gym') and not Path(change['path']).is_absolute() and
                '..' not in Path(change['path']).parts and
                all(isinstance(change[k],str) and len(change[k]) == 64 for k in ('before_sha256','after_sha256')),
                'Malformed exact source change')
    data_ref = request['data_acceptance']; data = bound_json(data_ref)
    require(data_ref['sha256'] == BALANCED_ACCEPTANCE_SHA and data['status'] == 'PASS' and
            data['data_sha256'] == BALANCED_DATA_SHA and data['rows'] == 12288,
            'Balanced data CPU acceptance differs')
    evidence_names = {'source':'GYM_CPU_DELTA_SOURCE.json','judge':'GYM_CPU_DELTA_JUDGE.json'}
    copies = {'GYM_SMOKE_CPU_DELTA.json':delta_ref,'GYM_CPU_DELTA_REQUEST.json':request_ref}
    for role, name in evidence_names.items():
        ref = delta['evidence'][role]; proof = bound_json(ref)
        job_key='job_id' if role=='source' else 'verification_job_id'
        request_key='source_cpu_job_id' if role=='source' else 'judge_verification_cpu_job_id'
        require(proof['status'] == 'PASS' and str(proof[job_key]) == str(request[request_key]) and
                str(proof[job_key]).isdigit() and proof['request_sha256'] == request_ref['sha256'] and
                proof['services_sha256'] == request['services_sha256'],
                'Current CPU evidence identity differs: '+role)
        if role == 'source':
            require(proof['source_pins'] == pins and proof['source_changes'] == request['source_changes'] and
                    proof['wrapper_changes'] == wrapper_changes and proof['actual_file_hashes_verified'] is True and
                    proof['git_diff_exact'] is True and
                    proof['services_changes'] == request['services_changes'] and
                    proof['exact_delta_verified'] is True and proof['focused_stop_tests_passed'] is True,
                    'Exact source/stop regression missing')
        else:
            require(proof['actual_api_job_ids']==request['judge_actual_api_job_ids'] and
                    proof['actual_api_calls_in_verification_job']==0 and
                    proof['qualification']==request['judge_qualification'] and
                    proof['tested_gym_commit'] == request['judge_tested_gym_commit'] and
                    proof['imported_judge_files'] == request['judge_imported_files'] and
                    bool(proof['imported_judge_files']) and proof['judge_enable_thinking'] is True and
                    all(proof[k] is True for k in ('native_scoring_preserved','native_directions_passed',
                         'strict_final_validity','failures_are_errors','cost_zero')),
                    'Current thinking judge qualification incomplete')
        copies[name] = ref
    return copies

def seal(run, acceptance, smoke_acceptance=None):
    read = lambda name: json.loads((run / name).read_text())
    contract = read('RUN_CONTRACT.json')
    require(contract['input_mode'] == 'gym', 'Gym gate must not classify legacy runs')
    role = read('ROLE_RUNTIME_PREFLIGHT.json')
    scheduler = read('NATIVE_SCHEDULER_PREFLIGHT.json')
    actor = read('GYM_ACTOR_FREEZE_PREFLIGHT.json')
    from gym_actor_runtime_gate import ACTOR,SERVICES
    pins = dict(line.split('=', 1) for line in (run/'SOURCE_RECEIPT.txt').read_text().splitlines())
    require(actor['bindings']['rl_commit']==pins['nemo_rl_applied'] and
            actor['bindings']['gym_commit']==pins['gym'],
            'Actual actor source differs from final frozen source receipt')
    # Qualification was validated against the resolved recipe during real actor gate.
    # Seal immutable receipts and keep the actor mapping separate from service imports.
    require(actor['status']=='PASS' and actor['job_id']==role['slurm_job_id']==os.environ['SLURM_JOB_ID'] and
            actor['run']==str(run) and actor['config_sha256']==sha(run/'config.yaml') and
            actor['selected_actor_python']==ACTOR and actor['services_python']==SERVICES and
            actor['runtime_env_sha256']==sha(run/'scripts/runtime_env.sh') and
            actor['qualification_sha256']==sha(run/'GYM_ACTOR_QUALIFICATION.json') and
            actor['roundtrip_sha256']==sha(run/'GYM_ACTOR_ROUNDTRIP.json') and
            actor['ray_owner_sha256']==sha(run/'GYM_CPU_RAY_OWNER.json') and
            role['actor_freeze_preflight_sha256']==sha(run/'GYM_ACTOR_FREEZE_PREFLIGHT.json') and
            role['runtime_mappings']['gym']==ACTOR and role['gym_services_python']==SERVICES,
            'Actual per-freeze actor runtime gate is missing or mismatched')
    for name, proof in [('role', role), ('scheduler', scheduler)]:
        require(proof['status'] == 'PASS', name + ' preflight did not pass')
        require(proof['config_sha256'] == sha(run / 'config.yaml'), name + ' config binding differs')
    require(scheduler['run_contract_sha256'] == sha(run / 'RUN_CONTRACT.json'), 'Scheduler run binding differs')
    require(scheduler['target_step'] == contract['scheduler_horizon'], 'Scheduler horizon differs')
    require(len(scheduler['strict_negative_checks']) == 9 and scheduler['override_flags_false'] and
            scheduler['remaining_updates_exact'], 'Native scheduler proof incomplete')
    require(scheduler['checkpoint'] == contract['expected_resume_checkpoint'], 'Scheduler source differs')
    # Materialization is not evaluation-leakage clearance. This separately
    # produced CPU receipt must bind exactly the frozen data SHA.
    clearance = json.loads(acceptance.read_text())
    require(clearance['status'] == 'PASS' and clearance['data_sha256'] == contract['data_sha256'],
            'Missing or mismatched data acceptance')
    shutil.copy2(acceptance, run / 'GYM_DATA_ACCEPTANCE.json')
    if contract['kind'] == 'train':
        require(smoke_acceptance is not None, 'Formal Gym submission requires accepted GPU smoke')
        smoke = json.loads(smoke_acceptance.read_text())
        delta_ref = read('RUNTIME_API_ACCEPTANCE.json').get('structural_smoke_cpu_delta')
        if contract['nodes'] == 10 or delta_ref is not None:
            require(delta_ref is not None, 'Reduced formal requires explicit CPU delta qualification')
            copies = validate_cpu_smoke_delta(delta_ref,
                {'path':str(smoke_acceptance.resolve(strict=True)), 'sha256':sha(smoke_acceptance)}, run)
            for name, ref in copies.items():
                shutil.copy2(Path(ref['path']), run/name)
        else:
            validate_smoke(smoke, run)
        shutil.copy2(smoke_acceptance, run / 'GYM_SMOKE_ACCEPTANCE.json')
        require(str(smoke['job_id']).isdigit(), 'Invalid GPU smoke job identity')
        (run / 'GYM_SMOKE_JOB').write_text(str(smoke['job_id']) + '\n')
    required = ['config.yaml', 'RUN_CONTRACT.json', 'GYM_LINEAGE_CONTRACT.json', 'INPUT_MANIFEST.json',
                'DATA_MIX_MANIFEST.json', 'GYM_ASSET_MANIFEST.json', 'GYM_EXTERNAL_ASSETS.json',
                'GYM_SERVER_CONFIGS.json',
                'DATA_RENDER_SUMMARY.json', 'DATA_RENDER_COMPLETE.json', 'TOKENIZER_CONTRACT.json',
                'MODEL_INPUT_AUDIT.json', 'ROLE_RUNTIME_PREFLIGHT.json', 'GYM_RUNTIME_PROBE.json',
                'GYM_NODE_HTTP_PREFLIGHT.json', 'GYM_NATIVE_RENDER_PREFLIGHT.json',
                'GYM_RAY_RUNTIME_PREFLIGHT.json', 'GYM_ACTOR_QUALIFICATION.json',
                'GYM_ACTOR_FREEZE_PREFLIGHT.json', 'GYM_ACTOR_ROUNDTRIP.json', 'GYM_CPU_RAY_OWNER.json',
                'NATIVE_SCHEDULER_PREFLIGHT.json', 'GYM_DATA_ACCEPTANCE.json',
                'RUNTIME_API_ACCEPTANCE.json', 'GYM_IMAGE_CONTRACT.json',
                'RUNTIME_IMAGE', 'SANDBOX_IMAGE',
                'RUNTIME_IMAGE_ACCEPTANCE.json', 'SANDBOX_IMAGE_ACCEPTANCE.json',
                'INPUT_MODE', 'OBJECTIVE', 'KIND', 'NODES', 'GYM_LINEAGE_DIR', 'GYM_PROJECT_DIR',
                'GYM_EXPECTED_RESUME', 'SOURCE_RECEIPT.txt', 'source.sha256', 'launcher.sha256',
                'LAUNCHER_PREFLIGHT.json', 'LAUNCHER_TESTS.log', 'ROLE_RUNTIME_PREFLIGHT.log',
                'NATIVE_SCHEDULER_PREFLIGHT.log']
    if role.get('initial_render_evidence_mode') == 'carried_forward':
        reuse = read('GYM_RENDER_REUSE.json')
        require(reuse['status'] == 'PASS' and reuse['evidence_mode'] == 'carried_forward' and
                reuse['new_run'] == str(run) and reuse['config_sha256'] == sha(run/'config.yaml') and
                reuse['data_sha256'] == contract['data_sha256'] and
                reuse['full_token_scan_executed_in_this_job'] is False and
                reuse['current_runtime_probe_sha256'] == sha(run/'GYM_RUNTIME_PROBE.json') and
                role['render_reuse_sha256'] == sha(run/'GYM_RENDER_REUSE.json') and
                reuse['request_sha256'] == sha(run/'RENDER_REUSE_REQUEST.json'),
                'Inherited full render lacks current source/config/probe provenance')
        required += ['GYM_RENDER_REUSE.json', 'GYM_NATIVE_RENDER_REUSE_PROBE.json', 'RENDER_REUSE_REQUEST.json']
    delta_names = ('GYM_SMOKE_CPU_DELTA.json','GYM_CPU_DELTA_REQUEST.json',
                   'GYM_CPU_DELTA_SOURCE.json','GYM_CPU_DELTA_JUDGE.json')
    if contract['kind'] == 'train' and contract['nodes'] == 10:
        required += list(delta_names)
    if (run/'GYM_HISTORICAL_SMOKE_TERMINAL.json').is_file():
        (run/'GYM_SMOKE_SITE').write_text('draco\n')
        (run/'GYM_HISTORICAL_SMOKE_TERMINAL_ACCEPTED').write_text('draco:13604692:COMPLETED:0:0\n')
        required += ['GYM_SMOKE_SITE','GYM_HISTORICAL_SMOKE_TERMINAL_ACCEPTED']
    for name in ('GYM_CS_ADMISSION_INPUTS.json','GYM_FILTER_ACCEPTANCE.json','GYM_SERVICE_CLOSURE_ACCEPTANCE.json','GYM_HISTORICAL_SMOKE_TERMINAL.json','GYM_PARENT_FILTER_ACCEPTANCE.json'):
        if (run/name).is_file(): required.append(name)
    if (run/'GYM_CS_ADMISSION_INPUTS.json').is_file() and cs_is_refit(read('GYM_CS_ADMISSION_INPUTS.json')['schema']):
        required += ['GYM_REFIT_SOURCE_MANIFEST.json','GYM_REFIT_CPU_ROOT.json','GYM_REFIT_CPU_SOURCE_STAGE.json',
                     'GYM_REFIT_GPU_ROOT.json','GYM_REFIT_GPU_PROBE.json','GYM_REFIT_GPU_PHASES.jsonl']
    if (run/'GYM_CS_ADMISSION_INPUTS.json').is_file() and read('GYM_CS_ADMISSION_INPUTS.json')['schema']==CS_LR1E6_INPUT_SCHEMA:
        required += ['GYM_LR1E6_PARENT_RECIPE.json','GYM_POSTMASK_SOURCE_MANIFEST.json',
                     'GYM_POSTMASK_METRICS_CPU.json','GYM_POSTMASK_METRICS_TESTS.log','GYM_POSTMASK_METRICS_TEST_RESULT.json']
    if (run/'GYM_CS_ADMISSION_INPUTS.json').is_file() and read('GYM_CS_ADMISSION_INPUTS.json')['schema']==CS_CODE_MATH_INPUT_SCHEMA:
        required += list(copies)+['GYM_SOURCE_ARCHIVE_REUSE.json']  # Exact v6 producer refs and current archive-tree proof.
    required += [str(p.relative_to(run)) for p in sorted((run / 'scripts').glob('*')) if p.is_file()]
    required += [str(p.relative_to(run)) for p in sorted((run / 'gym-configs').glob('*')) if p.is_file()]
    for optional in ('PENDING_HF_ACCEPTED.json', 'ORIGIN_CONFIG', 'CONTINUATION_ACCEPTANCE.json',
                     'GYM_PRIOR_JOB', 'GYM_PRIOR_RUN', 'GYM_SMOKE_ACCEPTANCE.json', 'GYM_SMOKE_JOB'):
        if (run / optional).is_file():
            required.append(optional)
    for name in required:
        path = run / name
        require(path.is_file() and not path.is_symlink(), 'Missing/symlink CPU proof: ' + name)
        require(path.stat().st_size <= 32 << 20, 'Submit proof unexpectedly large: ' + name)
    proof = {'schema': 'gym-frozen-ready.v1', 'status': 'PASS', 'cpu_job': os.environ.get('SLURM_JOB_ID'),
             'run': str(run), 'checkpoint_dir': contract['checkpoint_dir'],
             'config_sha256': sha(run / 'config.yaml'), 'data_sha256': contract['data_sha256'],
             'initial_render_evidence_mode': role.get('initial_render_evidence_mode', 'executed'),
             'role_scope': 'Initial Gym requests and CPU protocol fixtures; actual GPU Gym routes remain smoke-gated',
             'files': {name: sha(run / name) for name in sorted(set(required))}}
    if contract['kind'] == 'train' and contract['nodes'] == 10:
        proof.update(admission_mode='structural_gpu_plus_verified_cpu_delta', exact_gpu_match=False,
                     role_scope='Historical GPU structure plus current CPU delta; changed data/batch/idle has not had a matching GPU smoke')
    (run / 'GYM_FROZEN_READY.json').write_text(json.dumps(proof, indent=2) + '\n')
    required.append('GYM_FROZEN_READY.json')
    (run / 'GYM_SUBMIT_GATES.sha256').write_text(''.join(f'{sha(run / name)}  {name}\n' for name in sorted(set(required))))
    # Keep the legacy submitter's role manifest name, but the Gym submitter
    # validates the complete backend-specific manifest directly.
    (run / 'role-preflight.sha256').write_text(''.join(f'{sha(run / name)}  {name}\n' for name in
        ('config.yaml', 'ROLE_RUNTIME_PREFLIGHT.json', 'GYM_RUNTIME_PROBE.json')))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-acceptance', type=Path, required=True)
    parser.add_argument('--smoke-acceptance', type=Path)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID'), 'CPU allocation required')
    value = seal(args.run.resolve(strict=True), args.data_acceptance.resolve(strict=True),
                 args.smoke_acceptance.resolve(strict=True) if args.smoke_acceptance else None)
    print(json.dumps({'status': value['status'], 'run': value['run'], 'small_files': len(value['files'])}))


if __name__ == '__main__':
    main()
