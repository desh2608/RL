#!/usr/bin/env python3
"""Simulator-node launch chain: make_run.py arithmetic/refusals, launcher env plumbing, sim_role.sh.

Stdlib unittest only (no omegaconf/nemo_rl): the pure functions of make_run.py are imported by file,
the shell launchers are checked structurally, and sim_role.sh is exercised end to end against a fake
interpreter that answers /v1/models. Run inside the training container with the driver venv:
  /opt/nemotronvc/venvs/driver/bin/python scripts/test_make_run_sim.py
"""
import http.server
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_run = load('make_run')
ray_ports = load('ray_ports')
HUB_SERVICES = (PROJECT / 'config/gym-services-agentic-v1.yaml').read_text()
LOCALSIM_SERVICES = (PROJECT / 'config/gym-services-agentic-v1-localsim.yaml').read_text()


def sim_args(**overrides):
    values = {'sim_nodes': 1, 'sim_model_path': None, 'sim_served_name': 'gpt-oss-120b', 'sim_port': 18100,
              'sim_tp': 1, 'sim_dp': 8, 'sim_max_model_len': 32768, 'sim_gpu_memory_utilization': 0.90,
              'sim_max_num_seqs': 128, 'sim_extra_args': make_run.DEFAULT_SIM_EXTRA_ARGS, 'sim_python': sys.executable}
    values.update(overrides)
    return types.SimpleNamespace(**values)


class NodeLayout(unittest.TestCase):
    def test_nodes_file_is_ray_plus_sim(self):
        self.assertEqual(make_run.node_layout(3, 0), {'ray_nodes': 3, 'sim_nodes': 0, 'total_nodes': 3})
        self.assertEqual(make_run.node_layout(3, 1)['total_nodes'], 4)
        self.assertEqual(make_run.node_layout(10, 1), {'ray_nodes': 10, 'sim_nodes': 1, 'total_nodes': 11})
        self.assertEqual(make_run.node_layout(10, 2)['total_nodes'], 12)

    def test_invalid_counts_refused(self):
        for policy, sim in ((0, 1), (3, -1), ('3', 0), (3, '1')):
            with self.subTest(policy=policy, sim=sim), self.assertRaises(make_run.BuildError):
                make_run.node_layout(policy, sim)


class LauncherEnvRules(unittest.TestCase):
    def test_hub_services_without_sim_is_historical(self):
        self.assertEqual(make_run.classify_env_refs(HUB_SERVICES, 0), (['NVIDIA_API_KEY'], []))

    def test_localsim_services_with_sim_node_has_no_secret(self):
        secrets, launcher = make_run.classify_env_refs(LOCALSIM_SERVICES, 1)
        self.assertEqual(secrets, [])
        self.assertEqual(launcher, ['NVC_SIM_API_KEY', 'NVC_SIM_BASE_URL'])
        self.assertTrue(set(launcher) <= set(make_run.LAUNCHER_ENV_SOURCES))
        self.assertTrue(set(launcher).isdisjoint(make_run.SECRET_SOURCES))

    def test_localsim_services_without_sim_node_refused(self):
        with self.assertRaisesRegex(make_run.BuildError, r'NVC_SIM_(BASE_URL|API_KEY).*--sim-nodes 0'):
            make_run.classify_env_refs(LOCALSIM_SERVICES, 0)

    def test_hub_services_with_sim_node_refused(self):
        with self.assertRaisesRegex(make_run.BuildError, r'never references \$\{oc\.env:NVC_SIM_BASE_URL\}'):
            make_run.classify_env_refs(HUB_SERVICES, 1)

    def test_unknown_env_name_refused(self):
        with self.assertRaisesRegex(make_run.BuildError, 'knows no source'):
            make_run.classify_env_refs('"x": "${oc.env:SOMETHING_ELSE}"', 1)

    def test_localsim_yaml_only_changes_sim_gptoss(self):
        hub, local = json.loads(HUB_SERVICES), json.loads(LOCALSIM_SERVICES)
        self.assertEqual({k: v for k, v in hub.items() if k != 'sim_gptoss'},
                         {k: v for k, v in local.items() if k != 'sim_gptoss'})
        block = local['sim_gptoss']['responses_api_models']['openai_model']
        self.assertEqual(block['openai_base_url'], '${oc.env:NVC_SIM_BASE_URL}')
        self.assertEqual(block['openai_api_key'], '${oc.env:NVC_SIM_API_KEY}')
        self.assertEqual((block['openai_model'], block['max_concurrent_requests'], block['num_workers']), ('gpt-oss-120b', 768, 1))
        self.assertEqual(LOCALSIM_SERVICES, json.dumps(local, indent=2) + '\n', 'JSON-bodied style must match the hub file')
        self.assertEqual(HUB_SERVICES, json.dumps(hub, indent=2) + '\n')
        for name, services in (('hub', hub), ('localsim', local)):
            with self.subTest(services=name):
                self.assertNotIn('num_groups_nemo_rl', services['policy_model']['responses_api_models']['vllm_model'])
                tau2 = services['tau2']['responses_api_agents']['tau2']
                self.assertEqual(tau2['tool_not_found_substring'], "error: tool '")
                needle = tau2['tool_not_found_substring'].lower()
                self.assertIn(needle, "Error: Tool 'check_network_status' not found.".lower())
                self.assertNotIn(needle, 'Customer with phone number 555-0100 not found'.lower())
                self.assertNotIn(needle, "Error: Tool(s) not found: ['x']".lower())


class GymValidationGuard(unittest.TestCase):
    def cfg(self, period=10, at_start=True, gym=True):
        return {'env': {'should_use_nemo_gym': gym},
                'grpo': {'val_period': period, 'val_at_start': at_start, 'val_at_end': False, 'num_generations_per_prompt': 16}}

    def test_train_build_forces_gym_validation_off(self):
        config, applied = self.cfg(), {}
        changed = make_run.apply_train_gym_validation_guard(config, applied)
        self.assertEqual(changed, ['grpo.val_period', 'grpo.val_at_start'])
        self.assertEqual((config['grpo']['val_period'], config['grpo']['val_at_start'], config['grpo']['val_at_end']), (0, False, False))
        self.assertEqual(applied['grpo.val_period'], 0)
        self.assertIs(applied['grpo.val_at_start'], False)
        self.assertIn('gym_validation_guard', applied)

    def test_opt_out_and_no_ops(self):
        config, applied = self.cfg(), {}
        self.assertEqual(make_run.apply_train_gym_validation_guard(config, applied, keep=True), [])
        self.assertEqual((config['grpo']['val_period'], applied), (10, {}))
        config, applied = self.cfg(period=0, at_start=False), {}
        self.assertEqual(make_run.apply_train_gym_validation_guard(config, applied), [])
        self.assertEqual(applied, {})
        config, applied = self.cfg(gym=False), {}
        self.assertEqual(make_run.apply_train_gym_validation_guard(config, applied), [])
        self.assertEqual(config['grpo']['val_period'], 10)

    def test_cli_flag_exists(self):
        text = (SCRIPTS / 'make_run.py').read_text()
        self.assertIn("'--keep-gym-validation'", text)
        self.assertIn("apply_train_gym_validation_guard(config, applied, keep=args.keep_gym_validation)", text)


class SimSpec(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name) / 'gpt-oss-120b'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{"model_type": "gpt_oss"}\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_zero_sim_nodes_means_nothing_new(self):
        self.assertIsNone(make_run.sim_spec_from_args(sim_args(sim_nodes=0)))

    def test_defaults_are_recorded(self):
        spec = make_run.sim_spec_from_args(sim_args(sim_model_path=str(self.model)))
        self.assertEqual(spec['model_path'], str(self.model.resolve()))
        self.assertEqual((spec['served_name'], spec['port'], spec['tp'], spec['dp']), ('gpt-oss-120b', 18100, 1, 8))
        # argparse defaults must agree with the measured serving layout (SERVE_TEST.md: TP1 x DP8 on 8x A100).
        cli = vars(make_run.build_parser().parse_args(['--kind', 'smoke']))
        self.assertEqual({k: v for k, v in cli.items() if k.startswith('sim_')},
                         {'sim_nodes': 0, 'sim_served_name': 'gpt-oss-120b', 'sim_port': 18100, 'sim_tp': 1,
                                               'sim_dp': 8, 'sim_max_model_len': 32768, 'sim_gpu_memory_utilization': 0.9,
                                               'sim_max_num_seqs': 128, 'sim_extra_args': make_run.DEFAULT_SIM_EXTRA_ARGS,
                                               'sim_python': make_run.DEFAULT_SIM_PYTHON, 'sim_model_path': None})
        self.assertEqual((spec['max_model_len'], spec['gpu_memory_utilization'], spec['max_num_seqs']), (32768, 0.9, 128))
        self.assertEqual(spec['extra_args'], ['--enable-prefix-caching', '--enable-auto-tool-choice',
                                              '--tool-call-parser', 'openai', '--reasoning-parser', 'openai_gptoss'])
        self.assertEqual(spec['python'], sys.executable)
        self.assertEqual(spec['api_key'], 'nvc-local-sim')
        self.assertEqual(make_run.DEFAULT_SIM_PYTHON, '/opt/nemotronvc/venvs/vllm027/bin/python')

    def test_refusals(self):
        cases = {
            'model path required': sim_args(),
            'config.json required': sim_args(sim_model_path=self.tmp.name),
            'tp x dp within one node': sim_args(sim_model_path=str(self.model), sim_tp=4, sim_dp=4),
            'port range': sim_args(sim_model_path=str(self.model), sim_port=80),
            'interpreter must exist': sim_args(sim_model_path=str(self.model), sim_python='/nonexistent/python'),
            'unsafe extra arg': sim_args(sim_model_path=str(self.model), sim_extra_args='--x "$(rm -rf /)"'),
        }
        for reason, args in cases.items():
            with self.subTest(reason=reason), self.assertRaises(make_run.BuildError):
                make_run.sim_spec_from_args(args)
        with self.assertRaises(FileNotFoundError):
            make_run.sim_spec_from_args(sim_args(sim_model_path=self.tmp.name + '/missing'))


class SmokeGateWarning(unittest.TestCase):
    def test_hub_smoke_for_localsim_train_warns_but_is_allowed(self):
        for contract in ({}, {'sim_nodes': 0}, {'sim_nodes': None}):
            with self.subTest(contract=contract):
                warning = make_run.smoke_sim_warning(1, contract, '/runs/smoke-x')
                self.assertIn('did not exercise the simulator node', warning)
                self.assertIn('/runs/smoke-x', warning)

    def test_matching_or_hub_train_is_silent(self):
        self.assertIsNone(make_run.smoke_sim_warning(1, {'sim_nodes': 1}, 'x'))
        self.assertIsNone(make_run.smoke_sim_warning(0, {}, 'x'))


class SimServiceBindings(unittest.TestCase):
    def test_localsim_yaml_binds_sim_gptoss_to_the_served_name(self):
        bindings = make_run.sim_service_bindings(json.loads(LOCALSIM_SERVICES), 'gpt-oss-120b')
        self.assertEqual([(b['server'], b['impl'], b['openai_model']) for b in bindings], [('sim_gptoss', 'openai_model', 'gpt-oss-120b')])
        self.assertEqual(bindings[0]['openai_base_url'], '${oc.env:NVC_SIM_BASE_URL}')

    def test_served_name_mismatch_refused(self):
        with self.assertRaisesRegex(make_run.BuildError, r"openai_model 'gpt-oss-120b' .* --sim-served-name is 'other'"):
            make_run.sim_service_bindings(json.loads(LOCALSIM_SERVICES), 'other')

    def test_hub_yaml_has_no_simulator_binding(self):
        with self.assertRaisesRegex(make_run.BuildError, 'No responses_api_models entry uses openai_base_url'):
            make_run.sim_service_bindings(json.loads(HUB_SERVICES), 'gpt-oss-120b')

    def test_every_simulator_bound_entry_is_checked(self):
        services = json.loads(LOCALSIM_SERVICES)
        services['second'] = {'responses_api_models': {'openai_model': {
            'openai_base_url': '${oc.env:NVC_SIM_BASE_URL}', 'openai_model': 'wrong'}}}
        with self.assertRaisesRegex(make_run.BuildError, 'second.responses_api_models.openai_model'):
            make_run.sim_service_bindings(services, 'gpt-oss-120b')


class EpochBudget(unittest.TestCase):
    def test_formula_and_record(self):
        budget = make_run.epoch_budget(41027, 3, 256, 120, 2, 0.6)
        self.assertEqual((budget['required_prompt_rows'], budget['available_prompt_rows']), (78080, 123081))
        self.assertEqual(budget['max_zero_variance_drop'], 0.6)
        self.assertEqual(make_run.DEFAULT_MAX_ZERO_VARIANCE_DROP, 0.6)
        self.assertEqual(make_run.build_parser().parse_args(['--kind', 'smoke']).max_zero_variance_drop, 0.6)

    def test_insufficient_budget_refused(self):
        with self.assertRaisesRegex(make_run.BuildError, 'cannot supply'):
            make_run.epoch_budget(41027, 1, 256, 120, 2, 0.6)       # 41027 < 78080
        make_run.epoch_budget(41027, 1, 256, 120, 2, 0.0)           # 41027 >= 31232 without drop
        with self.assertRaisesRegex(make_run.BuildError, 'cannot supply'):
            make_run.epoch_budget(31231, 1, 256, 120, 2, 0.0)

    def test_invalid_inputs_refused(self):
        for args in ((100, 0, 16, 4, 2, 0.6), (100, 1, 16, 4, 2, 1.0), (100, 1, 16, 4, 2, -0.1), (100, '1', 16, 4, 2, 0.6)):
            with self.subTest(args=args), self.assertRaises(make_run.BuildError):
                make_run.epoch_budget(*args)


class SmokeGateServicesDrift(unittest.TestCase):
    SERVICES = ['a' * 64]
    SIM = {'served_name': 'gpt-oss-120b', 'model_path': '/models/sim', 'port': 18100, 'tp': 1}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        runs = Path(self.tmp.name) / 'runs'
        self.run = runs / 'train-v1-x'
        self.smoke = runs / 'smoke-v1-x'
        for d in (self.run, self.smoke):
            d.mkdir(parents=True)
        (self.smoke / 'JOB_ID').write_text('9001\n')
        (self.smoke / 'END_CODE').write_text('0\n')
        (self.smoke / 'SMOKE_PASS.json').write_text('{"resume_step": 4}\n')
        (self.smoke / 'source.sha256').write_text('b' * 64 + '  source.tar.gz\n')
        self.images = {'runtime': {'path': '/img/runtime.sqsh'}, 'sandbox': {'path': '/img/sandbox.sqsh'}}

    def tearDown(self):
        self.tmp.cleanup()

    def contract(self, **extra):
        (self.smoke / 'RUN_CONTRACT.json').write_text(json.dumps({
            'runtime_images': {k: {'path': v['path']} for k, v in self.images.items()}, **extra}))

    def gate(self, **kwargs):
        defaults = dict(our_services_sha=self.SERVICES, our_sim_spec=self.SIM, train_sim_nodes=1)
        defaults.update(kwargs)
        return make_run.smoke_gate(self.run, self.smoke, 'b' * 64, self.images, False, **defaults)

    def test_matching_services_and_sim_spec_pass_and_are_recorded(self):
        self.contract(services_sha256=self.SERVICES, sim_nodes=1, sim_spec=dict(self.SIM, tp=2))
        proof = self.gate()
        self.assertEqual((proof['services_match'], proof['sim_spec_match'], proof['services_drift'], proof['allow_services_drift']),
                         (True, True, False, False))
        self.assertEqual(proof['sim_spec_compared']['port'], {'smoke': 18100, 'train': 18100})
        self.assertEqual(json.loads((self.run / 'GYM_SMOKE_ACCEPTANCE.json').read_text())['services_match'], True)
        self.assertEqual((self.run / 'GYM_SMOKE_JOB').read_text().strip(), '9001')

    def test_services_sha_mismatch_refused_without_flag(self):
        self.contract(services_sha256=['c' * 64], sim_nodes=1, sim_spec=self.SIM)
        with self.assertRaisesRegex(make_run.BuildError, 'services_match=False.*--allow-services-drift'):
            self.gate()
        self.assertFalse((self.run / 'GYM_SMOKE_ACCEPTANCE.json').exists())
        proof = self.gate(allow_services_drift=True)
        self.assertEqual((proof['services_match'], proof['services_drift'], proof['allow_services_drift']), (False, True, True))
        self.assertEqual(proof['smoke_services_sha256'], ['c' * 64])

    def test_sim_spec_mismatch_refused_without_flag(self):
        for field, value in (('served_name', 'other'), ('model_path', '/models/other'), ('port', 18200)):
            with self.subTest(field=field):
                self.contract(services_sha256=self.SERVICES, sim_nodes=1, sim_spec=dict(self.SIM, **{field: value}))
                with self.assertRaisesRegex(make_run.BuildError, 'sim_spec_match=False'):
                    self.gate()
                proof = self.gate(allow_services_drift=True)
                self.assertEqual(proof['sim_spec_compared'][field], {'smoke': value, 'train': self.SIM[field]})
                (self.run / 'GYM_SMOKE_ACCEPTANCE.json').unlink()
                (self.run / 'GYM_SMOKE_JOB').unlink()

    def test_hub_smoke_for_localsim_train_needs_the_flag(self):
        self.contract(services_sha256=['c' * 64])
        with self.assertRaises(make_run.BuildError):
            self.gate()
        proof = self.gate(allow_services_drift=True)
        self.assertEqual(proof['sim_spec_match'], False)
        self.assertIn('did not exercise the simulator node', proof['simulator_node']['warning'])

    def test_hub_train_without_sim_spec_only_compares_services(self):
        self.contract(services_sha256=self.SERVICES)
        proof = self.gate(our_sim_spec=None, train_sim_nodes=0)
        self.assertEqual((proof['services_match'], proof['sim_spec_match'], proof['sim_spec_compared']), (True, None, None))


class LauncherScripts(unittest.TestCase):
    ray_role = (SCRIPTS / 'ray_role.sh').read_text()
    gpu_job = (SCRIPTS / 'gpu_job.sh').read_text()
    sim_role = (SCRIPTS / 'sim_role.sh').read_text()

    def test_shell_syntax(self):
        for name in make_run.RUNTIME_SCRIPTS:
            with self.subTest(script=name):
                subprocess.run(['bash', '-n', str(SCRIPTS / name)], check=True)
        self.assertIn('sim_role.sh', make_run.RUNTIME_SCRIPTS)

    def test_ray_launcher_contract_still_passes(self):
        self.assertEqual(ray_ports.validate_launcher(self.ray_role)['status'], 'PASS')

    def test_sim_branch_precedes_every_ray_step(self):
        branch = self.ray_role.index('exec bash "$run/scripts/sim_role.sh"')
        self.assertLess(branch, self.ray_role.index('"$ray" start'))
        self.assertLess(branch, self.ray_role.index('trap cleanup EXIT'))
        self.assertIn('if (( rank >= ${NVC_RAY_NODES:-$expected_nodes} )); then', self.ray_role)
        self.assertIn('export NVC_SIM_ROLE_RANK=$rank NVC_SIM_ROLE_JOB=$job', self.ray_role)
        self.assertIn('os.environ.get("NVC_RAY_NODES") or os.environ["NVC_EXPECTED_NODES"]', self.ray_role)
        wait = self.ray_role.index('READY-$sim_host.json')
        self.assertLess(self.ray_role.index('[SUPER35_RAY_READY]'), wait)
        self.assertLess(wait, self.ray_role.index('examples/run_grpo.py'))
        self.assertIn("printf '75\\n' > \"$run/END_CODE.tmp\"", self.ray_role)

    def test_head_reads_only_job_scoped_receipts_and_validates_job_and_port(self):
        head = self.ray_role[self.ray_role.index('if (( rank == 0 )); then'):]
        self.assertIn('sim_dir="$run/sim/$job"', head)
        self.assertIn('[[ -s "$sim_dir/HOSTS" ]]', head)
        self.assertIn('done < "$sim_dir/HOSTS"', head)
        self.assertIn('compgen -G "$sim_dir/FAILED-*.json"', head)
        self.assertNotIn('"$run/sim/HOSTS"', head)
        self.assertNotIn('"$run/sim/READY-', head)
        self.assertIn('json.load(open(sys.argv[1]))["port"]', head)
        validator = head[head.index('sim_receipt_state() {'):head.index('sim_deadline=$((SECONDS + 900))')]
        self.assertIn('"$python" - "$1" "$job" "$sim_port" <<\'PY\'', validator)
        self.assertIn("str(receipt.get('job_id')) == job", validator)
        self.assertIn("receipt.get('port') == int(port)", validator)
        self.assertIn("receipt.get('status') == 'ready'", validator)
        self.assertIn("*) sim_ready=false; sim_mismatch=", head)
        self.assertIn("[[ -n \"$sim_mismatch\" ]] || compgen", head)
        self.assertLess(head.index("[NVC_SIM_READY] job=%s port=%s"), head.index('examples/run_grpo.py'))

    def test_sim_receipt_validator_accepts_only_this_job_and_port(self):
        head = self.ray_role[self.ray_role.index('sim_receipt_state() {'):]
        body = head[head.index("<<'PY'\n") + len("<<'PY'\n"):head.index('\nPY\n')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'READY-h.json'
            def state(record):
                path.write_text(json.dumps(record) + '\n' if record is not None else '')
                return subprocess.run([sys.executable, '-c', body, str(path), '4242', '18100'],
                                      capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(subprocess.run([sys.executable, '-c', body, str(Path(tmp) / 'missing.json'), '4242', '18100'],
                                            capture_output=True, text=True, check=True).stdout.strip(), 'pending')
            self.assertEqual(state(None), 'pending')
            self.assertEqual(state({'status': 'ready', 'job_id': '4242', 'port': 18100}), 'ready')
            self.assertEqual(state({'status': 'ready', 'job_id': '4241', 'port': 18100}), 'mismatch')
            self.assertEqual(state({'status': 'ready', 'job_id': '4242', 'port': 18101}), 'mismatch')
            self.assertEqual(state({'status': 'process_exited_during_startup', 'job_id': '4242', 'port': 18100}), 'mismatch')
            path.write_text('{not json')
            self.assertEqual(subprocess.run([sys.executable, '-c', body, str(path), '4242', '18100'],
                                            capture_output=True, text=True, check=True).stdout.strip(), 'mismatch')

    def test_launcher_env_survives_ray_role_cleanup(self):
        match = re.search(r"awk -F= '/(\^\([A-Z|]+\)_)/", self.ray_role)
        self.assertIsNotNone(match, 'env cleanup regex changed; review this test')
        pattern = re.compile(match.group(1))
        for name in ('NVC_SIM_BASE_URL', 'NVC_SIM_API_KEY', 'NVC_SIM_NODES', 'NVC_RAY_NODES', 'NVC_EXPECTED_NODES', 'SLURMD_NODENAME'):
            self.assertIsNone(pattern.search(name), name)
        for name in ('SLURM_PROCID', 'PMIX_RANK', 'OMPI_COMM_WORLD_RANK'):
            self.assertIsNotNone(pattern.search(name), name)
        explicit = re.search(r'^unset (NVIDIA_API_KEY[^\n]*)$', self.ray_role, re.M).group(1).split()
        self.assertTrue(set(explicit).isdisjoint(make_run.LAUNCHER_ENV_SOURCES))

    def test_gpu_job_exports_launcher_env_from_last_hosts(self):
        for needle in ('export NVC_SIM_NODES=$(cat "$run/SIM_NODES" 2>/dev/null || printf 0)',
                       'export NVC_RAY_NODES=$((NVC_EXPECTED_NODES - NVC_SIM_NODES))',
                       'scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n "$NVC_SIM_NODES"',
                       "export NVC_SIM_BASE_URL=$(IFS=,; printf '%s' \"${sim_urls[*]}\")",
                       'export NVC_SIM_API_KEY=nvc-local-sim', 'sim_job_dir="$run/sim/$SLURM_JOB_ID"',
                       '[[ ! -e "$sim_job_dir" && ! -L "$sim_job_dir" ]] || {', 'exit 75; }', '> "$sim_job_dir/HOSTS"'):
            self.assertIn(needle, self.gpu_job)
        self.assertNotIn('"$run/sim/HOSTS"', self.gpu_job)
        self.assertLess(self.gpu_job.index('[[ ! -e "$sim_job_dir"'), self.gpu_job.index('mkdir -p "$sim_job_dir"'))
        # Secrets stay a closed list: launcher env names are never accepted through SECRETS_REQUIRED.
        self.assertIn('*) echo "Unknown secret requested by run: $secret_name" >&2; exit 2;;', self.gpu_job)

    def test_sim_role_never_uses_global_cleanup_or_prints_env(self):
        tokens = {t.rsplit('/', 1)[-1] for line in self.sim_role.splitlines() if not line.lstrip().startswith('#')
                  for t in line.split()}
        self.assertTrue(tokens.isdisjoint({'pkill', 'killall', 'printenv', 'env'}))
        self.assertNotIn('set -x', self.sim_role)
        self.assertIn('-m vllm.entrypoints.openai.api_server', self.sim_role)
        self.assertIn('export VLLM_LOGGING_LEVEL=INFO', self.sim_role)
        self.assertIn('job=${NVC_SIM_ROLE_JOB:?}', self.sim_role)
        self.assertIn('sim_dir="$run/sim/$job"', self.sim_role)


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


FAKE_SERVER = r'''
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index('--port') + 1])
served = sys.argv[sys.argv.index('--served-model-name') + 1]
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({'object': 'list', 'data': [{'id': served, 'object': 'model'}]}).encode()
        self.send_response(200 if self.path == '/v1/models' else 404)
        self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
HTTPServer(('0.0.0.0', port), H).serve_forever()
'''


class SimRoleEndToEnd(unittest.TestCase):
    """sim_role.sh with a fake interpreter: READY receipt, END_CODE follow-through, own process group only."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.run = root / 'run'
        self.local = root / 'local'
        self.model = root / 'model'
        for d in (self.run, self.local, self.model):
            d.mkdir()
        (self.model / 'config.json').write_text('{}\n')
        (root / 'fake_server.py').write_text(FAKE_SERVER)
        self.fake_python = root / 'fake-python'
        self.fake_python.write_text('#!/bin/bash\n# fake vllm interpreter: ignore "-m module", serve /v1/models\n'
                                    f'exec {sys.executable} {root}/fake_server.py "$@"\n')
        self.fake_python.chmod(0o755)
        self.dead_python = root / 'dead-python'
        self.dead_python.write_text('#!/bin/bash\necho "fake vllm: refusing to start" >&2\nexit 3\n')
        self.dead_python.chmod(0o755)
        self.port = free_port()
        self.env = dict(os.environ, NVC_DEBUG_RUN=str(self.run), NVC_DEBUG_LOCAL=str(self.local), SLURMD_NODENAME='simhost',
                        NVC_SIM_ROLE_JOB='4242')
        self.sim_dir = self.run / 'sim/4242'

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self, python):
        (self.run / 'SIM_SPEC.json').write_text(json.dumps({
            'model_path': str(self.model), 'served_name': 'gpt-oss-120b', 'port': self.port, 'tp': 2, 'dp': 4,
            'max_model_len': 32768, 'gpu_memory_utilization': 0.9, 'max_num_seqs': 128,
            'extra_args': ['--enable-prefix-caching'], 'python': str(python)}))

    def wait_for(self, path, seconds=40):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if path.is_file() and path.read_text().strip():
                return json.loads(path.read_text())
            time.sleep(0.2)
        self.fail('timed out waiting for ' + str(path))

    def test_ready_then_end_code_and_group_teardown(self):
        self.spec(self.fake_python)
        proc = subprocess.Popen(['bash', str(SCRIPTS / 'sim_role.sh')], env=self.env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        try:
            ready = self.wait_for(self.sim_dir / 'READY-simhost.json')
            self.assertEqual((ready['status'], ready['host'], ready['port']), ('ready', 'simhost', self.port))
            self.assertEqual((ready['job_id'], ready['served_name'], ready['schema']), ('4242', 'gpt-oss-120b', 'nemotronvc.sim_role.v2'))
            self.assertEqual(ready['models']['data'][0]['id'], 'gpt-oss-120b')
            self.assertFalse((self.run / 'sim/READY-simhost.json').exists(), 'receipts must be job-scoped')
            pid = ready['pid']
            os.kill(pid, 0)
            self.assertEqual(os.getpgid(pid), pid, 'server must lead its own process group (set -m)')
            self.assertFalse((self.sim_dir / 'FAILED-simhost.json').exists())
            (self.run / 'END_CODE').write_text('0\n')
            out, _ = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
        self.assertEqual(proc.returncode, 0, out)
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            self.fail('fake vLLM survived sim_role.sh exit')
        self.assertIn('[SIM_ROLE_READY] host=simhost', out)
        self.assertTrue((self.sim_dir / 'vllm-simhost.log').exists())
        for var in ('NVC_DEBUG_RUN=', 'PATH='):
            self.assertNotIn(var, out, 'sim_role.sh must not print environment variables')

    def test_nonzero_end_code_is_propagated(self):
        self.spec(self.fake_python)
        proc = subprocess.Popen(['bash', str(SCRIPTS / 'sim_role.sh')], env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        try:
            self.wait_for(self.sim_dir / 'READY-simhost.json')
            (self.run / 'END_CODE').write_text('75\n')
            proc.wait(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
        self.assertEqual(proc.returncode, 75)

    def test_dead_server_writes_failed_and_exits_1(self):
        self.spec(self.dead_python)
        result = subprocess.run(['bash', str(SCRIPTS / 'sim_role.sh')], env=self.env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        failed = json.loads((self.sim_dir / 'FAILED-simhost.json').read_text())
        self.assertEqual((failed['status'], failed['job_id'], failed['port']), ('process_exited_during_startup', '4242', self.port))
        self.assertFalse((self.sim_dir / 'READY-simhost.json').exists())
        self.assertIn('refusing to start', (self.sim_dir / 'vllm-simhost.log').read_text())

    def test_missing_or_non_numeric_job_id_refused_before_serving(self):
        self.spec(self.fake_python)
        for env in (dict(self.env, NVC_SIM_ROLE_JOB='abc'), {k: v for k, v in self.env.items() if k != 'NVC_SIM_ROLE_JOB'}):
            with self.subTest(job=env.get('NVC_SIM_ROLE_JOB')):
                result = subprocess.run(['bash', str(SCRIPTS / 'sim_role.sh')], env=env, capture_output=True, text=True, timeout=60)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.run / 'sim').exists(), 'no receipt directory without a job id')

    def test_term_signal_tears_down_server(self):
        self.spec(self.fake_python)
        proc = subprocess.Popen(['bash', str(SCRIPTS / 'sim_role.sh')], env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        try:
            ready = self.wait_for(self.sim_dir / 'READY-simhost.json')
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
        self.assertEqual(proc.returncode, 143)
        for _ in range(100):
            try:
                os.kill(ready['pid'], 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            self.fail('fake vLLM survived SIGTERM to sim_role.sh')


if __name__ == '__main__':
    unittest.main()
