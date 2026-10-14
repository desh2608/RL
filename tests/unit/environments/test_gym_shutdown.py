import ast
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

# tests/unit/environments -> repository root; no candidate-tree dependency.
ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('gym_shutdown', ROOT / 'nemo_rl/environments/gym_shutdown.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Helper:
    def __init__(self, process, behavior='clean', thread=None):
        self._processes = {'owned': process}
        self._head_server_thread = thread
        self.behavior = behavior

    def shutdown(self):
        if self.behavior == 'error':
            raise RuntimeError('native failure')
        if self.behavior == 'clean':
            for child in self._processes.values():
                child.terminate()
                child.wait(timeout=5)
            self._processes = {}


class ShutdownTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        (self.run / 'JOB_ID').write_text('123')
        self.children = []
        self.guards = []
        self.port_probe = socket.socket()
        self.port_probe.bind(('127.0.0.1', 0))
        self.endpoint = {'alias': 'owned', 'host': '127.0.0.1', 'port': self.port_probe.getsockname()[1]}
        self.port_probe.close()

    def tearDown(self):
        for guard in self.guards:
            guard.stop.set()
            guard.thread.join(timeout=2)
        for child in self.children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
        self.tmp.cleanup()

    def child(self, script='import time;time.sleep(30)'):
        child = subprocess.Popen([sys.executable, '-c', script])
        self.children.append(child)
        return child

    def guard(self, helper=None):
        if helper is None:
            helper = Helper(self.child())
        result = module.GymShutdownGuard(helper, run=self.run, job='123', phase='train', endpoints=[self.endpoint])
        self.guards.append(result)
        return result

    def test_actual_process_shutdown_then_pre_resume_gate(self):
        guard = self.guard()
        result = guard.shutdown(wait_seconds=.1)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(module.validate_shutdown_receipt(self.run, '123', 'train'), result)
        self.assertEqual(guard.shutdown(), result)

    def test_native_failure_receipt_preserved_and_not_swallowed(self):
        guard = self.guard(Helper(self.child(), behavior='error'))
        with self.assertRaises(RuntimeError):
            guard.shutdown(wait_seconds=.1)
        record = json.loads(guard.path.read_text())
        self.assertEqual(record['status'], 'FAIL')
        self.assertEqual(record['native_error_type'], 'RuntimeError')
        self.assertTrue(record['remaining_processes'])
        with self.assertRaises(RuntimeError):
            module.validate_shutdown_receipt(self.run, '123', 'train')

    def test_live_child_is_not_killed_to_make_gate_pass(self):
        child = self.child()
        guard = self.guard(Helper(child, behavior='leak'))
        with self.assertRaises(RuntimeError):
            guard.shutdown(wait_seconds=.1)
        self.assertIsNone(child.poll())
        self.assertEqual(json.loads(guard.path.read_text())['remaining_processes'][0]['pid'], child.pid)

    def test_owned_recursive_child_is_observed(self):
        child_file = self.run / 'childpid'
        script = f"import subprocess,sys,time,pathlib; c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);pathlib.Path({str(child_file)!r}).write_text(str(c.pid));c.wait()"
        parent = self.child(script)
        until = time.monotonic()+5
        while not child_file.exists():
            if time.monotonic()>until: self.fail('child did not start')
            time.sleep(.02)
        grandchild = module.psutil.Process(int(child_file.read_text()))
        guard = self.guard(Helper(parent, behavior='leak'))
        try:
            with self.assertRaises(RuntimeError):guard.shutdown(wait_seconds=.1)
            pids = {entry['pid'] for entry in json.loads(guard.path.read_text())['remaining_processes']}
            self.assertIn(grandchild.pid,pids)
            self.assertTrue(grandchild.is_running())
        finally:
            grandchild.terminate()
            parent.wait(timeout=5)

    def test_open_endpoint_blocks_even_after_owned_process_exits(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', self.endpoint['port']))
        listener.listen()
        try:
            guard = self.guard()
            with self.assertRaises(RuntimeError):guard.shutdown(wait_seconds=.1)
            result=json.loads(guard.path.read_text())
            self.assertEqual(result['remaining_processes'],[])
            self.assertTrue(result['open_endpoints'])
        finally:listener.close()

    def test_alive_native_head_thread_blocks(self):
        stop=threading.Event();thread=threading.Thread(target=stop.wait);thread.start()
        try:
            guard=self.guard(Helper(self.child(),thread=thread))
            with self.assertRaises(RuntimeError):guard.shutdown(wait_seconds=.1)
            self.assertTrue(json.loads(guard.path.read_text())['head_thread_alive'])
        finally:stop.set();thread.join(timeout=2)

    def test_wrong_run_job_phase_and_missing_receipt_rejected(self):
        guard=self.guard();guard.shutdown(wait_seconds=.1)
        with self.assertRaises(ValueError):module.validate_shutdown_receipt(self.run,'999','train')
        with self.assertRaises(FileNotFoundError):module.validate_shutdown_receipt(self.run,'123','train-resume')
        record=json.loads(guard.path.read_text());record['run']='/foreign';guard.path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):module.validate_shutdown_receipt(self.run,'123','train')

    def test_phase_environment_uses_persisted_job_not_slurm(self):
        with patch.dict(os.environ, {'NVC_DEBUG_RUN':str(self.run),'NVC_GYM_DRIVER_PHASE':'train'}, clear=True):
            self.assertEqual(module.identity_from_environment(),(self.run.resolve(),'123','train'))

    def test_pid_reuse_does_not_claim_unrelated_process(self):
        current=module.psutil.Process()
        self.assertEqual(module.remaining_identities([{'pid':current.pid,'create_time':current.create_time()-10,'alias':'old'}]),[])

    def test_upstream_model_and_sandbox_urls_are_excluded(self):
        helper=Helper(self.child())
        helper._server_client=types.SimpleNamespace(global_config_dict={'owned':{'responses_api_models':{'model':{'host':'127.0.0.1','port':12345,'base_url':'https://outside.invalid/v1','sandbox_port':9999}}}})
        actual=module.owned_endpoints(helper,'127.0.0.1',12346)
        self.assertEqual([x['port'] for x in actual],[12345,12346])

    def test_opt_in_strict_bool_and_legacy_default(self):
        self.assertFalse(module.extract_strict_shutdown({}))
        x={'strict_shutdown':True,'original':'preserved'}
        self.assertTrue(module.extract_strict_shutdown(x));self.assertEqual(x,{'original':'preserved'})
        with self.assertRaises(ValueError):module.extract_strict_shutdown({'strict_shutdown':'false'})

    def test_actual_grpo_helper_propagates_strict_errors_only(self):
        source=ast.parse((ROOT/'nemo_rl/algorithms/grpo.py').read_text())
        fn=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='shutdown_environments')
        fn.returns=None
        for arg in [*fn.args.args,*fn.args.kwonlyargs]:arg.annotation=None
        class FakeRay:
            calls=[]
            def get(self,value,timeout):
                self.calls.append(timeout);raise RuntimeError('native failed')
            def kill(self,env):pass
        ray=FakeRay();env=types.SimpleNamespace(shutdown=types.SimpleNamespace(remote=lambda:None))
        ns={'ray':ray};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_grpo_helper','exec'),ns)
        ns['shutdown_environments']({'gym':env},None)
        self.assertEqual(ray.calls, [10])
        with self.assertRaises(RuntimeError):ns['shutdown_environments']({'gym':env},None,strict_shutdown=True)
        self.assertEqual(ray.calls, [10,240])

    def test_gpu_release_remains_in_finally_on_strict_failure(self):
        source=ast.parse((ROOT/'examples/run_grpo.py').read_text())
        finally_nodes=[node for node in ast.walk(source) if isinstance(node,ast.Try) and node.finalbody]
        selected=next(node for node in finally_nodes if any(isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id=='shutdown_environments' for item in node.body for c in ast.walk(item)) and any(isinstance(c,ast.Attribute) and c.attr=='shutdown' for item in node.finalbody for c in ast.walk(item)))
        class Config:env={'nemo_gym':{'strict_shutdown':True}}
        calls=[]
        def fail(*args,**kwargs):raise RuntimeError('strict fail')
        ns={'shutdown_environments':fail,'task_to_env':{},'val_task_to_env':{},'config':Config(),'policy_generation':types.SimpleNamespace(shutdown=lambda:calls.append('released'))}
        with self.assertRaises(RuntimeError):exec(compile(ast.Module(body=[selected],type_ignores=[]),'actual_driver_finally','exec'),ns)
        self.assertEqual(calls,['released'])

    def test_actual_early_stop_releases_separate_learner_on_each_failure(self):
        tree=ast.parse((ROOT/'nemo_rl/algorithms/grpo.py').read_text())
        stop=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='stop_at_safe_boundary')
        cleanup=next(node for node in stop.body if isinstance(node,ast.Try))
        for failing_stage, same_policy in [('environment',False),('generation',False),(None,False),(None,True)]:
            with self.subTest(failing_stage=failing_stage,same_policy=same_policy):
                calls=[]
                def environment(*args,**kwargs):
                    calls.append('environment')
                    if failing_stage=='environment':raise RuntimeError('strict error')
                def generation():
                    calls.append('generation')
                    if failing_stage=='generation':raise RuntimeError('generation error')
                gen=types.SimpleNamespace(shutdown=generation)
                learner=gen if same_policy else types.SimpleNamespace(shutdown=lambda:calls.append('learner'))
                ns={'shutdown_environments':environment,'task_to_env':{},'val_task_to_env':{},'master_config':types.SimpleNamespace(env={'nemo_gym':{'strict_shutdown':True}}),'policy_generation':gen,'policy':learner}
                code=compile(ast.Module(body=[cleanup],type_ignores=[]),'actual_early_stop_cleanup','exec')
                if failing_stage:
                    with self.assertRaises(RuntimeError):exec(code,ns)
                else:exec(code,ns)
                self.assertEqual(calls,['environment','generation']+([] if same_policy else ['learner']))


    def test_actual_cleanup_call_sites_preserve_explicit_null_legacy_env(self):
        calls=[]
        for filename in ('nemo_rl/algorithms/grpo.py','examples/run_grpo.py'):
            tree=ast.parse((ROOT/filename).read_text())
            nodes=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='shutdown_environments']
            self.assertEqual(len(nodes),2 if filename=='nemo_rl/algorithms/grpo.py' else 1)
            for node in nodes:
                ns={'task_to_env':{},'val_task_to_env':None,
                    'master_config':types.SimpleNamespace(env={'nemo_gym':None}),
                    'config':types.SimpleNamespace(env={'nemo_gym':None}),
                    'shutdown_environments':lambda *args,**kwargs:calls.append(kwargs['strict_shutdown'])}
                expression=ast.Expression(body=node)
                eval(compile(expression,'actual_null_legacy_cleanup_call','eval'),ns)
        self.assertEqual(calls,[False,False,False])


if __name__=='__main__':unittest.main(verbosity=2)
