import ast,ctypes,importlib.util,json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
D=Path(os.environ["GYM_FIX_SOURCE"]) if "GYM_FIX_SOURCE" in os.environ else next(p for p in Path(__file__).resolve().parents if (p/"nemo_gym/cli/owned_processes.py").is_file())
spec=importlib.util.spec_from_file_location('owned',D/'nemo_gym/cli/owned_processes.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def wait_file(path):
 deadline=time.monotonic()+8
 while not path.exists() and time.monotonic()<deadline:time.sleep(.02)
 assert path.exists(),path

def stop(p):
 if p.poll() is None:
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
 p.wait(timeout=5)

class OwnedTests(unittest.TestCase):
 def setUp(self):self.ps=[]
 def tearDown(self):
  for p in self.ps:stop(p)
 def launch(self,code,**kwargs):
  p=subprocess.Popen([sys.executable,'-c',code],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,**kwargs);self.ps.append(p);return p
 def test_shell_tee_python_group_and_unowned_survives(self):
  unrelated=self.launch('import time;time.sleep(60)')
  reg=m.OwnedProcessRegistry()
  with tempfile.TemporaryDirectory() as tmp:
   ready=Path(tmp)/'ready'
   code=f'import time;from pathlib import Path;Path({str(ready)!r}).touch();time.sleep(60)'
   import shlex
   p=subprocess.Popen(shlex.join([sys.executable,'-c',code])+' | tee /dev/null',shell=True,executable='/bin/bash',start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);self.ps.append(p);reg.register('pipeline',p)
   wait_file(ready);time.sleep(.15)
   report=reg.shutdown(.5,.5,1)
   self.assertEqual(report['status'],'PASS',report);self.assertGreaterEqual(len(report['registered']),3)
   self.assertIsNone(unrelated.poll());self.assertIsNotNone(p.poll())
 def test_ignored_sigint_term_escalates(self):
  reg=m.OwnedProcessRegistry()
  with tempfile.TemporaryDirectory() as tmp:
   ready=Path(tmp)/'ready';p=self.launch(f'import signal,time;from pathlib import Path;signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN);Path({str(ready)!r}).touch();time.sleep(60)');reg.register('stubborn',p);wait_file(ready)
   report=reg.shutdown(.1,.1,1)
   self.assertEqual(report['status'],'PASS',report);self.assertEqual([x['signal'] for x in report['phases']],['SIGINT','SIGTERM','SIGKILL'])
 def test_setsided_child_observed_and_closed(self):
  reg=m.OwnedProcessRegistry()
  with tempfile.TemporaryDirectory() as tmp:
   ready=Path(tmp)/'child-ready'
   child=f'import os,time;from pathlib import Path;os.setsid();Path({str(ready)!r}).touch();time.sleep(60)'
   code=f'''import signal,subprocess,sys,time
c=subprocess.Popen([sys.executable,'-c',{child!r}])
def stop(sig,frame):
 c.wait(timeout=3)
 sys.exit(0)
signal.signal(signal.SIGINT,stop)
time.sleep(60)
'''
   p=self.launch(code);reg.register('escape',p);wait_file(ready);time.sleep(.2)
   report=reg.shutdown(1,1,1)
   self.assertEqual(report['status'],'PASS',report);self.assertEqual(len(report['escaped_processes']),1)
 def test_no_signal_to_empty_historical_group(self):
  old=self.launch('pass');reg=m.OwnedProcessRegistry();reg.register('old',old);old.wait()
  live=self.launch('import time;time.sleep(60)');reg.register('current',live);seen=[];original=m.os.killpg
  try:
   def record(pgid,sig):seen.append(pgid);return original(pgid,sig)
   m.os.killpg=record;report=reg.shutdown(.5,.5,1)
  finally:m.os.killpg=original
  self.assertEqual(report['status'],'PASS',report);self.assertNotIn(old.pid,seen)
 def test_early_root_exit_is_reaped(self):
  reg=m.OwnedProcessRegistry();p=self.launch('raise SystemExit(3)');reg.register('early',p);time.sleep(.1)
  report=reg.shutdown(.1,.1,1);self.assertEqual(report['status'],'PASS',report);self.assertEqual(p.returncode,3)
 def test_reaped_before_register_is_explicit_failure(self):
  reg=m.OwnedProcessRegistry();p=self.launch('pass');p.wait()
  with self.assertRaises(RuntimeError):reg.register('gone',p)
  report=reg.shutdown(.1,.1,1);self.assertEqual(report['status'],'FAIL');self.assertEqual(report['registration_failures'][0]['pid'],p.pid)
 def test_startup_error_invokes_native_cleanup(self):
  tree=ast.parse((D/'nemo_gym/cli/env.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RunHelper')
  methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('start','shutdown')]
  for method in methods:
   method.returns=None
   for arg in method.args.args:arg.annotation=None
  ns={};exec(compile(ast.Module(body=methods,type_ignores=[]),'actual_start_shutdown','exec'),ns)
  from types import SimpleNamespace
  owner=self
  class X:
   def _start(self,arg):
    self._owned_process_registry=m.OwnedProcessRegistry()
    self.p=owner.launch('import time;time.sleep(60)')
    self._owned_process_registry.register('started_sibling',self.p)
    self._processes={'started_sibling':self.p}
    self._head_server=SimpleNamespace(should_exit=False)
    self._head_server_thread=SimpleNamespace(join=lambda **kw:None,is_alive=lambda:False)
    raise ValueError('constructor failed')
   shutdown=ns['shutdown']
  x=X()
  with self.assertRaisesRegex(ValueError,'constructor'):ns['start'](x,None)
  self.assertIsNone(x._head_server);self.assertIsNotNone(x.p.poll());self.assertEqual(x._last_shutdown_report['status'],'PASS')
 def test_unregistered_new_child_is_not_killed_or_hidden(self):
  reg=m.OwnedProcessRegistry();p=self.launch('import time;time.sleep(60)')
  report=reg.shutdown(.1,.1,.1)
  self.assertEqual(report['status'],'FAIL');self.assertEqual(report['unattributed_children'][0]['pid'],p.pid);self.assertIsNone(p.poll())
 def test_ctrl_c_requires_and_uses_explicit_owned_shutdown(self):
  with tempfile.TemporaryDirectory() as tmp:
   ready=Path(tmp)/'ready';result=Path(tmp)/'result'
   code=f'''import importlib.util,json,signal,subprocess,sys,time
from pathlib import Path
s=importlib.util.spec_from_file_location('owned',{str(D/'nemo_gym/cli/owned_processes.py')!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=m.OwnedProcessRegistry()
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
r.register('service',p)
def stop(sig,frame):
 alive=p.poll() is None
 report=r.shutdown(.5,.5,1)
 Path({str(result)!r}).write_text(json.dumps({{'child_alive_before_explicit_shutdown':alive,'report':report}}))
 sys.exit(0)
signal.signal(signal.SIGINT,stop)
Path({str(ready)!r}).touch()
time.sleep(60)
'''
   p=self.launch(code);wait_file(ready);os.killpg(p.pid,signal.SIGINT);p.wait(timeout=8)
   x=json.loads(result.read_text());self.assertEqual(p.returncode,0);self.assertTrue(x['child_alive_before_explicit_shutdown']);self.assertEqual(x['report']['status'],'PASS')
 def test_subreaper_setting_is_scoped(self):
  if sys.platform!='linux':self.skipTest('Linux prctl is exercised on actual CPU node')
  libc=ctypes.CDLL(None);old=ctypes.c_int();self.assertEqual(libc.prctl(37,ctypes.byref(old),0,0,0),0)
  reg=m.OwnedProcessRegistry();during=ctypes.c_int();libc.prctl(37,ctypes.byref(during),0,0,0);self.assertEqual(during.value,1)
  report=reg.shutdown();after=ctypes.c_int();libc.prctl(37,ctypes.byref(after),0,0,0);self.assertEqual(after.value,old.value);self.assertEqual(report['subreaper_before'],old.value)
 def test_cached_pid_birth_cannot_claim_unrelated_descendant(self):
  from types import SimpleNamespace
  from unittest.mock import patch
  reg=m.OwnedProcessRegistry();reg._stop.set();reg._thread.join()
  rows=[SimpleNamespace(info={'pid':42,'ppid':1,'create_time':1.}),SimpleNamespace(info={'pid':43,'ppid':42,'create_time':2.})]
  for parent_time,child_time,child_ppid,accepted in [(99.,2.,42,False),(1.,99.,42,False),(1.,2.,44,False),(1.,2.,42,True)]:
   reg._known={42:{'pid':42,'create_time':1.,'server':'ours'}}
   def fresh(pid):return SimpleNamespace(create_time=lambda: parent_time if pid==42 else child_time,ppid=lambda: 1 if pid==42 else child_ppid)
   with patch.object(m.psutil,'process_iter',return_value=rows),patch.object(m.psutil,'Process',side_effect=fresh):reg._refresh()
   self.assertEqual(43 in reg._known,accepted)
  reg._known={};self.assertEqual(reg.shutdown()['status'],'PASS')

if __name__=='__main__':
 if sys.platform=='linux':
  value=ctypes.c_int();assert ctypes.CDLL(None).prctl(37,ctypes.byref(value),0,0,0)==0
  print(json.dumps({'test_runner_subreaper_before':value.value}),flush=True)
  assert value.value==0,'This suite must prove native cleanup without pre-existing subreaper state'
 unittest.main()
