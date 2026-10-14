#!/usr/bin/env python3
"""Own one allocation's sandbox children; run inside the pinned sandbox image.

No native start-with-nginx.sh is invoked: its global nginx cleanup is unsuitable
for shared nodes. Native /app/main.py and the session-affinity nginx template are
used unchanged. A child crash is fatal, never silently restarted mid-session.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def reserve_ports(host, workers, attempts=256):
    """Keep sockets until callers are about to bind. Release/bind is still a race.

    The proxy gets an OS-selected port. A randomized contiguous worker block is
    tested as a whole; occupied blocks are skipped, never killed or reused.
    """
    proxy = socket.socket()
    proxy.bind((host, 0))
    for _ in range(attempts):
        base = random.SystemRandom().randrange(20000, 60000 - workers)
        held = []
        try:
            for port in range(base, base + workers):
                sock = socket.socket()
                held.append(sock)
                sock.bind(('127.0.0.1', port))
            return proxy.getsockname()[1], base, [proxy] + held
        except OSError:
            for sock in held:
                sock.close()
    proxy.close()
    raise RuntimeError('No free contiguous sandbox worker port block')


def worker_environment(environ, worker, workers, proxy, base):
    # uWSGI translates UWSGI_* environment variables into options. The base
    # image's UWSGI_INI/CHEAPER otherwise load a second ini and invalidate our
    # single-process workers (or create shared sockets across workers).
    env = {key: value for key, value in environ.items() if not key.startswith('UWSGI_')}
    env.update(WORKER_NUM=str(worker), NUM_WORKERS=str(workers), NGINX_PORT=str(proxy),
               SANDBOX_WORKER_BASE_PORT=str(base), SANDBOX_FORCE_SINGLE_NODE='1')
    return env


def request_json(host, port, path, payload=None, session=None, method=None):
    headers = {'Content-Type': 'application/json'}
    if session:
        headers['X-Session-ID'] = session
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f'http://{host}:{port}{path}', body, headers, method=method)
    with urllib.request.urlopen(req, timeout=15) as response:
        if response.status != 200:
            raise RuntimeError('Sandbox request did not return HTTP 200')
        return json.load(response)


def execution_canary(host, port, marker):
    # Two calls through nginx must retain the same IPython kernel. A health-only
    # check or round-robin proxy would not establish this contract.
    session = 'nvc-canary-' + uuid.uuid4().hex
    try:
        for code in (f'nvc_canary = {marker!r}; print(nvc_canary)', 'print(nvc_canary)'):
            result = request_json(host, port, '/execute', {
                'generated_code': code, 'language': 'ipython', 'timeout': 10,
                'max_output_characters': 512,
            }, session)
            if result.get('process_status') != 'completed' or result.get('stdout', '').strip() != marker or result.get('stderr'):
                raise RuntimeError('Stateful sandbox execution canary failed')
        result = request_json(host, port, '/execute', {
            'generated_code': f"printf '%s\\n' '{marker}'", 'language': 'shell',
            'timeout': 10, 'max_output_characters': 512,
        }, session)
        if result.get('process_status') != 'completed' or result.get('stdout', '').strip() != marker or result.get('stderr'):
            raise RuntimeError('Shell sandbox execution canary failed')
    finally:
        # A failed first execution may not have created a kernel yet.
        try:
            request_json(host, port, '/sessions/' + session, session=session, method='DELETE')
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise


def process_snapshot():
    result = {}
    for entry in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = entry.read_text()
            fields = text[text.rfind(')') + 2:].split()
            result[int(entry.parent.name)] = {'state': fields[0], 'ppid': int(fields[1]), 'start': int(fields[19])}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def signal_identity(pid, started, signum):
    # Linux pidfds bind the signal to the process opened here; start-time checks
    # also reject a PID recycled between our ownership snapshot and pidfd_open.
    fd = None
    try:
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            fd = os.pidfd_open(pid)
        try:
            stat = Path('/proc', str(pid), 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            current_start = int(fields[19])
        except (FileNotFoundError, ProcessLookupError):
            return
        if current_start != started:
            return
        if fd is not None:
            signal.pidfd_send_signal(fd, signum)
        else:
            os.kill(pid, signum)
    except ProcessLookupError:
        pass
    finally:
        if fd is not None:
            os.close(fd)


class OwnedChildren:
    def __init__(self):
        self.children = []
        self.logs = []
        self.start_times = {}
        self.linux = Path('/proc/self/stat').is_file()
        self.preexisting = set()
        if self.linux:
            self.preexisting = {pid for pid, value in process_snapshot().items() if value['ppid'] == os.getpid()}
            # Native execute_shell/execute_python call setsid. Adopt orphaned
            # descendants so escaping a process group cannot escape cleanup.
            if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
                raise RuntimeError('Unable to set sandbox child-subreaper')

    def descendants(self):
        snapshot = process_snapshot()
        owned = {p.pid for _, p in self.children if p.pid in snapshot and snapshot[p.pid]['start'] == self.start_times.get(p.pid)}
        owned.update(pid for pid, value in snapshot.items() if value['ppid'] == os.getpid() and pid not in self.preexisting)
        while True:
            extra = {pid for pid, value in snapshot.items() if value['ppid'] in owned} - owned
            if not extra:
                break
            owned.update(extra)
        return {pid: snapshot[pid] for pid in owned}


    def start(self, name, command, log_path, **kwargs):
        log = open(log_path, 'ab', buffering=0)
        self.logs.append(log)
        child = subprocess.Popen(command, start_new_session=True, stdout=log, stderr=subprocess.STDOUT, **kwargs)
        self.children.append((name, child))
        if self.linux:
            identity = process_snapshot().get(child.pid)
            if identity is not None:
                self.start_times[child.pid] = identity['start']
        return child

    def check(self):
        failed = [(name, p.returncode) for name, p in self.children if p.poll() is not None]
        if failed:
            raise RuntimeError('Owned sandbox child exited: ' + repr(failed))

    def close(self, timeout=20):
        # Every direct child below was created here; Linux additionally follows
        # only their process ancestry/adoption, not command names or global PIDs.
        if self.linux:
            for pid, value in self.descendants().items():
                signal_identity(pid, value['start'], signal.SIGTERM)
        else:
            for _, child in self.children:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + timeout
        for _, child in self.children:
            try:
                child.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        if self.linux:
            for pid, value in self.descendants().items():
                if value['state'] not in ('Z', 'X'):
                    signal_identity(pid, value['start'], signal.SIGKILL)
        else:
            for _, child in self.children:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for _, child in self.children:
            child.wait(timeout=5)
        if self.linux:
            deadline = time.monotonic() + 5
            while True:
                descendants = self.descendants()
                for pid, value in descendants.items():
                    if value['state'] in ('Z', 'X') and value['ppid'] == os.getpid():
                        try:
                            os.waitpid(pid, os.WNOHANG)
                        except ChildProcessError:
                            pass
                    elif value['state'] not in ('Z', 'X'):
                        signal_identity(pid, value['start'], signal.SIGKILL)
                if not self.descendants():
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('Owned sandbox descendants were not fully reaped')
                time.sleep(.1)
        for log in self.logs:
            log.close()
        return [{'name': name, 'pid': p.pid, 'exit_code': p.returncode} for name, p in self.children]


def run(args):
    receipt = Path(args.receipt_dir)
    local = Path(args.local_dir)
    receipt.mkdir(parents=True, exist_ok=True)
    if any((receipt / name).exists() for name in ('READY.json', 'EXIT.json', 'STOP.json')):
        raise RuntimeError('Sandbox receipt directory has prior lifecycle artifacts')
    local.mkdir(parents=True, exist_ok=False)
    if not args.job_id.isdigit() or not 1 <= args.workers <= 64:
        raise ValueError('Invalid allocation identity or worker count')
    source = Path('/app/main.py')
    template = Path('/etc/nginx/nginx.conf.template')
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, template)}
    expected_hashes = {
        '/app/main.py': 'c79197dc02e33b63b58b30783e3cfe0076e8edf6ceb409835b5d0762756b621d',
        '/etc/nginx/nginx.conf.template': 'a665f1ae7784745eb6a9647c5755a8692c6a9d96691d24d2529f95cf875c06e9',
    }
    if hashes != expected_hashes:
        raise RuntimeError('Sandbox image source differs from audited image')
    identity = {'schema': 'gym-sandbox-sidecar.v1', 'job_id': args.job_id,
                'host': socket.gethostname(), 'address': args.host, 'workers': args.workers,
                'supervisor_pid': os.getpid(), 'source_hashes': hashes}
    children = OwnedChildren()
    held = []
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    exit_code = 1
    try:
        proxy, base, held = reserve_ports(args.host, args.workers)
        identity.update(port=proxy, worker_base_port=base)
        write_json(receipt / 'PORTS.json', identity)
        upstream = '\n'.join(f'        server 127.0.0.1:{base + i};' for i in range(args.workers))
        nginx = template.read_text().replace('${UPSTREAM_SERVERS}', upstream).replace('${NGINX_PORT}', f'{args.host}:{proxy}')
        if 'hash $hash_key consistent;' not in nginx or 'proxy_next_upstream off;' not in nginx:
            raise RuntimeError('Pinned native session-affinity template contract missing')
        nginx = 'pid ' + str(local / 'nginx.pid') + ';\n' + nginx
        nginx = nginx.replace('/var/log/nginx/access.log', str(local / 'nginx-access.log'))
        nginx = nginx.replace('/var/log/nginx/error.log', str(local / 'nginx-error.log'))
        conf = local / 'nginx.conf'
        conf.write_text(nginx)
        subprocess.run(['nginx', '-t', '-c', str(conf)], check=True)
        for i in range(args.workers):
            ini = local / f'worker-{i}.ini'
            ini.write_text(f'''[uwsgi]\nmodule = main\ncallable = app\nprocesses = 1\nhttp-socket = 127.0.0.1:{base+i}\nvacuum = true\nmaster = true\ndie-on-term = true\nlisten = 100\nhttp-timeout = 300\nsocket-timeout = 300\nlog-date = true\n''')
            held[i + 1].close()
            env = worker_environment(os.environ, i, args.workers, proxy, base)
            children.start(f'worker-{i}', ['uwsgi', '--ini', str(ini)], local / f'worker-{i}.log', cwd='/app', env=env)
        deadline = time.monotonic() + args.startup_timeout
        remaining = set(range(args.workers))
        while remaining and time.monotonic() < deadline and not stopping:
            children.check()
            for i in tuple(remaining):
                try:
                    health = request_json('127.0.0.1', base + i, '/health')
                    if health.get('status') == 'healthy' and health.get('worker') == str(i):
                        remaining.remove(i)
                except (OSError, urllib.error.URLError):
                    pass
            if remaining:
                time.sleep(.2)
        if remaining or stopping:
            raise RuntimeError('Sandbox worker startup incomplete')
        held[0].close()
        children.start('nginx', ['nginx', '-c', str(conf), '-g', 'daemon off;'], local / 'nginx-process.log')
        while time.monotonic() < deadline and not stopping:
            children.check()
            try:
                request_json(args.host, proxy, '/health')
                break
            except (OSError, urllib.error.URLError):
                time.sleep(.2)
        marker = 'NVC_' + args.job_id + '_' + uuid.uuid4().hex
        execution_canary(args.host, proxy, marker)
        children.check()
        write_json(receipt / 'READY.json', dict(identity, status='ready', stateful_execution_canary=True,
                   shell_execution_canary=True, canary_marker_sha256=hashlib.sha256(marker.encode()).hexdigest()))
        while not stopping:
            children.check()
            stop_file = receipt / 'STOP.json'
            if stop_file.exists():
                stop_request = json.loads(stop_file.read_text())
                if stop_request.get('job_id') != args.job_id:
                    raise RuntimeError('Foreign sandbox stop request')
                stopping = True
                break
            time.sleep(.5)
        exit_code = 0
    finally:
        for sock in held:
            sock.close()
        results = children.close()
        write_json(receipt / 'EXIT.json', dict(identity, exit_code=exit_code,
                   children_reaped=True, children=results, ready_removed=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--receipt-dir', required=True)
    p.add_argument('--local-dir', required=True)
    p.add_argument('--job-id', required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--startup-timeout', type=int, default=300)
    run(p.parse_args())
