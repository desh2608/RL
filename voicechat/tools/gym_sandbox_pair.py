#!/usr/bin/env python3
"""Join the owned sandbox Slurm step and training step as one allocation task."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from gym_idle_watchdog import IdleWatchdog


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def terminate_owned(child, grace=20):
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=10)


def validate_ready(value, job_id, host):
    if value.get('job_id') != job_id or value.get('address') != host:
        raise RuntimeError('Sandbox endpoint belongs to another allocation or host')
    if value.get('status') != 'ready' or value.get('stateful_execution_canary') is not True or value.get('shell_execution_canary') is not True:
        raise RuntimeError('Sandbox execution canaries were not accepted')
    port = value.get('port')
    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
        raise RuntimeError('Sandbox ready receipt has invalid dynamic port')
    return port


def run_paired(sidecar_cmd, main_builder, receipt, job_id, host, startup_timeout=600, stop_timeout=25, watchdog=None):
    receipt = Path(receipt)
    receipt.mkdir(parents=True, exist_ok=True)
    if any((receipt / name).exists() for name in ('READY.json', 'STOP.json', 'PAIR_EXIT.json')):
        raise RuntimeError('Refusing to reuse sandbox lifecycle directory')
    sandbox = main = None
    interrupted = False
    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
    previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    outcome = {'schema': 'gym-sandbox-pair.v1', 'job_id': job_id,
               'main_started': False, 'main_exit': None, 'sidecar_exit': None}
    try:
        sidecar_env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for key in ('NVIDIA_API_KEY', 'OPENAI_API_KEY', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'WANDB_API_KEY'):
            sidecar_env.pop(key, None)
        sandbox = subprocess.Popen(sidecar_cmd, start_new_session=True, env=sidecar_env)
        deadline = time.monotonic() + startup_timeout
        while not (receipt / 'READY.json').exists():
            if interrupted:
                raise RuntimeError('Allocation termination during sandbox startup')
            if sandbox.poll() is not None:
                raise RuntimeError('Sandbox step exited before readiness')
            if time.monotonic() >= deadline:
                raise RuntimeError('Sandbox startup timed out')
            time.sleep(.2)
        ready = json.loads((receipt / 'READY.json').read_text())
        port = validate_ready(ready, job_id, host)
        if sandbox.poll() is not None:
            raise RuntimeError('Sandbox step exited at readiness')
        env = dict(os.environ, NEMO_SKILLS_SANDBOX_HOST=host, NEMO_SKILLS_SANDBOX_PORT=str(port),
                   NVC_GYM_SANDBOX_JOB_ID=job_id)
        if watchdog is not None:
            env.update(watchdog.child_environment())
        main_command = main_builder()
        if watchdog is not None:
            main_command = watchdog.bind_command(main_command)
        main = subprocess.Popen(main_command, env=env, start_new_session=True)
        outcome['main_started'] = True
        while main.poll() is None:
            if interrupted:
                raise RuntimeError('Allocation termination during training')
            if sandbox.poll() is not None:
                raise RuntimeError('Sandbox step exited while training was active')
            if watchdog is not None and watchdog.tick(main):
                outcome['idle_watchdog_terminated_main'] = True
                watchdog.terminate_main_step(main)
                raise RuntimeError('Idle watchdog deadline: stopping owned main step for allocation cleanup')
            time.sleep(.2)
        outcome['main_exit'] = main.returncode
        atomic_json(receipt / 'STOP.json', {'job_id': job_id, 'reason': 'main_step_terminal'})
        sandbox.wait(timeout=stop_timeout)
        outcome['sidecar_exit'] = sandbox.returncode
        if main.returncode != 0 or sandbox.returncode != 0:
            raise RuntimeError('Training or sandbox step failed')
        exit_receipt = json.loads((receipt / 'EXIT.json').read_text())
        if exit_receipt.get('job_id') != job_id or exit_receipt.get('exit_code') != 0 or exit_receipt.get('children_reaped') is not True:
            raise RuntimeError('Sandbox child cleanup not accepted')
        outcome['status'] = 'complete'
    finally:
        # Main must be terminal before removing its service. If its Slurm step
        # needs escalation, this is limited to the Popen created above.
        terminate_owned(main)
        atomic_json(receipt / 'STOP.json', {'job_id': job_id, 'reason': 'allocation_pair_finally'})
        if sandbox is not None and sandbox.poll() is None:
            try:
                sandbox.wait(timeout=stop_timeout)
            except subprocess.TimeoutExpired:
                terminate_owned(sandbox)
        if main is not None:
            outcome['main_exit'] = main.returncode
        if sandbox is not None:
            outcome['sidecar_exit'] = sandbox.returncode
        outcome.setdefault('status', 'failed')
        atomic_json(receipt / 'PAIR_EXIT.json', outcome)
        for sig, previous_handler in previous.items():
            signal.signal(sig, previous_handler)


def build_commands(args, env):
    run = Path(args.run)
    nodes = int(env['NVC_EXPECTED_NODES'])
    job = env['SLURM_JOB_ID']
    local = env['NVC_DEBUG_LOCAL']
    host = socket.gethostbyname(args.head)
    receipt = run / 'sandbox' / job
    # --overlap shares this allocation's existing CPUs/memory/GRES. Explicit
    # --gres=none ensures the sidecar reserves no GPUs; no new allocation exists.
    sidecar = ['srun', '--overlap', '--nodes=1', '--ntasks=1', '--ntasks-per-node=1',
               '--nodelist=' + args.head, '--cpus-per-task=8', '--gres=none', '--mem=0',
               '--cpu-bind=none', '--mpi=none', '--kill-on-bad-exit=1', '--no-container-mount-home', '--no-container-entrypoint',
               '--container-image=' + args.sandbox_image,
               '--container-mounts=' + str(run / 'scripts') + ':/nvc/scripts:ro,' + str(receipt) + ':/nvc/receipts,' + local + '/sandbox-container:/nvc-owned,' + local + '/sandbox-tmp:/tmp',
               '--container-workdir=/app', '--output=' + str(run / 'sandbox-step-%j-%N.log'),
               'python3', '/nvc/scripts/gym_sandbox_supervisor.py', '--receipt-dir', '/nvc/receipts',
               '--local-dir', '/nvc-owned/runtime', '--job-id', job, '--host', host, '--workers', '32']
    main = ['srun', '--overlap', '--nodes=' + str(nodes), '--ntasks=' + str(nodes), '--ntasks-per-node=1',
            '--cpus-per-task=64', '--gpus-per-task=8', '--mem=0', '--gpu-bind=none', '--cpu-bind=none',
            '--mpi=none', '--kill-on-bad-exit=1', '--no-container-mount-home', '--container-image=' + args.image,
            '--container-mounts=' + args.mounts, '--container-workdir=/workspace/nemo-rl',
            '--output=' + str(run / 'rank-%t-%N.log'), 'bash', str(run / 'scripts/ray_role.sh')]
    return sidecar, main, receipt, job, host


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('run', 'head', 'image', 'sandbox-image', 'mounts'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    sidecar, main, receipt, job, host = build_commands(args, os.environ)
    watchdog = IdleWatchdog.from_run(args.run, job)
    run_paired(sidecar, lambda: main, receipt, job, host, watchdog=watchdog)
