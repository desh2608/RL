# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in host supervisor for one Popen-owned Gym main step; stdlib only.

No model requests, GPU kernels, process-name searches or allocation cancellation.
GPU utilization is diagnostic context only; decisions use phase/completion facts.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import pwd
import subprocess
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def frozen_policy(run):
    """Read only three flat keys in the native resolved checkpointing YAML.

    This is deliberately not a general YAML parser. Unexpected spellings of an
    opt-in key fail rather than silently disable the requested protection.
    """
    text = (Path(run) / 'config.yaml').read_text()
    keys = ('idle_watchdog_enabled', 'idle_watchdog_request_seconds',
            'idle_watchdog_terminate_seconds')
    values = {}
    in_checkpointing = False
    for line in text.splitlines():
        if line and not line[0].isspace() and not line.startswith('#'):
            in_checkpointing = line == 'checkpointing:'
        if not any(key in line for key in keys):
            continue
        match = re.fullmatch(r'  (idle_watchdog_\w+): (true|false|[0-9]+)', line)
        if not in_checkpointing or match is None or match[1] not in keys or match[1] in values:
            raise ValueError('Idle watchdog requires unique flat resolved checkpointing scalars')
        values[match[1]] = match[2]
    enabled = values.get(keys[0], 'false')
    if enabled not in ('true', 'false'):
        raise ValueError('Idle watchdog enabled must be a boolean')
    if enabled == 'false':
        return None
    request = int(values.get(keys[1], '1200'))
    terminate = int(values.get(keys[2], '1500'))
    # Reviewed range: the historical 20/25 min policy up to 60/65 min. Multi-turn agentic steps
    # (tau2 ~17 policy turns per episode, retries on simulator faults, gap-fill rounds) can keep the
    # learner idle well past 20 min without being stuck (smoke job 35894567 died at 1261 s of
    # learner idle while legitimately gap-filling). Anything outside the range is a config error.
    if not (1200 <= request <= 3600):
        raise ValueError('idle_watchdog_request_seconds must be within [1200, 3600]')
    if not (request + 300 <= terminate <= request + 900):
        raise ValueError('idle_watchdog_terminate_seconds must be request+300 .. request+900')
    return {'request_seconds': request, 'terminate_seconds': terminate}


# A job whose driver never publishes STATUS.json (simulator node never serving, Ray never forming,
# container import hanging) is terminated this long after the actual allocation start. Independent of
# the 1200/1500 s idle policy, which can be pinned while the job-bound simulator receipts are pending.
DRIVER_ABSENT_SECONDS = 2400
SIM_HOST_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')


def sim_pending_hosts(run, job_id):
    """Hosts listed in <run>/sim/<job>/HOSTS (gpu_job.sh) without a READY receipt from sim_role.sh.

    Only this allocation's job-scoped directory is read; another job's receipts never count. An
    absent HOSTS file (no simulator nodes) yields an empty list.
    """
    sim_dir = Path(run) / 'sim' / str(job_id)
    hosts_file = sim_dir / 'HOSTS'
    if not hosts_file.is_file() or hosts_file.is_symlink():
        return []
    pending = []
    for host in hosts_file.read_text().split():
        if SIM_HOST_PATTERN.fullmatch(host) is None:
            raise ValueError('Unsafe simulator host name in ' + str(hosts_file))
        ready = sim_dir / ('READY-' + host + '.json')
        if not (ready.is_file() and not ready.is_symlink() and ready.stat().st_size > 0):
            pending.append(host)
    return pending


class IdleWatchdog:
    def __init__(self, run, job_id, policy, *, clock=time.monotonic, wall=time.time,
                 allocation_start=None, driver_absent_seconds=DRIVER_ABSENT_SECONDS):
        self.run = Path(run).resolve(strict=True)
        self.job_id = str(job_id)
        if not self.job_id.isdigit():
            raise ValueError('Invalid Slurm job identity')
        driver_absent_seconds = float(driver_absent_seconds)
        if not math.isfinite(driver_absent_seconds) or driver_absent_seconds <= 0:
            raise ValueError('Invalid driver-absent cap')
        self.driver_absent_seconds = driver_absent_seconds
        self.clock, self.wall = clock, wall
        self.policy = policy
        source = (self.run / 'source.sha256').read_text().split()
        if not source or re.fullmatch('[0-9a-f]{64}', source[0]) is None:
            raise ValueError('Frozen source archive digest missing')
        self.binding = {
            'schema': 'gym-idle-guard.v1', 'job_id': self.job_id,
            'run': str(self.run), 'source_sha256': source[0],
            'config_sha256': hashlib.sha256((self.run / 'config.yaml').read_bytes()).hexdigest(),
            'token': uuid.uuid4().hex,
        }
        self.directory = self.run / 'idle-guard' / self.job_id
        self.directory.mkdir(parents=True, exist_ok=False)
        now = clock()
        elapsed = 0.0
        if allocation_start is not None:
            allocation_start = float(allocation_start)
            if not math.isfinite(allocation_start) or allocation_start <= 0:
                raise ValueError('Invalid actual allocation start')
            elapsed = max(0.0, wall() - allocation_start)
        # Both idle clocks and the driver-absent cap are pinned to the actual allocation start.
        self.allocation_clock_start = now - elapsed
        self.last_semantic = self.last_learner = self.allocation_clock_start
        self.previous = None
        self.request = None
        self.terminate_at = None
        self.last_poll = float('-inf')
        atomic_json(self.directory / 'OWNER.json', {
            **self.binding, 'supervisor_pid': os.getpid(),
            'actual_allocation_start': allocation_start,
            'scope': 'Only this supervisor-created main Popen; no allocation-wide signal',
            'policy': policy,
            'driver_absent_seconds': driver_absent_seconds,
            'sim_receipts': str(self.run / 'sim' / self.job_id),
        })

    @classmethod
    def from_run(cls, run, job_id):
        policy = frozen_policy(run)
        if policy is None:
            return None
        return cls(run, job_id, policy,
                   allocation_start=os.environ.get('NVC_ALLOCATION_START_EPOCH'))

    def bind_command(self, command):
        if not command or Path(command[0]).name != 'srun':
            raise ValueError('Idle supervisor requires its own synchronous main srun')
        if any(x.startswith(('--job-name', '--jobid', '--async')) for x in command[1:]):
            raise ValueError('Main srun already has an ambiguous job/step binding')
        self.step_name = 'nvc-idle-' + self.job_id + '-' + self.binding['token'][:16]
        return command[:1] + ['--jobid=' + self.job_id, '--job-name=' + self.step_name] + command[1:]

    def terminate_main_step(self, main, runner=subprocess.run):
        # srun SIGTERM can be forwarded as task SIGKILL. Use Slurm's exact
        # named step TERM first; the existing owned-Popen finally is only the
        # documented bounded fallback if this readback or join fails.
        if main.poll() is not None:
            return
        try:
            result = runner(['squeue', '--steps', '--jobs=' + self.job_id,
                             '--noheader', '--format=%i|%j|%u'],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=True, timeout=10)
            matches = []
            username = pwd.getpwuid(os.getuid()).pw_name
            for line in result.stdout.splitlines():
                parts = [x.strip() for x in line.split('|')]
                if len(parts) == 3 and parts[1] == self.step_name:
                    if not re.fullmatch(re.escape(self.job_id) + r'\.[0-9]+', parts[0]) or parts[2] != username:
                        raise RuntimeError('Named main step ownership mismatch')
                    matches.append(parts[0])
            if len(matches) != 1:
                raise RuntimeError('Expected exactly one current named main step')
            if main.poll() is not None:
                return
            step = matches[0]
            atomic_json(self.directory / 'MAIN_STEP_TERM.json', {
                **self.binding, 'step_id': step, 'step_name': self.step_name,
                'main_pid': main.pid, 'owner': username, 'signal': 'TERM',
                'time_epoch': self.wall(), 'status': 'intent',
            })
            runner(['scancel', '--signal=TERM', step], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            main.wait(timeout=60)
        except BaseException as exc:
            atomic_json(self.directory / 'MAIN_STEP_TERM_FAILURE.json', {
                **self.binding, 'main_pid': main.pid, 'error_type': type(exc).__name__,
                'time_epoch': self.wall(),
                'fallback': 'existing terminate_owned(main); srun TERM may forward task KILL',
            })
            raise

    def child_environment(self):
        # NeMo-RL init_ray strips SLURM_* before the native guard is built.
        # Carry the exact host-owned identity with the existing launch nonce.
        return {'NVC_IDLE_GUARD_TOKEN': self.binding['token'],
                'NVC_IDLE_GUARD_JOB_ID': self.job_id}

    def tick(self, main):
        """Return True only when the exact passed main Popen must be terminated.

        This function sends no signals. run_paired retains its existing owned
        terminate/wait/finally path and keeps the batch allocation for cleanup.
        """
        now = self.clock()
        if now - self.last_poll < 30:
            return False
        self.last_poll = now
        status = None
        path = self.directory / 'STATUS.json'
        if path.is_file() and not path.is_symlink():
            candidate = json.loads(path.read_text())
            if all(candidate.get(k) == v for k, v in self.binding.items()):
                if candidate.get('phase') not in {'startup', 'buffer_wait', 'training', 'refit', 'saving', 'stopping'}:
                    raise ValueError('Unrecognized idle phase')
                for key in ('semantic_seq', 'learner_seq'):
                    if type(candidate.get(key)) is not int or candidate[key] < 0:
                        raise ValueError('Invalid idle completion sequence')
                status = candidate
        # No driver status still has a live startup clock from actual allocation
        # start. File timestamps/log traffic alone never count as progress.
        if status is not None:
            previous = self.previous
            same_session = previous and previous.get('driver_session') == status.get('driver_session')
            old_semantic = previous['semantic_seq'] if same_session else 0
            old_learner = previous['learner_seq'] if same_session else 0
            if status['semantic_seq'] < old_semantic or status['learner_seq'] < old_learner:
                raise ValueError('Completion sequence regressed within one driver')
            if status['semantic_seq'] > old_semantic:
                self.last_semantic = now
            if status['learner_seq'] > old_learner:
                self.last_learner = now
            # Expected compute/refit/save is not learner-idle. It still has the
            # semantic no-completion timeout, and a phase label cannot reset it.
            if status['phase'] in {'training', 'refit', 'saving'}:
                self.last_learner = now
            # buffer_wait with a growing buffer is generation progress, not a hung learner, so the
            # learner clock never runs ahead of the semantic clock there. Train job 35904561 (P256,
            # three targets in flight, step 0) was killed at learner_idle 3003 s while the buffer
            # grew 0 -> 278 groups; a real stall still trips the no-semantic-progress budget.
            if status['phase'] == 'buffer_wait':
                self.last_learner = max(self.last_learner, self.last_semantic)
            self.previous = status
        # Before any driver STATUS.json has been seen, the simulator node wait in ray_role.sh (<= 900 s
        # after Ray formation) sits inside the 1200 s idle budget: while this job's HOSTS lists a host
        # without a READY receipt both clocks stay pinned to now. The driver-absent cap below is the
        # only limit in that state, so a job that never starts the driver still dies.
        driver_absent = status is None and self.previous is None
        pin_reason = None
        sim_pending = []
        if driver_absent:
            sim_pending = sim_pending_hosts(self.run, self.job_id)
            if sim_pending:
                pin_reason = 'sim_pending'
                self.last_semantic = self.last_learner = now
        semantic_age = max(0.0, now - self.last_semantic)
        learner_age = max(0.0, now - self.last_learner)
        reason = 'no_semantic_progress' if semantic_age >= learner_age else 'learner_idle_budget'
        age = max(semantic_age, learner_age)
        driver_absent_age = max(0.0, now - self.allocation_clock_start) if driver_absent else None
        snapshot = {**self.binding, 'phase': status['phase'] if status else 'startup',
                    'status': status, 'semantic_idle_seconds': semantic_age,
                    'learner_idle_seconds': learner_age,
                    'pin_reason': pin_reason, 'sim_pending_hosts': sim_pending,
                    'driver_absent_seconds': driver_absent_age,
                    'driver_absent_cap_seconds': self.driver_absent_seconds,
                    'main_pid': main.pid, 'time_epoch': self.wall(),
                    'gpu_utilization_used_as_stop_gate': False}
        atomic_json(self.directory / 'WATCHDOG_STATUS.json', snapshot)
        if self.request is None and driver_absent_age is not None and driver_absent_age >= self.driver_absent_seconds:
            # No driver exists to honor a stop request, so the latch terminates on this tick.
            self.terminate_at = now
            self.request = {**self.binding, 'request_id': uuid.uuid4().hex,
                            'reason': 'driver_absent', 'issued_at_epoch': self.wall(),
                            'not_after_epoch': self.wall(), 'snapshot': snapshot}
            atomic_json(self.directory / 'STOP_INPUT.json', self.request)
        if self.request is None and age >= self.policy['request_seconds']:
            self.terminate_at = now + max(0.0, self.policy['terminate_seconds'] - age)
            self.request = {**self.binding, 'request_id': uuid.uuid4().hex,
                            'reason': reason, 'issued_at_epoch': self.wall(),
                            'not_after_epoch': self.wall() + self.policy['terminate_seconds'] - age,
                            'snapshot': snapshot}
            atomic_json(self.directory / 'STOP_INPUT.json', self.request)
        if self.request is not None:
            # The request is latched. At most five more minutes are available
            # for a real boundary save; no reset by heartbeats or new requests.
            if now >= self.terminate_at and main.poll() is None:
                atomic_json(self.directory / 'MAIN_TERM_INTENT.json', {
                    **self.binding, 'request_id': self.request['request_id'],
                    'main_pid': main.pid, 'time_epoch': self.wall(),
                    'scope': 'Exact uniquely named owned JOB.STEP TERM; Popen fallback if identity/join fails; batch kept',
                })
                return True
        return False
