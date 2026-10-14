"""Opt-in observation of native Gym shutdown; never kill a process here."""

import json
import os
from pathlib import Path
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psutil


def extract_strict_shutdown(config):
    enabled = config.pop("strict_shutdown", False)
    if type(enabled) is not bool:
        raise ValueError("strict_shutdown must be a boolean")
    return enabled


def identity_from_environment():
    # SLURM_* is intentionally removed before creating the Ray actors.
    run = Path(os.environ["NVC_DEBUG_RUN"]).resolve(strict=True)
    job = (run / "JOB_ID").read_text().strip()
    phase = os.environ["NVC_GYM_DRIVER_PHASE"]
    if not job.isdecimal() or phase not in ("train", "train-resume"):
        raise ValueError("Invalid Gym shutdown run/job/phase identity")
    return run, job, phase


def atomic_json(path, value):
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite shutdown receipt {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, path)


def owned_endpoints(run_helper, node_ip, head_port):
    """Select local wrapper endpoints, never their upstream model/sandbox URLs."""
    config = run_helper._server_client.global_config_dict
    endpoints = []
    for alias in run_helper._processes:
        tree = config[alias]
        kind = next(iter(tree))
        if kind not in ("resources_servers", "responses_api_agents", "responses_api_models"):
            raise ValueError(f"Unknown owned service kind {kind}")
        inner = tree[kind]
        implementation = next(iter(inner))
        service = inner[implementation]
        host, port = service["host"], service["port"]
        if host in ("0.0.0.0", "::", "localhost", "127.0.0.1"):
            host = node_ip
        if host != node_ip or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError(f"Owned Gym endpoint identity is invalid for {alias}")
        endpoints.append({"alias": alias, "host": host, "port": port})
    endpoints.append({"alias": "__gym_head__", "host": node_ip, "port": head_port})
    return endpoints


def endpoint_open(endpoint):
    try:
        with socket.create_connection((endpoint["host"], endpoint["port"]), timeout=0.25):
            return True
    except OSError:
        return False


def remaining_identities(identities):
    remaining = []
    for identity in identities:
        try:
            process = psutil.Process(identity["pid"])
            if process.create_time() == identity["create_time"]:
                # Zombies are intentionally not accepted as removed.
                remaining.append({**identity, "status": process.status()})
        except psutil.NoSuchProcess:
            pass
    return remaining


class GymShutdownGuard:
    """Track only descendants of Popen objects owned by this RunHelper."""

    def __init__(self, run_helper, *, run, job, phase, endpoints):
        self.run_helper = run_helper
        self.run = Path(run).resolve(strict=True)
        if (self.run / "JOB_ID").read_text().strip() != job:
            raise ValueError("Gym shutdown job is not this run's owner")
        if phase not in ("train", "train-resume"):
            raise ValueError("Unknown Gym driver phase")
        self.job, self.phase = job, phase
        self.directory = self.run / "gym-shutdown" / job
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{phase}.json"
        self.start_path = self.directory / f"{phase}.START.json"
        self.endpoints = list(endpoints)
        self.observed = {}
        self.root_create_times = {}
        self.errors = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.result = None
        self.actor_identity = {"pid": os.getpid(), "create_time": psutil.Process().create_time()}
        self.head_thread = run_helper._head_server_thread
        self._observe()
        if not self.observed or not self.endpoints:
            raise RuntimeError("Strict Gym shutdown requires observed owned services")
        atomic_json(self.start_path, {
            "schema": "nvc.gym-shutdown.v1", "status": "STARTED", "job_id": job,
            "run": str(self.run), "phase": phase, "actor_identity": self.actor_identity,
            "endpoints": self.endpoints,
        })
        self.thread = threading.Thread(target=self._observe_loop, daemon=True)
        self.thread.start()

    def _observe(self):
        try:
            roots = list(self.run_helper._processes.items())
            captured = {}
            # children(recursive=True) rescans every process for every root.
            # Build one independent snapshot per round; never trust Gym's cleanup
            # registry as evidence that the cleanup itself succeeded.
            snapshot = {}
            children = {}
            for process in psutil.process_iter(["pid", "ppid", "create_time"]):
                try:
                    info = process.info
                    snapshot[info["pid"]] = info
                    children.setdefault(info["ppid"], []).append(info["pid"])
                except psutil.NoSuchProcess:
                    pass
            for alias, child in roots:
                try:
                    root = psutil.Process(child.pid)
                    root_time = root.create_time()
                    previous = self.root_create_times.setdefault(alias, (root.pid, root_time))
                    if previous != (root.pid, root_time):
                        raise RuntimeError("Owned service root PID identity changed")
                    root_info = snapshot.get(root.pid)
                    if root_info is None:
                        continue  # Spawned or disappeared while taking the snapshot.
                    if root_info["create_time"] != root_time:
                        raise RuntimeError("Owned service root snapshot identity changed")
                    pending = [root.pid]
                    visited = set()
                    while pending:
                        pid = pending.pop()
                        if pid in visited:
                            continue
                        visited.add(pid)
                        info = snapshot[pid]
                        try:
                            # process_iter caches Process instances. Recheck the
                            # live identity before attributing any snapshot entry.
                            process = psutil.Process(pid)
                            if process.create_time() != info["create_time"]:
                                if pid == root.pid:
                                    raise RuntimeError("Owned service root PID identity changed")
                                continue
                            identity = {"pid": pid, "create_time": info["create_time"], "alias": alias}
                            captured[(pid, identity["create_time"])] = identity
                            for descendant in children.get(pid, []):
                                created = snapshot[descendant]["create_time"]
                                if created is None:
                                    raise RuntimeError("Owned descendant identity unavailable")
                                # A reused parent PID cannot own an older process.
                                if created >= identity["create_time"]:
                                    pending.append(descendant)
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
            with self.lock:
                self.observed.update(captured)
        except Exception as error:
            with self.lock:
                self.errors.append(type(error).__name__)

    def _observe_loop(self):
        while not self.stop.wait(0.1):
            self._observe()

    def shutdown(self, *, wait_seconds=5):
        if self.result is not None:
            if self.result["status"] != "PASS":
                raise RuntimeError("A previous strict Gym shutdown failed")
            return self.result
        native_error = None
        self._observe()
        try:
            # The native cleanup is the tested subject, not replaced by a fallback.
            self.run_helper.shutdown()
        except Exception as error:
            native_error = type(error).__name__
        finally:
            self._observe()
            self.stop.set()
            self.thread.join(timeout=2)
        with self.lock:
            observed = list(self.observed.values())
            errors = list(self.errors)
        deadline = time.monotonic() + wait_seconds
        while True:
            remaining = remaining_identities(observed)
            with ThreadPoolExecutor(max_workers=16) as pool:
                alive = list(pool.map(endpoint_open, self.endpoints))
            open_endpoints = [endpoint for endpoint, open_ in zip(self.endpoints, alive) if open_]
            head_alive = bool(self.head_thread and self.head_thread.is_alive())
            if not (remaining or open_endpoints or head_alive) or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        passed = not (native_error or errors or remaining or open_endpoints or head_alive or self.thread.is_alive())
        self.result = {
            "schema": "nvc.gym-shutdown.v1", "status": "PASS" if passed else "FAIL",
            "job_id": self.job, "run": str(self.run), "phase": self.phase,
            "actor_identity": self.actor_identity, "native_error_type": native_error,
            "observation_errors": errors, "observed_processes": observed,
            "remaining_processes": remaining, "owned_endpoints": self.endpoints,
            "open_endpoints": open_endpoints, "head_thread_alive": head_alive,
            "observer_thread_alive": self.thread.is_alive(),
            "scope": "Native Gym owned service tree and local wrapper/head endpoints; excludes policy and sandbox",
        }
        atomic_json(self.path, self.result)
        if not passed:
            raise RuntimeError(f"Strict native Gym shutdown failed; receipt {self.path}")
        return self.result


def validate_shutdown_receipt(run, job, phase):
    run = Path(run).resolve(strict=True)
    if (run / "JOB_ID").read_text().strip() != job or not job.isdecimal():
        raise ValueError("Shutdown gate job does not own run")
    if phase not in ("train", "train-resume"):
        raise ValueError("Shutdown gate phase invalid")
    directory = run / "gym-shutdown" / job
    start = json.loads((directory / f"{phase}.START.json").read_text())
    result = json.loads((directory / f"{phase}.json").read_text())
    for receipt in (start, result):
        if receipt.get("schema") != "nvc.gym-shutdown.v1":
            raise ValueError("Shutdown gate schema mismatch")
        if (receipt.get("run"), receipt.get("job_id"), receipt.get("phase")) != (str(run), job, phase):
            raise ValueError("Shutdown gate identity mismatch")
    if result.get("status") != "PASS" or not result.get("observed_processes") or not result.get("owned_endpoints"):
        raise RuntimeError("Missing successful native Gym shutdown proof")
    for field in ("remaining_processes", "open_endpoints", "observation_errors", "native_error_type", "head_thread_alive", "observer_thread_alive"):
        if field not in result or result[field]:
            raise RuntimeError(f"Shutdown gate rejected {field}")
    if start.get("status") != "STARTED" or not start.get("actor_identity"):
        raise RuntimeError("Shutdown proof is missing its started actor identity")
    if start["actor_identity"] != result.get("actor_identity") or start.get("endpoints") != result.get("owned_endpoints"):
        raise RuntimeError("Shutdown proof differs from original owned service identity")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--phase", choices=("train", "train-resume"), required=True)
    args = parser.parse_args()
    validate_shutdown_receipt(args.run, args.job, args.phase)
    print(f"[GYM_SHUTDOWN_GATE_PASS] job={args.job} phase={args.phase}")
