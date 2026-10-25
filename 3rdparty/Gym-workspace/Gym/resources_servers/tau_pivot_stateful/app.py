# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-turn binary stateful verifier; archived native runtimes live in subprocesses."""

import asyncio
import hashlib
import json
import os
import select
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.tau_pivot_stateful import afterquery_v21, combined_static


class VerificationRef(BaseModel):
    record_id: str
    dataset: str
    verifier_version: str
    expected_turn_kind: str


class Config(BaseResourcesServerConfig):
    cache_manifest: str
    runtime_manifest: str
    v21_field_table: str | None = None
    v21_cache_manifest: str | None = None
    cache_entries: int = 2
    replicas_per_runtime: int = 1
    worker_timeout_seconds: float = 300
    dispatch_policy: Literal["round_robin", "record_affinity", "affinity_steal"] = "affinity_steal"


class VerifyRequest(BaseVerifyRequest):
    verification_ref: VerificationRef


class VerifyResponse(BaseVerifyResponse):
    verification_ref: VerificationRef
    diagnostics: dict


class WorkerExited(RuntimeError):
    pass


def record_slot(record_id: str, pool_size: int) -> int:
    """Stable affinity independent of Python's randomized hash seed."""
    return int(hashlib.sha256(record_id.encode()).hexdigest()[:8], 16) % pool_size


class Engine:
    def __init__(
        self,
        cache_manifest,
        runtime_manifest,
        cache_entries,
        replicas_per_runtime,
        worker_timeout_seconds,
        dispatch_policy,
        v21_field_table=None,
        v21_cache_manifest=None,
    ):
        if cache_entries < 1 or replicas_per_runtime < 1:
            raise ValueError("cache entries and replicas must be positive")
        self.index = json.loads(Path(cache_manifest).read_text())["records"]
        self.v21_index = json.loads(Path(v21_cache_manifest).read_text())["records"] if v21_cache_manifest else {}
        self.v21_table = json.loads(Path(v21_field_table).read_text()) if v21_field_table else None
        self.runtimes = json.loads(Path(runtime_manifest).read_text())
        self.cache_entries = cache_entries
        self.timeout = worker_timeout_seconds
        if dispatch_policy not in ("round_robin", "record_affinity", "affinity_steal"):
            raise ValueError("Unknown dispatch policy")
        self.dispatch_policy = dispatch_policy
        sizes = {
            key: runtime["replicas"] if "replicas" in runtime else replicas_per_runtime
            for key, runtime in self.runtimes.items()
        }
        if any(n < 1 for n in sizes.values()) or sum(sizes.values()) > 32:
            raise ValueError("Runtime pools require 1..32 workers in total")
        self.workers = {key: [None] * sizes[key] for key in self.runtimes}
        self.locks = {key: [threading.Lock() for _ in range(sizes[key])] for key in self.runtimes}
        self.cursor = {key: 0 for key in self.runtimes}
        self.dispatch_lock = threading.Lock()
        self.dispatch_counts = {"owner_idle": 0, "stolen": 0, "owner_queued": 0}

    def spawn(self, key):
        runtime = self.runtimes[key]
        env = os.environ.copy()
        env.update(runtime["env"])
        return subprocess.Popen(
            runtime["command"] + ["--cache-entries", str(self.cache_entries)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

    def replace(self, key, slot):
        worker = self.workers[key][slot]
        if worker is not None:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=10)
            worker.stdin.close()
            worker.stdout.close()
        self.workers[key][slot] = self.spawn(key)

    def next_slot(self, key: str, record_id: str) -> int:
        if self.dispatch_policy in ("record_affinity", "affinity_steal"):
            return record_slot(record_id, len(self.workers[key]))
        return self.cursor[key]

    def acquire_slot(self, key, record_id):
        # Reserve an idle slot under the dispatch lock; never block while holding it.
        with self.dispatch_lock:
            owner = self.next_slot(key, record_id)
            if self.dispatch_policy == "round_robin":
                self.cursor[key] = (owner + 1) % len(self.workers[key])
            if self.dispatch_policy == "affinity_steal":
                candidates = [owner] + [i for i in range(len(self.workers[key])) if i != owner]
                for slot in candidates:
                    if self.locks[key][slot].acquire(blocking=False):
                        self.dispatch_counts["owner_idle" if slot == owner else "stolen"] += 1
                        return slot
                self.dispatch_counts["owner_queued"] += 1
        self.locks[key][owner].acquire()
        return owner

    def verify(self, ref, output):
        item = self.index.get(ref["record_id"])
        if item is not None and item.get("mode") == "combined_static":
            return combined_static.verify(item, ref, output)
        if ref.get("verifier_version", "").startswith("afterquery-v2.1:"):
            return self.verify_v21(self.v21_index[ref["record_id"]], ref, output)
        item = self.index[ref["record_id"]]
        key = item["runtime_key"]
        slot = self.acquire_slot(key, ref["record_id"])
        try:
            try:
                if self.workers[key][slot] is None:
                    self.workers[key][slot] = self.spawn(key)
                worker = self.workers[key][slot]
                if worker.poll() is not None:
                    raise WorkerExited("exit status " + str(worker.returncode))
                worker.stdin.write(json.dumps({"item": item, "ref": ref, "output": output}) + "\n")
                worker.stdin.flush()
                if not select.select([worker.stdout], [], [], self.timeout)[0]:
                    raise WorkerExited("response timeout")
                line = worker.stdout.readline()
                if not line:
                    raise WorkerExited("worker EOF")
                result = json.loads(line)
            except (OSError, WorkerExited, json.JSONDecodeError) as exc:
                detail = str(exc)
                try:
                    self.replace(key, slot)
                except Exception as respawn_error:
                    detail += "; respawn: " + str(respawn_error)
                return {
                    "reward": 0,
                    "predicates": {},
                    "error": "worker_exited",
                    "detail": detail,
                    "runtime_key": key,
                    "slot": slot,
                }
            if not result["ok"]:
                return {"reward": 0, "predicates": {}, "error": result["error"], "worker": result.get("worker")}
            return {**result["result"], "worker": result.get("worker"), "slot": slot}

        finally:
            self.locks[key][slot].release()

    def verify_v21(self, item, ref, output):
        if self.v21_table is None:
            raise ValueError("v2 field table is required")
        data = Path(item["entry"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["entry_sha256"]:
            raise ValueError("v2 cache entry changed")
        entry = json.loads(data)
        for key in ("record_id", "dataset", "verifier_version", "expected_turn_kind"):
            if entry[key] != ref[key]:
                raise ValueError("v2 reference mismatch: " + key)
        if afterquery_v21.digest(entry["state"]) != entry["state_hash_at_boundary"]:
            raise ValueError("v2 boundary hash mismatch")
        if entry["state"]["ledger"] != entry["remaining_gold"]:
            raise ValueError("v2 boundary ledger mismatch")
        if afterquery_v21.ledger(entry) != entry["v2_state"]["ledger"]:
            raise ValueError("v2 public ledger mapping mismatch")
        return afterquery_v21.evaluate(entry["verification"], output, entry["v2_state"], self.v21_table)

    def close(self):
        for pool in self.workers.values():
            for worker in pool:
                if worker is not None:
                    if worker.poll() is None:
                        worker.kill()
                    worker.wait(timeout=10)
                    worker.stdin.close()
                    worker.stdout.close()


class TauPivotStateful(SimpleResourcesServer):
    config: Config
    _engine: Engine | None = PrivateAttr(default=None)

    _init_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    async def verify(self, body: VerifyRequest) -> VerifyResponse:
        started = time.perf_counter()
        try:
            with self._init_lock:
                if self._engine is None:
                    self._engine = Engine(
                        self.config.cache_manifest,
                        self.config.runtime_manifest,
                        self.config.cache_entries,
                        self.config.replicas_per_runtime,
                        self.config.worker_timeout_seconds,
                        self.config.dispatch_policy,
                        self.config.v21_field_table,
                        self.config.v21_cache_manifest,
                    )
            result = await asyncio.to_thread(
                self._engine.verify,
                body.verification_ref.model_dump(),
                [x.model_dump(exclude_none=True) for x in body.response.output],
            )
            diagnostics = {
                "predicates": result["predicates"],
                "native_reason": (result.get("native") or {}).get("reason"),
            }
            diagnostics.update(
                {k: result[k] for k in ("error", "detail", "worker", "runtime_key", "slot") if k in result}
            )
            reward = float(result["reward"])
        except Exception as exc:
            reward = 0.0
            diagnostics = {"error": type(exc).__name__ + ": " + str(exc)}
        diagnostics["verify_latency_ms"] = (time.perf_counter() - started) * 1000
        return VerifyResponse(**body.model_dump(), reward=reward, diagnostics=diagnostics)


if __name__ == "__main__":
    TauPivotStateful.run_webserver()
