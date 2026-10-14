# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Version-pinned CS node-owned AsyncLLM helpers (vLLM 0.27.1)."""

import asyncio
import math
import queue
import threading
import time
from typing import Any


def node_engine_dp_size(config: dict[str, Any]) -> int:
    """Absent optional field retains the upstream single-engine topology."""
    if "node_engine_data_parallel_size" not in config:
        return 1
    value = config["node_engine_data_parallel_size"]
    if isinstance(value, bool) or not isinstance(value, int) or value not in {1, 8}:
        raise ValueError("node_engine_data_parallel_size must be 1 or 8")
    return value

def _init_node_engine_placement_groups(cluster: Any) -> list[Any]:
    """Reserve one atomic eight-GPU bundle on every rollout node."""

    # Ray is optional for the CPU queue/RPC contract tests.
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group

    physical_actors = cluster.node_count()
    if (
        physical_actors <= 0
        or cluster.world_size() != physical_actors * 8
        or cluster.num_gpus_per_node != 8
        or not cluster.use_gpus
    ):
        raise RuntimeError(
            "node-owned rollout requires exactly eight GPUs on every rollout node"
        )
    existing = getattr(cluster, "_node_placement_groups", None)
    if existing is not None:
        if len(existing) != physical_actors or any(
            len(pg.bundle_specs) != 1
            or float(pg.bundle_specs[0].get("GPU", 0)) != 8.0
            for pg in existing
        ):
            raise RuntimeError("rollout placement groups are not atomic GPU8 bundles")
        return existing

    constraints = getattr(cluster, "node_resource_constraints", None)
    placement_groups: list[Any] = []
    try:
        for actor_index in range(physical_actors):
            bundle: dict[str, float] = {"CPU": 1.0, "GPU": 8.0}
            if constraints and constraints[actor_index]:
                node_constraints = constraints[actor_index]
                if "CPU" in node_constraints or "GPU" in node_constraints:
                    raise RuntimeError(
                        "node resource constraints must not override CPU/GPU reservations"
                    )
                bundle.update(node_constraints)
            placement_groups.append(
                placement_group(
                    bundles=[bundle],
                    strategy="STRICT_PACK",
                    name=f"node-rollout-dp8-{actor_index}",
                )
            )
        ray.get([pg.ready() for pg in placement_groups], timeout=180)
    except BaseException:
        for pg in placement_groups:
            remove_placement_group(pg)
        raise
    cluster._node_placement_groups = placement_groups
    print(
        "[VLLM_NODE_ENGINE_PLACEMENT_READY] "
        f"physical_actors={physical_actors} actor_gpus=8 "
        f"internal_ranks={physical_actors * 8}",
        flush=True,
    )
    return placement_groups

class _AsyncReplicaReadyQueue:
    """Thread-safe bounded admission queue shared by async collector loops."""

    def __init__(self, replica_count: int, max_inflight_per_replica: int):
        if replica_count <= 0 or max_inflight_per_replica <= 0:
            raise ValueError("async ready queue dimensions must be positive")
        self.replica_count = replica_count
        self.max_inflight_per_replica = max_inflight_per_replica
        self.capacity = replica_count * max_inflight_per_replica
        self._slots: queue.Queue[int] = queue.Queue(maxsize=self.capacity)
        # Interleave replicas by slot layer so initial admission is balanced.
        for _slot in range(max_inflight_per_replica):
            for replica_idx in range(replica_count):
                self._slots.put_nowait(replica_idx)
        self._lock = threading.Lock()
        self._inflight = [0] * replica_count
        self._completed = 0
        self._wait_seconds_total = 0.0
        self._wait_seconds_max = 0.0

    async def acquire(self) -> tuple[int, float]:
        wait_started = time.perf_counter()
        # Queue is safe across collector threads/event loops. A cancelled
        # to_thread(Queue.get) would remain blocked and later steal a slot, so
        # acquisition must not leave a blocking waiter behind on cancellation.
        while True:
            try:
                replica_idx = self._slots.get_nowait()
                break
            except queue.Empty:
                await asyncio.sleep(0.001)
        wait_seconds = time.perf_counter() - wait_started
        with self._lock:
            self._inflight[replica_idx] += 1
            if self._inflight[replica_idx] > self.max_inflight_per_replica:
                raise RuntimeError(
                    "async ready queue exceeded per-replica admission bound"
                )
            self._wait_seconds_total += wait_seconds
            self._wait_seconds_max = max(self._wait_seconds_max, wait_seconds)
        return replica_idx, wait_seconds

    def release(self, replica_idx: int) -> dict[str, float | int] | None:
        with self._lock:
            if not 0 <= replica_idx < self.replica_count:
                raise RuntimeError(f"invalid async ready queue replica {replica_idx}")
            if self._inflight[replica_idx] <= 0:
                raise RuntimeError(
                    f"async ready queue double release for replica {replica_idx}"
                )
            self._inflight[replica_idx] -= 1
            self._completed += 1
            completed = self._completed
            report = None
            if completed % self.capacity == 0:
                report = {
                    "completed": completed,
                    "inflight": sum(self._inflight),
                    "min_inflight": min(self._inflight),
                    "max_inflight": max(self._inflight),
                    "wait_seconds_mean": self._wait_seconds_total / completed,
                    "wait_seconds_max": self._wait_seconds_max,
                }
        self._slots.put_nowait(replica_idx)
        return report

async def _collective_rpc_all_node_engines(
    llm: Any,
    method: str,
    *,
    args: tuple[Any, ...],
    expected_engines: int,
) -> list[Any]:
    """Return every EngineCore result from vLLM 0.27 internal DP.

    ``DPLBAsyncMPClient.call_utility_async`` executes a utility on every DP
    engine but deliberately returns only the first engine's reply.  Refit
    coverage needs all replies, so invoke the same per-engine utility primitive
    directly and flatten the one TP/PP result list returned by each EngineCore.
    This path is version-locked to the node-owned DP8 profile and fails closed
    if the pinned vLLM client shape changes.
    """
    engine_client = getattr(llm, "engine_core", None)
    if (
        engine_client is None
        or type(engine_client).__name__ != "DPLBAsyncMPClient"
    ):
        raise RuntimeError(
            "node-owned vLLM collective refit requires DPLBAsyncMPClient"
        )
    core_engines = tuple(getattr(engine_client, "core_engines", ()))
    if len(core_engines) != expected_engines:
        raise RuntimeError(
            "node-owned vLLM collective refit found the wrong number of "
            f"EngineCores: expected={expected_engines} actual={len(core_engines)}"
        )
    call_utility_async = getattr(engine_client, "_call_utility_async", None)
    if not callable(call_utility_async):
        raise RuntimeError(
            "node-owned vLLM collective refit is missing the pinned "
            "per-EngineCore utility method"
        )

    per_engine_results = await asyncio.gather(
        *(
            call_utility_async(
                "collective_rpc", method, None, args, None, engine=engine
            )
            for engine in core_engines
        )
    )
    worker_results: list[Any] = []
    for engine_index, results in enumerate(per_engine_results):
        if not isinstance(results, list | tuple) or len(results) != 1:
            raise RuntimeError(
                "node-owned vLLM collective refit returned an invalid "
                f"EngineCore result at engine_index={engine_index}"
            )
        worker_results.extend(results)
    return worker_results


def node_rollout_rank(
    *, rank_prefix: int, data_parallel_rank: int, rollout_world_size: int
) -> int:
    """Map one local DP8 rank to the global inference refit communicator."""
    if rank_prefix < 0 or rank_prefix % 8 or not 0 <= data_parallel_rank < 8:
        raise ValueError("node refit requires an eight-aligned prefix and local DP rank in [0, 8)")
    rollout_rank = rank_prefix + data_parallel_rank
    if not 0 <= rollout_rank < rollout_world_size:
        raise ValueError("node-owned refit rank exceeds rollout world size")
    return rollout_rank


def shared_response_allowed_tokens(config: dict[str, Any]) -> list[int] | None:
    """Build the exact common text support used by sampling and OPD scoring."""
    excluded = config.get("_shared_response_excluded_token_ids")
    if excluded is None:
        return None
    if any(isinstance(token, bool) or not isinstance(token, int) for token in excluded):
        raise ValueError("shared response exclusions must be integer token IDs")
    vocab_size = config["_shared_response_vocab_size"]
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("shared response vocabulary must be positive")
    excluded_set = set(excluded)
    if any(not 0 <= token < vocab_size for token in excluded_set):
        raise ValueError("shared response exclusion is outside the common vocabulary")
    allowed = [token for token in range(vocab_size) if token not in excluded_set]
    if not allowed:
        raise ValueError("shared response support cannot be empty")
    return allowed


def sampled_token_logprobs(generated_token_ids: list[int], logprobs: Any) -> list[float]:
    """Reject missing/nonfinite sampler scores instead of substituting logprob 0."""
    if logprobs is None or len(logprobs) != len(generated_token_ids):
        raise RuntimeError("vLLM sampled token logprob count does not match completion")
    result = []
    for token_id, token_logprobs in zip(generated_token_ids, logprobs):
        if token_logprobs is None or token_id not in token_logprobs:
            raise RuntimeError("vLLM did not return a sampled token logprob")
        value = float(token_logprobs[token_id].logprob)
        if not math.isfinite(value):
            raise RuntimeError("vLLM sampled token logprob is nonfinite")
        result.append(value)
    return result
