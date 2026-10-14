# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pytest

from nemo_rl.models.generation.vllm import vllm_generation


def _generation():
    generation = vllm_generation.VllmGeneration.__new__(
        vllm_generation.VllmGeneration
    )
    generation.__setstate__(
        {
            "_async_replica_ready_queue_enabled": True,
            "_async_replica_ready_queue_max_inflight": 64,
            "dp_size": 12,
        }
    )
    return generation


def test_target_threads_share_one_admission_queue(monkeypatch):
    """Hold construction open while the next target thread requests admission."""
    generation = _generation()
    original_queue = vllm_generation._AsyncReplicaReadyQueue
    constructing = threading.Event()
    release_constructor = threading.Event()
    second_call_started = threading.Event()
    queues = []

    def delayed_queue(*args):
        ready_queue = original_queue(*args)
        queues.append(ready_queue)
        constructing.set()
        assert release_constructor.wait(timeout=5)
        return ready_queue

    def second_target():
        second_call_started.set()
        return generation._ensure_async_replica_ready_queue()

    monkeypatch.setattr(vllm_generation, "_AsyncReplicaReadyQueue", delayed_queue)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(generation._ensure_async_replica_ready_queue)
        try:
            assert constructing.wait(timeout=5)
            second = executor.submit(second_target)
            assert second_call_started.wait(timeout=5)
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
        finally:
            release_constructor.set()
        first_queue = first.result(timeout=5)
        second_queue = second.result(timeout=5)

    assert first_queue is second_queue
    assert generation._ensure_async_replica_ready_queue() is first_queue
    assert queues == [first_queue]
    assert first_queue.capacity == 768


def test_pickle_rebuilds_process_local_lock_and_empty_queue():
    generation = _generation()
    original_queue = generation._ensure_async_replica_ready_queue()
    state = generation.__getstate__()
    assert "_async_replica_ready_queue_lock" not in state
    assert state["_async_replica_ready_queue"] is None

    restored = pickle.loads(pickle.dumps(generation))
    assert restored._async_replica_ready_queue is None
    assert (
        restored._async_replica_ready_queue_lock
        is not generation._async_replica_ready_queue_lock
    )
    restored_queue = restored._ensure_async_replica_ready_queue()
    assert restored_queue is not original_queue
    assert restored_queue.capacity == 768
    assert restored._ensure_async_replica_ready_queue() is restored_queue


def test_disabled_ready_queue_keeps_existing_route():
    generation = _generation()
    generation._async_replica_ready_queue_enabled = False
    assert generation._ensure_async_replica_ready_queue() is None
    assert generation._async_replica_ready_queue is None
