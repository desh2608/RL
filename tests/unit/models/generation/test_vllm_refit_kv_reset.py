# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""KV-cache reset ordering around the collective (DP8) refit.

Covers VllmAsyncGenerationWorkerImpl.resume_after_collective_refit_async: with
vllm_cfg.reset_running_requests_on_cache_invalidation=true every DP engine's
reset_prefix_cache(reset_running_requests=True) must run while the schedulers are
still paused, i.e. before resume_scheduler_async; with the flag off no preempt
happens. reset_prefix_cache_async() itself defaults to a plain reset so that
finish_generation after validation never preempts live rollouts, while
VllmGeneration.invalidate_kv_cache keeps passing the flag explicitly.

The tests drive the coroutines with asyncio.run so they also run under the
container's pytest stand-in (scripts/pytest_stub), which has no asyncio plugin.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nemo_rl.models.generation.vllm import vllm_generation as vllm_generation_module
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


class _FakeDPClient:
    """Records the order of per-engine utility calls and scheduler resume."""

    def __init__(self, n_engines: int = 8, per_engine_results=None, raise_on=None):
        self.core_engines = [f"engine-{i}" for i in range(n_engines)]
        self.calls: list[tuple] = []
        self._per_engine_results = per_engine_results or {}
        self._raise_on = raise_on

    async def _call_utility_async(self, method, *args, engine):
        self.calls.append(("utility", method, args, engine))
        if self._raise_on is not None and engine == self._raise_on:
            raise Exception("Failed to reset KV cache even when all the running requests are preempted")
        return self._per_engine_results.get(engine, True)

    async def resume_scheduler_async(self):
        self.calls.append(("resume_scheduler_async",))

    async def pause_scheduler_async(self, mode="abort", clear_cache=True):
        self.calls.append(("pause_scheduler_async", mode, clear_cache))


class _FakeEngine0OnlyClient:
    """A multi-engine client that hides the per-engine utility (older vLLM shape)."""

    def __init__(self, n_engines: int = 8):
        self.core_engines = [f"engine-{i}" for i in range(n_engines)]
        self.calls: list[tuple] = []

    async def resume_scheduler_async(self):
        self.calls.append(("resume_scheduler_async",))


def _make_paused_worker(client, *, reset_running_requests_on_cache_invalidation: bool):
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "vllm_cfg": {
            "async_engine": True,
            "reset_running_requests_on_cache_invalidation": reset_running_requests_on_cache_invalidation,
        }
    }
    worker.llm = SimpleNamespace(
        engine_core=client,
        reset_prefix_cache=AsyncMock(return_value=True),
    )
    worker._collective_refit_state = "paused"
    # Bypass the pinned DP8/DPLBAsyncMPClient type checks; ordering is what is under test.
    worker._collective_refit_scheduler_client = lambda: client
    return worker


def _reset_calls(client):
    return [c for c in client.calls if c[0] == "utility" and c[1] == "reset_prefix_cache"]


def test_resume_after_collective_refit_preempts_every_engine_before_resume(capsys):
    client = _FakeDPClient()
    worker = _make_paused_worker(client, reset_running_requests_on_cache_invalidation=True)

    assert asyncio.run(worker.resume_after_collective_refit_async()) is True

    assert worker._collective_refit_state == "idle"
    # Eight per-engine preempt+reset calls, then exactly one resume, in that order.
    assert len(client.calls) == 9
    assert client.calls[-1] == ("resume_scheduler_async",)
    resets = _reset_calls(client)
    assert len(resets) == 8
    assert all(call[2] == (True,) for call in resets)
    assert sorted(call[3] for call in resets) == sorted(client.core_engines)
    # The AsyncLLM entry point (engine-0-only result) was not used.
    worker.llm.reset_prefix_cache.assert_not_awaited()
    log = capsys.readouterr().out
    assert "[REFIT_KV_RESET] status=ok reset_running_requests=True path=per_engine engines=8" in log


def test_resume_after_collective_refit_does_not_preempt_when_flag_off(capsys):
    client = _FakeDPClient()
    worker = _make_paused_worker(client, reset_running_requests_on_cache_invalidation=False)

    assert asyncio.run(worker.resume_after_collective_refit_async()) is True

    assert worker._collective_refit_state == "idle"
    assert client.calls == [("resume_scheduler_async",)]
    worker.llm.reset_prefix_cache.assert_not_awaited()
    log = capsys.readouterr().out
    assert "[REFIT_KV_RESET] status=skipped" in log
    assert "reset_running_requests_on_cache_invalidation=false" in log


def test_resume_after_collective_refit_ands_results_across_engines(capsys):
    client = _FakeDPClient(per_engine_results={"engine-3": False})
    worker = _make_paused_worker(client, reset_running_requests_on_cache_invalidation=True)

    # A False from one engine is reported, but the resume still proceeds.
    assert asyncio.run(worker.resume_after_collective_refit_async()) is True
    assert worker._collective_refit_state == "idle"
    assert client.calls[-1] == ("resume_scheduler_async",)
    log = capsys.readouterr().out
    assert "[REFIT_KV_RESET] status=failed reset_running_requests=True path=per_engine engines=8" in log
    assert "False" in log


def test_resume_after_collective_refit_reset_error_marks_failed_and_keeps_pause(capsys):
    client = _FakeDPClient(raise_on="engine-5")
    worker = _make_paused_worker(client, reset_running_requests_on_cache_invalidation=True)

    with pytest.raises(Exception) as excinfo:
        asyncio.run(worker.resume_after_collective_refit_async())

    assert "Failed to reset KV cache" in str(excinfo.value)
    assert worker._collective_refit_state == "failed"
    # Never resume a DP group whose KV state is only half reset.
    assert ("resume_scheduler_async",) not in client.calls
    log = capsys.readouterr().out
    assert "[REFIT_KV_RESET] status=error" in log


def test_resume_after_collective_refit_falls_back_to_llm_when_per_engine_hidden(capsys):
    client = _FakeEngine0OnlyClient()
    worker = _make_paused_worker(client, reset_running_requests_on_cache_invalidation=True)
    order: list[str] = []
    worker.llm.reset_prefix_cache = AsyncMock(
        side_effect=lambda **kwargs: order.append(f"reset:{kwargs}") or True
    )

    async def _resume():
        order.append("resume")

    client.resume_scheduler_async = _resume

    assert asyncio.run(worker.resume_after_collective_refit_async()) is True

    assert order == ["reset:{'reset_running_requests': True}", "resume"]
    log = capsys.readouterr().out
    assert "[REFIT_KV_RESET] status=ok reset_running_requests=True path=llm_engine0_only" in log
    assert "caveat=DPLBAsyncMPClient.call_utility_async_returns_only_engine0_result" in log


def test_resume_after_collective_refit_requires_paused_state():
    client = _FakeDPClient()
    worker = _make_paused_worker(client, reset_running_requests_on_cache_invalidation=True)
    worker._collective_refit_state = "idle"

    with pytest.raises(RuntimeError, match="was not paused"):
        asyncio.run(worker.resume_after_collective_refit_async())
    assert client.calls == []


def test_reset_prefix_cache_async_default_is_plain_reset_even_with_flag_on():
    """finish_generation (validation) calls reset_prefix_cache_async() with no args and
    must not preempt live rollouts, whatever the refit flag says."""
    client = _FakeDPClient()
    worker = _make_paused_worker(client, reset_running_requests_on_cache_invalidation=True)

    assert asyncio.run(worker.reset_prefix_cache_async()) is True

    resets = _reset_calls(client)
    assert len(resets) == 8
    assert all(call[2] == () for call in resets)
    assert ("resume_scheduler_async",) not in client.calls


def test_reset_prefix_cache_async_preempts_only_when_requested(capsys):
    client = _FakeDPClient(per_engine_results={"engine-1": False})
    worker = _make_paused_worker(client, reset_running_requests_on_cache_invalidation=False)

    # Explicit request wins over the (off) config flag; result is the AND over engines.
    assert asyncio.run(worker.reset_prefix_cache_async(reset_running_requests=True)) is False

    resets = _reset_calls(client)
    assert len(resets) == 8
    assert all(call[2] == (True,) for call in resets)
    log = capsys.readouterr().out
    assert "prefix cache reset reported failure (reset_running_requests=True, path=per_engine" in log


def test_reset_prefix_cache_async_type_error_fallback_on_single_engine(capsys):
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {"vllm_cfg": {"async_engine": True}}
    worker.llm = SimpleNamespace(
        reset_prefix_cache=AsyncMock(side_effect=[TypeError("unexpected keyword"), True])
    )

    assert asyncio.run(worker.reset_prefix_cache_async(reset_running_requests=True)) is True

    assert worker.llm.reset_prefix_cache.await_count == 2
    assert worker.llm.reset_prefix_cache.await_args_list[0].kwargs == {
        "reset_running_requests": True
    }
    assert worker.llm.reset_prefix_cache.await_args_list[1].kwargs == {}
    log = capsys.readouterr().out
    assert "no reset_running_requests argument" in log


def test_reset_prefix_cache_async_rejects_sync_engine():
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {"vllm_cfg": {"async_engine": False}}
    worker.llm = SimpleNamespace(reset_prefix_cache=AsyncMock(return_value=True))

    with pytest.raises(RuntimeError, match="async_engine=True"):
        asyncio.run(worker.reset_prefix_cache_async())


def _make_generation(vllm_cfg: dict):
    gen = VllmGeneration.__new__(VllmGeneration)
    gen.cfg = {"vllm_cfg": vllm_cfg}
    # VllmGeneration.__del__ calls shutdown(); keep it a silent no-op for the bare object.
    gen.weight_synchronizer = None
    recorded: list[tuple] = []

    class _WorkerGroup:
        def run_all_workers_single_data(self, method_name, *args, run_rank_0_only_axes=None, **kwargs):
            recorded.append((method_name, args, run_rank_0_only_axes, kwargs))
            return [True, True]

        def shutdown(self, cleanup_method=None):
            return True

    gen.worker_group = _WorkerGroup()
    return gen, recorded


def test_invalidate_kv_cache_passes_preempt_flag_to_async_workers():
    gen, recorded = _make_generation(
        {"async_engine": True, "reset_running_requests_on_cache_invalidation": True}
    )
    with patch.object(vllm_generation_module.ray, "get", lambda futures: futures):
        assert gen.invalidate_kv_cache() is True
    assert recorded == [
        (
            "reset_prefix_cache_async",
            (),
            ["tensor_parallel", "pipeline_parallel"],
            {"reset_running_requests": True},
        )
    ]


def test_invalidate_kv_cache_defaults_to_plain_reset_without_flag():
    gen, recorded = _make_generation({"async_engine": True})
    with patch.object(vllm_generation_module.ray, "get", lambda futures: futures):
        assert gen.invalidate_kv_cache() is True
    assert recorded[0][0] == "reset_prefix_cache_async"
    assert recorded[0][3] == {"reset_running_requests": False}


def test_invalidate_kv_cache_sync_engine_call_shape_unchanged():
    gen, recorded = _make_generation(
        {"async_engine": False, "reset_running_requests_on_cache_invalidation": True}
    )
    with patch.object(vllm_generation_module.ray, "get", lambda futures: futures):
        assert gen.invalidate_kv_cache() is True
    assert recorded == [
        ("reset_prefix_cache", (), ["tensor_parallel", "pipeline_parallel"], {})
    ]
