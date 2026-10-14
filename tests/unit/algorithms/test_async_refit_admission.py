# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU scheduling tests for the actual collector's refit/start boundary."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from nemo_rl.algorithms.async_utils import trajectory_collector as module


class Batch:
    size = 1

    def __init__(self, before_repeat=None):
        self.before_repeat = before_repeat

    def get(self, key):
        return [{module.NEMO_GYM_TASK_INDEX_KEY: 101}] if key == "extra_env_info" else None

    def slice(self, start, end):
        return self

    def repeat_interleave(self, repeats, *, share_immutable_media=False):
        if self.before_repeat is not None:
            self.before_repeat()
        return self


def make_collector(monkeypatch, finish_worker):
    cls = module.AsyncTrajectoryCollector.__ray_metadata__.modified_class
    collector = cls.__new__(cls)
    collector.current_weight_version = 4
    collector.running = True
    collector._num_generations_per_prompt = 2
    collector._num_prompts_per_step = 1
    collector._max_trajectory_age_steps = 1
    collector._threads_lock = threading.Lock()
    collector._inflight_threads = set()
    collector._outstanding_lock = threading.Lock()
    collector._outstanding_task_indices = set()
    collector._generation_check_lock = threading.Lock()
    collector._generating_targets = {5}
    collector._get_next_target_for_generation = lambda version: 5
    collector._refit_pause_cleared = threading.Event()
    collector._refit_pause_cleared.set()
    collector._generation_limit_cleared = threading.Event()
    collector._generation_limit_cleared.set()
    collector._debug_payload_metrics = False
    collector._deduplicate_multimodal_data = False
    collector._efficiency_timer = SimpleNamespace(time=lambda name: nullcontext())
    collector.master_config = SimpleNamespace(
        policy={"generation": {"backend": "vllm", "vllm_cfg": {"async_engine": True}}}
    )
    collector.async_config = SimpleNamespace(
        in_flight_weight_updates=False,
        recompute_kv_cache_after_weight_updates=False,
    )
    collector.replay_buffer = SimpleNamespace(
        get_trajectories_needed=SimpleNamespace(remote=lambda *args: 1)
    )
    observed = []
    worker_started = threading.Event()

    async def rollout_worker(**kwargs):
        observed.append((kwargs["generation_weight_version"], kwargs["target_weight_version"]))
        worker_started.set()
        while not finish_worker.is_set():
            await asyncio.sleep(0.001)

    collector._run_rollout_batch_worker = rollout_worker
    monkeypatch.setattr(module.ray, "get", lambda value: value)
    monkeypatch.setattr(module, "should_use_nemo_gym", lambda config: False)
    return collector, observed, worker_started


def test_refit_cannot_retire_registered_thread_before_start(monkeypatch):
    before_start = threading.Event()
    release_start = threading.Event()
    finish_worker = threading.Event()
    collector, observed, worker_started = make_collector(monkeypatch, finish_worker)
    actual_threads = []

    class DelayedThread:
        def __init__(self, *, target, daemon):
            self.inner = threading.Thread(target=target, daemon=daemon)
            actual_threads.append(self.inner)

        def start(self):
            before_start.set()
            assert release_start.wait(timeout=5)
            self.inner.start()

        def is_alive(self):
            return self.inner.is_alive()

    monkeypatch.setattr(module, "_threading", SimpleNamespace(Thread=DelayedThread))
    with ThreadPoolExecutor(max_workers=2) as executor:
        process = executor.submit(collector._process_batch, Batch())
        try:
            assert before_start.wait(timeout=5)
            refit = executor.submit(collector.prepare_for_refit)
            # A registered but not-yet-started worker must not disappear from drain.
            with pytest.raises(TimeoutError):
                refit.result(timeout=0.1)
            release_start.set()
            assert worker_started.wait(timeout=5)
            assert process.result(timeout=5) is None
            with pytest.raises(TimeoutError):
                refit.result(timeout=0.1)
            finish_worker.set()
            refit.result(timeout=5)
        finally:
            release_start.set()
            finish_worker.set()
            collector._refit_pause_cleared.set()
    for worker in actual_threads:
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert observed == [(4, 5)]


def test_pause_after_early_check_delays_start_and_restamps_version(monkeypatch):
    after_early_check = threading.Event()
    release_repeat = threading.Event()
    finish_worker = threading.Event()
    collector, observed, worker_started = make_collector(monkeypatch, finish_worker)

    def pause_after_check():
        after_early_check.set()
        assert release_repeat.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        process = executor.submit(collector._process_batch, Batch(pause_after_check))
        try:
            assert after_early_check.wait(timeout=5)
            collector.prepare_for_refit()
            release_repeat.set()
            assert not worker_started.wait(timeout=0.1)
            with pytest.raises(TimeoutError):
                process.result(timeout=0.1)
            collector.set_weight_version(5)
            collector.resume_after_refit()
            assert worker_started.wait(timeout=5)
            assert process.result(timeout=5) is None
        finally:
            release_repeat.set()
            finish_worker.set()
            collector._refit_pause_cleared.set()
    collector.wait_for_pending_generations()
    assert observed == [(5, 5)]


def test_start_failure_releases_registration_and_target(monkeypatch):
    finish_worker = threading.Event()
    collector, observed, _ = make_collector(monkeypatch, finish_worker)

    class FailingThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError("injected thread.start failure")

        def is_alive(self):
            return False

    monkeypatch.setattr(module, "_threading", SimpleNamespace(Thread=FailingThread))
    batch = Batch()
    assert collector._process_batch(batch) is batch
    assert not collector._inflight_threads
    assert not collector._outstanding_task_indices
    assert not collector._generating_targets
    assert observed == []


def test_complete_refit_between_checks_uses_new_weight_version(monkeypatch):
    after_early_check = threading.Event()
    release_repeat = threading.Event()
    finish_worker = threading.Event()
    collector, observed, worker_started = make_collector(monkeypatch, finish_worker)

    def pause_after_check():
        after_early_check.set()
        assert release_repeat.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        process = executor.submit(collector._process_batch, Batch(pause_after_check))
        try:
            assert after_early_check.wait(timeout=5)
            collector.prepare_for_refit()
            collector.set_weight_version(5)
            collector.resume_after_refit()
            # The event is set at both observations (ABA); its state alone
            # cannot show that the previously captured version is stale.
            release_repeat.set()
            assert worker_started.wait(timeout=5)
            assert process.result(timeout=5) is None
        finally:
            release_repeat.set()
            finish_worker.set()
            collector._refit_pause_cleared.set()
    collector.wait_for_pending_generations()
    assert observed == [(5, 5)]
