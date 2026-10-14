# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Async GRPO stall fixes: gap-fill window, lost wakeup, masked-row advantages, unshaped reward.

Observed on smoke jobs 35879179/35881263 (2026-10-01): after zero-variance drops left
a target short, the collector could never refill the step the trainer was waiting on.
"""

import inspect
import re
import threading
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import torch

import nemo_rl.experience.rollouts as rollouts_mod
from nemo_rl.algorithms import grpo as grpo_mod
from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.async_utils import trajectory_collector as tc
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.timer import ThreadSafeTimer

_meta = getattr(tc.AsyncTrajectoryCollector, "__ray_metadata__", None)
_plain = _meta.modified_class if _meta is not None else tc.AsyncTrajectoryCollector


# --------------------------------------------------------------------------------------
# (1) target window includes the current version
# --------------------------------------------------------------------------------------


class _FakeRef:
    def __init__(self, value):
        self.value = value


class _FakeRay:
    @staticmethod
    def get(ref):
        return ref.value if isinstance(ref, _FakeRef) else ref


@contextmanager
def _fake_ray():
    real = tc.ray
    tc.ray = _FakeRay()
    try:
        yield
    finally:
        tc.ray = real


class _FakeBuffer:
    """Replay buffer stand-in: last consumed target + per-target shortfall."""

    def __init__(self, *, last_consumed, needed):
        self.calls = []
        self.get_last_target_weight_already_generated = SimpleNamespace(
            remote=lambda: _FakeRef(last_consumed)
        )

        def _needed(target, num_prompts, max_age):
            self.calls.append((target, num_prompts, max_age))
            return _FakeRef(needed.get(target, num_prompts))

        self.get_trajectories_needed = SimpleNamespace(remote=_needed)


class _WindowCollector:
    """The target-selection slice of AsyncTrajectoryCollector without the Ray actor."""

    def __init__(self, *, current, buffer, initial=0, lead=2, num_prompts=8):
        self.current_weight_version = current
        self.initial_weight_version = initial
        self._generation_lead_steps = lead
        self._max_trajectory_age_steps = lead
        self._num_prompts_per_step = num_prompts
        self.replay_buffer = buffer
        self._generation_check_lock = threading.Lock()
        self._generating_targets = set()

    _calculate_target_weights = _plain._calculate_target_weights
    _get_next_target_for_generation = _plain._get_next_target_for_generation
    _should_pause_for_generation_limits = _plain._should_pause_for_generation_limits


def test_target_window_starts_at_current_version_for_every_version():
    collector = _WindowCollector(current=0, buffer=None, initial=0, lead=2)
    assert collector._calculate_target_weights(0) == [0, 1, 2]
    # Previously [6, 7]: the trainer waiting on target 5 could never be gap-filled.
    assert collector._calculate_target_weights(5) == [5, 6, 7]
    collector = _WindowCollector(current=10, buffer=None, initial=0, lead=4)
    assert collector._calculate_target_weights(10) == [10, 11, 12, 13, 14]
    # Resume: initial version other than 0 behaves the same way.
    collector = _WindowCollector(current=30, buffer=None, initial=30, lead=2)
    assert collector._calculate_target_weights(30) == [30, 31, 32]
    assert collector._calculate_target_weights(31) == [31, 32, 33]


def test_short_current_target_is_selected_for_gap_fill():
    """Current version t, target t short by 3 (zero-variance drops): reserve t."""
    t = 7
    buffer = _FakeBuffer(last_consumed=t - 1, needed={t: 3, t + 1: 0, t + 2: 0})
    collector = _WindowCollector(current=t, buffer=buffer, lead=2)
    with _fake_ray():
        assert collector._should_pause_for_generation_limits() is False
        assert collector._get_next_target_for_generation(t) == t
        assert collector._generating_targets == {t}
        # While the gap-fill worker holds t and the lookahead is full, pause ...
        assert collector._should_pause_for_generation_limits() is True
        # ... and wake up to re-check once the worker releases the target.
        collector._generating_targets.discard(t)
        assert collector._should_pause_for_generation_limits() is False
    assert buffer.calls[0][0] == t


def test_consumed_current_target_is_skipped():
    """Once the trainer has sampled step t (last_consumed == t) t is never regenerated."""
    t = 7
    buffer = _FakeBuffer(last_consumed=t, needed={t: 3, t + 1: 8, t + 2: 8})
    collector = _WindowCollector(current=t, buffer=buffer, lead=2)
    with _fake_ray():
        assert collector._get_next_target_for_generation(t) == t + 1
    assert all(target != t for target, _, _ in buffer.calls)


def test_full_window_pauses_until_a_target_is_short():
    t = 3
    buffer = _FakeBuffer(last_consumed=t - 1, needed={t: 0, t + 1: 0, t + 2: 0})
    collector = _WindowCollector(current=t, buffer=buffer, lead=2)
    with _fake_ray():
        assert collector._get_next_target_for_generation(t) is None
        assert collector._should_pause_for_generation_limits() is True
        assert collector._generating_targets == set()


# --------------------------------------------------------------------------------------
# (2) lost wakeup between the pause check and clear()
# --------------------------------------------------------------------------------------


class _LoopHarness:
    """Minimal state for AsyncTrajectoryCollector._collection_loop.

    ``_should_pause_for_generation_limits`` returns a *stale* "pause" verdict: it
    decides from the pre-release state, then (still inside the check) performs the
    batch worker's finally-block — release the target and set() the wake event —
    before returning. With check-then-clear ordering that wake is lost and the loop
    sleeps until the next weight update; with clear-then-check it survives.
    """

    worker_target = 5

    def __init__(self, *, stale_pause: bool):
        self.stale_pause = stale_pause
        self.running = True
        self.data_exhausted = False
        self.collection_failed = False
        self.collection_error = None
        self._failure_lock = threading.Lock()
        self.dataloader = [{"batch": 0}]
        for name in ("_manual_pause_cleared", "_refit_pause_cleared", "_generation_limit_cleared"):
            event = threading.Event()
            event.set()
            setattr(self, name, event)
        self._efficiency_timer = ThreadSafeTimer(context={"worker": "collector"})
        self._last_limit_warning_version = None
        self.current_weight_version = 4
        self._pending_lock = threading.Lock()
        self._pending_batch = None
        self._next_nemo_gym_task_index = 0
        self._snapshot_lock = threading.Lock()
        self._dataloader_snapshots = deque()
        self._generating_targets = {self.worker_target} if stale_pause else set()
        self.worker_finished = False
        self.processed = threading.Event()
        self.verdicts = []

    _collection_loop = _plain._collection_loop
    _mark_collection_failed = _plain._mark_collection_failed

    def _capture_dataloader_state(self):
        return None

    def _stamp_task_indices(self, batch):
        return False

    def _filter_covered_rows(self, batch):
        return batch

    def _calculate_target_weights(self, version):
        return [version, version + 1]

    def _should_pause_for_generation_limits(self):
        verdict = self.worker_target in self._generating_targets
        if verdict and not self.worker_finished:
            # The batch worker finishes short *during* the check: _release_target + wake.
            self._generating_targets.discard(self.worker_target)
            self._generation_limit_cleared.set()
            self.worker_finished = True
        self.verdicts.append(verdict)
        return verdict

    def _process_batch(self, batch):
        self.processed.set()
        self.running = False  # one gap-fill is all this harness needs
        return None

    def run(self, timeout=3.0):
        thread = threading.Thread(target=self._collection_loop, daemon=True)
        thread.start()
        reached = self.processed.wait(timeout=timeout)
        if not reached:
            self.running = False
            self._generation_limit_cleared.set()
        thread.join(timeout=timeout)
        assert not thread.is_alive()
        return reached


def test_worker_wake_during_pause_check_is_not_lost():
    harness = _LoopHarness(stale_pause=True)
    assert harness.run() is True, "collection loop lost the worker's wake and never gap-filled"
    assert harness.verdicts[0] is True  # it did decide to pause on the stale state
    assert harness.worker_finished is True
    assert harness.data_exhausted is False


def test_not_pausing_leaves_wake_event_set():
    harness = _LoopHarness(stale_pause=False)
    assert harness.run() is True
    assert harness.verdicts == [False]
    # clear-before-check must restore the event when the loop does not pause, so
    # set_weight_version keeps seeing a non-paused collector.
    assert harness._generation_limit_cleared.is_set()


def test_collection_loop_clears_wake_event_before_pause_check():
    source = inspect.getsource(_plain._collection_loop)
    clear_at = source.index("self._generation_limit_cleared.clear()")
    check_at = source.index("if self._should_pause_for_generation_limits() and self.running:")
    assert clear_at < check_at
    assert source.count("self._generation_limit_cleared.clear()") == 1


# --------------------------------------------------------------------------------------
# (4) masked rows never enter the group baseline on the async path
# --------------------------------------------------------------------------------------


def _estimator():
    config = AdvEstimatorConfig(name="grpo", normalize_rewards=True, use_leave_one_out_baseline=True)
    return GRPOAdvantageEstimator(config, loss_config=None)


def test_masked_row_does_not_shift_valid_siblings_advantages():
    rewards = torch.tensor([1.0] * 7 + [0.0])
    prompt_ids = grpo_mod.build_rollout_group_ids(8, 8)
    mask = torch.ones(8, 3)
    valid = torch.tensor([1.0] * 7 + [0.0])  # the 0-reward row is a masked placeholder

    advantages = _estimator().compute_advantage(prompt_ids=prompt_ids, rewards=rewards, mask=mask, reward_valid_mask=valid)
    assert advantages.shape == (8, 3)
    assert torch.all(advantages == 0), advantages

    # Without the validity mask the placeholder leaks into the baseline/std.
    leaked = _estimator().compute_advantage(prompt_ids=prompt_ids, rewards=rewards, mask=mask)
    assert torch.any(leaked[:7] != 0)


def test_singleton_valid_group_has_zero_finite_advantage():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
    prompt_ids = grpo_mod.build_rollout_group_ids(4, 4)
    valid = torch.tensor([1.0, 0.0, 0.0, 0.0])
    advantages = _estimator().compute_advantage(
        prompt_ids=prompt_ids, rewards=rewards, mask=torch.ones(4, 2), reward_valid_mask=valid
    )
    assert torch.isfinite(advantages).all()
    assert torch.all(advantages == 0)


def test_async_grpo_passes_final_sample_mask_as_reward_valid_mask():
    source = inspect.getsource(grpo_mod.async_grpo_train)
    assert (
        re.search(r"reward_valid_mask=sample_mask\s*\n\s*if master_config\.grpo\.reward_training\.enabled", source)
        is None
    ), "async advantage call still gates reward_valid_mask on reward_training.enabled"
    call = source[source.index("train_data[\"advantages\"] = adv_estimator.compute_advantage(") :]
    call = call[: call.index('hasattr(adv_estimator, "last_metrics")')]  # the statement after the call
    assert "reward_valid_mask=sample_mask," in call, call
    assert 'sample_mask=train_data["sample_mask"],' in call, call


# --------------------------------------------------------------------------------------
# (3) unshaped base reward column on NeMo-Gym groups
# --------------------------------------------------------------------------------------


def test_unshaped_total_reward_reads_tau2_reward_shaping():
    unshaped = rollouts_mod._unshaped_total_reward
    penalised = {"reward": -0.05, "reward_shaping": {"reward": -0.05, "raw_reward": 0.0, "zeroed_on_transfer": False, "tool_not_found_penalty": 0.05}}
    assert unshaped(penalised) == 0.0
    transferred = {"reward": 0.0, "reward_shaping": {"reward": 0.0, "raw_reward": 1.0, "zeroed_on_transfer": True, "tool_not_found_penalty": 0.0}}
    assert unshaped(transferred) == 0.0
    solved = {"reward": 0.95, "reward_shaping": {"reward": 0.95, "raw_reward": 1.0, "zeroed_on_transfer": False, "tool_not_found_penalty": 0.05}}
    assert unshaped(solved) == 1.0
    assert unshaped({"reward": 0.7}) == 0.7  # env without shaping
    assert unshaped({"reward": 1, "reward_shaping": {}}) == 1.0  # tau2 default before shaping ran


def _assistant(n_tokens):
    return {
        "role": "assistant",
        "content": "a",
        "token_ids": torch.tensor([2] * n_tokens),
        "generation_logprobs": torch.tensor([-0.1] * n_tokens),
    }


def _user(n_tokens):
    return {"role": "user", "content": "u", "token_ids": torch.tensor([1] * n_tokens)}


def _result(full_result):
    log = [_user(1), _assistant(2)]
    return {"input_message_log": log[:1], "message_log": log, "full_result": full_result}


def _postprocess(results):
    rows = [{"agent_ref": {"name": "tau2"}}] * len(results)
    return rollouts_mod._postprocess_single_nemo_gym_group(
        nemo_gym_rows=rows,
        results=results,
        timer=rollouts_mod.Timer(),
        timer_prefix="timing/rollout",
        policy_generation=type("_PolicyGeneration", (), {"cfg": {"max_new_tokens": 8, "vllm_cfg": {"max_model_len": 128}}})(),
        input_batch=BatchedDataDict({"loss_multiplier": torch.ones(len(results))}),
        tokenizer=type("_Tokenizer", (), {"pad_token_id": 0})(),
        log_full_result_tables=False,
    )


def test_postprocess_group_carries_unshaped_total_reward():
    shaping = lambda reward, raw, transfer=False: {"reward": reward, "raw_reward": raw, "zeroed_on_transfer": transfer, "tool_not_found_penalty": raw - reward if not transfer else 0.0}
    results = [
        _result({"reward": 0.0, "reward_shaping": shaping(0.0, 0.0)}),
        _result({"reward": -0.05, "reward_shaping": shaping(-0.05, 0.0)}),
        _result({"reward": 0.0, "reward_shaping": shaping(0.0, 1.0, transfer=True)}),
        _result({"reward": 0.4}),  # no shaping block: base == reward
    ]
    final_batch = _postprocess(results).final_batch
    # Both columns are float32 tensors: compare with tolerance, not Python-float equality.
    assert torch.allclose(final_batch["total_reward"], torch.tensor([0.0, -0.05, 0.0, 0.4])), final_batch["total_reward"]
    assert torch.allclose(final_batch["unshaped_total_reward"], torch.tensor([0.0, 0.0, 0.0, 0.4])), final_batch["unshaped_total_reward"]
    assert final_batch["unshaped_total_reward"].shape == final_batch["total_reward"].shape
    assert final_batch["unshaped_total_reward"].dtype == torch.float32

    # The shaping-only group is zero-variance for the collector even though total_reward varies.
    penalty_only = _postprocess(results[:2]).final_batch
    assert torch.allclose(penalty_only["total_reward"], torch.tensor([0.0, -0.05]))
    assert tc._group_rewards_identical(penalty_only) is True
    # Without the base-reward column the same group would have been kept.
    assert tc._group_rewards_identical(BatchedDataDict({"total_reward": penalty_only["total_reward"]})) is False
