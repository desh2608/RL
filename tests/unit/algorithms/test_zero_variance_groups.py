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
"""env.nemo_gym.drop_zero_variance_groups: collector-side K-group filter."""

import asyncio
import threading
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.algorithms.async_utils import trajectory_collector as tc
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.nemo_gym import (
    DROP_ZERO_VARIANCE_GROUPS_KEY,
    should_drop_zero_variance_groups,
)
from nemo_rl.experience.interfaces import NEMO_RL_EMPTY_RESPONSE_OUTPUT_KEY


def test_flag_defaults_off_and_validates_type():
    assert should_drop_zero_variance_groups({}) is False
    assert should_drop_zero_variance_groups({"nemo_gym": {}}) is False
    assert should_drop_zero_variance_groups({"nemo_gym": {DROP_ZERO_VARIANCE_GROUPS_KEY: True}}) is True
    with pytest.raises(ValueError):
        should_drop_zero_variance_groups({"nemo_gym": {DROP_ZERO_VARIANCE_GROUPS_KEY: "yes"}})


def test_flag_is_stripped_before_gym_sees_the_global_config():
    # spinup_nemo_gym_actor pops the knob like invalid_tool_call_patterns / thinking_tags.
    import nemo_rl.environments.nemo_gym as ng

    source = open(ng.__file__).read()
    spin = source[source.index("def spinup_nemo_gym_actor") :]
    assert f'nemo_gym_dict.pop({DROP_ZERO_VARIANCE_GROUPS_KEY!r}' not in spin  # popped via the constant
    assert "nemo_gym_dict.pop(DROP_ZERO_VARIANCE_GROUPS_KEY, None)" in spin
    actor = source[source.index("def _spinup") : source.index("def spinup_nemo_gym_actor")]
    assert "initial_global_config_dict.pop(DROP_ZERO_VARIANCE_GROUPS_KEY, None)" in actor


def _batch(rewards, agent="tool_agent", *, mask_sample=None, empty=None, unshaped=None):
    data = {
        "total_reward": torch.tensor(rewards, dtype=torch.float32),
        "agent_ref": [{"type": "responses_api_agents", "name": agent}] * len(rewards),
    }
    if mask_sample is not None:
        data["mask_sample"] = torch.tensor(mask_sample, dtype=torch.bool)
    if empty is not None:
        data[NEMO_RL_EMPTY_RESPONSE_OUTPUT_KEY] = torch.tensor(empty, dtype=torch.bool)
    if unshaped is not None:
        data["unshaped_total_reward"] = torch.tensor(unshaped, dtype=torch.float32)
    return BatchedDataDict(data)


def test_group_rewards_identical():
    assert tc._group_rewards_identical(_batch([1.0, 1.0, 1.0, 1.0])) is True
    assert tc._group_rewards_identical(_batch([0.0, 0.0])) is True
    assert tc._group_rewards_identical(_batch([0.0, 1.0, 0.0, 0.0])) is False
    assert tc._group_rewards_identical(_batch([0.95, 1.0])) is False  # shaped rewards count as variance
    # A single valid rollout has no baseline to compare against: no advantage anywhere, droppable.
    assert tc._group_rewards_identical(_batch([1.0])) is True
    assert tc._group_rewards_identical(BatchedDataDict({"agent_ref": [{"name": "x"}]})) is False


def test_group_rewards_identical_ignores_masked_and_placeholder_rows():
    seven_ones_and_a_zero = [1.0] * 7 + [0.0]
    # The lone differing row is env-masked (mask_sample): it gets loss_multiplier 0 and is
    # excluded from the baseline, so the seven valid rows are all identical -> drop.
    assert tc._group_rewards_identical(_batch(seven_ones_and_a_zero, mask_sample=[False] * 7 + [True])) is True
    # Same when the differing row is an empty-response placeholder.
    assert tc._group_rewards_identical(_batch(seven_ones_and_a_zero, empty=[False] * 7 + [True])) is True
    # Unmasked, the zero is real variance -> keep.
    assert tc._group_rewards_identical(_batch(seven_ones_and_a_zero)) is False
    assert tc._group_rewards_identical(_batch([1.0] + [0.0] * 7)) is False
    # Masking a row that agrees with the rest changes nothing.
    assert tc._group_rewards_identical(_batch(seven_ones_and_a_zero, mask_sample=[True] + [False] * 7)) is False
    # Fewer than two valid rows left -> droppable, even if the masked rows differ.
    assert tc._group_rewards_identical(_batch([1.0, 0.0, 0.5], mask_sample=[False, True, True])) is True
    assert tc._group_rewards_identical(_batch([1.0, 0.0], mask_sample=[True, True])) is True
    # A mask column of the wrong length is ignored rather than trusted.
    assert tc._group_rewards_identical(_batch([1.0, 0.0], mask_sample=[True])) is False


def test_group_rewards_identical_prefers_unshaped_base_reward():
    # tau2 tool_not_found penalty: shaped [0,...,0,-0.05] with base reward all 0 is a
    # shaping-only delta -> drop (otherwise normalize_rewards turns 0.05 into +-0.378 advantages).
    shaped = [0.0] * 7 + [-0.05]
    assert tc._group_rewards_identical(_batch(shaped, unshaped=[0.0] * 8)) is True
    # Without the base-reward column the shaped delta still counts as variance (legacy behaviour).
    assert tc._group_rewards_identical(_batch(shaped)) is False
    # Real task signal in the base reward is kept even when shaping is identical.
    assert tc._group_rewards_identical(_batch([0.95] * 8, unshaped=[1.0] + [0.0] * 7)) is False
    # Masking still applies on top of the unshaped column.
    assert tc._group_rewards_identical(
        _batch([1.0] * 7 + [-0.05], unshaped=[1.0] * 7 + [0.0], mask_sample=[False] * 7 + [True])
    ) is True
    assert tc._group_reward_values(_batch([1.0, 0.0, 0.5], unshaped=[1.0, 1.0, 0.0], mask_sample=[False, False, True])) == [1.0, 1.0]


def test_group_agent_name_sources():
    assert tc._group_agent_name(_batch([0.0, 1.0], agent="tau2")) == "tau2"
    only_rows = BatchedDataDict({"extra_env_info": [{"agent_ref": {"name": "conv"}}]})
    assert tc._group_agent_name(only_rows) == "conv"
    assert tc._group_agent_name(BatchedDataDict({})) == "unknown"


class _Collector:
    """Just the counters and the two methods under test, without the Ray actor."""

    def __init__(self, enabled=True):
        self._drop_zero_variance_groups = enabled
        self._group_counts_lock = threading.Lock()
        self._groups_seen = Counter()
        self._groups_dropped_zero_variance = Counter()
        self._groups_seen_reported = Counter()
        self._groups_dropped_reported = Counter()

    # AsyncTrajectoryCollector is a Ray ActorClass; unwrap to the plain class for its methods.
    _plain = getattr(tc.AsyncTrajectoryCollector, "__ray_actor_class__", tc.AsyncTrajectoryCollector)
    _record_group_outcome = _plain._record_group_outcome
    get_zero_variance_drop_metrics = _plain.get_zero_variance_drop_metrics


def test_drop_metrics_are_per_agent_interval_and_cumulative():
    collector = _Collector()
    for _ in range(3):
        collector._record_group_outcome("tau2", dropped=True)
    collector._record_group_outcome("tau2", dropped=False)
    collector._record_group_outcome("single_step", dropped=False)
    collector._record_group_outcome("single_step", dropped=False)

    first = collector.get_zero_variance_drop_metrics()
    assert first["zero_variance_groups/enabled"] == 1.0
    assert first["zero_variance_groups/seen"] == 6
    assert first["zero_variance_groups/dropped"] == 3
    assert first["zero_variance_groups/dropped_fraction"] == pytest.approx(0.5)
    assert first["zero_variance_groups/tau2/dropped_fraction"] == pytest.approx(0.75)
    assert first["zero_variance_groups/single_step/dropped_fraction"] == 0.0
    assert first["zero_variance_groups/tau2/dropped_total"] == 3

    # Next step: only new groups count toward the interval fraction; totals accumulate.
    collector._record_group_outcome("tau2", dropped=False)
    second = collector.get_zero_variance_drop_metrics()
    assert second["zero_variance_groups/seen"] == 1
    assert second["zero_variance_groups/dropped"] == 0
    assert second["zero_variance_groups/tau2/dropped_fraction"] == 0.0
    assert second["zero_variance_groups/tau2/dropped_total"] == 3
    assert second["zero_variance_groups/seen_total"] == 7
    # Idle agents report zero for the interval but keep their totals.
    assert second["zero_variance_groups/single_step/seen"] == 0
    assert second["zero_variance_groups/single_step/seen_total"] == 2


def test_drop_metrics_when_disabled_and_empty():
    metrics = _Collector(enabled=False).get_zero_variance_drop_metrics()
    assert metrics == {
        "zero_variance_groups/enabled": 0.0,
        "zero_variance_groups/seen": 0.0,
        "zero_variance_groups/dropped": 0.0,
        "zero_variance_groups/dropped_fraction": 0.0,
        "zero_variance_groups/seen_total": 0.0,
        "zero_variance_groups/dropped_total": 0.0,
    }


def test_batch_worker_always_wakes_collector_after_release():
    """After a worker releases its target the collection loop must be woken
    unconditionally: with zero-variance drops a successfully finished worker leaves the
    target short and the loop would otherwise sleep until the next weight update."""
    import inspect
    import re

    from nemo_rl.algorithms.async_utils import trajectory_collector as tc

    cls = tc.AsyncTrajectoryCollector
    cls = getattr(cls, "__ray_actor_class__", cls)
    source = inspect.getsource(cls._run_rollout_batch_worker)
    finally_block = source.split("finally:", 1)[1]
    assert "_release_target(target_weight_version)" in finally_block
    # The set() must not be guarded by the failure-only flag any more.
    assert re.search(r"\n\s*if wake_generation_limits_after_cleanup:\n\s*self\._generation_limit_cleared\.set\(\)", finally_block) is None
    assert "self._generation_limit_cleared.set()" in finally_block


class _KeepPathReached(Exception):
    """Raised by the harness once _enqueue_rollout_group decides to keep a group."""


class _FinalBatch(BatchedDataDict):
    def to(self, *args, **kwargs):  # first call on the keep path, right after the drop decision
        raise _KeepPathReached()


class _EnqueueCollector(_Collector):
    """Enough of AsyncTrajectoryCollector to run the drop/keep decision in _enqueue_rollout_group."""

    def __init__(self):
        super().__init__(enabled=True)
        self._outstanding_lock = threading.Lock()
        self._outstanding_task_indices = {11}
        self.master_config = None  # not a GRPO MasterConfig -> no target-version column
        self._has_distillation_teachers = False
        self.on_policy_distillation_cfg = {}

    _enqueue_rollout_group = _Collector._plain._enqueue_rollout_group

    def decide(self, final_batch):
        """Return "dropped" or "kept" for one K-group."""
        handled = set()
        result = SimpleNamespace(
            final_batch=_FinalBatch(dict(final_batch)),
            group_index=3,
            task_index=11,
            rollout_metrics={},
        )
        try:
            asyncio.run(
                self._enqueue_rollout_group(
                    result,
                    generation_weight_version=4,
                    target_weight_version=5,
                    expected_prompt_groups=8,
                    buffered_group_indices=handled,
                    collection_started_at=0.0,
                )
            )
        except _KeepPathReached:
            assert handled == set()
            return "kept"
        assert handled == {3}
        assert 11 not in self._outstanding_task_indices
        return "dropped"


def test_enqueue_drop_decision_uses_valid_rows_and_unshaped_reward(capsys):
    seven_ones_and_a_zero = [1.0] * 7 + [0.0]
    collector = _EnqueueCollector()
    assert collector.decide(_batch(seven_ones_and_a_zero, mask_sample=[False] * 7 + [True])) == "dropped"
    assert "valid_rows=7" in capsys.readouterr().out
    collector = _EnqueueCollector()
    assert collector.decide(_batch([0.0] * 7 + [-0.05], unshaped=[0.0] * 8)) == "dropped"
    collector = _EnqueueCollector()
    assert collector.decide(_batch([1.0] + [0.0] * 7)) == "kept"
    assert collector.get_zero_variance_drop_metrics()["zero_variance_groups/dropped"] == 0

