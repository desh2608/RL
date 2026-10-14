# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the actual collector method AST without starting Ray or loading models."""

import ast
import asyncio
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace


class GRPOConfigFixture:
    def __init__(self, enabled):
        self.grpo = SimpleNamespace(reward_training=SimpleNamespace(enabled=enabled))
        self.policy = {"max_total_sequence_length": 16384}


class RewardStreamingModeTests(unittest.TestCase):
    def invoke(self, config, opd_config):
        path = (
            Path(__file__).resolve().parents[3]
            / "nemo_rl/algorithms/async_utils/trajectory_collector.py"
        )
        source = ast.parse(path.read_text())
        cls = next(
            node
            for node in source.body
            if isinstance(node, ast.ClassDef)
            and node.name == "AsyncTrajectoryCollector"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_iter_rollout_groups"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                copy.deepcopy(method),
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        observed = []
        sentinel = object()

        async def fake_native_rollout(**kwargs):
            observed.append(kwargs)
            yield sentinel

        namespace = {
            "GRPOMasterConfig": GRPOConfigFixture,
            "run_async_multi_turn_rollout_groups": fake_native_rollout,
        }
        exec(compile(module, str(path), "exec"), namespace)
        instance = SimpleNamespace(
            master_config=config,
            on_policy_distillation_cfg=opd_config,
            policy_generation=None,
            tokenizer=None,
            task_to_env={},
            _max_rollout_turns=1,
            _deduplicate_multimodal_data=False,
        )

        async def collect():
            return [
                item
                async for item in namespace["_iter_rollout_groups"](
                    instance,
                    repeated_batch=object(),
                    num_generations=4,
                    use_nemo_gym=False,
                    task_index_to_group_index={},
                )
            ]

        self.assertEqual(asyncio.run(collect()), [sentinel])
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["num_generations"], 4)
        return observed[0]["stream_prompt_groups"]

    def test_reward_mode_streams_with_shared_or_absent_disabled_opd_mode(self):
        for opd_config in ({"enabled": False, "teacher_input_mode": "shared"}, {}):
            with self.subTest(opd_config=opd_config):
                self.assertIs(self.invoke(GRPOConfigFixture(True), opd_config), True)

    def test_legacy_independent_text_still_streams(self):
        self.assertIs(
            self.invoke(
                GRPOConfigFixture(False), {"teacher_input_mode": "independent_text"}
            ),
            True,
        )

    def test_ordinary_nonreward_grpo_retains_whole_batch_semantics(self):
        self.assertIs(
            self.invoke(GRPOConfigFixture(False), {"teacher_input_mode": "shared"}),
            False,
        )

    def test_ppo_union_member_does_not_read_a_nonexistent_grpo_field(self):
        ppo = SimpleNamespace(policy={"max_total_sequence_length": 16384})
        self.assertIs(self.invoke(ppo, {}), False)
        self.assertIs(
            self.invoke(ppo, {"teacher_input_mode": "independent_text"}), True
        )


if __name__ == "__main__":
    unittest.main()
