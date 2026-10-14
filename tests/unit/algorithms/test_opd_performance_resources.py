# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU throughput-denominator contracts for separate OPD teacher allocations."""

import contextlib
import io
import unittest
from types import SimpleNamespace

from nemo_rl.algorithms.opd import OnPolicyDistillationConfig
from nemo_rl.algorithms.utils import print_performance_metrics


def config(nodes, rollout_nodes, teacher_nodes=2, *, colocated=False):
    return SimpleNamespace(
        cluster={"num_nodes": nodes, "gpus_per_node": 8},
        policy={
            "generation": {
                "colocated": {
                    "enabled": colocated,
                    "resources": {"num_nodes": rollout_nodes, "gpus_per_node": 8},
                }
            }
        },
        on_policy_distillation={
            "enabled": True,
            "teacher_model_by_agent_name": {"nano": "/models/nano"},
            "non_colocated_teachers": {
                "enabled": True,
                "default_teacher_cfg": {"num_nodes": teacher_nodes, "gpus_per_node": 8},
            },
        },
    )


def measure(master_config, timing_overrides=None):
    timing = {
        "policy_training": 10,
        "policy_and_reference_logprobs": 2,
        "exposed_generation": 4,
        "total_step_time": 20,
    }
    timing.update(timing_overrides or {})
    with contextlib.redirect_stdout(io.StringIO()):
        return print_performance_metrics(
            {},
            {"total_num_tokens": 16000},
            timing,
            master_config,
            num_prompts_per_step=512,
            num_generations_per_prompt=2,
            is_async_rl=True,
        )


class OPDPerformanceResourcesTest(unittest.TestCase):
    def test_exposed_wait_above_tenth_second_is_reported_against_total_step(self):
        perf = measure(config(16, 12))
        self.assertEqual(perf["training_worker_idle_time_ratio"], 4 / 20)

    def test_idle_ratio_handles_zero_total_step_without_dividing_by_zero(self):
        perf = measure(config(16, 12), {"total_step_time": 0})
        self.assertEqual(perf["training_worker_idle_time_ratio"], 0)

    def test_formal_and_smoke_learner_counts_leave_e2e_allocation_count_unchanged(self):
        for nodes, rollout_nodes, teacher_nodes, learner_gpus in [(16, 12, 2, 16), (3, 1, 1, 8)]:
            with self.subTest(nodes=nodes):
                perf = measure(config(nodes, rollout_nodes, teacher_nodes))
                self.assertEqual(perf["training_num_gpus"], learner_gpus)
                self.assertEqual(perf["generation_num_gpus"], rollout_nodes * 8)
                self.assertEqual(perf["teacher_num_gpus"], teacher_nodes * 8)
                self.assertEqual(perf["total_num_gpus"], nodes * 8)
                self.assertEqual(perf["policy_training_tokens_per_sec_per_gpu"], 1600 / learner_gpus)
                self.assertEqual(perf["tokens_per_sec_per_gpu"], 800 / (nodes * 8))

    def test_checkpoint_aliases_share_first_resolved_teacher_with_overrides(self):
        master = config(16, 12)
        opd = master.on_policy_distillation
        opd["teacher_model_by_agent_name"]["nano_alias"] = "/models/nano"
        opd["non_colocated_teachers"]["teacher_overrides"] = {
            "nano": {"num_nodes": 1}, "nano_alias": {"num_nodes": 3}
        }
        # Exercise the same Pydantic config shape as the training driver.
        master.on_policy_distillation = OnPolicyDistillationConfig(**opd)
        perf = measure(master)
        self.assertEqual(perf["teacher_num_gpus"], 8)
        self.assertEqual(perf["training_num_gpus"], 24)

    def test_explicit_independent_teachers_and_unique_models_are_counted(self):
        for second_model, deduplicate in [("/models/nano", False), ("/models/other", True)]:
            with self.subTest(second_model=second_model, deduplicate=deduplicate):
                master = config(18, 12)
                master.on_policy_distillation["teacher_model_by_agent_name"]["second"] = second_model
                master.on_policy_distillation["deduplicate_shared_teacher_checkpoints"] = deduplicate
                perf = measure(master)
                self.assertEqual(perf["teacher_num_gpus"], 32)
                self.assertEqual(perf["training_num_gpus"], 16)

    def test_disabled_or_missing_opd_preserves_non_opd_accounting(self):
        for mode in ("missing", "disabled", "colocated_teacher"):
            with self.subTest(mode=mode):
                master = config(16, 12)
                if mode == "missing":
                    del master.on_policy_distillation
                elif mode == "disabled":
                    master.on_policy_distillation["enabled"] = False
                else:
                    master.on_policy_distillation["non_colocated_teachers"]["enabled"] = False
                perf = measure(master)
                self.assertEqual(perf["teacher_num_gpus"], 0)
                self.assertEqual(perf["training_num_gpus"], 32)

    def test_colocated_policy_excludes_reserved_teacher_nodes_and_reports_actual_teacher_gpus(self):
        master = config(4, 0, teacher_nodes=1, colocated=True)
        master.on_policy_distillation["non_colocated_teachers"]["default_teacher_cfg"]["gpus_per_node"] = 4
        perf = measure(master)
        self.assertEqual(perf["teacher_num_gpus"], 4)
        self.assertEqual(perf["teacher_reserved_num_gpus"], 8)
        self.assertEqual(perf["training_num_gpus"], 24)
        self.assertEqual(perf["generation_num_gpus"], 24)
        self.assertEqual(perf["total_num_gpus"], 32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
