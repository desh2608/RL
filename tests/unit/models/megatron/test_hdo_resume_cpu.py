# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU coverage for the opt-in HDO resume numerical acceptance checks."""

import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import (
    HybridDeviceOptimizer,
)

from nemo_rl.models.megatron.hdo_resume import (
    capture_hdo_resume_steps,
    verify_hdo_resume,
    verify_master_matches_model,
)


def make_optimizer(original_dtype=torch.bfloat16):
    original = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=original_dtype))
    master = torch.nn.Parameter(torch.tensor([1.000123, -2.000321]))
    inner = torch.optim.AdamW([master], fused=True)
    inner.param_groups[0]["step"] = 3
    state = {
        "master_param": master.detach().clone(),
        "exp_avg": torch.tensor([0.1, -0.2]),
        "exp_avg_sq": torch.tensor([0.01, 0.02]),
        "step": torch.tensor(3.0),
    }
    inner.state[master] = state
    hdo = object.__new__(HybridDeviceOptimizer)
    hdo.param_groups = [{"params": [original], "step": 3}]
    hdo.state = defaultdict(dict, {original: state})
    hdo.cpu_optimizers = [inner]
    hdo.gpu_optimizer = None
    hdo.param_to_inner_param = {original: master}
    hdo.param_to_fp32_param = (
        {original: master} if original_dtype != torch.float32 else {}
    )
    hdo.inner_param_to_orig_param = {master: original}
    hdo.param_update_in_fp32 = True
    wrapper = SimpleNamespace(optimizer=hdo)
    return wrapper, original, master, inner


class TestHDOResumeCPU(unittest.TestCase):
    def test_native_fp32_parameter_is_valid_without_a_promotion_mapping(self):
        optimizer, original, master, _ = make_optimizer(torch.float32)
        hdo = optimizer.optimizer
        self.assertNotIn(original, hdo.param_to_fp32_param)
        # Real HDO records master_param for native FP32 parameters as well.
        hdo._sync_sub_optimizers_state_to_hdo()
        self.assertIs(hdo.state[original]["master_param"], master)
        receipt = verify_hdo_resume(
            optimizer, capture_hdo_resume_steps(optimizer), rank=0
        )
        self.assertTrue(receipt["passed"])

    def test_checks_every_chained_optimizer_and_reports_small_receipt(self):
        first, *_ = make_optimizer()
        second, *_ = make_optimizer()
        optimizer = SimpleNamespace(chained_optimizers=[first, second])
        result = verify_hdo_resume(
            optimizer, capture_hdo_resume_steps(optimizer), rank=7
        )
        self.assertEqual(result["parameter_count"], 2)
        self.assertEqual(result["moment_count"], 4)
        self.assertEqual(result["optimizer_steps"], [3])
        self.assertEqual(result["max_master_error"], 0)
        self.assertEqual(result["rank"], 7)
        self.assertTrue(result["passed"])

    def test_rejects_stale_master_even_with_correct_moments_and_step(self):
        optimizer, _, master, _ = make_optimizer()
        expected = capture_hdo_resume_steps(optimizer)
        master.data.add_(1)
        with self.assertRaisesRegex(RuntimeError, "FP32 master mismatch"):
            verify_hdo_resume(optimizer, expected, rank=0)

    def test_rejects_detached_wrong_moment_state(self):
        optimizer, _, master, inner = make_optimizer()
        expected = capture_hdo_resume_steps(optimizer)
        inner.state[master] = dict(inner.state[master])
        inner.state[master]["exp_avg"] = torch.zeros_like(master)
        with self.assertRaisesRegex(RuntimeError, "exp_avg mismatch"):
            verify_hdo_resume(optimizer, expected, rank=0)

    def test_rejects_moment_dtype_mismatch_even_if_shared(self):
        optimizer, original, _, _ = make_optimizer()
        expected = capture_hdo_resume_steps(optimizer)
        optimizer.optimizer.state[original]["exp_avg_sq"] = torch.ones(
            2, dtype=torch.bfloat16
        )
        with self.assertRaisesRegex(RuntimeError, "exp_avg_sq mismatch"):
            verify_hdo_resume(optimizer, expected, rank=0)

    def test_rejects_dummy_step_after_resume(self):
        optimizer, original, _, _ = make_optimizer()
        expected = capture_hdo_resume_steps(optimizer)
        optimizer.optimizer.state[original]["step"] = torch.tensor(1.0)
        with self.assertRaisesRegex(
            RuntimeError,
            r"parameter step mismatch.*actual=1, expected=3, suboptimizer=torch.optim.adamw.AdamW",
        ):
            verify_hdo_resume(optimizer, expected, rank=0)

    def test_rejects_missing_checkpoint_step_metadata(self):
        optimizer, *_ = make_optimizer()
        del optimizer.optimizer.param_groups[0]["step"]
        with self.assertRaisesRegex(RuntimeError, "invalid restored step"):
            capture_hdo_resume_steps(optimizer)

    def test_rejects_changed_group_step(self):
        optimizer, *_ = make_optimizer()
        expected = capture_hdo_resume_steps(optimizer)
        optimizer.optimizer.param_groups[0]["step"] = 4
        with self.assertRaisesRegex(RuntimeError, "group step changed"):
            verify_hdo_resume(optimizer, expected, rank=0)

    def test_master_matches_model_accepts_rounded_view(self):
        optimizer, original, master, _ = make_optimizer()
        result = verify_master_matches_model(optimizer)
        self.assertEqual(result["master_model_checked_params"], 1)
        self.assertEqual(result["master_model_mismatch_params"], 0)
        self.assertTrue(result["master_model_exact_equal"])
        self.assertFalse(result["master_model_strict"])
        self.assertTrue(
            torch.equal(original.detach(), master.detach().to(torch.bfloat16))
        )

    def test_master_matches_model_reports_mismatch_without_raising(self):
        optimizer, original, _, _ = make_optimizer()
        # The restored master no longer rounds to the bf16 model view.
        optimizer.optimizer.state[original]["master_param"].add_(1)
        result = verify_master_matches_model(optimizer)
        self.assertEqual(result["master_model_checked_params"], 1)
        self.assertEqual(result["master_model_mismatch_params"], 1)
        self.assertEqual(result["master_model_mismatch_elements"], 2)
        self.assertFalse(result["master_model_exact_equal"])
        self.assertGreaterEqual(result["master_model_max_abs_diff"], 0.5)
        with patch.dict(os.environ, {"NVC_HDO_MASTER_MODEL_STRICT": "1"}):
            with self.assertRaisesRegex(RuntimeError, r"bf16\(master\) != model shard"):
                verify_master_matches_model(optimizer)

    def test_master_matches_model_skips_native_fp32(self):
        optimizer, *_ = make_optimizer(torch.float32)
        optimizer.optimizer._sync_sub_optimizers_state_to_hdo()
        result = verify_master_matches_model(optimizer)
        self.assertEqual(result["master_model_checked_params"], 0)
        self.assertEqual(result["master_model_skipped_params"], 1)
        self.assertIsNone(result["master_model_exact_equal"])

    def test_worker_receipt_carries_master_model_fields(self):
        from nemo_rl.models.policy.workers.megatron_policy_worker import (
            MegatronPolicyWorkerImpl,
        )

        optimizer, *_ = make_optimizer()
        worker = SimpleNamespace(
            optimizer=optimizer,
            _hdo_resume_steps=capture_hdo_resume_steps(optimizer),
            _hdo_resume_source_checkpoint="/run/checkpoints/step_12/policy",
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"NVC_HDO_RESUME_RECEIPT_DIR": directory}):
                with patch("torch.distributed.get_rank", return_value=3):
                    MegatronPolicyWorkerImpl._verify_hdo_resume_before_update(worker)
            receipt = json.loads((Path(directory) / "rank-3.json").read_text())
        self.assertTrue(receipt["passed"])
        self.assertTrue(receipt["master_model_exact_equal"])
        self.assertEqual(receipt["master_model_checked_params"], 1)

    def test_worker_receipt_and_one_time_check(self):
        from nemo_rl.models.policy.workers.megatron_policy_worker import (
            MegatronPolicyWorkerImpl,
        )

        optimizer, *_ = make_optimizer()
        worker = SimpleNamespace(
            optimizer=optimizer,
            _hdo_resume_steps=capture_hdo_resume_steps(optimizer),
            _hdo_resume_source_checkpoint="/run/checkpoints/step_3/policy",
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"NVC_HDO_RESUME_RECEIPT_DIR": directory}):
                with patch("torch.distributed.get_rank", return_value=7):
                    MegatronPolicyWorkerImpl._verify_hdo_resume_before_update(
                        worker, after_optimizer_load=True
                    )
                    early = json.loads((Path(directory) / "load-rank-7.json").read_text())
                    self.assertTrue(early["passed"])
                    self.assertEqual(early["phase"], "after_optimizer_load")
                    self.assertIsNotNone(worker._hdo_resume_steps)
                    self.assertFalse((Path(directory) / "rank-7.json").exists())
                    MegatronPolicyWorkerImpl._verify_hdo_resume_before_update(worker)
            receipt = json.loads((Path(directory) / "rank-7.json").read_text())
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["phase"], "before_first_resumed_train")
            self.assertEqual(
                receipt["source_checkpoint"], worker._hdo_resume_source_checkpoint
            )
            self.assertIsNone(worker._hdo_resume_steps)
            # A fresh/non-resume worker and an already-verified worker both skip
            # the check without touching distributed initialization.
            with patch("torch.distributed.get_rank", side_effect=AssertionError):
                MegatronPolicyWorkerImpl._verify_hdo_resume_before_update(worker)
                MegatronPolicyWorkerImpl._verify_hdo_resume_before_update(
                    SimpleNamespace()
                )

    def test_worker_writes_failure_receipt_and_raises(self):
        from nemo_rl.models.policy.workers.megatron_policy_worker import (
            MegatronPolicyWorkerImpl,
        )

        optimizer, _, master, _ = make_optimizer()
        worker = SimpleNamespace(
            optimizer=optimizer,
            _hdo_resume_steps=capture_hdo_resume_steps(optimizer),
            _hdo_resume_source_checkpoint="/run/checkpoints/step_3/policy",
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"NVC_HDO_RESUME_RECEIPT_DIR": directory}):
                with patch("torch.distributed.get_rank", return_value=2):
                    MegatronPolicyWorkerImpl._verify_hdo_resume_before_update(
                        worker, after_optimizer_load=True
                    )
                    master.data.add_(1)
                    with self.assertRaisesRegex(RuntimeError, "FP32 master mismatch"):
                        MegatronPolicyWorkerImpl._verify_hdo_resume_before_update(
                            worker
                        )
            receipt = json.loads((Path(directory) / "rank-2.json").read_text())
            self.assertFalse(receipt["passed"])
            self.assertIn("FP32 master mismatch", receipt["error"])
            self.assertIsNotNone(worker._hdo_resume_steps)
            self.assertTrue(
                json.loads((Path(directory) / "load-rank-2.json").read_text())["passed"]
            )


if __name__ == "__main__":
    unittest.main()
