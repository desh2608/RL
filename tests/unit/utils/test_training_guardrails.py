"""Stdlib-only regression tests; runnable directly without model dependencies."""

import ast
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[3] / "nemo_rl/utils/training_guardrails.py"
SPEC = importlib.util.spec_from_file_location("training_guardrails_test_target", SOURCE)
guards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guards)


class TestAllocationDeadline(unittest.TestCase):
    def test_uses_allocation_epoch_not_python_start(self):
        now = [12500]
        guard = guards.AllocationDeadline(
            {"allocation_deadline_env": "DEADLINE", "deadline_step_reserve_seconds": 900},
            {"DEADLINE": "13200"}, clock=lambda: now[0],
        )
        self.assertTrue(guard.should_save())
        self.assertFalse(guard.should_save(allow_prediction=False))
        now[0] = 13200
        self.assertTrue(guard.should_save(allow_prediction=False))

    def test_slow_step_increases_admission_reserve(self):
        guard = guards.AllocationDeadline(
            {"allocation_deadline_env": "D", "deadline_step_reserve_seconds": 100},
            {"D": "5000"}, clock=lambda: 3000,
        )
        self.assertFalse(guard.should_save())
        guard.record_step(2500)
        self.assertTrue(guard.should_save())
        guard.record_step(float("nan"))
        self.assertEqual(len(guard.durations), 1)

    def test_required_deadline_does_not_silently_fall_back(self):
        with self.assertRaises(ValueError):
            guards.AllocationDeadline({"allocation_deadline_env": "D"}, {})
        with self.assertRaises(ValueError):
            guards.AllocationDeadline({"allocation_deadline_env": "D"}, {"D": "nan"})
        self.assertFalse(guards.AllocationDeadline({}, {}).should_save())


class TestDriverDeadlineBoundaries(unittest.TestCase):
    """Execute the actual driver branches with fake RPCs, without importing CUDA."""

    def branch(self, prefix):
        path = SOURCE.parents[1] / "algorithms/grpo.py"
        tree = ast.parse(path.read_text())
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.If) and ast.unparse(node.test).startswith(prefix)]
        self.assertEqual(len(matches), 1)
        return compile(ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[])), str(path), "exec")

    def test_deadline_crossed_during_refit_forces_recovery_before_export(self):
        receipts, pauses = [], []
        env = {
            "should_save_by_timeout": False,
            "timeout": SimpleNamespace(check_save=lambda: False),
            "allocation_deadline": SimpleNamespace(should_save=lambda: True, receipt=lambda: {"reason": "allocation_deadline"}),
            "master_config": SimpleNamespace(checkpointing={"enabled": True}),
            "ray": SimpleNamespace(get=lambda value: value),
            "trajectory_collector": SimpleNamespace(pause=SimpleNamespace(remote=lambda: pauses.append(True))),
            "atomic_json": lambda path, value: receipts.append(value),
            "os": os, "monitor_dir": "/unused", "step": 9,
        }
        exec(self.branch("not should_save_by_timeout and"), env)
        self.assertTrue(env["saving_this_step"])
        self.assertTrue(env["guard_stop"])
        self.assertEqual(pauses, [True])
        self.assertEqual(receipts[0]["boundary"], "after_refit")

    def test_due_hf_export_is_deferred_after_deadline_recovery(self):
        receipts = []
        env = {
            "hf_export_period": 10, "step": 9, "should_save_by_timeout": True,
            "allocation_deadline": SimpleNamespace(should_save=lambda: True),
            "master_config": SimpleNamespace(checkpointing={"hf_export_dir": "/unused"}, policy={"model_name": "source"}),
            "checkpointer": SimpleNamespace(get_latest_checkpoint_path=lambda: "/recovery/step_10"),
            "atomic_json": lambda path, value: receipts.append(value),
            "export_current_hf": lambda _: self.fail("must not start an unsafe HF export"),
            "os": os,
        }
        exec(self.branch("hf_export_period and (step + 1) % hf_export_period"), env)
        self.assertEqual(receipts[0]["step"], 10)
        self.assertEqual(receipts[0]["recovery_checkpoint"], "/recovery/step_10")


class TestLengthMonitor(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.monitor = guards.LengthMonitor({"enabled": True}, self.temp.name)

    def observe(self, step, length=1000, cap=False, repeat=False, domain="tools", count=8):
        return self.monitor.observe_rows({domain: [(length, cap, repeat)] * count}, step)

    def test_healthy_baseline_then_sustained_inflation(self):
        for step in range(1, 4):
            self.assertFalse(self.observe(step)["stop"])
        for step in (4, 5):
            result = self.observe(step, 10000, cap=True)
            self.assertFalse(result["stop"])
            self.assertEqual(len(result["alerts"]), 1)
        self.assertTrue(self.observe(6, 10000, cap=True)["stop"])
        alert = json.loads((Path(self.temp.name) / "LENGTH_ALERT.json").read_text())
        self.assertTrue(alert["stop"])

    def test_legitimate_long_code_does_not_trigger_on_length_alone(self):
        for step in range(1, 9):
            self.assertFalse(self.observe(step, 50000, domain="code")["stop"])
        self.assertEqual(self.monitor.state["streak"]["code"], 0)

    def test_extreme_repetition_detectable_before_baseline(self):
        self.assertFalse(self.observe(1, 65536, True, True)["stop"])
        self.assertFalse(self.observe(2, 65536, True, True)["stop"])
        self.assertTrue(self.observe(3, 65536, True, True)["stop"])
        self.assertEqual(self.monitor.state["baseline"]["tools"], [])

    def test_missing_domain_breaks_consecutive_streak(self):
        self.observe(1, 65536, True, True)
        self.observe(2, 65536, True, True)
        self.observe(3, domain="code")
        self.assertFalse(self.observe(4, 65536, True, True)["stop"])
        self.assertEqual(self.monitor.state["streak"]["tools"], 1)

    def test_too_few_samples_do_not_stop(self):
        for step in range(1, 8):
            self.assertFalse(self.observe(step, 65536, True, True, count=1)["stop"])

    def test_state_resume_preserves_baseline_and_streak(self):
        for step in range(1, 4):
            self.observe(step)
        self.observe(4, 10000, True)
        self.observe(5, 10000, True)
        saved = json.loads(json.dumps(self.monitor.state))
        resumed = guards.LengthMonitor({"enabled": True}, self.temp.name, saved)
        self.assertTrue(resumed.observe_rows({"tools": [(10000, True, False)] * 8}, 6)["stop"])

    def test_only_new_assistant_tokens_are_counted(self):
        batch = {"extra_env_info": [{"domain": "tools"}], "truncated": [False], "message_log": [[
            {"role": "assistant", "token_ids": [1] * 500},
            {"role": "user", "token_ids": [2] * 20},
            {"role": "assistant", "token_ids": [3] * 50, "generation_logprobs": [0] * 50},
        ]]}
        result = self.monitor.observe(batch, 1, 65536)
        self.assertEqual(result["domains"]["tools"]["mean"], 50)
        self.assertIsNone(self.monitor.observe(batch, 1, 65536))

    def test_tail_repetition_is_bounded_and_distinguishes_unique_tail(self):
        self.assertGreater(guards.tail_repetition([4] * 100000), 0.99)
        self.assertEqual(guards.tail_repetition(list(range(10000))), 0)


class TestHFPublication(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def export(path):
        path = Path(path)
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"fake-test-only")

    def test_no_automatic_hf_deletion_and_idempotent_resume(self):
        for step in (10, 20):
            guards.export_hf_atomically(self.export, self.root, step, source_model="original")
        guards.export_hf_atomically(lambda _: self.fail("must not export again"), self.root, 20, source_model="original")
        self.assertTrue((self.root / "step_10/EXPORT_COMPLETE.json").exists())
        self.assertTrue((self.root / "step_20/EXPORT_COMPLETE.json").exists())
        self.assertFalse((self.root / "HF_EXPORT_PENDING.json").exists())

    def test_failure_cannot_publish_or_delete_previous_export(self):
        guards.export_hf_atomically(self.export, self.root, 10, source_model="original")
        with self.assertRaises(RuntimeError):
            guards.export_hf_atomically(lambda _: None, self.root, 20, source_model="original")
        self.assertFalse((self.root / "step_20").exists())
        self.assertTrue((self.root / "step_10").exists())
        self.assertTrue((self.root / "HF_EXPORT_PENDING.json").exists())
        guards.export_hf_atomically(self.export, self.root, 20, source_model="original")
        self.assertTrue((self.root / "step_20").exists())

    def test_missing_shard_fails_before_publication(self):
        def bad_export(path):
            self.export(path)
            (Path(path) / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
        with self.assertRaises(RuntimeError):
            guards.export_hf_atomically(bad_export, self.root, 10, source_model="original")
        self.assertFalse((self.root / "step_10").exists())

    def test_conflicting_source_not_silently_reused(self):
        guards.export_hf_atomically(self.export, self.root, 10, source_model="original")
        with self.assertRaises(ValueError):
            guards.export_hf_atomically(self.export, self.root, 10, source_model="different")


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires the real training environment")
class TestSingleRecoveryCheckpoint(unittest.TestCase):
    def setUp(self):
        from nemo_rl.utils.checkpoint import CheckpointManager
        self.manager_class = CheckpointManager
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = {
            "enabled": True, "checkpoint_dir": self.temp.name,
            "metric_name": None, "higher_is_better": False,
            "save_period": 10, "keep_top_k": 0, "ft_keep_latest_k": 1,
            "save_optimizer": True, "require_single_recovery_checkpoint": True,
        }

    def test_old_checkpoint_survives_until_new_save_finishes(self):
        manager = self.manager_class(self.config)
        self.addCleanup(manager.shutdown)
        first = manager.init_tmp_checkpoint(1, {"current_step": 1})
        manager.begin_finalization(first)
        manager.finalize_pending()
        second = manager.init_tmp_checkpoint(2, {"current_step": 2})
        old = Path(self.temp.name) / "step_1"

        def finish_write():
            self.assertTrue(old.exists())
            self.assertFalse((Path(self.temp.name) / "step_2").exists())

        manager.begin_finalization(second, wait_fn=finish_write)
        manager.finalize_pending()
        self.assertFalse(old.exists())
        self.assertTrue((Path(self.temp.name) / "step_2").exists())

    def test_cleanup_failure_cannot_silently_accumulate_checkpoints(self):
        manager = self.manager_class(self.config)
        self.addCleanup(manager.shutdown)
        temporary = manager.init_tmp_checkpoint(1, {"current_step": 1})
        with patch.object(manager, "remove_old_checkpoints", side_effect=OSError("delete failed")):
            manager.begin_finalization(temporary)
            with self.assertRaisesRegex(RuntimeError, "Background checkpoint finalization failed"):
                manager.finalize_pending()
        self.assertTrue((Path(self.temp.name) / "step_1").exists())

    def test_conflicting_retention_is_rejected(self):
        self.config["keep_top_k"] = 3
        with self.assertRaisesRegex(ValueError, "keep_top_k=0"):
            self.manager_class(self.config)


class TestDomainTokenAccounting(unittest.TestCase):
    def test_masked_long_response_does_not_dominate_valid_share(self):
        with tempfile.TemporaryDirectory() as root:
            monitor = guards.LengthMonitor({"enabled": True}, root)
            result = monitor.record_training_tokens(
                [{"domain": "tool"}, {"domain": "code"}, {"domain": "code"}],
                [100, 65000, 300], [1, 0, 1], 7,
            )
            self.assertEqual(result["domain_tokens/code/raw_response_tokens"], 65300)
            self.assertEqual(result["domain_tokens/code/unmasked_responses"], 1)
            self.assertEqual(result["domain_tokens/tool/sample_mask_valid_token_share"], .25)
            report = json.loads((Path(root) / "domain_token_metrics.jsonl").read_text())
            self.assertEqual(report["step"], 7)
            self.assertIn("excludes token importance weights", report["scope"])

    def test_zero_masks_and_unaligned_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            monitor = guards.LengthMonitor({"enabled": True}, root)
            result = monitor.record_training_tokens([{}], [12], [0], 1)
            self.assertEqual(result["domain_tokens/unknown/sample_mask_valid_token_share"], 0)
            with self.assertRaisesRegex(ValueError, "not aligned"):
                monitor.record_training_tokens([], [12], [1], 1)


if __name__ == "__main__":
    unittest.main()
