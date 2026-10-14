# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise actual async stop/logging blocks without Ray or GPU imports.

AST extraction preserves the production branch and logger implementation. It
does not stand in for a distributed checkpoint/resume or model execution test.
"""

import ast
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[3]
GRPO = ROOT / "nemo_rl/algorithms/grpo.py"
UTILS = ROOT / "nemo_rl/algorithms/utils.py"


def only(nodes):
    nodes = list(nodes)
    assert len(nodes) == 1, f"Expected one production block, found {len(nodes)}"
    return copy.deepcopy(nodes[0])


def calls(node, name):
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == name
        for child in ast.walk(node)
    )


def compile_blocks(nodes, filename):
    module = ast.Module(body=nodes, type_ignores=[])
    return compile(ast.fix_missing_locations(module), str(filename), "exec")


class AsyncStopGenerationMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train = only(
            node for node in ast.parse(GRPO.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "async_grpo_train"
        )
        reset = only(
            node for node in ast.walk(train)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "generation_logger_metrics"
                    for t in node.targets)
            and isinstance(node.value, ast.Constant) and node.value.value is None
        )
        blocks_training = only(
            node for node in ast.walk(train)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "policy_generation.blocks_training()"
        )
        stop = only(
            node for node in ast.walk(train)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name) and node.test.id == "guard_stop"
        )
        # Select the direct call's enclosing if, not any outer loop/try/if.
        log = only(
            node for node in ast.walk(train)
            if isinstance(node, ast.If)
            and any(isinstance(child, ast.Expr) and calls(child, "log_generation_metrics")
                    for child in node.body)
        )
        cls.stop_code = compile_blocks([reset, blocks_training, stop, log], GRPO)
        cls.log_code = compile_blocks([log], GRPO)
        helper = only(
            node for node in ast.parse(UTILS.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "log_generation_metrics"
        )
        helper_scope = {"Any": object, "Logger": object}
        exec(compile_blocks([helper], UTILS), helper_scope)
        cls.real_logger = staticmethod(helper_scope["log_generation_metrics"])

    def scope(self, *, metrics=None, enabled=True, colocated=False, timeout=True):
        logger = SimpleNamespace(log_plot_per_worker_timeline_metrics=Mock())
        policy = SimpleNamespace(
            blocks_training=lambda: colocated,
            get_logger_metrics=Mock(return_value=metrics),
            finish_generation=Mock(),
        )
        collector = SimpleNamespace(
            pause=SimpleNamespace(remote=Mock(return_value="pause")),
            prepare_for_refit=SimpleNamespace(remote=Mock(return_value="drain")),
            set_weight_version=SimpleNamespace(remote=Mock()),
            resume_after_refit=SimpleNamespace(remote=Mock()),
        )
        return {
            "generation_logger_metrics": metrics,
            "guard_stop": True,
            "defer_wake_for_save": False,
            "weight_version": 48,
            "policy": object(),
            "colocated_inference": colocated,
            "refit_policy_generation": Mock(return_value={}),
            "should_save_by_timeout": timeout,
            "allocation_deadline": SimpleNamespace(receipt=lambda: {"reason": "allocation_deadline"}),
            "length_report": {"alerts": ["length-collapse"]},
            "monitor_dir": "/mock/monitor",
            "os": __import__("os"),
            "atomic_json": Mock(),
            "print": Mock(),
            "policy_generation": policy,
            "trajectory_collector": collector,
            "ray": SimpleNamespace(get=lambda value: value),
            "timer": SimpleNamespace(time=lambda name: nullcontext(), start=Mock(), stop=Mock()),
            "master_config": SimpleNamespace(policy={"generation": {"vllm_cfg": {
                "enable_vllm_metrics_logger": enabled,
                "vllm_metrics_logger_interval": 0.1,
            }}}),
            "step": 48,
            "logger": logger,
            "log_generation_metrics": self.real_logger,
        }

    def test_deadline_stop_does_not_drain_or_log_missing_timeline(self):
        scope = self.scope()
        exec(self.stop_code, scope)
        scope["trajectory_collector"].pause.remote.assert_called_once()
        scope["trajectory_collector"].prepare_for_refit.remote.assert_not_called()
        scope["policy_generation"].get_logger_metrics.assert_not_called()
        scope["logger"].log_plot_per_worker_timeline_metrics.assert_not_called()
        self.assertIsNone(scope["generation_logger_metrics"])
        scope["atomic_json"].assert_called_once_with(
            "/mock/monitor/STOP_REQUESTED.json", {"reason": "allocation_deadline"}
        )

    def test_length_stop_also_preserves_missing_timeline(self):
        scope = self.scope(timeout=False)
        exec(self.stop_code, scope)
        scope["logger"].log_plot_per_worker_timeline_metrics.assert_not_called()
        scope["atomic_json"].assert_called_once_with(
            "/mock/monitor/STOP_REQUESTED.json",
            {"reason": "sustained_length_collapse", "step": 49, "alerts": ["length-collapse"]},
        )

    def test_colocated_stop_logs_previously_collected_timeline(self):
        metrics = {"running": {0: [2, 1]}}
        scope = self.scope(metrics=metrics, colocated=True)
        exec(self.stop_code, scope)
        scope["policy_generation"].get_logger_metrics.assert_called_once()
        scope["policy_generation"].finish_generation.assert_called_once_with(release_gpu=True)
        scope["logger"].log_plot_per_worker_timeline_metrics.assert_called_once_with(
            metrics["running"], step=49, prefix="generation_metrics", name="running",
            timeline_interval=0.1,
        )

    def test_normal_metrics_are_logged_unchanged(self):
        metrics = {"running": {0: [2]}, "pending": {1: [3]}}
        scope = self.scope(metrics=metrics)
        exec(self.log_code, scope)
        self.assertEqual(scope["logger"].log_plot_per_worker_timeline_metrics.call_count, 2)
        self.assertIs(scope["generation_logger_metrics"], metrics)

    def test_normal_refit_collects_and_logs_timeline(self):
        metrics = {"running": {0: [2, 1]}}
        scope = self.scope(metrics=metrics)
        scope["guard_stop"] = False
        exec(self.stop_code, scope)
        scope["trajectory_collector"].pause.remote.assert_not_called()
        scope["trajectory_collector"].prepare_for_refit.remote.assert_called_once()
        scope["policy_generation"].get_logger_metrics.assert_called_once()
        scope["refit_policy_generation"].assert_called_once()
        scope["trajectory_collector"].resume_after_refit.remote.assert_called_once()
        scope["logger"].log_plot_per_worker_timeline_metrics.assert_called_once()
        self.assertIs(scope["generation_logger_metrics"], metrics)

    def test_empty_dictionary_is_safe(self):
        scope = self.scope(metrics={})
        exec(self.log_code, scope)
        scope["logger"].log_plot_per_worker_timeline_metrics.assert_not_called()

    def test_disabled_logger_remains_disabled(self):
        scope = self.scope(metrics={"running": {0: [1]}}, enabled=False)
        exec(self.log_code, scope)
        scope["logger"].log_plot_per_worker_timeline_metrics.assert_not_called()

    def test_backend_error_is_not_suppressed(self):
        scope = self.scope(metrics={"running": {0: [1]}})
        scope["logger"].log_plot_per_worker_timeline_metrics.side_effect = RuntimeError("backend failed")
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            exec(self.log_code, scope)

    def test_malformed_non_none_payload_is_not_suppressed(self):
        scope = self.scope(metrics=[])
        with self.assertRaises(AttributeError):
            exec(self.log_code, scope)


if __name__ == "__main__":
    unittest.main()
