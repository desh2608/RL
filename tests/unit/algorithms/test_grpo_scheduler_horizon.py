"""CPU-only checks of the production GRPO scheduler-horizon setup block."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[3] / "nemo_rl/algorithms/grpo.py"


def configure(policy, maximum=80, epochs=4, batches=22):
    tree = ast.parse(SOURCE.read_text())
    blocks = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "total_train_iters"
                for t in stmt.targets
            ):
                for j in range(i + 1, len(node.body)):
                    end = node.body[j]
                    if isinstance(end, ast.Assign) and any(
                        ast.unparse(t) == "policy_config['megatron_cfg']['train_iters']"
                        for t in end.targets
                    ):
                        blocks.append(node.body[i : j + 1])
                        break
    assert len(blocks) == 1, "Expected one production scheduler-horizon block"
    scope = {
        "policy_config": policy,
        "grpo_config": SimpleNamespace(max_num_steps=maximum, max_num_epochs=epochs),
        "train_sample_count": batches,
    }
    exec(compile(ast.Module(body=blocks[0], type_ignores=[]), str(SOURCE), "exec"), scope)
    return policy["megatron_cfg"]["train_iters"]


class SchedulerHorizonTest(unittest.TestCase):
    def test_default_retains_target_and_data_bound(self):
        self.assertEqual(configure({"megatron_cfg": {}}), 80)
        self.assertEqual(configure({"megatron_cfg": {}}, maximum=100), 88)

    def test_original_short_run_is_unchanged(self):
        self.assertEqual(configure({"megatron_cfg": {}}, maximum=20, epochs=1), 20)

    def test_none_uses_computed_default(self):
        self.assertEqual(configure({"megatron_cfg": {"train_iters": None}}), 80)

    def test_explicit_resume_horizon_preserves_other_scheduler_fields(self):
        policy = {"megatron_cfg": {"train_iters": 20, "scheduler": {
            "lr_decay_iters": 100, "lr_warmup_iters": 10,
            "lr_decay_style": "constant", "start_weight_decay": 0,
            "end_weight_decay": 0,
        }}}
        before = dict(policy["megatron_cfg"]["scheduler"])
        self.assertEqual(configure(policy, maximum=80), 20)
        self.assertEqual(policy["megatron_cfg"]["scheduler"], before)

    def test_explicit_long_horizon_is_honored(self):
        self.assertEqual(configure({"megatron_cfg": {"train_iters": 100}}), 100)

    def test_invalid_horizon_fails_before_workers_start(self):
        for value in (False, True, 0, -1, 20.0, "20"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                configure({"megatron_cfg": {"train_iters": value}})


if __name__ == "__main__":
    unittest.main()
