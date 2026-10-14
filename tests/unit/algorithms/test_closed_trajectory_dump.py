"""Unit tests for the bounded closed-trajectory dump (env.closed_trajectory_dump)."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from nemo_rl.algorithms.grpo import (
    _closed_trajectory_dump_config,
    _dump_closed_trajectories,
    build_closed_trajectory_records,
)


def _batch(num_rows=4, max_len=12):
    train_data = {
        "input_ids": torch.arange(num_rows * max_len).reshape(num_rows, max_len),
        "token_mask": torch.ones(num_rows, max_len),
        "sample_mask": torch.ones(num_rows),
        "advantages": torch.full((num_rows, max_len), 0.5),
        "generation_logprobs": torch.full((num_rows, max_len), -1.0),
        "prev_logprobs": torch.full((num_rows, max_len), -1.5),
    }
    repeated_batch = {
        "agent_ref": [{"name": f"agent{i % 2}"} for i in range(num_rows)],
        "total_reward": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "truncated": torch.tensor([False, True, False, False]),
        "rollout_group_ids": torch.tensor([0, 0, 1, 1]),
    }
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    input_lengths = torch.tensor([5, 12, 7, 3])
    content = [f"content {i}" for i in range(num_rows)]
    return train_data, repeated_batch, rewards, input_lengths, content


class ClosedTrajectoryDumpTest(unittest.TestCase):
    def test_config_disabled_by_default(self):
        master = SimpleNamespace(env={})
        self.assertIsNone(_closed_trajectory_dump_config(master))
        master = SimpleNamespace(env={"closed_trajectory_dump": {"enabled": False}})
        self.assertIsNone(_closed_trajectory_dump_config(master))

    def test_config_values(self):
        master = SimpleNamespace(
            env={"closed_trajectory_dump": {"enabled": True, "every_n_steps": 5, "max_samples": 64}}
        )
        self.assertEqual(_closed_trajectory_dump_config(master), {"every_n_steps": 5, "max_samples": 64})
        with self.assertRaises(ValueError):
            _closed_trajectory_dump_config(
                SimpleNamespace(env={"closed_trajectory_dump": {"enabled": True, "max_samples": 0}})
            )

    def test_records_are_trimmed_and_bounded(self):
        train_data, repeated_batch, rewards, input_lengths, content = _batch()
        records = build_closed_trajectory_records(
            step=6,
            repeated_batch=repeated_batch,
            train_data=train_data,
            flat_messages_content=content,
            rewards=rewards,
            input_lengths=input_lengths,
            max_samples=3,
        )
        self.assertEqual(len(records), 3)
        for record, length in zip(records, (5, 12, 7)):
            self.assertEqual(record["step"], 7)
            self.assertEqual(record["input_length"], length)
            for key in ("token_ids", "token_loss_mask", "advantages", "generation_logprobs", "prev_logprobs"):
                self.assertEqual(len(record[key]), length, key)
        self.assertEqual(records[0]["token_ids"], list(range(5)))
        self.assertEqual(records[1]["truncated"], True)
        self.assertEqual(records[2]["agent_ref"], {"name": "agent0"})
        self.assertEqual(records[2]["rollout_group_ids"], 1)
        self.assertEqual(records[0]["content"], "content 0")
        self.assertEqual(records[0]["sample_loss_mask"], 1.0)
        json.dumps(records)  # JSON serialisable without default=str

    def test_records_are_stratified_over_agent_refs(self):
        from nemo_rl.algorithms.grpo import _closed_trajectory_row_selection

        batch = {"agent_ref": [{"name": n} for n in ("a", "a", "a", "a", "b", "c")]}
        self.assertEqual(_closed_trajectory_row_selection(batch, 6, 3), [0, 4, 5])
        self.assertEqual(_closed_trajectory_row_selection(batch, 6, 5), [0, 1, 4, 5, 2][:5] if False else [0, 1, 2, 4, 5])
        self.assertEqual(_closed_trajectory_row_selection(batch, 6, 6), list(range(6)))
        self.assertEqual(_closed_trajectory_row_selection({}, 6, 3), [0, 1, 2])
        train_data, repeated_batch, rewards, input_lengths, content = _batch(num_rows=4)
        repeated_batch["agent_ref"] = [{"name": "x"}, {"name": "x"}, {"name": "x"}, {"name": "y"}]
        records = build_closed_trajectory_records(
            step=0, repeated_batch=repeated_batch, train_data=train_data, flat_messages_content=content,
            rewards=rewards, input_lengths=input_lengths, max_samples=2,
        )
        self.assertEqual([r["idx"] for r in records], [0, 3])
        self.assertEqual([r["agent_ref"]["name"] for r in records], ["x", "y"])

    def test_dump_writes_jsonl_only_on_cadence(self):
        train_data, repeated_batch, rewards, input_lengths, content = _batch()
        with tempfile.TemporaryDirectory() as tmp:
            logger = SimpleNamespace(base_log_dir=tmp)
            master = SimpleNamespace(
                env={"closed_trajectory_dump": {"enabled": True, "every_n_steps": 2, "max_samples": 8}}
            )
            common = dict(
                logger=logger, master_config=master, repeated_batch=repeated_batch, train_data=train_data,
                flat_messages_content=content, rewards=rewards, input_lengths=input_lengths,
            )
            self.assertIsNone(_dump_closed_trajectories(step=0, **common))  # step 1: off cadence
            path = _dump_closed_trajectories(step=1, **common)  # step 2
            self.assertEqual(path, os.path.join(tmp, "closed_trajectories", "step2.jsonl"))
            lines = open(path).read().splitlines()
            self.assertEqual(len(lines), 4)
            self.assertEqual(json.loads(lines[3])["input_length"], 3)
            self.assertFalse(os.path.exists(path + ".tmp"))

    def test_dump_never_raises(self):
        master = SimpleNamespace(env={"closed_trajectory_dump": {"enabled": True}})
        logger = SimpleNamespace(base_log_dir="/nonexistent-root-for-test/x")
        broken_train_data = {"input_ids": "not a tensor"}
        result = _dump_closed_trajectories(
            logger=logger, master_config=master, step=0, repeated_batch={}, train_data=broken_train_data,
            flat_messages_content=[], rewards=torch.tensor([0.0]), input_lengths=torch.tensor([1]),
        )
        self.assertIsNone(result)

    def test_disabled_writes_nothing(self):
        train_data, repeated_batch, rewards, input_lengths, content = _batch()
        with tempfile.TemporaryDirectory() as tmp:
            result = _dump_closed_trajectories(
                logger=SimpleNamespace(base_log_dir=tmp), master_config=SimpleNamespace(env={}), step=0,
                repeated_batch=repeated_batch, train_data=train_data, flat_messages_content=content,
                rewards=rewards, input_lengths=input_lengths,
            )
            self.assertIsNone(result)
            self.assertFalse(os.path.exists(os.path.join(tmp, "closed_trajectories")))


if __name__ == "__main__":
    unittest.main()
