"""Dependency-free production-class tests for finite async epoch continuation.

Run with Python directly. The checkpointable loader fixture isolates stream
semantics; a real torchdata state roundtrip remains part of CPU runtime checks.
"""

import ast
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from typing import Any, Optional, cast


ROOT = Path(__file__).resolve().parents[3]


def production_node(relative_path, name):
    tree = ast.parse((ROOT / relative_path).read_text())
    return next(node for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)


namespace = {"StatefulDataLoader": Any, "Iterator": Iterator, "BatchedDataDict": Any, "Any": Any}
exec(compile(ast.Module(body=[production_node("nemo_rl/data/dataloader.py", "CyclingDataLoader")], type_ignores=[]), "production_dataloader", "exec"), namespace)
CyclingDataLoader = namespace["CyclingDataLoader"]


class StatefulRows:
    """Three deterministic two-row batches, with restorable iterator state."""

    def __init__(self, size=6):
        self.size = size
        self.cursor = 0
        self.finished = False

    def __len__(self):
        return self.size // 2

    def __iter__(self):
        if self.finished:
            self.cursor = 0
            self.finished = False
        while self.cursor < self.size:
            result = list(range(self.cursor, self.cursor + 2))
            self.cursor += 2
            yield result
        self.finished = True

    def state_dict(self):
        return {"cursor": self.cursor, "finished": self.finished}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]
        self.finished = state["finished"]


class Batch(dict):
    def select_indices(self, indices):
        return Batch({key: [value[index] for index in indices] for key, value in self.items()})


class FiniteEpochTests(unittest.TestCase):
    def test_finite_budget_and_unchanged_epoch_length(self):
        loader = CyclingDataLoader(StatefulRows(), max_epochs=4)
        self.assertEqual(len(loader), 3)
        self.assertEqual(list(loader), [[0, 1], [2, 3], [4, 5]] * 4)
        self.assertEqual(loader.completed_epochs, 4)
        self.assertEqual(list(loader), [])

    def test_every_snapshot_roundtrips_across_epoch_boundaries(self):
        loader = CyclingDataLoader(StatefulRows(), max_epochs=4)
        iterator = iter(loader)
        expected = [[0, 1], [2, 3], [4, 5]] * 4
        for index in range(len(expected)):
            state = deepcopy(loader.state_dict())
            resumed = CyclingDataLoader(StatefulRows(), max_epochs=4)
            resumed.load_state_dict(state)
            self.assertEqual(list(resumed), expected[index:], f"snapshot before batch {index}")
            self.assertEqual(next(iterator), expected[index])

    def test_old_one_pass_checkpoint_keeps_cursor(self):
        raw = StatefulRows()
        iterator = iter(raw)
        next(iterator)
        next(iterator)
        resumed = CyclingDataLoader(StatefulRows(), max_epochs=4)
        resumed.load_state_dict(raw.state_dict())
        self.assertEqual(list(resumed), [[4, 5]] + [[0, 1], [2, 3], [4, 5]] * 3)

    def test_end_of_old_first_pass_is_counted_once(self):
        raw = StatefulRows()
        iterator = iter(raw)
        for _ in range(3):
            next(iterator)
        resumed = CyclingDataLoader(StatefulRows(), max_epochs=4)
        resumed.load_state_dict(raw.state_dict())
        self.assertEqual(list(resumed), [[0, 1], [2, 3], [4, 5]] * 3)

    def test_complete_checkpoint_does_not_repeat(self):
        loader = CyclingDataLoader(StatefulRows(), max_epochs=2)
        list(loader)
        resumed = CyclingDataLoader(StatefulRows(), max_epochs=2)
        resumed.load_state_dict(loader.state_dict())
        self.assertEqual(list(resumed), [])

    def test_dataset_length_and_shrunk_epoch_budget_rejected(self):
        loader = CyclingDataLoader(StatefulRows(), max_epochs=4)
        list(loader)
        with self.assertRaises(ValueError):
            CyclingDataLoader(StatefulRows(), max_epochs=3).load_state_dict(loader.state_dict())
        with self.assertRaises(ValueError):
            CyclingDataLoader(StatefulRows(8), max_epochs=4).load_state_dict(loader.state_dict())

    def test_unbounded_mode_keeps_legacy_checkpoint_format(self):
        raw = StatefulRows()
        loader = CyclingDataLoader(raw)
        iterator = iter(loader)
        self.assertEqual([next(iterator) for _ in range(8)], [[0, 1], [2, 3], [4, 5]] * 2 + [[0, 1], [2, 3]])
        self.assertEqual(loader.state_dict(), raw.state_dict())

    def test_empty_dataset_fails_instead_of_spinning(self):
        with self.assertRaisesRegex(RuntimeError, "two consecutive epochs"):
            list(CyclingDataLoader(StatefulRows(0), max_epochs=4))

    def test_frontier_filter_preserves_global_ordinals_after_epoch_boundary(self):
        # Use the real collector filter, including a previously trained group
        # above the cut. Local row positions repeat; global ordinals must not.
        method = production_node("nemo_rl/algorithms/async_utils/trajectory_collector.py", "_filter_covered_rows")
        scope = {"Optional": Optional, "BatchedDataDict": dict, "DatumSpec": Any, "cast": cast,
                 "NEMO_GYM_TASK_INDEX_KEY": "ordinal"}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "production_frontier_filter", "exec"), scope)
        loader = CyclingDataLoader(StatefulRows(), max_epochs=4)
        iterator = iter(loader)
        for _ in range(5):
            next(iterator)
        snapshot = deepcopy(loader.state_dict())  # global base 10, within pass 2
        # Simulate an in-flight collector having progressed into pass 3.
        for _ in range(3):
            next(iterator)
        resumed = CyclingDataLoader(StatefulRows(), max_epochs=4)
        resumed.load_state_dict(snapshot)
        owner = SimpleNamespace(_resume_frontier_ordinal=11, _skip_horizon=14, _covered_task_indices={14})
        actual = []
        ordinal = 10
        for values in resumed:
            batch = Batch(values=values, extra_env_info=[{"ordinal": ordinal + index} for index in range(len(values))])
            filtered = scope["_filter_covered_rows"](owner, batch)
            if filtered is not None:
                actual.extend(zip([row["ordinal"] for row in filtered["extra_env_info"]], filtered["values"]))
            ordinal += len(values)
        self.assertEqual(actual, [(i, i % 6) for i in range(11, 24) if i != 14])

    def test_grpo_wraps_before_restore_only_for_async_single_multi_epoch(self):
        function = production_node("nemo_rl/algorithms/grpo.py", "init_train_dataloader")
        for asynchronous, multiple, epochs in ((False, False, 4), (True, True, 4), (True, False, 1), (True, False, 4)):
            restored = []
            scope = {"StatefulDataLoader": lambda *args, **kwargs: StatefulRows(),
                     "CyclingDataLoader": CyclingDataLoader, "dataloader_batch_size": 2,
                     "data_config": {"shuffle": False, "num_workers": 0, "use_multiple_dataloader": multiple},
                     "grpo_config": SimpleNamespace(async_grpo=SimpleNamespace(enabled=asynchronous), max_num_epochs=epochs),
                     "rl_collate_fn": None, "last_checkpoint_path": "checkpoint",
                     "load_dataloader_state": lambda loader, *args: restored.append(loader)}
            exec(compile(ast.Module(body=[function], type_ignores=[]), "production_grpo_loader", "exec"), scope)
            result = scope["init_train_dataloader"](None)
            self.assertIs(restored[0], result)
            self.assertEqual(isinstance(result, CyclingDataLoader), asynchronous and not multiple and epochs > 1)


if __name__ == "__main__":
    unittest.main()
