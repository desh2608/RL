"""Real torchdata CPU roundtrips; run in the existing driver CPU runtime."""

import ast
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any
import unittest

from torchdata.stateful_dataloader import StatefulDataLoader


ROOT = Path(__file__).resolve().parents[3]
tree = ast.parse((ROOT / "nemo_rl/data/dataloader.py").read_text())
node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CyclingDataLoader")
scope = {"StatefulDataLoader": StatefulDataLoader, "Iterator": Iterator, "BatchedDataDict": Any, "Any": Any}
exec(compile(ast.Module(body=[node], type_ignores=[]), "production_dataloader", "exec"), scope)
CyclingDataLoader = scope["CyclingDataLoader"]


def raw_loader(rows=12, batch=2, workers=0):
    return StatefulDataLoader(range(rows), batch_size=batch, shuffle=False, drop_last=True, num_workers=workers)


def unpack(iterator):
    return [batch.tolist() for batch in iterator]


class TorchdataFiniteEpochTests(unittest.TestCase):
    def test_every_snapshot_across_four_epochs(self):
        for workers in (0, 1):
            with self.subTest(workers=workers):
                loader = CyclingDataLoader(raw_loader(workers=workers), max_epochs=4)
                iterator = iter(loader)
                expected = [[i, i + 1] for i in range(0, 12, 2)] * 4
                for index in range(len(expected)):
                    snapshot = deepcopy(loader.state_dict())
                    resumed = CyclingDataLoader(raw_loader(workers=workers), max_epochs=4)
                    resumed.load_state_dict(snapshot)
                    self.assertEqual(unpack(resumed), expected[index:], f"workers={workers}, batch={index}")
                    self.assertEqual(next(iterator).tolist(), expected[index])
                self.assertEqual(list(iterator), [])
                resumed = CyclingDataLoader(raw_loader(workers=workers), max_epochs=4)
                resumed.load_state_dict(deepcopy(loader.state_dict()))
                self.assertEqual(list(resumed), [])

    def test_formal_11264_row_legacy_step20_resume(self):
        # Formal P512, 22 batches per pass: resume the exact old cursor at20,
        # train the reserve at21/22, then cycle; no data bytes or IDs change.
        old = raw_loader(rows=11264, batch=512, workers=1)
        old_iterator = iter(old)
        for _ in range(20):
            next(old_iterator)
        state = deepcopy(old.state_dict())
        resumed = CyclingDataLoader(raw_loader(rows=11264, batch=512, workers=1), max_epochs=4)
        resumed.load_state_dict(state)
        actual = [value for batch in resumed for value in batch.tolist()]
        self.assertEqual(actual, list(range(10240, 11264)) + list(range(11264)) * 3)
        self.assertEqual(len(actual), 68 * 512)
        # Learner max_num_steps80 cuts this 88-batch stream after60 new batches.
        self.assertEqual(actual[:60 * 512][-1], 7167)

    def test_formal_epoch_boundaries_and_rewound_prefetch_snapshot(self):
        loader = CyclingDataLoader(raw_loader(rows=11264, batch=512, workers=1), max_epochs=4)
        iterator = iter(loader)
        snapshots = {}
        for batch_index in range(88):
            if batch_index in {20, 21, 22, 23, 43, 44, 65, 66, 67, 80, 87}:
                snapshots[batch_index] = deepcopy(loader.state_dict())
            actual = next(iterator).tolist()
            offset = (batch_index % 22) * 512
            self.assertEqual(actual, list(range(offset, offset + 512)))
        self.assertEqual(list(iterator), [])
        expected_stream = list(range(11264)) * 4
        # Restore snapshots only after the live producer finished all4passes:
        # the snapshot epoch must win over the producer's newer prefetched one.
        for base, state in snapshots.items():
            with self.subTest(global_batch=base):
                resumed = CyclingDataLoader(raw_loader(rows=11264, batch=512, workers=1), max_epochs=4)
                resumed.load_state_dict(state)
                actual = [value for batch in resumed for value in batch.tolist()]
                self.assertEqual(actual, expected_stream[base * 512:])


if __name__ == "__main__":
    unittest.main()
