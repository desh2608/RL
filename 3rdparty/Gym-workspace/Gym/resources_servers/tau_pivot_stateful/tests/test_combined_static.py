# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static dispatch isolation, reference binding, and G6 due-call controls."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from resources_servers.tau_pivot_stateful import combined_static


class CombinedStaticTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.path = root / "entry.json"
        self.call = {"name": "get_order_details", "arguments": {"order_id": "O1"}}
        self.ref = dict(
            record_id="test", dataset="v3-stage-a-static", verifier_version="pinned", expected_turn_kind="call"
        )
        self.entry = dict(
            self.ref,
            mode="combined_static",
            evaluator="g6_error_kinds",
            state_hash_at_boundary="state",
            verification=dict(
                contract="G6-INBOX50-v2",
                error_kind="K0",
                prefix_state_consistent=True,
                state_hash_at_boundary="state",
                approved_calls=[self.call],
                no_gold_call_due=False,
            ),
        )
        self.path.write_text(json.dumps(self.entry))
        self.item = dict(entry=str(self.path), entry_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())

    def tearDown(self):
        combined_static.load_entry.cache_clear()
        self.temp.cleanup()

    def test_due_call_passes_filler_and_wrong_identifier_fail(self):
        good = [dict(type="function_call", **self.call)]
        self.assertEqual(combined_static.verify(self.item, self.ref, good)["reward"], 1)
        self.assertEqual(
            combined_static.verify(
                self.item,
                self.ref,
                [dict(type="message", role="assistant", content="I can continue helping with your request.")],
            )["reward"],
            0,
        )
        self.assertEqual(
            combined_static.verify(
                self.item, self.ref, [dict(type="function_call", name=self.call["name"], arguments={"order_id": "O2"})]
            )["reward"],
            0,
        )

    def test_reference_rebinding_fails(self):
        for key in self.ref:
            with self.assertRaisesRegex(ValueError, "reference mismatch"):
                combined_static.verify(self.item, dict(self.ref, **{key: "changed"}), [])

    def test_entry_hash_and_boundary_tamper_fail(self):
        self.path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            combined_static.verify(self.item, self.ref, [])
        self.entry["state_hash_at_boundary"] = "wrong"
        self.path.write_text(json.dumps(self.entry))
        self.item["entry_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "boundary hash differs"):
            combined_static.verify(self.item, self.ref, [])


if __name__ == "__main__":
    unittest.main()
