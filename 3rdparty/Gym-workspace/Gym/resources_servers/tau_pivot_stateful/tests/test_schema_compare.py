# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema projection preserves meaningful mutations and drops undeclared fields."""

import unittest

from resources_servers.tau_pivot_stateful import schema_compare
from resources_servers.tau_pivot_stateful.native_runtime import schema_native_check


class SchemaProjectionTest(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "name": "change",
                "parameters": {
                    "$defs": {"Flight": {"type": "object", "properties": {"number": {"type": "string"}}}},
                    "type": "object",
                    "properties": {
                        "flights": {
                            "type": "array",
                            "items": {
                                "anyOf": [{"$ref": "#/$defs/Flight"}, {"type": "object", "additionalProperties": True}]
                            },
                        },
                        "enabled": {"type": "boolean"},
                        "open_map": {"type": "object", "additionalProperties": True},
                    },
                },
            }
        ]

    def test_nested_reference_beats_generic_dict_fallback(self):
        value = {"name": "change", "arguments": {"flights": [{"number": "AB1", "origin": "BOS"}], "extra": 9}}
        self.assertEqual(schema_compare.call(value, self.tools)["arguments"], {"flights": [{"number": "AB1"}]})
        self.assertIn("origin", value["arguments"]["flights"][0])

    def test_declared_values_types_and_array_order_are_preserved(self):
        value = {"name": "change", "arguments": {"flights": [{"number": "B"}, {"number": "A"}], "enabled": False}}
        result = schema_compare.call(value, self.tools)
        self.assertEqual(result, value)
        self.assertIs(result["arguments"]["enabled"], False)

    def test_explicit_open_map_and_untouched_encoded_arguments(self):
        value = {"name": "change", "arguments": '{"open_map":{"custom":1}}'}
        self.assertEqual(schema_compare.call(value, self.tools), value)

    def test_parsed_json_string_gold_keeps_required_leaves(self):
        tools = [{"name": "add_vehicle", "parameters": {"type": "object", "properties": {"vehicle": {"type": "string"}}}}]
        gold = {"name": "add_vehicle", "arguments": {"vehicle": {"VIN": "123", "garaging_address": "Charlotte"}}}
        candidate = {"name": "add_vehicle", "arguments": {"vehicle": '{"VIN":"123"}'}}
        self.assertEqual(schema_compare.call(gold, tools), gold)
        self.assertEqual(schema_compare.call(candidate, tools), candidate)
        self.assertNotEqual(schema_compare.call(gold, tools), schema_compare.call(candidate, tools))

    def test_unknown_tools_and_malformed_arguments_remain_invalid(self):
        for value in [{"name": "unknown", "arguments": {"extra": 1}}, {"name": "change", "arguments": "bad JSON"}]:
            self.assertEqual(schema_compare.call(value, self.tools), value)

    def test_frozen_native_ledger_accepts_only_schema_equal_reference(self):
        gold = {"name": "change", "arguments": {"flights": [{"number": "AB1", "origin": "BOS"}]}}
        seen = []

        def frozen(candidate):
            seen.append(candidate)
            return {"feasible": candidate == gold}

        check = schema_native_check(frozen, {"gold": gold}, self.tools)
        correct = schema_compare.call(gold, self.tools)
        self.assertTrue(check(correct)["feasible"])
        self.assertEqual(seen[-1], gold)
        wrong = {"name": "change", "arguments": {"flights": [{"number": "WRONG"}]}}
        self.assertFalse(check(wrong)["feasible"])
        self.assertEqual(seen[-1], wrong)


if __name__ == "__main__":
    unittest.main()
