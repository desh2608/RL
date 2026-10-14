# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable with unittest; no GPU, Ray service, network or fake reward."""

import copy
import importlib.util
import unittest
from unittest.mock import patch

from nemo_rl.environments.terminal_text_verifier import (
    RewardContractError,
    RewardInfrastructureError,
    final_response,
    score_terminal_response,
    validate_reward_spec,
)


def tool_metadata():
    return {
        "domain": "dtau",
        "source_id": "fixture:strict-tool",
        "original_verifier": {
            "schema_version": 2,
            "type": "tool_action",
            "payload": {
                "strict": True,
                "parallel_tool_calls": False,
                "expected": {
                    "type": "function_call",
                    "name": "lookup",
                    "arguments": {"city": "Boston", "count": 2, "active": False},
                },
            },
        },
    }


def tool_schemas():
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "count": {"type": "integer"},
                        "active": {"type": "boolean"},
                    },
                },
            },
        }
    ]


GOOD_CALL = "<tool_call><function=lookup><parameter=city>Boston</parameter><parameter=count>2</parameter><parameter=active>false</parameter></function></tool_call>"


class TerminalTextVerifierTests(unittest.TestCase):
    def score(self, response, *, metadata=None, truncated=False):
        data = tool_metadata() if metadata is None else metadata
        spec = validate_reward_spec(data, tools=tool_schemas())
        return score_terminal_response(
            response,
            spec=spec,
            source_id=data["source_id"],
            domain=data["domain"],
            response_truncated=truncated,
        )

    def test_real_strict_action_matches_all_arguments(self):
        result = self.score("reasoning</think>" + GOOD_CALL)
        self.assertEqual(
            (result.status, result.outcome, result.score), ("valid", "success", 1.0)
        )

    def test_wrong_name_or_argument_fails(self):
        for response in [
            GOOD_CALL.replace("lookup", "delete"),
            GOOD_CALL.replace("Boston", "Austin"),
            GOOD_CALL.replace(">2<", ">3<"),
        ]:
            self.assertEqual(self.score("</think>" + response).score, 0.0)

    def test_full_call_set_is_checked_not_only_first(self):
        result = self.score(
            "</think>" + GOOD_CALL + GOOD_CALL.replace("lookup", "delete")
        )
        self.assertEqual(result.score, 0.0)

    def test_extra_missing_duplicate_parameters_fail(self):
        variants = [
            GOOD_CALL.replace(
                "</function>", "<parameter=extra>x</parameter></function>"
            ),
            GOOD_CALL.replace("<parameter=count>2</parameter>", ""),
            GOOD_CALL.replace(
                "</function>", "<parameter=city>Boston</parameter></function>"
            ),
        ]
        self.assertTrue(
            all(self.score("</think>" + text).score == 0.0 for text in variants)
        )

    def test_garbage_does_not_match_false_boolean(self):
        self.assertEqual(
            self.score("</think>" + GOOD_CALL.replace(">false<", ">banana<")).score, 0.0
        )

    def test_non_call_text_is_not_silently_discarded(self):
        self.assertEqual(self.score("</think>ignored " + GOOD_CALL).score, 0.0)

    def test_truncated_correct_prefix_is_valid_budget_failure(self):
        result = self.score("</think>" + GOOD_CALL, truncated=True)
        self.assertEqual(
            (result.status, result.score, result.outcome),
            ("valid", 0.0, "budget_exhausted"),
        )

    def test_empty_and_unclosed_thinking_are_model_failures(self):
        for response in [
            "",
            "still reasoning",
            "reasoning</think>",
            "<think>bad</think>again</think>x",
        ]:
            result = self.score(response)
            self.assertEqual(
                (result.status, result.outcome, result.score),
                ("valid", "malformed", 0.0),
            )

    def test_final_extraction_does_not_grade_reasoning(self):
        self.assertEqual(
            final_response("orchid reasoning</think>rose<|im_end|>"), "rose"
        )

    def test_weak_message_and_first_call_checker_are_unsupported(self):
        for kind in ["message", "weak"]:
            metadata = tool_metadata()
            if kind == "message":
                metadata["original_verifier"]["payload"]["expected"] = {
                    "type": "message"
                }
            else:
                metadata["original_verifier"]["payload"]["strict"] = False
            with self.assertRaises(RewardContractError):
                validate_reward_spec(metadata, tools=tool_schemas())

    def test_bad_reference_never_becomes_model_zero(self):
        metadata = tool_metadata()
        metadata["original_verifier"]["payload"]["expected"]["arguments"] = (
            '{"x":1,"x":2}'
        )
        with self.assertRaises(RewardContractError):
            validate_reward_spec(metadata, tools=tool_schemas())

    def test_type_sensitive_reference(self):
        metadata = tool_metadata()
        metadata["original_verifier"]["payload"]["expected"]["arguments"]["count"] = 2.0
        self.assertEqual(
            self.score("</think>" + GOOD_CALL, metadata=metadata).score, 0.0
        )

    def test_validation_does_not_mutate_private_data(self):
        metadata, schemas = tool_metadata(), tool_schemas()
        before = copy.deepcopy((metadata, schemas))
        validate_reward_spec(metadata, tools=schemas)
        self.assertEqual((metadata, schemas), before)

    def test_unknown_verifier_is_fail_closed(self):
        metadata = tool_metadata()
        metadata["original_verifier"]["type"] = "teacher_only"
        with self.assertRaises(RewardContractError):
            validate_reward_spec(metadata, tools=tool_schemas())

    def test_missing_if_dependency_never_fills_zero(self):
        metadata = {
            "verifier": {
                "schema_version": 1,
                "type": "instruction_following",
                "payload": {
                    "instruction_id_list": ["keywords:existence"],
                    "kwargs": [{"keywords": ["orchid"]}],
                },
            }
        }
        with patch(
            "nemo_rl.environments.terminal_text_verifier._instruction_registry",
            side_effect=RewardInfrastructureError("missing dependency"),
        ):
            with self.assertRaises(RewardInfrastructureError):
                validate_reward_spec(metadata, tools=[])


@unittest.skipUnless(
    importlib.util.find_spec("verifiable_instructions")
    and importlib.util.find_spec("langdetect"),
    "Run in the real pinned CPU/container environment for actual IF checker",
)
class RealInstructionFollowingTests(unittest.TestCase):
    def test_actual_registry_checks_final_all_constraints(self):
        spec = validate_reward_spec(
            {
                "verifier": {
                    "schema_version": 1,
                    "type": "instruction_following",
                    "payload": {
                        "instruction_id_list": ["keywords:existence"],
                        "kwargs": [{"keywords": ["orchid", "violet"]}],
                    },
                }
            },
            tools=[],
        )
        for response, expected in [
            ("orchid in reasoning</think>violet", 0.0),
            ("</think>orchid violet", 1.0),
        ]:
            result = score_terminal_response(
                response,
                spec=spec,
                source_id="fixture:if",
                domain="if",
                response_truncated=False,
            )
            self.assertEqual(result.score, expected)


if __name__ == "__main__":
    unittest.main()
