# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real actor implementation and metadata privacy tests; run in CPU container."""

import copy
import importlib.util
import json
import unittest
from unittest.mock import patch

DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ["torch", "ray", "transformers", "pydantic"]
)


def fixture_datum():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    view = {
        "messages": [{"role": "user", "content": "Perform the requested lookup."}],
        "tools": tools,
        "modality": "text",
        "multimodal_payload": {},
    }
    return {
        "source_id": "fixture:private-reference",
        "student_view": view,
        "teacher_view": copy.deepcopy(view),
        "teacher_alias": "nano",
        "task_name": "terminal_text_reward",
        "extra_env_info": {
            "domain": "dtau",
            "source": {"trajectory_id": "trajectory-fixture"},
            "verifier": {"schema_version": 1, "type": "teacher_only", "payload": {}},
            "original_verifier": {
                "schema_version": 2,
                "type": "tool_action",
                "payload": {
                    "strict": True,
                    "expected": {
                        "type": "function_call",
                        "name": "lookup",
                        "arguments": {"city": "PRIVATE_REFERENCE_CANARY"},
                    },
                },
            },
        },
    }


@unittest.skipUnless(DEPENDENCIES, "Run with the real pinned training CPU interpreter")
class TerminalTextRewardIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Optional runtime imports keep this file discoverable on a CPU-only
        # workstation; the sealed container must run these without skipping.
        import torch

        from nemo_rl.data.interfaces import TaskDataSpec
        from nemo_rl.data.terminal_text_reward import terminal_text_reward_processor
        from nemo_rl.environments.terminal_text_reward import TerminalTextRewardCore

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return json.dumps({"messages": messages, "tools": kwargs["tools"]})

            def __call__(self, text, **kwargs):
                return {
                    "input_ids": torch.tensor(
                        [[ord(char) for char in text]], dtype=torch.long
                    )
                }

        self.tokenizer = Tokenizer()
        self.spec = TaskDataSpec(task_name="terminal_text_reward")
        self.processor = terminal_text_reward_processor
        self.env = TerminalTextRewardCore(
            {
                "response_scope": "final_after_think",
                "require_think_close": True,
                "reject_truncated": True,
            }
        )

    def process(self):
        datum = fixture_datum()
        before = copy.deepcopy(datum)
        output = self.processor(datum, self.spec, self.tokenizer, 8192, 7)
        self.assertEqual(datum, before)
        return output

    def test_collector_processing_does_not_import_scoring_dependencies(self):
        from nemo_rl.environments.terminal_text_verifier import RewardInfrastructureError

        datum = fixture_datum()
        datum["student_view"]["tools"] = []
        datum["teacher_view"]["tools"] = []
        datum["extra_env_info"]["original_verifier"] = {
            "schema_version": 1, "type": "instruction_following",
            "payload": {"instruction_id_list": ["keywords:existence"],
                        "kwargs": [{"keywords": ["orchid"]}]},
        }
        with patch("nemo_rl.environments.terminal_text_verifier._instruction_registry",
                   side_effect=RewardInfrastructureError("Scoring package unavailable")) as registry:
            output = self.processor(datum, self.spec, self.tokenizer, 8192, 7)
            registry.assert_not_called()
            metadata = output["extra_env_info"]
            for truncated in (False, True):
                metadata["_terminal_reward_runtime"] = {"response_truncated": truncated}
                with self.assertRaises(RewardInfrastructureError):
                    self.env.step([[{"role": "assistant", "content": "</think>orchid"}]], [metadata])
            self.assertEqual(registry.call_count, 2)

    def test_unknown_if_is_rejected_by_scoring_even_after_portable_processing(self):
        from nemo_rl.environments.terminal_text_verifier import RewardContractError

        datum = fixture_datum()
        datum["extra_env_info"]["original_verifier"] = {
            "schema_version": 1, "type": "instruction_following",
            "payload": {"instruction_id_list": ["unknown:checker"], "kwargs": [{}]},
        }
        with patch("nemo_rl.environments.terminal_text_verifier._instruction_registry", return_value={}):
            output = self.processor(datum, self.spec, self.tokenizer, 8192, 7)
            metadata = output["extra_env_info"]
            metadata["_terminal_reward_runtime"] = {"response_truncated": True}
            with self.assertRaises(RewardContractError):
                self.env.step([[{"role": "assistant", "content": "</think>text"}]], [metadata])

    def test_reference_never_enters_student_or_teacher_prompt(self):
        output = self.process()
        self.assertNotIn(
            "PRIVATE_REFERENCE_CANARY", output["message_log"][0]["content"]
        )
        self.assertNotIn("PRIVATE_REFERENCE_CANARY", json.dumps(output["teacher_view"]))
        self.assertEqual(
            output["extra_env_info"]["source"]["trajectory_id"], "trajectory-fixture"
        )
        self.assertEqual(
            output["extra_env_info"]["source_id"], "fixture:private-reference"
        )

    def test_real_terminal_reward_and_empty_observation(self):
        output = self.process()
        metadata = output["extra_env_info"]
        metadata["_terminal_reward_runtime"] = {"response_truncated": False}
        good = "</think><tool_call><function=lookup><parameter=city>PRIVATE_REFERENCE_CANARY</parameter></function></tool_call>"
        result = self.env.step(
            [output["message_log"] + [{"role": "assistant", "content": good}]],
            [metadata],
        )
        self.assertEqual(result.rewards.tolist(), [1.0])
        self.assertEqual(result.observations, [{"role": "environment", "content": ""}])
        self.assertEqual(result.terminateds.tolist(), [True])
        self.assertEqual(result.metadata[0]["_terminal_reward"]["status"], "valid")
        self.assertNotIn("_terminal_reward", metadata)
        result2 = self.env.step(
            [
                output["message_log"]
                + [{"role": "assistant", "content": "</think>Wrong action"}]
            ],
            [metadata],
        )
        self.assertEqual(result2.rewards.tolist(), [0.0])
        self.assertEqual(
            result2.metadata[0]["_terminal_reward"]["outcome"], "task_failure"
        )

    def test_reserved_runtime_fields_cannot_be_forged_by_dataset(self):
        for key in [
            "_terminal_reward_contract",
            "_terminal_reward_runtime",
            "_terminal_reward",
            "reward_tools",
        ]:
            datum = fixture_datum()
            datum["extra_env_info"][key] = {"response_truncated": False}
            with self.assertRaises(ValueError):
                self.processor(datum, self.spec, self.tokenizer, 8192, 7)

    def test_missing_truncation_metadata_fails_closed(self):
        output = self.process()
        with self.assertRaises(ValueError):
            self.env.step(
                [[{"role": "assistant", "content": "</think>response"}]],
                [output["extra_env_info"]],
            )

    def test_truncation_is_valid_model_failure_not_infra(self):
        output = self.process()
        metadata = output["extra_env_info"]
        metadata["_terminal_reward_runtime"] = {"response_truncated": True}
        result = self.env.step(
            [[{"role": "assistant", "content": "still reasoning"}]], [metadata]
        )
        self.assertEqual(result.rewards.tolist(), [0.0])
        self.assertEqual(
            result.metadata[0]["_terminal_reward"]["outcome"], "budget_exhausted"
        )

    def test_historical_assistant_cannot_supply_the_current_answer(self):
        output = self.process()
        metadata = output["extra_env_info"]
        metadata["_terminal_reward_runtime"] = {"response_truncated": False}
        with self.assertRaises(ValueError):
            self.env.step(
                [
                    [
                        {"role": "assistant", "content": "</think>old answer"},
                        {"role": "user", "content": "new question"},
                    ]
                ],
                [metadata],
            )


if __name__ == "__main__":
    unittest.main()
