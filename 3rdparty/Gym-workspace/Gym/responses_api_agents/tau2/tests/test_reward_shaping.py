# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tau2 reward shaping: tool-not-found penalty and transfer zeroing (NemotronVC agentic-teacher-v1)."""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.server_utils import ServerClient


# Import the adapter before Tau2 so it can set TAU2_DATA_DIR before Tau2
# initializes its module-level data paths.
# isort: off
from responses_api_agents.tau2.app import (
    ModelServerRef,
    Tau2Agent,
    Tau2Config,
    Tau2RunRequest,
    shape_tau2_reward,
)
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
# isort: on


TELECOM = dict(
    tool_not_found_penalty=0.05,
    tool_not_found_penalty_cap=0.3,
    zero_reward_on_transfer_domains=["telecom"],
)


def _not_found(tool: str, requestor: str = "assistant") -> ToolMessage:
    return ToolMessage(
        id=f"call-{tool}",
        role="tool",
        content=f"Error: Tool '{tool}' not found.",
        requestor=requestor,
        error=True,
    )


def _transfer() -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="t1", name="transfer_to_human_agents", arguments={"summary": "x"})],
    )


class TestShapeTau2Reward:
    def test_defaults_leave_reward_untouched(self) -> None:
        messages = [_not_found("check_network_status")] * 3 + [_transfer()]
        shaped = shape_tau2_reward(
            1.0,
            messages,
            domain="telecom",
            tool_not_found_penalty=0.0,
            tool_not_found_penalty_cap=0.0,
            zero_reward_on_transfer_domains=[],
        )
        assert shaped["reward"] == 1.0
        assert shaped["tool_not_found_count"] == 3
        assert shaped["transfer_to_human_agents"] is True
        assert shaped["zeroed_on_transfer"] is False

    def test_penalty_per_not_found_message(self) -> None:
        messages = [
            AssistantMessage.text("checking"),
            _not_found("check_network_status"),
            UserMessage.text("still broken"),
            _not_found("toggle_airplane_mode"),
        ]
        shaped = shape_tau2_reward(1.0, messages, domain="telecom", **TELECOM)
        assert shaped["tool_not_found_count"] == 2
        assert shaped["tool_not_found_penalty"] == pytest.approx(0.10)
        assert shaped["reward"] == pytest.approx(0.90)

    def test_penalty_is_capped(self) -> None:
        messages = [_not_found(f"tool_{i}") for i in range(9)]
        shaped = shape_tau2_reward(0.0, messages, domain="telecom", **TELECOM)
        assert shaped["tool_not_found_count"] == 9
        assert shaped["tool_not_found_penalty"] == pytest.approx(0.3)
        assert shaped["reward"] == pytest.approx(-0.3)

    def test_user_side_tool_results_do_not_count(self) -> None:
        messages = [_not_found("reboot_device", requestor="user"), _not_found("x", requestor="assistant")]
        shaped = shape_tau2_reward(1.0, messages, domain="telecom", **TELECOM)
        assert shaped["tool_not_found_count"] == 1
        assert shaped["reward"] == pytest.approx(0.95)

    def test_substring_match_is_case_insensitive_and_configurable(self) -> None:
        message = ToolMessage(id="a", role="tool", content="ERROR: tool 'x' NOT FOUND", requestor="assistant")
        assert shape_tau2_reward(1.0, [message], domain="telecom", **TELECOM)["tool_not_found_count"] == 1
        assert (
            shape_tau2_reward(
                1.0, [message], domain="telecom", tool_not_found_substring="does not exist", **TELECOM
            )["tool_not_found_count"]
            == 0
        )

    def test_transfer_zeroes_only_configured_domains(self) -> None:
        messages = [_transfer(), UserMessage.text("ok")]
        telecom = shape_tau2_reward(1.0, messages, domain="telecom", **TELECOM)
        assert telecom["zeroed_on_transfer"] is True and telecom["reward"] == 0.0
        airline = shape_tau2_reward(1.0, messages, domain="airline", **TELECOM)
        assert airline["zeroed_on_transfer"] is False and airline["reward"] == 1.0

    def test_transfer_zeroing_precedes_penalty(self) -> None:
        messages = [_not_found("a"), _not_found("b"), _transfer()]
        shaped = shape_tau2_reward(1.0, messages, domain="telecom", **TELECOM)
        assert shaped["reward"] == pytest.approx(-0.10)
        assert shaped["raw_reward"] == 1.0

    def test_accepts_message_dicts(self) -> None:
        messages = [
            {"role": "tool", "content": "Error: Tool 'x' not found.", "requestor": "assistant"},
            {"role": "assistant", "tool_calls": [{"name": "transfer_to_human_agents"}]},
        ]
        shaped = shape_tau2_reward(1.0, messages, domain="telecom", **TELECOM)
        assert shaped["reward"] == pytest.approx(-0.05)


class TestRunAppliesShaping:
    def _server(self, **shaping) -> Tau2Agent:
        config = Tau2Config(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name=""),
            user_model_server=ModelServerRef(type="responses_api_models", name=""),
            max_steps=4,
            **shaping,
        )
        return Tau2Agent(config=config, server_client=MagicMock(spec=ServerClient))

    def _body(self, domain: str) -> Tau2RunRequest:
        example_jsonl = Path(__file__).parent.parent / "data" / "example.jsonl"
        with example_jsonl.open() as f:
            row = json.loads(f.readline())
        row["config"]["domain"] = domain
        return Tau2RunRequest.model_validate(row)

    def _simulation(self, messages, reward: float) -> SimulationRun:
        return SimulationRun(
            id="fake-simulation",
            task_id="fake-task",
            start_time="2026-05-05T00:00:00",
            end_time="2026-05-05T00:00:01",
            duration=1.0,
            num_steps=len(messages),
            termination_reason=TerminationReason.USER_STOP,
            reward_info=RewardInfo(reward=reward),
            messages=messages,
        )

    async def _run(self, server: Tau2Agent, body: Tau2RunRequest, messages, reward: float):
        with (
            patch("responses_api_agents.tau2.app.get_server_url", return_value="dummy base url"),
            patch(
                "responses_api_agents.tau2.app.run_single_task",
                AsyncMock(return_value=self._simulation(messages, reward)),
            ),
        ):
            return await server.run(body)

    async def test_run_default_config_is_upstream_behaviour(self) -> None:
        messages = [AssistantMessage.text("hello"), _not_found("x"), UserMessage.text("hi"), _transfer()]
        response = await self._run(self._server(), self._body("telecom"), messages, 1.0)
        assert response.reward == 1.0
        assert response.reward_shaping["tool_not_found_count"] == 1
        assert response.reward_shaping["zeroed_on_transfer"] is False

    async def test_run_applies_penalty_and_transfer_zeroing(self) -> None:
        messages = [
            AssistantMessage.text("hello"),
            _not_found("check_network_status"),
            UserMessage.text("hi"),
            _not_found("toggle_airplane_mode"),
            _transfer(),
            UserMessage.text("bye"),
        ]
        response = await self._run(self._server(**TELECOM), self._body("telecom"), messages, 1.0)
        assert response.reward == pytest.approx(-0.10)
        assert response.reward_shaping == {
            "reward": pytest.approx(-0.10),
            "raw_reward": 1.0,
            "tool_not_found_count": 2,
            "tool_not_found_penalty": pytest.approx(0.10),
            "transfer_to_human_agents": True,
            "zeroed_on_transfer": True,
        }
        dumped = response.model_dump(mode="json")
        assert dumped["reward"] == pytest.approx(-0.10)
        assert dumped["reward_shaping"]["raw_reward"] == 1.0

    async def test_run_other_domain_keeps_reward_but_still_penalizes(self) -> None:
        messages = [AssistantMessage.text("hello"), _not_found("x"), _transfer()]
        response = await self._run(self._server(**TELECOM), self._body("airline"), messages, 1.0)
        assert response.reward == pytest.approx(0.95)
        assert response.reward_shaping["zeroed_on_transfer"] is False

    def test_compute_metrics_reports_shaping_when_present(self) -> None:
        server = self._server(**TELECOM)
        task = {
            "config": {"domain": "telecom"},
            "task": {"id": "[mms_issue]x"},
            "reward": -0.1,
            "reward_shaping": {
                "reward": -0.1,
                "raw_reward": 1.0,
                "tool_not_found_count": 2,
                "tool_not_found_penalty": 0.1,
                "transfer_to_human_agents": True,
                "zeroed_on_transfer": True,
            },
            "result": {
                "termination_reason": "user_stop",
                # compute_metrics divides by the assistant-message count; one message with
                # the raw provider payload keeps the upstream percentages defined.
                "messages": [
                    {
                        "role": "tool",
                        "content": "Error: Tool 'check_network_status' not found.",
                        "requestor": "assistant",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"name": "transfer_to_human_agents", "arguments": {}}],
                        "raw_data": {
                            "choices": [
                                {
                                    "finish_reason": "tool_calls",
                                    "message": {"content": None, "tool_calls": [{"name": "transfer_to_human_agents"}]},
                                }
                            ]
                        },
                    },
                ],
            },
        }
        metrics = server.compute_metrics([[task]])
        assert metrics["trajectory_transfer_to_human_agents/count"] == 1
        assert metrics["tool_call_hallucination/count/total"] == 1
        assert metrics["telecom/reward"] == pytest.approx(-0.1)
        assert metrics["telecom/raw_reward"] == 1.0
        assert metrics["reward_shaping/tool_not_found_penalty/mean"] == pytest.approx(0.1)
        assert metrics["reward_shaping/zeroed_on_transfer/count"] == 1
        assert metrics["reward_shaping/zeroed_on_transfer/pct"] == 1.0
