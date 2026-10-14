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
"""Behavioural contract for parallel tool-call (``function_call_batch``) verification.

WHY THIS FILE EXISTS
--------------------
static-v2 training rows only ever expected 0 or 1 tool call, and the verifier scored any response with
more than one call as 0.0 (``MULTIPLE_TOOL_CALLS_FOUND``). After 27 GRPO steps BFCL v3 fell from 69.10
to 67.71, with the parallel categories down 12.5 / 7.5 pp: the policy had been taught never to emit
two calls at once. static-v3 adds a ``parallel_tool_calls_..._agent`` route whose rows carry
``function_call_batch`` gold, and this file pins down (a) that the shipped route yaml really turns the
four comparator knobs on in the way its comment promises, (b) how the knob values chosen for the
static-v3 route score the failure modes we care about, and (c) that the shipped example rows verify.

STATIC-V3 KNOBS
---------------
``parallel_tool_call_rewarding: true, allow_subset: false, allow_superset: true, reward mode f1``.
``allow_subset: false`` is the one deliberate departure from the worked-example yaml in ``configs/``:
emitting one call when several were asked for is exactly the behaviour that cost the BFCL parallel
categories, so it must score ZERO rather than the f1 partial credit the example config would give.
``allow_superset: true`` with ``f1`` keeps over-calling admissible but docked, so the policy is not
pushed back towards under-calling.

TEST STYLE
----------
Module-level, zero-argument, synchronous functions, mirroring ``tests/common/test_bc_v4_8_config.py``'s
yaml-loading pattern but without fixtures: the training container has no pytest, and
``scripts/pytest_stub/run_selected.py`` can only call ``module::function`` with no arguments and does
not await coroutines, so ``verify()`` is driven through ``asyncio.run``.
"""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import yaml
from pydantic import TypeAdapter

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.single_step_tool_use_with_argument_comparison.app import (
    SingleStepToolUseArgumentComparisonResourcesServer,
    SingleStepToolUseArgumentComparisonResourcesServerConfig,
    SingleStepToolUseArgumentComparisonVerifyRequest,
    SingleStepToolUseArgumentComparisonVerifyResponse,
)
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ExpectedAction,
    ExpectedFunctionCall,
    ExpectedFunctionCallBatch,
    ExpectedMessage,
    ParallelToolCallRewardMode,
    StepRewardCategory,
    ToolCallComparatorConfig,
)


SERVER_DIR = Path(__file__).parents[1]
PARALLEL_YAML = SERVER_DIR / "configs" / "parallel_tool_calls_single_step_tool_use_with_argument_comparison.yaml"
PARALLEL_EXAMPLE_JSONL = SERVER_DIR / "data" / "parallel_example.jsonl"
PARALLEL_RESOURCES_SERVER = "parallel_tool_calls_single_step_tool_use_with_argument_comparison_resources_server"
PARALLEL_AGENT = "parallel_tool_calls_single_step_tool_use_with_argument_comparison_agent"
IMPLEMENTATION = "single_step_tool_use_with_argument_comparison"

# The ``tool_call_comparator_config`` block config/gym-services-static-v3.yaml gives the parallel route.
# Kept inline (not loaded from that file) so this server's tests stay self-contained; the make_run.py
# gates own the services yaml.
STATIC_V3_PARALLEL_COMPARATOR_CONFIG: dict[str, Any] = {
    "word_count_similarity_threshold": 0.1,
    "parallel_tool_call_rewarding": True,
    "allow_subset": False,
    "allow_superset": True,
    "parallel_tool_call_reward_mode": "f1",
}

SEARCH_TOOL_PARAM: dict[str, Any] = {
    "type": "function",
    "name": "search",
    "strict": None,
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


def _raw_yaml() -> dict[str, Any]:
    return yaml.safe_load(PARALLEL_YAML.read_text())


def _yaml_comparator_config() -> dict[str, Any]:
    return _raw_yaml()[PARALLEL_RESOURCES_SERVER]["resources_servers"][IMPLEMENTATION]["tool_call_comparator_config"]


def _build_server(comparator_config: dict[str, Any]) -> SingleStepToolUseArgumentComparisonResourcesServer:
    """Same wiring as the ``resources_server`` fixture in tests/test_app.py, fed a raw yaml-shaped dict."""
    resources_server_config = SingleStepToolUseArgumentComparisonResourcesServerConfig(
        host="127.0.0.1",
        port=20002,
        entrypoint="",
        name="tool_argument_comparison_server",
        tool_call_comparator_config=ToolCallComparatorConfig(**comparator_config),
    )
    return SingleStepToolUseArgumentComparisonResourcesServer(
        config=resources_server_config,
        server_client=MagicMock(spec=ServerClient),
    )


def _response(output: list[NeMoGymResponseOutputItem]) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="parallel_batch",
        created_at=1001,
        model="test_model",
        object="response",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _search_calls(queries: list[str]) -> list[NeMoGymResponseFunctionToolCall]:
    return [
        NeMoGymResponseFunctionToolCall(call_id=f"call_{index}", name="search", arguments=json.dumps({"query": query}))
        for index, query in enumerate(queries)
    ]


def _search_batch(queries: list[str]) -> ExpectedFunctionCallBatch:
    return ExpectedFunctionCallBatch(
        type="function_call_batch",
        calls=[
            ExpectedFunctionCall(type="function_call", name="search", arguments=json.dumps({"query": query}))
            for query in queries
        ],
    )


def _verify(
    resources_server: SingleStepToolUseArgumentComparisonResourcesServer,
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
    output: list[NeMoGymResponseOutputItem],
    expected_action: ExpectedAction,
) -> SingleStepToolUseArgumentComparisonVerifyResponse:
    verify_request = SingleStepToolUseArgumentComparisonVerifyRequest(
        responses_create_params=responses_create_params,
        response=_response(output),
        expected_action=expected_action,
    )
    return asyncio.run(resources_server.verify(verify_request))


def _verify_search(
    resources_server: SingleStepToolUseArgumentComparisonResourcesServer,
    actual_queries: list[str],
    expected_queries: list[str],
) -> tuple[float, StepRewardCategory]:
    responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
        input=[NeMoGymEasyInputMessage(role="user", content="Search for several related facts.")],
        parallel_tool_calls=True,
        tools=[SEARCH_TOOL_PARAM],
    )
    verify_response = _verify(
        resources_server, responses_create_params, _search_calls(actual_queries), _search_batch(expected_queries)
    )
    return verify_response.reward, verify_response.category


def _close(reward: float, expected_reward: float) -> bool:
    return abs(reward - expected_reward) < 1e-9


# --------------------------------------------------------------------------
# The shipped route yaml
# --------------------------------------------------------------------------


def test_parallel_yaml_loads_into_comparator_config() -> None:
    """The four knobs land on ToolCallComparatorConfig with the values the yaml comment promises."""
    raw = _yaml_comparator_config()
    config = ToolCallComparatorConfig(**raw)

    assert config.word_count_similarity_threshold == 0.1
    assert config.parallel_tool_call_rewarding is True
    assert config.allow_subset is True
    assert config.allow_superset is True
    assert config.parallel_tool_call_reward_mode == ParallelToolCallRewardMode.F1

    # The yaml is only about parallelism: every other knob keeps its default, so this route compares
    # arguments exactly like the plain single_step route does.
    defaults = ToolCallComparatorConfig(word_count_similarity_threshold=0.1)
    for field_name in ToolCallComparatorConfig.model_fields:
        if field_name not in raw:
            assert getattr(config, field_name) == getattr(defaults, field_name), field_name


def test_parallel_yaml_agent_wiring() -> None:
    """Same implementation and agent as the single_step entries in the services yaml, new route name."""
    raw = _raw_yaml()
    assert set(raw) == {PARALLEL_RESOURCES_SERVER, PARALLEL_AGENT}

    resources_servers = raw[PARALLEL_RESOURCES_SERVER]["resources_servers"]
    assert set(resources_servers) == {IMPLEMENTATION}
    assert resources_servers[IMPLEMENTATION]["entrypoint"] == "app.py"

    agents = raw[PARALLEL_AGENT]["responses_api_agents"]
    assert set(agents) == {"tool_simulation_agent"}
    agent = agents["tool_simulation_agent"]
    assert agent["resources_server"] == {"type": "resources_servers", "name": PARALLEL_RESOURCES_SERVER}
    assert agent["model_server"] == {"type": "responses_api_models", "name": "policy_model"}


def test_parallel_yaml_knobs_are_honoured_end_to_end() -> None:
    """Reproduce the yaml comment's measured-rewards table, then flip each knob and watch it move."""
    yaml_config = _yaml_comparator_config()
    server = _build_server(yaml_config)
    golds = ["alpha", "beta"]
    both_reversed = ["beta", "alpha"]
    one_wrong = ["alpha", "wrong"]
    only_one = ["alpha"]
    spam = ["alpha", "beta", "x", "y", "z"]

    assert _verify_search(server, both_reversed, golds) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL_BATCH)
    reward, category = _verify_search(server, one_wrong, golds)
    assert _close(reward, 0.5) and category == StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
    reward, category = _verify_search(server, only_one, golds)
    assert _close(reward, 2 / 3) and category == StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT
    reward, category = _verify_search(server, spam, golds)
    assert _close(reward, 4 / 7) and category == StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT

    # parallel_tool_call_rewarding: off means the count is ignored (spam is free) but a missing call
    # still has to be there.
    off = _build_server({**yaml_config, "parallel_tool_call_rewarding": False})
    assert _verify_search(off, spam, golds) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL_BATCH)
    assert _verify_search(off, only_one, golds) == (0.0, StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT)

    # allow_subset: closing it zeroes under-calling and leaves over-calling untouched.
    no_subset = _build_server({**yaml_config, "allow_subset": False})
    assert _verify_search(no_subset, only_one, golds) == (0.0, StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT)
    assert _close(_verify_search(no_subset, spam, golds)[0], 4 / 7)

    # allow_superset: closing it zeroes over-calling and leaves under-calling untouched.
    no_superset = _build_server({**yaml_config, "allow_superset": False})
    assert _verify_search(no_superset, spam, golds) == (0.0, StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT)
    assert _close(_verify_search(no_superset, only_one, golds)[0], 2 / 3)

    # parallel_tool_call_reward_mode: the other two modes turn the open gates into free passes.
    binary = _build_server({**yaml_config, "parallel_tool_call_reward_mode": "binary_strict"})
    assert _verify_search(binary, one_wrong, golds) == (0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT)
    assert _verify_search(binary, only_one, golds) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL_BATCH)
    assert _verify_search(binary, spam, golds) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL_BATCH)

    fractional = _build_server({**yaml_config, "parallel_tool_call_reward_mode": "fractional"})
    reward, category = _verify_search(fractional, one_wrong, golds)
    assert _close(reward, 0.5) and category == StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
    assert _verify_search(fractional, only_one, golds) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL_BATCH)
    assert _verify_search(fractional, spam, golds) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL_BATCH)


# --------------------------------------------------------------------------
# The static-v3 route's knob values
# --------------------------------------------------------------------------


def test_static_v3_parallel_route_zeroes_undercalling_and_docks_overcalling() -> None:
    config = ToolCallComparatorConfig(**STATIC_V3_PARALLEL_COMPARATOR_CONFIG)
    assert (
        config.parallel_tool_call_rewarding,
        config.allow_subset,
        config.allow_superset,
        config.parallel_tool_call_reward_mode,
    ) == (True, False, True, ParallelToolCallRewardMode.F1)

    server = _build_server(STATIC_V3_PARALLEL_COMPARATOR_CONFIG)
    golds = ["alpha", "beta"]

    # Exactly the expected set, in any order, is the only way to full credit.
    assert _verify_search(server, ["beta", "alpha"], golds) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL_BATCH)

    # Under-calling is the BFCL failure mode: hard zero, not f1 partial credit.
    assert _verify_search(server, ["alpha"], golds) == (0.0, StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT)
    assert _verify_search(server, ["beta"], golds) == (0.0, StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT)

    # Right count, one call wrong: 2 * 1 / (2 + 2), and the near miss is what gets reported.
    reward, category = _verify_search(server, ["alpha", "wrong"], golds)
    assert _close(reward, 0.5) and category == StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

    # Over-calling is admissible but docked: 2 * 2 / (2 + 3), then 2 * 2 / (2 + 5).
    reward, category = _verify_search(server, ["alpha", "beta", "junk"], golds)
    assert _close(reward, 0.8) and category == StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT
    reward, category = _verify_search(server, ["alpha", "beta", "x", "y", "z"], golds)
    assert _close(reward, 4 / 7) and category == StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT

    # Three expected, two right plus one junk: 2 * 2 / (3 + 3); the unmatched gold call's near miss
    # (right tool, wrong argument) is the category.
    reward, category = _verify_search(server, ["alpha", "beta", "junk"], ["alpha", "beta", "gamma"])
    assert _close(reward, 2 / 3) and category == StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

    # A batch of one scores like the single-call row it came from, except that over-calling is docked
    # rather than zeroed: 2 * 1 / (1 + 2).
    assert _verify_search(server, ["alpha"], ["alpha"]) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL)
    reward, category = _verify_search(server, ["alpha", "junk"], ["alpha"])
    assert _close(reward, 2 / 3) and category == StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT


# --------------------------------------------------------------------------
# The shipped example rows
# --------------------------------------------------------------------------


def test_parallel_example_rows_verify_with_the_yaml_config() -> None:
    """The five example rows parse as the three gold shapes and score as the route yaml says they should."""
    rows = [json.loads(line) for line in PARALLEL_EXAMPLE_JSONL.read_text().splitlines() if line.strip()]
    adapter = TypeAdapter(ExpectedAction)
    actions = [adapter.validate_python(row["expected_action"]) for row in rows]
    assert [type(action) for action in actions] == [
        ExpectedFunctionCallBatch,
        ExpectedFunctionCallBatch,
        ExpectedFunctionCallBatch,
        ExpectedFunctionCall,
        ExpectedMessage,
    ]
    assert [len(action.calls) for action in actions[:3]] == [3, 2, 2]

    server = _build_server(_yaml_comparator_config())
    for row, action in zip(rows[:3], actions[:3]):
        responses_create_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            row["responses_create_params"]
        )
        # The tools come from the row itself, so this also exercises the declared-schema check on
        # Responses-format tools with ``strict: true`` / ``additionalProperties: false``.
        assert len(responses_create_params.tools) >= 1

        # Re-emitting the gold calls in reverse order is full credit.
        output = [
            NeMoGymResponseFunctionToolCall(call_id=f"call_{index}", name=call.name, arguments=call.arguments)
            for index, call in enumerate(reversed(action.calls))
        ]
        verify_response = _verify(server, responses_create_params, output, action)
        assert verify_response.reward == 1.0, row["responses_create_params"]["input"][-1]["content"]
        assert verify_response.category == StepRewardCategory.EXPECTED_TOOL_CALL_BATCH

        # Dropping one call is admitted by this yaml's allow_subset and scored 2 (n - 1) / (n + n - 1).
        expected_count = len(action.calls)
        verify_response = _verify(server, responses_create_params, output[1:], action)
        assert _close(verify_response.reward, 2 * (expected_count - 1) / (2 * expected_count - 1))
        assert verify_response.category == StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT

    # The single-call row keeps the single-call contract even on this route: two calls is a hard zero.
    single_row, single_action = rows[3], actions[3]
    responses_create_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        single_row["responses_create_params"]
    )
    one_call = [
        NeMoGymResponseFunctionToolCall(call_id="call_0", name=single_action.name, arguments=single_action.arguments)
    ]
    verify_response = _verify(server, responses_create_params, one_call, single_action)
    assert (verify_response.reward, verify_response.category) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL)
    verify_response = _verify(server, responses_create_params, one_call + one_call, single_action)
    assert (verify_response.reward, verify_response.category) == (0.0, StepRewardCategory.MULTIPLE_TOOL_CALLS_FOUND)

    # The message row is unchanged: any assistant text is full credit.
    message_row, message_action = rows[4], actions[4]
    responses_create_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        message_row["responses_create_params"]
    )
    chat_message = NeMoGymResponseOutputMessage(
        id="chat_message",
        content=[NeMoGymResponseOutputText(annotations=[], text="You're welcome!")],
    )
    verify_response = _verify(server, responses_create_params, [chat_message], message_action)
    assert (verify_response.reward, verify_response.category) == (1.0, StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND)
