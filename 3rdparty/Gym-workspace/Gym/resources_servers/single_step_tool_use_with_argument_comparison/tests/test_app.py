# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import json
from unittest.mock import MagicMock

from openai.types.responses import FunctionTool
from pytest import approx, fixture

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.single_step_tool_use_with_argument_comparison.app import (
    SingleStepToolUseArgumentComparisonResourcesServer,
    SingleStepToolUseArgumentComparisonResourcesServerConfig,
    SingleStepToolUseArgumentComparisonVerifyRequest,
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


class TestApp:
    @fixture
    def resources_server(self) -> SingleStepToolUseArgumentComparisonResourcesServer:
        tool_call_comparator_config = ToolCallComparatorConfig(word_count_similarity_threshold=0.1)
        resources_server_config = SingleStepToolUseArgumentComparisonResourcesServerConfig(
            host="127.0.0.1",
            port=20002,
            entrypoint="",
            name="tool_argument_comparison_server",
            tool_call_comparator_config=tool_call_comparator_config,
        )
        return SingleStepToolUseArgumentComparisonResourcesServer(
            config=resources_server_config,
            server_client=MagicMock(spec=ServerClient),
        )

    async def _verify_and_compare_response(
        self,
        resources_server: SingleStepToolUseArgumentComparisonResourcesServer,
        responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
        tool: FunctionTool,
        expected_action: ExpectedAction,
        response_id: str,
        output_item: NeMoGymResponseOutputItem,
        expected_reward: float,
        expected_reward_category: StepRewardCategory,
    ) -> None:
        response = NeMoGymResponse(
            id=response_id,
            created_at=1001,
            model="test_model",
            object="response",
            output=[output_item],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[tool],
        )
        verify_request = SingleStepToolUseArgumentComparisonVerifyRequest(
            responses_create_params=responses_create_params,
            response=response,
            expected_action=expected_action,
        )
        verify_response = await resources_server.verify(verify_request)
        assert verify_response.responses_create_params == responses_create_params
        assert verify_response.response == response
        assert verify_response.expected_action == expected_action
        assert verify_response.reward == approx(expected_reward)
        assert verify_response.category == expected_reward_category

    async def test_verify(self, resources_server: SingleStepToolUseArgumentComparisonResourcesServer) -> None:
        tool = FunctionTool(
            type="function",
            name="set_metric_count",
            parameters={
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                    },
                    "metric_count": {
                        "type": "integer",
                    },
                },
                "required": [
                    "metric_name",
                    "metric_count",
                ],
            },
        )
        tool_param = tool.model_dump(exclude={"defer_loading"})
        tool_call_responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(
                    role="user",
                    content="Set the views metric count to 75.",
                )
            ],
            tools=[tool_param],
        )

        expected_arguments = {
            "metric_name": "views",
            "metric_count": 75,
        }
        expected_arguments_string = json.dumps(expected_arguments)
        expected_tool_call = ExpectedFunctionCall(
            type="function_call",
            name="set_metric_count",
            arguments=expected_arguments_string,
        )

        reasoning_item = NeMoGymResponseReasoningItem(
            id="reasoning_item",
            summary=[
                NeMoGymSummary(
                    type="summary_text",
                    text="this is reasoning",
                )
            ],
        )
        await self._verify_and_compare_response(
            resources_server,
            tool_call_responses_create_params,
            tool,
            expected_tool_call,
            "no_output",
            reasoning_item,
            0.0,
            StepRewardCategory.NO_ACTION_FOUND,
        )

        different_arguments = {
            "metric_name": "views",
            "metric_count": "75",
        }
        different_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="different_value",
            name="set_metric_count",
            arguments=json.dumps(different_arguments),
        )
        await self._verify_and_compare_response(
            resources_server,
            tool_call_responses_create_params,
            tool,
            expected_tool_call,
            "different_arguments",
            different_tool_call,
            0.0,
            StepRewardCategory.TOOL_SCHEMA_VALIDATION_FAILED,
        )

        matching_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="matching_arguments",
            name="set_metric_count",
            arguments=expected_arguments_string,
        )
        await self._verify_and_compare_response(
            resources_server,
            tool_call_responses_create_params,
            tool,
            expected_tool_call,
            "matching_tool_call",
            matching_tool_call,
            1.0,
            StepRewardCategory.EXPECTED_TOOL_CALL,
        )

        extra_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="extra_matching_call",
            name="set_metric_count",
            arguments=expected_arguments_string,
        )
        multi_tool_response = NeMoGymResponse(
            id="multi_tool_call",
            created_at=1001,
            model="test_model",
            object="response",
            output=[matching_tool_call, extra_tool_call],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[tool],
        )
        multi_tool_verify_request = SingleStepToolUseArgumentComparisonVerifyRequest(
            responses_create_params=tool_call_responses_create_params,
            response=multi_tool_response,
            expected_action=expected_tool_call,
        )
        multi_tool_verify_response = await resources_server.verify(multi_tool_verify_request)
        assert multi_tool_verify_response.reward == approx(0.0)
        assert multi_tool_verify_response.category == StepRewardCategory.MULTIPLE_TOOL_CALLS_FOUND

        chat_message = NeMoGymResponseOutputMessage(
            id="chat_message",
            content=[
                NeMoGymResponseOutputText(
                    annotations=[],
                    text="How can I help you?",
                )
            ],
        )
        await self._verify_and_compare_response(
            resources_server,
            tool_call_responses_create_params,
            tool,
            expected_tool_call,
            "chat_message_instead_of_tool",
            chat_message,
            0.0,
            StepRewardCategory.NO_EXPECTED_TOOL_CALL,
        )

        chat_message_responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(
                    role="user",
                    content="This is a greeting.",
                )
            ],
            tools=[tool_param],
        )
        expected_message = ExpectedMessage(
            type="message",
            content="This is a message.",
        )

        await self._verify_and_compare_response(
            resources_server,
            chat_message_responses_create_params,
            tool,
            expected_message,
            "different_chat_message",
            chat_message,
            1.0,
            StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND,
        )
        await self._verify_and_compare_response(
            resources_server,
            chat_message_responses_create_params,
            tool,
            expected_message,
            "tool_call_instead_of_chat_message",
            matching_tool_call,
            0.0,
            StepRewardCategory.NO_EXPECTED_CHAT_MESSAGE,
        )

    async def test_verify_list_f1_match_details(self) -> None:
        tool_call_comparator_config = ToolCallComparatorConfig(
            word_count_similarity_threshold=0.3,
            use_f1_for_list=True,
            use_list_f1_threshold=False,
        )
        resources_server_config = SingleStepToolUseArgumentComparisonResourcesServerConfig(
            host="127.0.0.1",
            port=20002,
            entrypoint="",
            name="tool_argument_comparison_server",
            tool_call_comparator_config=tool_call_comparator_config,
        )
        resources_server = SingleStepToolUseArgumentComparisonResourcesServer(
            config=resources_server_config,
            server_client=MagicMock(spec=ServerClient),
        )

        tool = FunctionTool(
            type="function",
            name="search",
            parameters={
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["items"],
            },
        )
        expected_action = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps({"items": [1, 2, 3]}),
        )
        actual_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="test_f1",
            name="search",
            arguments=json.dumps({"items": [1, 2]}),
        )
        response = NeMoGymResponse(
            id="f1_test",
            created_at=1001,
            model="test_model",
            object="response",
            output=[actual_tool_call],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[tool],
        )
        verify_request = SingleStepToolUseArgumentComparisonVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="Search for items.")],
                tools=[tool.model_dump(exclude={"defer_loading"})],
            ),
            response=response,
            expected_action=expected_action,
        )
        verify_response = await resources_server.verify(verify_request)
        assert abs(verify_response.reward - 0.8) < 1e-6
        assert verify_response.category == StepRewardCategory.ARGUMENT_LIST_F1_PARTIAL
        assert len(verify_response.list_f1_match_details) == 1
        detail = verify_response.list_f1_match_details[0]
        assert detail.expected_values == [1, 2, 3]
        assert detail.actual_values == [1, 2]
        assert detail.tp == 2
        assert detail.matched_pairs == [(0, 0), (1, 1)]
        assert abs(detail.f1 - 0.8) < 1e-6


# ---------------------------------------------------------------------------------------------------
# Parallel tool calls through verify() (``function_call_batch`` gold), ported from upstream Gym's
# test_app.py and extended with this tree's declared-schema check and single-call regression lock.
#
# Module-level, zero-argument, synchronous functions on purpose: the training container has no pytest,
# and scripts/pytest_stub/run_selected.py can only call ``module::function`` with no fixtures and does
# not await coroutines, so verify() is driven through ``asyncio.run``.
# ---------------------------------------------------------------------------------------------------


def _build_resources_server(**comparator_overrides: object) -> SingleStepToolUseArgumentComparisonResourcesServer:
    """Same wiring as the ``resources_server`` fixture in TestApp, with the comparator knobs overridable."""
    tool_call_comparator_config = ToolCallComparatorConfig(word_count_similarity_threshold=0.1, **comparator_overrides)
    resources_server_config = SingleStepToolUseArgumentComparisonResourcesServerConfig(
        host="127.0.0.1",
        port=20002,
        entrypoint="",
        name="tool_argument_comparison_server",
        tool_call_comparator_config=tool_call_comparator_config,
    )
    return SingleStepToolUseArgumentComparisonResourcesServer(
        config=resources_server_config,
        server_client=MagicMock(spec=ServerClient),
    )


def _search_tool() -> FunctionTool:
    return FunctionTool(
        type="function",
        name="search",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    )


def _search_call(index: int, query: str, name: str = "search") -> NeMoGymResponseFunctionToolCall:
    return NeMoGymResponseFunctionToolCall(call_id=f"call_{index}", name=name, arguments=json.dumps({"query": query}))


def _parallel_verify_request(
    tool: FunctionTool,
    output: list[NeMoGymResponseOutputItem],
    expected_queries: list[str],
) -> SingleStepToolUseArgumentComparisonVerifyRequest:
    responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
        input=[NeMoGymEasyInputMessage(role="user", content="Search for two related facts.")],
        parallel_tool_calls=True,
        tools=[tool.model_dump(exclude={"defer_loading"})],
    )
    response = NeMoGymResponse(
        id="parallel_tool_calls",
        created_at=1001,
        model="test_model",
        object="response",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[tool],
    )
    expected_action = ExpectedFunctionCallBatch(
        type="function_call_batch",
        calls=[
            ExpectedFunctionCall(type="function_call", name="search", arguments=json.dumps({"query": query}))
            for query in expected_queries
        ],
    )
    return SingleStepToolUseArgumentComparisonVerifyRequest(
        responses_create_params=responses_create_params,
        response=response,
        expected_action=expected_action,
    )


def _queries_request(
    tool: FunctionTool, actual_queries: list[str], expected_queries: list[str]
) -> SingleStepToolUseArgumentComparisonVerifyRequest:
    output = [_search_call(index, query) for index, query in enumerate(actual_queries)]
    return _parallel_verify_request(tool, output, expected_queries)


def _verify(
    resources_server: SingleStepToolUseArgumentComparisonResourcesServer,
    verify_request: SingleStepToolUseArgumentComparisonVerifyRequest,
):
    """verify() is async; run it to completion so these tests stay plain sync functions (see note above)."""
    verify_response = asyncio.run(resources_server.verify(verify_request))
    assert verify_response.expected_action == verify_request.expected_action
    assert verify_response.response == verify_request.response
    return verify_response


def test_verify_parallel_tool_calls() -> None:
    tool = _search_tool()
    default_server = _build_resources_server()
    counting_server = _build_resources_server(parallel_tool_call_rewarding=True)

    # The response emits the expected calls in the opposite order, which does not matter.
    for resources_server in (default_server, counting_server):
        verify_response = _verify(resources_server, _queries_request(tool, ["beta", "alpha"], ["alpha", "beta"]))
        assert verify_response.reward == 1.0
        assert verify_response.category == StepRewardCategory.EXPECTED_TOOL_CALL_BATCH
        assert verify_response.list_f1_match_details == []

    surplus = _queries_request(tool, ["alpha", "beta", "gamma"], ["alpha", "beta"])

    # With the shipped default (parallel_tool_call_rewarding off) the call count is ignored.
    verify_response = _verify(default_server, surplus)
    assert verify_response.reward == 1.0
    assert verify_response.category == StepRewardCategory.EXPECTED_TOOL_CALL_BATCH

    # With it on, a surplus call is disqualifying unless the cardinality gate is opened.
    verify_response = _verify(counting_server, surplus)
    assert verify_response.reward == 0.0
    assert verify_response.category == StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT


def test_verify_parallel_tool_calls_with_f1_reward_mode() -> None:
    resources_server = _build_resources_server(
        parallel_tool_call_rewarding=True,
        allow_subset=True,
        allow_superset=True,
        parallel_tool_call_reward_mode=ParallelToolCallRewardMode.F1,
    )
    tool = _search_tool()

    verify_response = _verify(resources_server, _queries_request(tool, ["beta", "alpha"], ["alpha", "beta"]))
    assert verify_response.reward == 1.0
    assert verify_response.category == StepRewardCategory.EXPECTED_TOOL_CALL_BATCH

    # Both correct calls are present, but so are three junk calls: 2 * 2 / (2 + 5).
    verify_response = _verify(
        resources_server, _queries_request(tool, ["alpha", "beta", "x", "y", "z"], ["alpha", "beta"])
    )
    assert abs(verify_response.reward - 4 / 7) < 1e-9
    assert verify_response.category == StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT


def test_verify_parallel_batch_gold_without_tool_calls() -> None:
    tool = _search_tool()
    resources_server = _build_resources_server(parallel_tool_call_rewarding=True)
    chat_message = NeMoGymResponseOutputMessage(
        id="chat_message",
        content=[NeMoGymResponseOutputText(annotations=[], text="How can I help you?")],
    )
    reasoning_item = NeMoGymResponseReasoningItem(
        id="reasoning_item",
        summary=[NeMoGymSummary(type="summary_text", text="this is reasoning")],
    )

    # Answering in prose when calls were expected is the same miss it is for a single-call row.
    verify_response = _verify(resources_server, _parallel_verify_request(tool, [chat_message], ["alpha", "beta"]))
    assert verify_response.reward == 0.0
    assert verify_response.category == StepRewardCategory.NO_EXPECTED_TOOL_CALL

    verify_response = _verify(resources_server, _parallel_verify_request(tool, [reasoning_item], ["alpha", "beta"]))
    assert verify_response.reward == 0.0
    assert verify_response.category == StepRewardCategory.NO_ACTION_FOUND


def test_verify_parallel_batch_validates_every_call_against_declared_schema() -> None:
    """A malformed call anywhere in the batch is a hard zero, exactly as it is for a single call.

    The server is deliberately permissive (superset admitted, f1 scoring): without the schema check
    each of these responses would still earn 2 * 1 / (2 + 2) = 0.5 for its one good call.
    """
    tool = _search_tool()
    resources_server = _build_resources_server(
        parallel_tool_call_rewarding=True,
        allow_superset=True,
        parallel_tool_call_reward_mode=ParallelToolCallRewardMode.F1,
    )
    good_call = _search_call(0, "alpha")

    wrong_type_call = NeMoGymResponseFunctionToolCall(
        call_id="call_1", name="search", arguments=json.dumps({"query": 75})
    )
    verify_response = _verify(
        resources_server, _parallel_verify_request(tool, [good_call, wrong_type_call], ["alpha", "beta"])
    )
    assert verify_response.reward == 0.0
    assert verify_response.category == StepRewardCategory.TOOL_SCHEMA_VALIDATION_FAILED

    undeclared_tool_call = _search_call(1, "beta", name="lookup")
    verify_response = _verify(
        resources_server, _parallel_verify_request(tool, [good_call, undeclared_tool_call], ["alpha", "beta"])
    )
    assert verify_response.reward == 0.0
    assert verify_response.category == StepRewardCategory.TOOL_SCHEMA_NOT_FOUND

    undecodable_call = NeMoGymResponseFunctionToolCall(call_id="call_1", name="search", arguments="query=beta")
    verify_response = _verify(
        resources_server, _parallel_verify_request(tool, [good_call, undecodable_call], ["alpha", "beta"])
    )
    assert verify_response.reward == 0.0
    assert verify_response.category == StepRewardCategory.ARGUMENTS_DECODE_ERROR

    # The same two calls with valid arguments score normally, so the zeros above came from the schema check.
    verify_response = _verify(resources_server, _queries_request(tool, ["alpha", "beta"], ["alpha", "beta"]))
    assert verify_response.reward == 1.0
    assert verify_response.category == StepRewardCategory.EXPECTED_TOOL_CALL_BATCH


def test_verify_single_call_and_message_gold_still_reject_multiple_calls() -> None:
    """Regression lock: the parallel knobs never relax the single-call or message contracts.

    TestApp.test_verify already asserts MULTIPLE_TOOL_CALLS_FOUND on the default server; this repeats
    it on a server with every parallel knob opened, where a batch gold WOULD have scored.
    """
    tool = _search_tool()
    two_calls = NeMoGymResponse(
        id="multi_tool_call",
        created_at=1001,
        model="test_model",
        object="response",
        output=[_search_call(0, "alpha"), _search_call(1, "alpha")],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[tool],
    )
    responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
        input=[NeMoGymEasyInputMessage(role="user", content="Search for alpha.")],
        parallel_tool_calls=True,
        tools=[tool.model_dump(exclude={"defer_loading"})],
    )
    single_call_gold = ExpectedFunctionCall(
        type="function_call", name="search", arguments=json.dumps({"query": "alpha"})
    )
    message_gold = ExpectedMessage(type="message", content="Anything at all.")

    permissive_server = _build_resources_server(
        parallel_tool_call_rewarding=True,
        allow_subset=True,
        allow_superset=True,
        parallel_tool_call_reward_mode=ParallelToolCallRewardMode.F1,
    )
    for resources_server in (_build_resources_server(), permissive_server):
        for expected_action, expected_category in (
            (single_call_gold, StepRewardCategory.MULTIPLE_TOOL_CALLS_FOUND),
            (message_gold, StepRewardCategory.NO_EXPECTED_CHAT_MESSAGE),
        ):
            verify_request = SingleStepToolUseArgumentComparisonVerifyRequest(
                responses_create_params=responses_create_params,
                response=two_calls,
                expected_action=expected_action,
            )
            verify_response = _verify(resources_server, verify_request)
            assert verify_response.reward == 0.0
            assert verify_response.category == expected_category

    # And the batch gold for the same two calls on the permissive server does score, so the lock above
    # is about the gold's type, not about the response.
    batch_request = _parallel_verify_request(tool, list(two_calls.output), ["alpha", "alpha"])
    verify_response = _verify(permissive_server, batch_request)
    assert verify_response.reward == 1.0
    assert verify_response.category == StepRewardCategory.EXPECTED_TOOL_CALL_BATCH
