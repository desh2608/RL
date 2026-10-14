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
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputRefusal,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from resources_servers.single_step_tool_use_with_argument_comparison.common.response_utils import (
    extract_function_calls,
    extract_tool_call_or_text,
)


class TestResponseUtils:
    def _create_response(self, output_list: list[NeMoGymResponseOutputItem]) -> NeMoGymResponse:
        return NeMoGymResponse(
            id="test_response",
            created_at=101.0,
            model="test_model",
            object="response",
            output=output_list,
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

    def test_extract_tool_call_or_text(self) -> None:
        assert extract_tool_call_or_text(self._create_response([])) is None

        reasoning_item = NeMoGymResponseReasoningItem(
            id="reasoning_item",
            summary=[
                NeMoGymSummary(
                    type="summary_text",
                    text="this is reasoning text",
                )
            ],
        )
        assert extract_tool_call_or_text(self._create_response([reasoning_item])) is None

        first_output_text = NeMoGymResponseOutputText(
            annotations=[],
            text="this is the first output text",
        )
        single_text_message = NeMoGymResponseOutputMessage(
            id="single_text",
            content=[first_output_text],
        )
        assert extract_tool_call_or_text(self._create_response([single_text_message])) is first_output_text

        second_output_text = NeMoGymResponseOutputText(
            annotations=[],
            text="this is the second output text",
        )
        multiple_texts_message = NeMoGymResponseOutputMessage(
            id="multiple_texts",
            content=[
                second_output_text,
                first_output_text,
            ],
        )
        assert (
            extract_tool_call_or_text(
                self._create_response(
                    [
                        reasoning_item,
                        multiple_texts_message,
                        single_text_message,
                    ]
                )
            )
            is second_output_text
        )

        refusal_message = NeMoGymResponseOutputMessage(
            id="refusal",
            content=[
                NeMoGymResponseOutputRefusal(refusal="this is a refusal"),
            ],
        )
        assert (
            extract_tool_call_or_text(
                self._create_response(
                    [
                        reasoning_item,
                        refusal_message,
                        single_text_message,
                    ]
                )
            )
            is first_output_text
        )

        tool_call = NeMoGymResponseFunctionToolCall(
            call_id="tool_call",
            name="respond",
            arguments="",
        )
        assert extract_tool_call_or_text(self._create_response([tool_call])) is tool_call
        assert (
            extract_tool_call_or_text(
                self._create_response(
                    [
                        single_text_message,
                        tool_call,
                    ]
                )
            )
            is tool_call
        )
        assert (
            extract_tool_call_or_text(
                self._create_response(
                    [
                        tool_call,
                        single_text_message,
                    ]
                )
            )
            is tool_call
        )
        assert (
            extract_tool_call_or_text(
                self._create_response(
                    [
                        reasoning_item,
                        single_text_message,
                        refusal_message,
                        tool_call,
                        multiple_texts_message,
                    ]
                )
            )
            is tool_call
        )


# ---------------------------------------------------------------------------------------------------
# ``extract_function_calls`` (the batch view used by ``function_call_batch`` gold).
#
# Module-level, zero-argument, synchronous functions on purpose: the training container has no pytest,
# and scripts/pytest_stub/run_selected.py can only call ``module::function`` with no fixtures.
# ---------------------------------------------------------------------------------------------------


def _response(output_list: list[NeMoGymResponseOutputItem]) -> NeMoGymResponse:
    """Same shape as ``TestResponseUtils._create_response`` above, reachable without a class instance."""
    return NeMoGymResponse(
        id="test_response",
        created_at=101.0,
        model="test_model",
        object="response",
        output=output_list,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def test_extract_function_calls_returns_every_call_in_emission_order() -> None:
    first_tool_call = NeMoGymResponseFunctionToolCall(call_id="first_tool_call", name="respond", arguments="")
    second_tool_call = NeMoGymResponseFunctionToolCall(
        call_id="second_tool_call", name="lookup", arguments='{"query": "alpha"}'
    )
    text_message = NeMoGymResponseOutputMessage(
        id="single_text",
        content=[NeMoGymResponseOutputText(annotations=[], text="this is the output text")],
    )
    reasoning_item = NeMoGymResponseReasoningItem(
        id="reasoning_item",
        summary=[NeMoGymSummary(type="summary_text", text="this is reasoning text")],
    )

    assert extract_function_calls(_response([first_tool_call])) == [first_tool_call]
    assert extract_function_calls(
        _response([reasoning_item, text_message, first_tool_call, second_tool_call, text_message])
    ) == [first_tool_call, second_tool_call]
    assert extract_function_calls(_response([second_tool_call, first_tool_call])) == [
        second_tool_call,
        first_tool_call,
    ]

    # The single-action view and the batch view agree on the first call, so the two gold paths never
    # disagree about which call a one-call response made.
    assert extract_tool_call_or_text(_response([text_message, first_tool_call, second_tool_call])) is first_tool_call


def test_extract_function_calls_is_empty_without_tool_calls() -> None:
    reasoning_item = NeMoGymResponseReasoningItem(
        id="reasoning_item",
        summary=[NeMoGymSummary(type="summary_text", text="this is reasoning text")],
    )
    text_message = NeMoGymResponseOutputMessage(
        id="single_text",
        content=[NeMoGymResponseOutputText(annotations=[], text="this is the output text")],
    )
    refusal_message = NeMoGymResponseOutputMessage(
        id="refusal",
        content=[NeMoGymResponseOutputRefusal(refusal="this is a refusal")],
    )

    assert extract_function_calls(_response([])) == []
    assert extract_function_calls(_response([reasoning_item])) == []
    assert extract_function_calls(_response([text_message])) == []
    assert extract_function_calls(_response([reasoning_item, refusal_message, text_message])) == []
