"""The vLLM token bundle kept in tau2's AssistantMessage.raw_data must survive to_litellm_messages."""

from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.utils import llm_utils

from responses_api_agents.tau2.app import _assistant_token_info, to_litellm_messages


def _raw(prompt=(1, 2, 3), gen=(4, 5), lp=(-0.1, -0.2)):
    return {"choices": [{"message": {"role": "assistant", "content": "x", "prompt_token_ids": list(prompt),
                                     "generation_token_ids": list(gen), "generation_log_probs": list(lp)}}]}


def test_token_info_extracted_from_raw_data():
    m = AssistantMessage(role="assistant", content="hi", raw_data=_raw())
    assert _assistant_token_info(m) == {"prompt_token_ids": [1, 2, 3], "generation_token_ids": [4, 5], "generation_log_probs": [-0.1, -0.2]}


def test_token_info_absent_when_raw_data_lacks_bundle():
    assert _assistant_token_info(AssistantMessage(role="assistant", content="hi", raw_data={"choices": [{"message": {"content": "hi"}}]})) == {}
    assert _assistant_token_info(AssistantMessage(role="assistant", content="hi")) == {}


def test_to_litellm_messages_merges_bundle_and_keeps_shape():
    msgs = [
        UserMessage(role="user", content="hello"),
        AssistantMessage(role="assistant", content=None, tool_calls=[ToolCall(id="c1", name="f", arguments={"a": 1})], raw_data=_raw(gen=(9,), lp=(-1.0,))),
        AssistantMessage(role="assistant", content="plain"),
    ]
    out = to_litellm_messages(msgs)
    assert [d["role"] for d in out] == ["user", "assistant", "assistant"]
    assert out[1]["generation_token_ids"] == [9] and out[1]["prompt_token_ids"] == [1, 2, 3]
    assert out[1]["tool_calls"][0]["function"]["name"] == "f"
    assert "generation_token_ids" not in out[2]
    # tau2's generate() picks up the patched converter through its module globals.
    assert llm_utils.to_litellm_messages is to_litellm_messages


def test_user_simulator_generate_retries_malformed_tool_call_json():
    import asyncio
    import json

    import tau2.user.user_simulator as user_sim
    from responses_api_agents.tau2 import app as tau2_app

    calls = {"n": 0}

    async def flaky_generate(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            json.loads('{"x')  # raises JSONDecodeError like tau2's ToolCall construction
        return "ok"

    original = tau2_app._tau2_llm_utils.generate
    tau2_app._tau2_llm_utils.generate = flaky_generate
    try:
        assert user_sim.generate is tau2_app._user_generate_with_retries
        assert asyncio.run(user_sim.generate()) == "ok"
        assert calls["n"] == 3
        calls["n"] = -10  # never recovers within the retry budget
        try:
            asyncio.run(user_sim.generate())
        except json.JSONDecodeError:
            pass
        else:
            raise AssertionError("expected the last JSONDecodeError to propagate")
    finally:
        tau2_app._tau2_llm_utils.generate = original
