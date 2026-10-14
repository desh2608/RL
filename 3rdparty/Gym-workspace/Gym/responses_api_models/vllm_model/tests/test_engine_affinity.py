# NemotronVC: per-rollout node + engine affinity for multi-turn prefix caching.
from unittest.mock import MagicMock, patch

import nemo_gym.server_utils
from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def _server(base_url, dp=None):
    config = VLLMModelConfig(
        host="0.0.0.0", port=8081, base_url=base_url, api_key="k", model="m", entrypoint="", name="",
        return_token_id_information=False, uses_reasoning_parser=False, engine_data_parallel_size=dp,
    )
    with patch.object(nemo_gym.server_utils, "get_global_config_dict", MagicMock(return_value=dict())):
        return VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))


def _request(path):
    request = MagicMock()
    request.url.path = path
    request.session = {"session_id": "s-" + path}
    return request


def test_affinity_key_prefers_rollout_path_then_conversation_head():
    assert VLLMModel.affinity_key("/ng-rollout/abc-1/v1/chat/completions", {"messages": []}) == "rollout:abc-1"
    sys_, greet = {"role": "system", "content": "POLICY"}, {"role": "assistant", "content": "Hi! How can I help you today?"}
    # tau2 shape: fixed system + fixed agent greeting, then the simulated user's first utterance.
    ep_a = [sys_, greet, {"role": "user", "content": "I cannot send MMS"}]
    ep_b = [sys_, greet, {"role": "user", "content": "My bill is wrong"}]
    assert VLLMModel.affinity_key(None, {"messages": ep_a}) != VLLMModel.affinity_key(None, {"messages": ep_b})
    # Turn 1 of the K samples of one prompt share a key; from the first sampled reply on they differ,
    # and every later turn of a sample keeps that sample's key.
    k1 = ep_a + [{"role": "assistant", "content": "Let me check your APN"}]
    k2 = ep_a + [{"role": "assistant", "content": "Is mobile data on?"}]
    assert VLLMModel.affinity_key(None, {"messages": ep_a}) == VLLMModel.affinity_key(None, {"messages": ep_a})
    assert VLLMModel.affinity_key(None, {"messages": k1}) != VLLMModel.affinity_key(None, {"messages": k2})
    later = k1 + [{"role": "user", "content": "APN looks fine"}, {"role": "assistant", "content": "Toggle data"}]
    assert VLLMModel.affinity_key(None, {"messages": k1}) == VLLMModel.affinity_key(None, {"messages": later})
    # Tool-call first replies (content None) still key per sample.
    t1 = ep_a + [{"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "get_line", "arguments": "{}"}}]}]
    t2 = ep_a + [{"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "function": {"name": "get_bill", "arguments": "{}"}}]}]
    assert VLLMModel.affinity_key(None, {"messages": t1}) != VLLMModel.affinity_key(None, {"messages": t2})
    # Responses API: instructions + input items follow the same rule.
    resp = {"instructions": "S", "input": [{"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}, {"role": "user", "content": "U2"}]}
    assert VLLMModel.affinity_key(None, resp) == VLLMModel.affinity_key(None, {"instructions": "S", "input": resp["input"][:2]})
    assert VLLMModel.affinity_key(None, resp) != VLLMModel.affinity_key(None, {"instructions": "S", "input": [{"role": "user", "content": "U"}, {"role": "assistant", "content": "B"}]})
    assert VLLMModel.affinity_key(None, None) is None


def test_tau2_like_episodes_spread_over_nodes_and_engines():
    server = _server(["http://a:1/v1", "http://b:1/v1", "http://c:1/v1", "http://d:1/v1"], dp=8)
    sys_, greet = {"role": "system", "content": "POLICY" * 500}, {"role": "assistant", "content": "Hi! How can I help you today?"}
    targets = set()
    for i in range(400):
        msgs = [sys_, greet, {"role": "user", "content": f"issue {i}"}, {"role": "assistant", "content": f"reply {i}"}]
        c = server._resolve_client(_request("/v1/chat/completions"), {"messages": msgs})
        targets.add((c.base_url, c.default_headers["X-data-parallel-rank"]))
    assert len(targets) >= 24   # 32 possible (node, engine) slots; a constant opener would give exactly 1


def test_same_rollout_always_gets_same_endpoint_and_engine_header():
    server = _server(["http://a:1/v1", "http://b:1/v1", "http://c:1/v1"], dp=8)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    first = server._resolve_client(_request("/ng-rollout/r-7/v1/chat/completions"), body)
    again = server._resolve_client(_request("/ng-rollout/r-7/v1/chat/completions"), {"messages": body["messages"] * 3})
    assert first is again
    assert first.base_url in {"http://a:1/v1", "http://b:1/v1", "http://c:1/v1"}
    rank = int(first.default_headers["X-data-parallel-rank"])
    assert 0 <= rank < 8
    # Different rollouts spread over endpoints and engines.
    seen = {(server._resolve_client(_request(f"/ng-rollout/r-{i}/v1/chat/completions"), body).base_url,
             server._resolve_client(_request(f"/ng-rollout/r-{i}/v1/chat/completions"), body).default_headers["X-data-parallel-rank"])
            for i in range(200)}
    assert len(seen) > 12


def test_without_engine_dp_no_header_and_single_url_still_works():
    server = _server("http://a:1/v1")
    client = server._resolve_client(_request("/ng-rollout/r-1/v1/chat/completions"), {"messages": []})
    assert client is server._clients[0]
    assert "X-data-parallel-rank" not in client.default_headers


def test_session_fallback_when_no_key():
    server = _server(["http://a:1/v1", "http://b:1/v1"], dp=8)
    request = _request("/v1/chat/completions")
    client = server._resolve_client(request, None)
    assert client in server._clients
    assert server._resolve_client(request, None) is client


def _server_error(status=500):
    from aiohttp.client_exceptions import ClientResponseError
    e = ClientResponseError(request_info=MagicMock(), history=(), status=status, message="Internal Server Error")
    e.response_content = b"Internal Server Error"
    return e


def _patched_chat(dead_urls, status=500):
    """Class-level stand-in for NeMoGymAsyncOpenAI.create_chat_completion (pydantic models reject
    per-instance attribute patches): endpoints in ``dead_urls`` raise HTTP ``status``, others answer."""
    from nemo_gym.openai_utils import NeMoGymAsyncOpenAI

    async def fake(self, **kwargs):
        if self.base_url in dead_urls:
            raise _server_error(status)
        return {"ok": self.base_url, "rank": self.default_headers.get("X-data-parallel-rank")}

    return patch.object(NeMoGymAsyncOpenAI, "create_chat_completion", fake)


def test_5xx_marks_endpoint_unhealthy_and_fails_over_once():
    import asyncio

    server = _server(["http://a:1/v1", "http://b:1/v1", "http://c:1/v1"], dp=8)
    req = _request("/ng-rollout/r-9/v1/chat/completions")
    body = {"messages": [{"role": "user", "content": "hi"}]}
    first = server._resolve_client(req, body)
    dead_url = first.base_url
    with _patched_chat({dead_url}):
        result, used = asyncio.run(server._call_upstream(req, body, "create_chat_completion", body))
    assert used is not first and used.base_url != dead_url
    assert result["ok"] == used.base_url and result["rank"] is not None
    # While the cooldown lasts, every resolution hashed to the dead endpoint moves to a healthy one
    # and keeps a per-rank header.
    again = server._resolve_client(req, body)
    assert again.base_url != dead_url and "X-data-parallel-rank" in again.default_headers
    assert all(server._resolve_client(_request(f"/ng-rollout/x-{i}/v1/chat/completions"), body).base_url != dead_url
               for i in range(100))
    # Cooldown expiry restores the original affinity.
    idx = server._client_to_endpoint[id(first)]
    server._endpoint_unhealthy_until[idx] = 0.0
    assert server._resolve_client(req, body) is first


def test_4xx_and_single_endpoint_do_not_fail_over():
    import asyncio
    import pytest

    server = _server(["http://a:1/v1", "http://b:1/v1"], dp=8)
    req = _request("/ng-rollout/r-1/v1/chat/completions")
    body = {"messages": [{"role": "user", "content": "hi"}]}
    client = server._resolve_client(req, body)
    with _patched_chat({client.base_url}, status=400):
        with pytest.raises(Exception):
            asyncio.run(server._call_upstream(req, body, "create_chat_completion", body))
    assert not server._endpoint_unhealthy_until
    single = _server("http://a:1/v1")
    with _patched_chat({"http://a:1/v1"}):
        with pytest.raises(Exception):
            asyncio.run(single._call_upstream(req, body, "create_chat_completion", body))
    assert not single._endpoint_unhealthy_until
