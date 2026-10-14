# NemotronVC: several simulator nodes behind one openai_model server (comma-separated openai_base_url).
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig, split_base_urls


def test_split_base_urls():
    assert split_base_urls("http://a:1/v1") == ["http://a:1/v1"]
    assert split_base_urls(" http://a:1/v1 , http://b:1/v1,") == ["http://a:1/v1", "http://b:1/v1"]
    with pytest.raises(ValueError):
        split_base_urls(" , ")


def _server(base_url: str) -> SimpleModelServer:
    config = SimpleModelServerConfig(
        host="localhost", port=0, entrypoint="", name="sim",
        openai_base_url=base_url, openai_api_key="k", openai_model="m",
    )
    return SimpleModelServer(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))


def test_one_url_keeps_single_client():
    server = _server("http://a:1/v1")
    assert len(server._clients) == 1 and server._client is server._clients[0]


def test_requests_go_to_least_loaded_upstream():
    server = _server("http://a:1/v1,http://b:1/v1")
    assert len(server._clients) == 2
    calls = []

    def make(name):
        async def create_chat_completion(**body):
            calls.append(name)
            await asyncio.sleep(0.01 if name == "a" else 0.0)
            return {"id": name, "object": "chat.completion", "created": 0, "model": "m",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": name}}]}
        client = MagicMock(spec=NeMoGymAsyncOpenAI)
        client.create_chat_completion = AsyncMock(side_effect=create_chat_completion)
        return client

    server._clients = [make("a"), make("b")]
    server._inflight = [0, 0]
    body = MagicMock()
    body.model_dump.return_value = {"messages": [{"role": "user", "content": "hi"}]}

    async def run():
        # Two concurrent requests: first picks a (tie -> lowest index), second sees a busy -> picks b.
        return await asyncio.gather(server.chat_completions(body), server.chat_completions(body))

    out = asyncio.run(run())
    assert sorted(calls) == ["a", "b"]
    assert server._inflight == [0, 0]
    assert {o.choices[0].message.content for o in out} == {"a", "b"}


def test_single_upstream_still_uses_replaced_client_attribute():
    server = _server("http://a:1/v1")
    body = MagicMock()
    body.model_dump.return_value = {"messages": [{"role": "user", "content": "hi"}]}
    replaced = MagicMock(spec=NeMoGymAsyncOpenAI)
    replaced.create_chat_completion = AsyncMock(return_value={
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]})
    server._client = replaced
    out = asyncio.run(server.chat_completions(body))
    replaced.create_chat_completion.assert_awaited_once()
    assert out.choices[0].message.content == "ok"
