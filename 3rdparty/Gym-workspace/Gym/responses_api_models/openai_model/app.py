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
from contextlib import nullcontext
from typing import Any, Dict, Optional

from pydantic import Field

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


class SimpleModelServerConfig(BaseResponsesAPIModelConfig):
    # One base URL, or several comma-separated URLs (NemotronVC: one vLLM API server per simulator
    # node). Every request goes to the upstream with the fewest in-flight requests from this process.
    openai_base_url: str
    openai_api_key: str
    openai_model: str

    extra_body: Dict[str, Any] = Field(default_factory=dict)
    openai_default_headers: Dict[str, str] = Field(default_factory=dict)

    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description=(
            "Cap on in-flight upstream requests from this server (per-process "
            "asyncio.Semaphore). Set on rate-limited endpoints (e.g. Gemini) "
            "to stay under quota; None = unlimited."
        ),
    )

    drop_input_reasoning_items: bool = Field(
        default=False,
        description=(
            "Strip type=reasoning items from the Responses API input before the "
            "upstream call. Workaround for endpoints (e.g. NVIDIA-hosted gpt-oss) "
            "that 500 with KeyError 'content' on their own content-less reasoning "
            "items when echoed back across tool-use turns."
        ),
    )


def split_base_urls(value: str) -> list[str]:
    """Comma-separated base URLs -> list; whitespace ignored; at least one non-empty URL required."""
    urls = [part.strip() for part in str(value).split(",")]
    urls = [url for url in urls if url]
    if not urls:
        raise ValueError("openai_base_url must contain at least one URL")
    return urls


class SimpleModelServer(SimpleResponsesAPIModel):
    config: SimpleModelServerConfig

    def model_post_init(self, context):
        base_urls = split_base_urls(self.config.openai_base_url)
        self._clients = [
            NeMoGymAsyncOpenAI(
                base_url=base_url,
                api_key=self.config.openai_api_key,
                default_headers=self.config.openai_default_headers,
            )
            for base_url in base_urls
        ]
        self._client = self._clients[0]
        self._inflight = [0] * len(self._clients)
        self._semaphore = (
            asyncio.Semaphore(self.config.max_concurrent_requests)
            if self.config.max_concurrent_requests is not None
            else nullcontext()
        )

        return super().model_post_init(context)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        if self.config.drop_input_reasoning_items:
            input_items = body_dict.get("input")
            if isinstance(input_items, list):
                body_dict["input"] = [
                    item for item in input_items if not (isinstance(item, dict) and item.get("type") == "reasoning")
                ]
        async with self._semaphore:
            index = self._pick_upstream()
            self._inflight[index] += 1
            try:
                openai_response_dict = await self._upstream(index).create_response(**body_dict)
            finally:
                self._inflight[index] -= 1
        return NeMoGymResponse.model_validate(openai_response_dict)

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        async with self._semaphore:
            index = self._pick_upstream()
            self._inflight[index] += 1
            try:
                openai_response_dict = await self._upstream(index).create_chat_completion(**body_dict)
            finally:
                self._inflight[index] -= 1
        return NeMoGymChatCompletion.model_validate(openai_response_dict)

    def _pick_upstream(self) -> int:
        """Index of the upstream with the fewest in-flight requests (ties -> lowest index)."""
        return min(range(len(self._clients)), key=lambda i: (self._inflight[i], i))

    def _upstream(self, index: int) -> NeMoGymAsyncOpenAI:
        # Single upstream keeps using ``self._client`` (tests and subclasses replace that attribute).
        return self._client if len(self._clients) == 1 else self._clients[index]


if __name__ == "__main__":
    SimpleModelServer.run_webserver()
