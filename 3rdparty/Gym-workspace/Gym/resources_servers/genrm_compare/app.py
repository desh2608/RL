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

"""
GenRM Pairwise Comparison Resources Server.

Compares multiple candidate responses using a GenRM model via pairwise comparisons.
The GenRM model expects OpenAI-format messages with special roles 'response_1' and 'response_2'.

Input:
- conversation_history: List of user/assistant messages
- response_objs: List of N candidate Response API objects to compare

Output:
- Per-response rewards after pairwise aggregation
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import raise_for_status
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from resources_servers.genrm_compare.utils import (
    GenRMOutputParseError,
    aggregate_scores,
    extract_output_text,
    generate_comparison_pairs,
    get_prompt_key_from_input,
    parse_genrm_output,
)


logger = logging.getLogger(__name__)

# Cohort state for verify(): buffer by prompt_key until num_rollouts_per_prompt received (Difference 1)
_cohort_failures: Dict[str, str] = {}
_cohort_buffers: Dict[str, List[Tuple[Any, asyncio.Future]]] = defaultdict(list)
_cohort_jit_tasks: Dict[str, List[asyncio.Task]] = defaultdict(list)


class GenRMCompareConfig(BaseResourcesServerConfig):
    """Configuration for the GenRM compare server.

    Attributes:
        genrm_model_server: Target GenRM model server (default: genrm_model from config)
        genrm_responses_create_params: Base create params for GenRM calls
        comparison_strategy: "all_pairs" or "circular"
        num_judges_per_comparison: Number of judge passes per pair (majority voting)
        aggregator_method: Method for aggregating scores
        reasoning_bonus: Bonus for shortest reasoning content among top performers
        answer_bonus: Bonus for shortest answer among top performers
        top_percentile: Percentile threshold for applying bonuses
        group_reasoning_length_penalty_coeff: Coefficient for reasoning length penalty
        group_answer_length_penalty_coeff: Coefficient for answer length penalty
        group_style_penalty_coeff: Coefficient for style density penalty
        default_score: Default neutral score when parsing fails
        default_ranking: Default neutral ranking when parsing fails
        debug_logging: Enable verbose logging for debugging
        genrm_parse_retries: Number of retries on parse failures
        genrm_parse_retry_sleep_s: Sleep duration between parse retries
        use_principle: Enable principle-based comparison
        default_principle: Default principle when none provided in request
    """

    name: str = "genrm_compare"
    genrm_model_server: ModelServerRef  # Default: genrm_model (see config)
    genrm_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    # Cohort-based verify: number of rollouts per prompt before running comparison (Difference 1)
    # When > 1, verify() buffers by prompt and runs comparison when cohort is full; rewards are relative to cohort.
    # When <= 1, verify() returns default_score (no comparison).
    num_rollouts_per_prompt: int = 1

    # Comparison strategy
    comparison_strategy: str = "circular"  # "all_pairs" or "circular"
    num_judges_per_comparison: int = 1

    # Principle-based GenRM settings
    use_principle: bool = False
    default_principle: str = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
        "to the user prompt. Begin your evaluation by generating your own answer to the prompt. You must provide "
        "your answer before judging any answers. When evaluating the assistants' answers, compare both assistants' "
        "answers with your answer. You must identify and correct any mistakes or inaccurate information. Then "
        "consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly "
        "responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than "
        "one interpretation, it is more helpful and appropriate to ask for clarifications or more information from "
        "the user than providing an answer based on assumptions. Relevant means all parts of the response closely "
        "connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or "
        "excessive. Then consider the creativity and novelty of the assistant's answers when needed. Finally, "
        "identify any missing important information in the assistants' answers that would be beneficial to include "
        "when responding to the user prompt."
    )

    # Aggregator settings (only "simple_tiebreaker" is currently implemented)
    aggregator_method: str = "simple_tiebreaker"

    # Length bonus config (only for simple_tiebreaker)
    reasoning_bonus: float = 0.0
    answer_bonus: float = 0.0
    top_percentile: float = 0.2
    group_reasoning_length_penalty_coeff: float = 0.0
    group_answer_length_penalty_coeff: float = 0.0
    group_style_penalty_coeff: float = 0.0

    # Default neutral scores when parsing fails
    default_score: float = 3.0
    default_ranking: float = 3.5

    # Debug logging
    debug_logging: bool = False

    # Retry config for parse failures
    genrm_parse_retries: int = 3
    genrm_parse_retry_sleep_s: float = 0.2
    # Opt in for externally hosted judges: never replace API or parse failures
    # with plausible neutral training labels. Existing recipes retain defaults.
    fail_on_error: bool = False


class GenRMCompareVerifyRequest(BaseVerifyRequest):
    """Verify request with optional principle for cohort-based GenRM comparison."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    principle: Optional[str] = None  # Principle for principle-based GenRM; forwarded by agent when provided
    task_index: Optional[int] = Field(default=None, alias=TASK_INDEX_KEY_NAME)
    rollout_index: Optional[int] = Field(default=None, alias=ROLLOUT_INDEX_KEY_NAME)
    prompt_id: Optional[str] = None  # Optional stable prompt identifier from the caller


class GenRMCompareRequest(BaseModel):
    """Request payload for GenRM pairwise comparison."""

    conversation_history: List[Dict[str, str]]  # User/assistant messages before the responses
    response_objs: List[Dict[str, Any]]  # Raw Response API objects from policy model
    principle: Optional[str] = None  # Principle for principle-based GenRM (e.g., "The response should be helpful")


class GenRMCompareResponse(BaseModel):
    """Response payload with per-response rewards."""

    rewards: List[float]  # One reward per response, in same order as input
    comparison_results: Optional[List[Dict[str, Any]]] = None  # Detailed pairwise results
    metrics: Optional[Dict[str, float]] = None  # Aggregation metrics


def _input_to_conversation_history(input_messages: Any) -> List[Dict[str, str]]:
    """Convert Response API input messages to conversation_history list of {role, content}."""
    out: List[Dict[str, str]] = []
    items = list(input_messages) if input_messages else []
    for m in items:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
        else:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "") or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "output_text"
            )
        out.append({"role": str(role), "content": str(content)})
    return out


class GenRMCompareResourcesServer(SimpleResourcesServer):
    """Resources server for GenRM pairwise comparison of multiple responses.

    Supports two modes:
    - Cohort-based verify (Difference 1): When num_rollouts_per_prompt > 1, verify() buffers by prompt;
      when the cohort is full, runs comparison and returns per-rollout rewards. Callers await until
      their cohort is complete and get their reward.
    - Batch /compare: Direct comparison of N response_objs (e.g. for rollout_collection or tests).
    """

    config: GenRMCompareConfig

    async def verify(self, body: GenRMCompareVerifyRequest) -> BaseVerifyResponse:
        if self.config.num_rollouts_per_prompt <= 1:
            if self.config.fail_on_error:
                raise ValueError("Fail-closed GenRM requires at least two rollouts per prompt")
            return await self._verify_impl(body)
        messages = getattr(body.responses_create_params, "input", None) or []
        key = self._get_verify_cohort_key(body, list(messages), body.principle)
        if key in _cohort_failures:
            raise RuntimeError("GenRM cohort previously failed; refusing implicit retry")
        return await self._verify_impl(body)

    async def _abort_cohort(self, prompt_key: str, cohort_buf: List[Tuple[Any, asyncio.Future]]) -> None:
        # A completed caller must never cancel a newer cohort reusing this key.
        if _cohort_buffers.get(prompt_key) is not cohort_buf:
            return
        _cohort_failures[prompt_key] = "external_judge_failure"
        _cohort_buffers.pop(prompt_key)
        tasks = _cohort_jit_tasks.pop(prompt_key, [])
        for task in tasks:
            if not task.done():
                task.cancel()
        for _, future in cohort_buf:
            if not future.done():
                future.set_exception(RuntimeError("GenRM cohort judge failed"))
                future.exception()  # Mark retrieved; awaiting still raises.
        # Each task owns and drains its judge children before it exits.
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _verify_impl(self, body: GenRMCompareVerifyRequest) -> BaseVerifyResponse:
        """Buffer a cohort; aggregate only after every registered comparison completes."""
        cfg = self.config
        principle = body.principle
        if cfg.num_rollouts_per_prompt <= 1:
            return BaseVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=body.response,
                reward=cfg.default_score,
            )

        input_messages = getattr(body.responses_create_params, "input", None) or []
        prompt_key = self._get_verify_cohort_key(
            body,
            input_messages if isinstance(input_messages, list) else list(input_messages),
            principle,
        )
        future: asyncio.Future[float] = asyncio.get_running_loop().create_future()
        cohort_buf = _cohort_buffers[prompt_key]
        cohort_buf.append((body, future))
        try:
            conversation_history = _input_to_conversation_history(input_messages)
            response_objs = [
                (b.response.model_dump() if hasattr(b.response, "model_dump") else b.response) for b, _ in cohort_buf
            ]
            assert len(response_objs) <= cfg.num_rollouts_per_prompt
            principle_val = getattr(body, "principle", None) or principle

            # Register before the first await: requests in this server share an
            # event loop, and each arrival owns pairs involving its newest index.
            comparison_task = asyncio.create_task(
                self._run_jit_compare_using_most_recent_response_obj(
                    conversation_history, response_objs, [], principle_val
                )
            )
            cohort_tasks = _cohort_jit_tasks[prompt_key]
            cohort_tasks.append(comparison_task)
            # Every request observes its own task errors, even before K arrivals.
            await asyncio.shield(comparison_task)

            if len(response_objs) == cfg.num_rollouts_per_prompt:
                # All tasks already run concurrently. Shielding leaves this
                # cohort's single abort path in charge of cancellation/draining.
                outcomes = [await asyncio.shield(task) for task in cohort_tasks]
                results = [result for new_results, _ in outcomes for result in new_results]
                metadata = [item for _, new_metadata in outcomes for item in new_metadata]
                expected = [
                    (i, j, judge_idx)
                    for judge_idx in range(cfg.num_judges_per_comparison)
                    for i, j in generate_comparison_pairs(cfg.comparison_strategy, cfg.num_rollouts_per_prompt)
                ]
                if sorted(metadata) != sorted(expected) or len(results) != len(expected):
                    raise RuntimeError("Incomplete or duplicate GenRM cohort comparisons")
                # Preserve the original judge/pair order and reward algorithm.
                ordered = sorted(zip(results, metadata), key=lambda pair: (pair[1][2], pair[1][0], pair[1][1]))
                existing_results, existing_metadata = zip(*ordered)
                rewards, _, _, _ = aggregate_scores(
                    comparison_results=existing_results,
                    comparison_metadata=existing_metadata,
                    response_objs=response_objs,
                    aggregator_method=cfg.aggregator_method,
                    default_score=cfg.default_score,
                    reasoning_bonus=cfg.reasoning_bonus,
                    answer_bonus=cfg.answer_bonus,
                    top_percentile=cfg.top_percentile,
                    group_reasoning_length_penalty_coeff=cfg.group_reasoning_length_penalty_coeff,
                    group_answer_length_penalty_coeff=cfg.group_answer_length_penalty_coeff,
                    group_style_penalty_coeff=cfg.group_style_penalty_coeff,
                )
                if len(rewards) != len(cohort_buf):
                    raise RuntimeError("Invalid GenRM cohort reward count")
                _cohort_jit_tasks.pop(prompt_key)
                _cohort_buffers.pop(prompt_key)
                for i, (_, f) in enumerate(cohort_buf):
                    if not f.done():
                        f.set_result(rewards[i])

            reward = await future
            return BaseVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=body.response,
                reward=reward,
            )
        except (Exception, asyncio.CancelledError):
            # Cancellation is not caught by Exception. Wake all result waiters
            # and cancel only this cohort's registered judge tasks in both modes.
            # Legacy per-judge default-score policy remains in _run_single_comparison.
            await self._abort_cohort(prompt_key, cohort_buf)
            raise

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/compare")(self.compare)
        return app

    def _get_verify_cohort_key(
        self,
        body: GenRMCompareVerifyRequest,
        input_messages: List[Any],
        principle: Optional[str] = None,
    ) -> str:
        """Prefer task-scoped keys when available so identical prompt text from different tasks does not collide."""
        prompt_key = get_prompt_key_from_input(input_messages, principle)
        if body.task_index is not None:
            return f"task_idx::{body.task_index}::{prompt_key}"
        if body.prompt_id is not None:
            return f"prompt_id::{body.prompt_id}::{prompt_key}"
        return prompt_key

    async def _run_jit_compare_using_most_recent_response_obj(
        self,
        conversation_history: List[Dict[str, str]],
        response_objs: List[Dict[str, Any]],
        seen_comparison_metadata: List[Tuple[int, int, int]],
        principle: Optional[str] = None,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
        # Cannot run comparison with only 1 result
        if len(response_objs) == 1:
            return [], []

        cfg = self.config
        this_response_idx = len(response_objs) - 1

        comparison_pairs = generate_comparison_pairs(cfg.comparison_strategy, cfg.num_rollouts_per_prompt)
        comparison_tasks = []
        comparison_metadata: List[Tuple[int, int, int]] = []
        for judge_idx in range(cfg.num_judges_per_comparison):
            for i, j in comparison_pairs:
                # If one of the indices has not yet been run, continue
                if not (i < len(response_objs) and j < len(response_objs)):
                    continue

                # At least one of the indices must be this index
                if i != this_response_idx and j != this_response_idx:
                    continue

                this_comparison_metadata = (i, j, judge_idx)

                # Don't double count since this will trigger when both i and j are finished.
                if this_comparison_metadata in seen_comparison_metadata:
                    continue

                comparison_tasks.append(
                    asyncio.create_task(
                        self._run_single_comparison(
                            conversation_history,
                            response_objs[i],
                            response_objs[j],
                            pair_idx=(i, j),
                            principle=principle,
                        )
                    )
                )
                comparison_metadata.append(this_comparison_metadata)

        group = asyncio.gather(*comparison_tasks)
        try:
            comparison_results = await asyncio.shield(group)
        except (Exception, asyncio.CancelledError):
            # gather alone does not cancel siblings when one judge fails.
            # Cancel/join every child before this registered task can finish.
            for task in comparison_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*comparison_tasks, return_exceptions=True)
            if not group.cancelled():
                group.exception()  # Retrieve a failure from the shielded group.
            raise

        return comparison_results, comparison_metadata

    async def _run_compare(
        self,
        conversation_history: List[Dict[str, str]],
        response_objs: List[Dict[str, Any]],
        principle: Optional[str] = None,
    ) -> Tuple[List[float], Dict[str, float], List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
        """Run pairwise comparison; return (rewards, metrics, comparison_results, comparison_metadata)."""
        cfg = self.config
        num_responses = len(response_objs)
        if num_responses < 2:
            return [cfg.default_score] * num_responses, {}, [], []

        comparison_pairs = generate_comparison_pairs(cfg.comparison_strategy, num_responses)
        comparison_tasks = []
        comparison_metadata: List[Tuple[int, int, int]] = []
        for judge_idx in range(cfg.num_judges_per_comparison):
            for i, j in comparison_pairs:
                comparison_tasks.append(
                    self._run_single_comparison(
                        conversation_history,
                        response_objs[i],
                        response_objs[j],
                        pair_idx=(i, j),
                        principle=principle,
                    )
                )
                comparison_metadata.append((i, j, judge_idx))
        comparison_results = await asyncio.gather(*comparison_tasks)
        rewards, metrics, _, _ = aggregate_scores(
            comparison_results=list(comparison_results),
            comparison_metadata=comparison_metadata,
            response_objs=response_objs,
            aggregator_method=cfg.aggregator_method,
            default_score=cfg.default_score,
            reasoning_bonus=cfg.reasoning_bonus,
            answer_bonus=cfg.answer_bonus,
            top_percentile=cfg.top_percentile,
            group_reasoning_length_penalty_coeff=cfg.group_reasoning_length_penalty_coeff,
            group_answer_length_penalty_coeff=cfg.group_answer_length_penalty_coeff,
            group_style_penalty_coeff=cfg.group_style_penalty_coeff,
        )
        return rewards, metrics, list(comparison_results), comparison_metadata

    async def compare(self, body: GenRMCompareRequest) -> GenRMCompareResponse:
        """Compare multiple responses using GenRM pairwise comparisons (batch API)."""
        cfg = self.config
        response_objs = body.response_objs
        conversation_history = body.conversation_history
        num_responses = len(response_objs)
        if cfg.debug_logging:
            logger.info(f"[GenRM] Compare request: {num_responses} responses")
        if num_responses < 2:
            return GenRMCompareResponse(
                rewards=[cfg.default_score],
                comparison_results=None,
                metrics=None,
            )
        rewards, metrics, comparison_results, comparison_metadata = await self._run_compare(
            conversation_history, response_objs, principle=body.principle
        )
        detailed_results = [
            {
                "response_i": i,
                "response_j": j,
                "judge_idx": judge_idx,
                "score_1": score_1,
                "score_2": score_2,
                "ranking": ranking,
            }
            for (score_1, score_2, ranking), (i, j, judge_idx) in zip(comparison_results, comparison_metadata)
        ]
        if cfg.debug_logging:
            logger.info(f"[GenRM] Final rewards: {[f'{r:.4f}' for r in rewards]}")
        return GenRMCompareResponse(
            rewards=rewards,
            comparison_results=detailed_results,
            metrics=metrics,
        )

    async def _run_single_comparison(
        self,
        conversation_history: List[Dict[str, str]],
        response_obj_1: Dict[str, Any],
        response_obj_2: Dict[str, Any],
        pair_idx: Tuple[int, int] = (0, 0),
        principle: Optional[str] = None,
    ) -> Tuple[float, float, float]:
        """Run a single pairwise comparison via GenRM.

        Args:
            conversation_history: The conversation context
            response_obj_1: First Response API object
            response_obj_2: Second Response API object
            pair_idx: Tuple of (i, j) for logging
            principle: Optional principle for principle-based comparison

        Returns:
            Tuple of (score_1, score_2, ranking)
        """
        cfg = self.config

        # Extract final answer from Response API objects (GenRM only takes the final answer, not reasoning)
        response_1 = extract_output_text(response_obj_1)
        response_2 = extract_output_text(response_obj_2)

        # input carries only the conversation history (standard OpenAI roles).
        # The comparison payload is passed via metadata so the request schema stays
        # generic and GenRMModelMixin._preprocess_chat_completion_create_params can
        # inject the GenRM-specific roles (response_1, response_2, principle) server-side.
        messages: List[NeMoGymEasyInputMessage] = [
            NeMoGymEasyInputMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                type="message",
            )
            for msg in conversation_history
        ]

        metadata = {"response_1": response_1, "response_2": response_2}
        if cfg.use_principle:
            metadata["principle"] = principle if principle else cfg.default_principle

        # Build the request params
        responses_create_params = cfg.genrm_responses_create_params.model_copy(deep=True)
        responses_create_params.input = messages
        responses_create_params.metadata = metadata

        try:
            # Retry logic for parse failures (not connection errors, which are handled elsewhere)
            max_attempts = max(1, int(cfg.genrm_parse_retries) + 1)

            for attempt_idx in range(max_attempts):
                # Call the GenRM model via /v1/responses endpoint (server name from config, e.g. genrm_model)
                response = await self.server_client.post(
                    server_name=cfg.genrm_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )
                if cfg.fail_on_error:
                    await raise_for_status(response)
                raw_response = await response.json()

                # Extract output_text from GenRM response (skip reasoning, only parse the final JSON scores)
                genrm_answer = extract_output_text(raw_response)

                try:
                    score_1, score_2, ranking = parse_genrm_output(
                        genrm_answer,
                        cfg.default_score,
                        cfg.default_ranking,
                        raise_on_fail=True,
                        strict=cfg.fail_on_error,
                    )
                    return score_1, score_2, ranking

                except GenRMOutputParseError:
                    if attempt_idx < max_attempts - 1:
                        await asyncio.sleep(float(cfg.genrm_parse_retry_sleep_s))
                        continue

                    if cfg.fail_on_error:
                        raise
                    # Give up: fall back to defaults
                    logger.warning(
                        f"[GenRM] Parse failed for pair {pair_idx} after {max_attempts} attempts; "
                        f"falling back to defaults."
                    )
                    return cfg.default_score, cfg.default_score, cfg.default_ranking

            return cfg.default_score, cfg.default_score, cfg.default_ranking

        except Exception as e:
            if cfg.fail_on_error:
                logger.error("[GenRM] Comparison failed closed for pair %s (%s)", pair_idx, type(e).__name__)
                raise
            logger.error(f"[GenRM] Error in comparison for pair {pair_idx}: {e}")
            return cfg.default_score, cfg.default_score, cfg.default_ranking


if __name__ == "__main__":
    GenRMCompareResourcesServer.run_webserver()
