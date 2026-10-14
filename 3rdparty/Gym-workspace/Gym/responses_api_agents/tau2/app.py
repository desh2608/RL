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

from collections import defaultdict
from os import environ
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Optional

from responses_api_agents.tau2.source import ensure_tau2_data_dir


DATA_DIR = Path(__file__).parent / "tau2_data"
environ["TAU2_DATA_DIR"] = str(DATA_DIR)

from fastapi import Body
from loguru import logger
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_server_url, is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMConverter, split_responses_input_output_items
from tau2.data_model.simulation import SimulationRun, TextRunConfig
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner.batch import run_single_task
from tau2.utils import llm_utils as _tau2_llm_utils
from tau2.data_model.message import AssistantMessage as _Tau2AssistantMessage


_TOKEN_INFO_KEYS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs")


def _assistant_token_info(message) -> dict:
    """Token bundle vLLM attached to the policy's chat completion, as kept by tau2 in raw_data.

    NeMo Gym's vllm_model returns ``prompt_token_ids`` / ``generation_token_ids`` /
    ``generation_log_probs`` on ``choices[0].message``; tau2-bench stores the whole litellm
    response in ``AssistantMessage.raw_data`` but ``to_litellm_messages`` only re-emits
    role/content/tool_calls. Without the bundle (a) the policy's history is re-rendered from
    text, so NeMo-RL's exact-prefix replay never triggers and multi-turn trajectories are not
    token-contiguous, and (b) no output item is trainable ("no generation data").
    """
    raw = getattr(message, "raw_data", None)
    if not isinstance(raw, dict):
        return {}
    choices = raw.get("choices") or []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    msg = first.get("message") if isinstance(first, dict) else None
    if not isinstance(msg, dict) or not all(isinstance(msg.get(k), list) for k in _TOKEN_INFO_KEYS):
        return {}
    info = {k: msg[k] for k in _TOKEN_INFO_KEYS}
    if msg.get("routed_experts") is not None:
        info["routed_experts"] = msg["routed_experts"]
    return info


_to_litellm_messages_plain = _tau2_llm_utils.to_litellm_messages


def to_litellm_messages(messages: list) -> list[dict]:
    """tau2's converter plus the vLLM token bundle on assistant messages that carry one."""
    dicts: list[dict] = []
    for message in messages:
        converted = _to_litellm_messages_plain([message])
        if len(converted) == 1 and isinstance(message, _Tau2AssistantMessage):
            converted[0].update(_assistant_token_info(message))
        dicts.extend(converted)
    return dicts


# tau2.utils.llm_utils.generate() resolves `to_litellm_messages` from its module globals on each call,
# so this also makes every per-turn policy request carry the token bundle of earlier assistant turns.
_tau2_llm_utils.to_litellm_messages = to_litellm_messages


# --- user-simulator robustness -------------------------------------------------------------------
# The hosted/local user simulator (gpt-oss-120b) occasionally returns a tool call whose `arguments`
# is not valid JSON (observed: "Unterminated string starting at: line 1 column 2"). tau2's
# `generate()` does `json.loads(tool_call.function.arguments)` and raises, which aborts the whole
# episode and surfaces as a NeMo-RL row failure (smoke job 35894567, resume phase: 6 occurrences on
# 5 tau2 tasks, enough to starve a 16-prompt step). A malformed *simulator* turn is not the policy's
# fault, so regenerate it a few times before giving up. Only the user simulator is wrapped; the
# agent's (policy's) own malformed calls keep tau2's default handling.
import json as _json
import sys as _sys

USER_SIM_MALFORMED_TOOL_CALL_RETRIES = 3


async def _user_generate_with_retries(*args, **kwargs):
    last_error = None
    for attempt in range(1, USER_SIM_MALFORMED_TOOL_CALL_RETRIES + 1):
        try:
            return await _tau2_llm_utils.generate(*args, **kwargs)
        except _json.JSONDecodeError as error:
            last_error = error
            print(
                f"USER_SIM_MALFORMED_TOOL_CALL event=retry attempt={attempt}/{USER_SIM_MALFORMED_TOOL_CALL_RETRIES} "
                f"error={error.msg} pos={error.pos} doc={error.doc[:200]!r}",
                file=_sys.stdout,
                flush=True,
            )
    assert last_error is not None
    raise last_error


try:
    import tau2.user.user_simulator as _tau2_user_simulator

    _tau2_user_simulator.generate = _user_generate_with_retries
except Exception as _patch_error:  # pragma: no cover - import layout changed
    print(f"WARNING: could not install the tau2 user-simulator retry wrapper: {_patch_error!r}", flush=True)


class Tau2Config(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    user_model_server: ModelServerRef
    user_llm_args: dict = Field(default_factory=dict)
    debug: bool = False
    print_step_counts: bool = False
    # Tau2 default
    max_steps: int = 200
    max_agent_steps: Optional[int] = None
    turns_remaining_interval: int = 1
    # Reward shaping on top of the tau2 evaluator's 0/1 reward. All off by default
    # (upstream behaviour). NemotronVC agentic-teacher-v1 enables it for telecom:
    # - every tool-result message the *agent* requested whose content contains
    #   ``tool_not_found_substring`` (e.g. "Error: Tool 'X' not found.", i.e. the
    #   agent called a user-side device tool) subtracts ``tool_not_found_penalty``,
    #   capped at ``tool_not_found_penalty_cap`` per episode;
    # - in ``zero_reward_on_transfer_domains`` any assistant ``transfer_tool_name``
    #   call zeroes the evaluator reward before the penalty is applied.
    tool_not_found_penalty: float = Field(default=0.0, ge=0.0)
    tool_not_found_penalty_cap: float = Field(default=0.0, ge=0.0)
    tool_not_found_substring: str = "not found"
    zero_reward_on_transfer_domains: List[str] = Field(default_factory=list)
    transfer_tool_name: str = "transfer_to_human_agents"


def _message_attr(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def shape_tau2_reward(
    reward: float,
    messages: Optional[List[Any]],
    *,
    domain: Optional[str],
    tool_not_found_penalty: float,
    tool_not_found_penalty_cap: float,
    zero_reward_on_transfer_domains: List[str],
    tool_not_found_substring: str = "not found",
    transfer_tool_name: str = "transfer_to_human_agents",
) -> Dict[str, Any]:
    """Apply the configurable tau2 reward shaping; returns the shaped reward and its parts.

    ``messages`` are the full simulation messages (tau2 ``Message`` objects or their
    dict dumps). Tool results requested by the user simulator (``requestor == "user"``)
    are never counted: the penalty targets the agent calling tools it does not have.
    Zeroing on transfer happens before the penalty, so a transfer plus hallucinated
    tool calls ranks below a bare transfer inside the same GRPO group.
    """
    raw_reward = float(reward)
    tool_not_found_count = 0
    transferred = False
    needle = tool_not_found_substring.lower() if tool_not_found_substring else ""
    for message in messages or []:
        role = _message_attr(message, "role")
        if role == "tool":
            if _message_attr(message, "requestor", "assistant") != "assistant":
                continue
            content = _message_attr(message, "content") or ""
            if needle and needle in str(content).lower():
                tool_not_found_count += 1
        elif role == "assistant":
            for tool_call in _message_attr(message, "tool_calls") or []:
                if _message_attr(tool_call, "name") == transfer_tool_name:
                    transferred = True
    zeroed_on_transfer = bool(
        transferred and domain is not None and domain in zero_reward_on_transfer_domains
    )
    base_reward = 0.0 if zeroed_on_transfer else raw_reward
    penalty = 0.0
    if tool_not_found_penalty > 0.0 and tool_not_found_count:
        penalty = min(tool_not_found_count * tool_not_found_penalty, tool_not_found_penalty_cap)
    return {
        "reward": base_reward - penalty,
        "raw_reward": raw_reward,
        "tool_not_found_count": tool_not_found_count,
        "tool_not_found_penalty": penalty,
        "transfer_to_human_agents": transferred,
        "zeroed_on_transfer": zeroed_on_transfer,
    }


class Tau2RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    config: TextRunConfig
    task: Task
    seed: int
    evaluation_type: EvaluationType
    save_dir: Literal[None]
    user_voice_settings: Literal[None]
    user_persona_config: Literal[None]
    verbose_logs: Literal[False]
    audio_debug: Literal[False]
    audio_taps: Literal[False]
    auto_review: Literal[False]
    review_mode: Literal["full"]
    hallucination_feedback: Literal[None]


class Tau2VerifyResponse(Tau2RunRequest, BaseVerifyResponse):
    result: SimulationRun
    duration: float
    num_steps: int
    agent_steps: Optional[int]
    max_agent_steps: Optional[int]
    num_agent_calls: int
    min_prompt_tokens: Optional[float]
    min_completion_tokens: Optional[float]
    mean_prompt_tokens: Optional[float]
    mean_completion_tokens: Optional[float]
    max_prompt_tokens: Optional[float]
    max_completion_tokens: Optional[float]
    # Parts of the shaped reward (see shape_tau2_reward); ``reward`` above is the shaped value.
    reward_shaping: Dict[str, Any] = Field(default_factory=dict)


class Tau2Agent(SimpleResponsesAPIAgent):
    config: Tau2Config

    __key_metrics: Optional[List[str]] = None

    def setup_webserver(self):
        ensure_tau2_data_dir(DATA_DIR)

        if not self.config.debug:
            print("Removing loguru logging since `debug=False`")
            logger.remove()

        if self.config.print_step_counts:
            environ["NEMO_GYM_TAU2_STEP_COUNT_PRINT"] = "true"

        return super().setup_webserver()

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError

    async def run(self, body: Tau2RunRequest) -> Tau2VerifyResponse:
        body_dict = {name: getattr(body, name) for name in Tau2RunRequest.model_fields}
        responses_create_params = body_dict.pop("responses_create_params").model_dump(exclude_unset=True)

        config: TextRunConfig = body_dict["config"]

        # Need `openai/` provider prefix for LiteLLM
        config.llm_user = "openai/dummy user model"
        config.llm_args_user |= {
            "api_base": f"{self.base_url_for_run(get_server_url(self.config.user_model_server.name), body)}/v1",
            "api_key": "dummy api key",  # pragma: allowlist secret
        } | self.config.user_llm_args

        extra_agent_args = {k: v for k, v in responses_create_params.items() if k in ("temperature", "top_p")}
        if responses_create_params.get("max_output_tokens"):
            # Convert to chat completions
            extra_agent_args["max_tokens"] = responses_create_params["max_output_tokens"]

        # Need `openai/` provider prefix for LiteLLM
        config.llm_agent = "openai/dummy agent model"
        config.llm_args_agent = {
            "api_base": f"{self.base_url_for_run(get_server_url(self.config.model_server.name), body)}/v1",
            "api_key": "dummy api key",  # pragma: allowlist secret
        } | extra_agent_args

        config.max_steps = self.config.max_steps
        config.max_agent_steps = self.config.max_agent_steps
        config.turns_remaining_interval = self.config.turns_remaining_interval

        result = await run_single_task(**body_dict)

        result_messages = result.messages or []
        reward_shaping = shape_tau2_reward(
            result.reward_info.reward,
            result_messages,
            domain=getattr(config, "domain", None),
            tool_not_found_penalty=self.config.tool_not_found_penalty,
            tool_not_found_penalty_cap=self.config.tool_not_found_penalty_cap,
            zero_reward_on_transfer_domains=self.config.zero_reward_on_transfer_domains,
            tool_not_found_substring=self.config.tool_not_found_substring,
            transfer_tool_name=self.config.transfer_tool_name,
        )
        if result.agent_messages is not None:
            messages_to_convert = result.agent_messages
        else:
            messages_to_convert = []
            for message in result_messages:
                if message.role == "user" and message.tool_calls:
                    continue
                elif message.role == "tool" and message.requestor == "user":
                    continue
                messages_to_convert.append(message)

        message_dicts = to_litellm_messages(messages_to_convert)

        converter = VLLMConverter(return_token_id_information=True)
        all_items = converter.chat_completions_messages_to_responses_items(message_dicts)
        input_items_1, output_items = split_responses_input_output_items(all_items)
        # Tau starts trajectories with an assistant message
        input_items_1 += output_items[:1]
        input_items_2, output_items = split_responses_input_output_items(output_items[1:])

        prompt_usages = []
        completion_usages = []
        num_agent_calls = 0
        for message in result_messages:
            if not message.role == "assistant":
                continue

            num_agent_calls += 1
            if message.usage:
                prompt_usages.append(message.usage["prompt_tokens"])
                completion_usages.append(message.usage["completion_tokens"])

        min_prompt_tokens = None
        min_completion_tokens = None
        mean_prompt_tokens = None
        mean_completion_tokens = None
        max_prompt_tokens = None
        max_completion_tokens = None
        if prompt_usages:
            min_prompt_tokens = min(prompt_usages)
            min_completion_tokens = min(completion_usages)
            mean_prompt_tokens = sum(prompt_usages) / len(prompt_usages)
            mean_completion_tokens = sum(completion_usages) / len(completion_usages)
            max_prompt_tokens = max(prompt_usages)
            max_completion_tokens = max(completion_usages)

        return Tau2VerifyResponse(
            **body_dict,
            responses_create_params=dict(
                input=body.responses_create_params.input + input_items_1 + input_items_2,
                model=body.responses_create_params.model or "",
                parallel_tool_calls=body.responses_create_params.parallel_tool_calls,
                tool_choice=body.responses_create_params.tool_choice,
                tools=body.responses_create_params.tools,
            ),
            response=dict(
                id=f"tau2-{body.config.domain}-{body.task.id}",
                created_at=int(time()),
                object="response",
                output=output_items,
                model=body.responses_create_params.model or "",
                parallel_tool_calls=body.responses_create_params.parallel_tool_calls,
                tool_choice=body.responses_create_params.tool_choice,
                tools=body.responses_create_params.tools,
            ),
            reward=reward_shaping["reward"],
            reward_shaping=reward_shaping,
            result=result,
            duration=result.duration,
            num_steps=len(result_messages),
            agent_steps=result.agent_steps,
            max_agent_steps=result.max_agent_steps,
            num_agent_calls=num_agent_calls,
            min_prompt_tokens=min_prompt_tokens,
            min_completion_tokens=min_completion_tokens,
            mean_prompt_tokens=mean_prompt_tokens,
            mean_completion_tokens=mean_completion_tokens,
            max_prompt_tokens=max_prompt_tokens,
            max_completion_tokens=max_completion_tokens,
        )

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Override to select headline metrics for this benchmark.

        Default: all mean/* entries from agent_metrics.
        """
        res = super().get_key_metrics(agent_metrics)
        del (
            res["mean/seed"],
            res["mean/verbose_logs"],
            res["mean/audio_debug"],
            res["mean/audio_taps"],
            res["mean/auto_review"],
        )
        return res | {k: agent_metrics[k] for k in self.__key_metrics}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        domain_to_rewards = defaultdict(list)
        domain_to_unique_samples = defaultdict(int)
        termination_reason_domain_count = defaultdict(int)
        termination_reason_count = defaultdict(int)
        finish_reasons_count = defaultdict(int)
        hallucination_count = defaultdict(int)
        transfer_to_human_agents = 0
        total_count = 0
        missing_tool_call = 0
        incomplete_reasoning = 0
        telecom_subtask_rewards = defaultdict(list)
        domain_to_raw_rewards = defaultdict(list)
        shaping_penalties = []
        zeroed_on_transfer_count = 0
        for task_group in tasks:
            for task in task_group:
                domain = task["config"]["domain"]
                domain_to_rewards[domain].append(task["reward"])
                shaping = task.get("reward_shaping") or {}
                if shaping:
                    domain_to_raw_rewards[domain].append(shaping["raw_reward"])
                    shaping_penalties.append(shaping["tool_not_found_penalty"])
                    zeroed_on_transfer_count += bool(shaping["zeroed_on_transfer"])

                if domain == "telecom":
                    subtask = task["task"]["id"].split("]")[0].removeprefix("[")
                    telecom_subtask_rewards[f"telecom/{subtask}/reward"].append(task["reward"])

                termination_reason = task["result"]["termination_reason"]
                termination_reason_count[f"trajectory_termination_reason/{termination_reason}/count"] += 1
                termination_reason_domain_count[
                    f"{domain}/trajectory_termination_reason/{termination_reason}/count"
                ] += 1

                this_task_transfer_to_human_agents = False
                has_tool_call = False
                for message in task["result"]["messages"]:
                    if message["role"] == "tool":
                        # e.g. `Error: Tool 'run_speed_test' not found.`
                        if "Error: Tool" and "not found" in message["content"]:
                            tool_name = message["content"].removeprefix("Error: Tool '").removesuffix(" not found.")
                            hallucination_count[f"tool_call_hallucination/{tool_name}/count"] += 1

                    if message["role"] != "assistant":
                        continue

                    if message["raw_data"]:
                        finish_reason = message["raw_data"]["choices"][0]["finish_reason"]
                        finish_reasons_count[f"message_finish_reason/{finish_reason}/count"] += 1

                        raw_message = message["raw_data"]["choices"][0]["message"]
                        has_reasoning = raw_message.get("reasoning_content") is not None
                        is_empty = not (raw_message.get("content") or raw_message.get("tool_calls"))
                        incomplete_reasoning += is_empty and has_reasoning

                    if not message.get("tool_calls"):
                        continue

                    has_tool_call = True

                    if message["tool_calls"][0]["name"] == "transfer_to_human_agents":
                        this_task_transfer_to_human_agents = True

                missing_tool_call += not has_tool_call
                transfer_to_human_agents += this_task_transfer_to_human_agents
                total_count += 1

            domain_to_unique_samples[f"{domain}/num_samples_unique"] += 1

        total_num_assistant_messages = sum(finish_reasons_count.values())
        finish_reasons_pct = {
            f"{k.removesuffix('/count')}/pct": v / total_num_assistant_messages
            for k, v in finish_reasons_count.items()
        }

        telecom_subtask_avg_reward = {k: sum(v) / len(v) for k, v in telecom_subtask_rewards.items()}

        domain_to_average_reward: Dict[str, float] = dict()
        domain_to_counts: Dict[str, int] = dict()
        for domain, rewards in domain_to_rewards.items():
            domain_to_counts[f"{domain}/num_samples_total"] = len(rewards)
            domain_to_average_reward[f"{domain}/reward"] = (
                sum(rewards) / domain_to_counts[f"{domain}/num_samples_total"]
            )

        macro_average = sum(domain_to_average_reward.values()) / len(domain_to_average_reward)

        termination_reason_pct = {
            f"{k.removesuffix('/count')}/pct": v / total_count for k, v in termination_reason_count.items()
        }
        termination_reason_domain_pct = dict()
        for k, v in termination_reason_domain_count.items():
            for domain, domain_count in domain_to_counts.items():
                if k.startswith(domain):
                    termination_reason_domain_pct[f"{k.removesuffix('/count')}/pct"] = v / domain_count
                    break

        shaping_metrics: Dict[str, Any] = dict()
        if shaping_penalties:
            for domain, raw_rewards in domain_to_raw_rewards.items():
                shaping_metrics[f"{domain}/raw_reward"] = sum(raw_rewards) / len(raw_rewards)
            shaping_metrics["reward_shaping/tool_not_found_penalty/mean"] = sum(shaping_penalties) / len(
                shaping_penalties
            )
            shaping_metrics["reward_shaping/zeroed_on_transfer/count"] = zeroed_on_transfer_count
            shaping_metrics["reward_shaping/zeroed_on_transfer/pct"] = zeroed_on_transfer_count / total_count

        res = {
            "macro_average": macro_average,
            **shaping_metrics,
            **domain_to_unique_samples,
            **domain_to_counts,
            **domain_to_average_reward,
            **telecom_subtask_avg_reward,
            **termination_reason_domain_count,
            **termination_reason_count,
            **termination_reason_pct,
            **termination_reason_domain_pct,
            **finish_reasons_count,
            **finish_reasons_pct,
            "trajectory_transfer_to_human_agents/count": transfer_to_human_agents,
            "trajectory_transfer_to_human_agents/pct": transfer_to_human_agents / total_count,
            **hallucination_count,
            "tool_call_hallucination/count/total": sum(hallucination_count.values()),
            "trajectory_missing_tool_call/count": missing_tool_call,
            "trajectory_missing_tool_call/pct": missing_tool_call / total_count,
            "messages_with_incomplete_reasoning/count": incomplete_reasoning,
            "messages_with_incomplete_reasoning/pct": incomplete_reasoning / total_num_assistant_messages,
        }
        self.__key_metrics = list(res.keys())
        return res


if __name__ == "__main__":
    Tau2Agent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = Tau2Agent.run_webserver()  # noqa: F401
