# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility adapter for sealed independent-view text and static tool OPD data.

The original student/teacher views and verifier metadata are retained. This
one-step integration recipe uses a zero-reward environment because the new
upstream pure OPD objective consumes teacher/student token logprobs, not reward.
"""

import copy
from typing import Any

import ray
import torch

from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


from nemo_rl.data.text_tool_prompt import (
    validate_text_tool_messages,
    validate_text_tool_schemas,
    text_tool_render_messages,
)


def _validate_independent_text_messages(messages: Any, *, owner: str):
    """Keep the public processor entrypoint while accepting complete tool history."""
    return validate_text_tool_messages(messages, owner=owner)


def independent_text_opd_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Render a text student view while preserving a raw text teacher view.

    The two views may use different model templates.  Only the student view is
    rendered here; the selected MCore or vLLM teacher renders its own raw
    messages at scoring time. Tool schemas are an explicit one-turn prompt
    contract: histories and schemas are rendered for both models, but no new
    tool is executed and no new tool result is appended.
    """
    student_view = copy.deepcopy(datum_dict.get("student_view"))
    teacher_view = copy.deepcopy(datum_dict.get("teacher_view"))
    teacher_alias = datum_dict.get("teacher_alias")
    if not isinstance(student_view, dict) or not isinstance(teacher_view, dict):
        raise ValueError(
            "Independent text OPD requires student_view and teacher_view"
        )
    if not isinstance(teacher_alias, str) or not teacher_alias.strip():
        raise ValueError("Independent text OPD requires a non-empty teacher_alias")

    allowed_view_keys = {"messages", "modality", "multimodal_payload", "tools"}
    for owner, view in (("student", student_view), ("teacher", teacher_view)):
        unexpected = sorted(set(view) - allowed_view_keys)
        if unexpected:
            raise ValueError(
                f"Independent {owner} text view contains unsupported fields: "
                f"{unexpected}"
            )
        if view.get("modality") != "text":
            raise ValueError(
                f"Independent {owner} view must declare modality='text'"
            )
        if view.get("multimodal_payload") != {}:
            raise ValueError(
                f"Independent {owner} text view must have empty multimodal_payload"
            )

    responses_params = datum_dict.get("responses_create_params")
    if datum_dict.get("tools") is not None:
        raise ValueError(
            "Independent text OPD requires tool schemas inside both materialized views"
        )
    if responses_params is not None:
        raise ValueError(
            "Independent text OPD requires materialized views; raw "
            "responses_create_params is forbidden"
        )
    tool_control_fields = (
        datum_dict.get("tool_choice"),
        datum_dict.get("functions"),
        datum_dict.get("function_call"),
        responses_params.get("tool_choice")
        if isinstance(responses_params, dict)
        else None,
        responses_params.get("functions")
        if isinstance(responses_params, dict)
        else None,
        responses_params.get("function_call")
        if isinstance(responses_params, dict)
        else None,
    )
    if any(value not in (None, [], "none") for value in tool_control_fields):
        raise NotImplementedError(
            "Independent text OPD tool controls require a multi-turn tool runner"
        )

    student_messages = _validate_independent_text_messages(
        student_view.get("messages"), owner="student"
    )
    teacher_messages = _validate_independent_text_messages(
        teacher_view.get("messages"), owner="teacher"
    )
    student_tools = validate_text_tool_schemas(student_view.get("tools", []), owner="student")
    teacher_tools = validate_text_tool_schemas(teacher_view.get("tools", []), owner="teacher")
    if student_tools != teacher_tools:
        raise ValueError("Independent text OPD requires identical tool schemas")
    # Only the explicit messages/tools views are rendered. Verifier answers and
    # provenance remain in extra_env_info. Do not recursively ban schema property
    # names such as expected_answer: those can be valid historical tool arguments.

    student_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    formatted_student_prompt = student_tokenizer.apply_chat_template(
        text_tool_render_messages(student_messages, owner="student"),
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
        tools=student_tools,
    )
    if not isinstance(formatted_student_prompt, str) or not formatted_student_prompt:
        raise ValueError("Student text chat template produced an empty prompt")
    tokenized = student_tokenizer(
        formatted_student_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    tokenized_ids = tokenized["input_ids"]
    if not isinstance(tokenized_ids, torch.Tensor):
        tokenized_ids = torch.as_tensor(tokenized_ids, dtype=torch.long)
    if tokenized_ids.ndim != 2 or tokenized_ids.shape[0] != 1:
        raise TypeError(
            "Student text prompt tokenizer must return input_ids with shape [1, S]"
        )
    student_prompt_ids = tokenized_ids[0].detach().to(dtype=torch.long)
    if student_prompt_ids.numel() == 0:
        raise ValueError("Student text prompt tokenization is empty")
    if student_prompt_ids.numel() >= max_seq_length:
        raise ValueError(
            "Independent text student prompt exceeds max_input_seq_length: "
            f"length={student_prompt_ids.numel()}, max_seq_length={max_seq_length}"
        )

    processed_student_view = {
        "messages": copy.deepcopy(student_messages),
        "modality": "text",
        "multimodal_payload": {},
        "rendered_prompt": formatted_student_prompt,
        "prompt_token_ids": student_prompt_ids.clone(),
        "tools": copy.deepcopy(student_tools),
    }
    processed_teacher_view = {
        "messages": copy.deepcopy(teacher_messages),
        "modality": "text",
        "multimodal_payload": {},
        "tools": copy.deepcopy(teacher_tools),
    }
    extra_env_info = copy.deepcopy(datum_dict.get("extra_env_info"))
    if extra_env_info is not None and not isinstance(extra_env_info, dict):
        raise TypeError("Independent text OPD extra_env_info must be a mapping or null")

    output: DatumSpec = {
        "message_log": [
            {
                "role": "user",
                "content": formatted_student_prompt,
                "token_ids": student_prompt_ids,
            }
        ],
        "length": int(student_prompt_ids.numel()),
        "extra_env_info": extra_env_info,
        "loss_multiplier": 1.0,
        "idx": idx,
        "student_view": processed_student_view,
        "teacher_view": processed_teacher_view,
        "teacher_alias": teacher_alias.strip(),
    }
    task_name = datum_dict.get("task_name", task_data_spec.task_name)
    if isinstance(task_name, str) and task_name:
        output["task_name"] = task_name
    # The upstream native collator retains extra_env_info, and native rollout
    # carries it through unchanged. Keep teacher-only inputs out of the student
    # message log and out of verifier payloads.
    metadata = copy.deepcopy(extra_env_info) if extra_env_info is not None else {}
    if "_independent_text_opd" in metadata:
        raise ValueError("Input metadata reserves _independent_text_opd")
    metadata["_independent_text_opd"] = {
        "teacher_view": processed_teacher_view,
        "teacher_alias": teacher_alias.strip(),
        "source_id": datum_dict["source_id"],
    }
    output["extra_env_info"] = metadata
    return output


@ray.remote  # pragma: no cover
class PureTextOPDEnvironment(EnvironmentInterface):
    """Finish the single response without fabricating a verification score."""

    def __init__(self, config: dict[str, Any]):
        if config.get("verification_enabled") is not False:
            raise ValueError("PureTextOPDEnvironment requires verification_enabled=false")

    def step(self, message_log_batch, metadata):
        count = len(message_log_batch)
        if count != len(metadata):
            raise ValueError("Response and metadata batch sizes differ")
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": ""} for _ in range(count)],
            metadata=metadata,
            next_stop_strings=[None] * count,
            rewards=torch.zeros(count, dtype=torch.float32),
            terminateds=torch.ones(count, dtype=torch.bool),
            answers=[None] * count,
        )

    def global_post_process_and_metrics(self, batch: BatchedDataDict):
        return batch, {"verification/skipped_for_pure_opd": 1.0}


def register_legacy_text_opd() -> None:
    """Register the adapter explicitly from the project entrypoint."""
    from nemo_rl.data.processors import register_processor
    from nemo_rl.environments.utils import register_env
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

    ACTOR_ENVIRONMENT_REGISTRY["nemo_rl.data.legacy_text_opd.PureTextOPDEnvironment"] = PY_EXECUTABLES.SYSTEM

    register_processor("independent_text_opd_processor", independent_text_opd_processor)
    register_env("pure_text_opd", "nemo_rl.data.legacy_text_opd.PureTextOPDEnvironment")
