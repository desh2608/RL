# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-reward adapter retaining the existing student/teacher prompt contract."""

import copy
from typing import Any

from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec, TokenizerType
from nemo_rl.data.legacy_text_opd import independent_text_opd_processor
from nemo_rl.environments.terminal_text_verifier import (
    RewardContractError,
    validate_reward_spec,
)


def terminal_text_reward_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Validate private reward metadata without adding it to actor/teacher text."""
    row = copy.deepcopy(datum_dict)
    metadata = row.get("extra_env_info")
    if not isinstance(metadata, dict):
        raise RewardContractError("Terminal reward requires extra_env_info")
    reserved = {
        "_terminal_reward",
        "_terminal_reward_runtime",
        "_terminal_reward_contract",
        "reward_tools",
    }
    if reserved.intersection(metadata):
        raise RewardContractError(
            "Dataset cannot supply reserved reward/runtime fields"
        )
    source_id, domain = row.get("source_id"), metadata.get("domain")
    if (
        not isinstance(source_id, str)
        or not source_id
        or not isinstance(domain, str)
        or not domain
    ):
        raise RewardContractError("Reward data needs source_id and domain")
    if "source_id" in metadata and metadata["source_id"] != source_id:
        raise RewardContractError("Conflicting reward source_id")
    tools = row["student_view"].get("tools", [])
    # Collector DataLoader workers use the vLLM interpreter. Checker packages
    # belong to the scoring actor; full validation also runs in CPU sealing.
    validate_reward_spec(metadata, tools=tools, validate_if_runtime=False)
    # The reused processor renders only explicit messages/tools, preserves the
    # separate teacher view, and attaches private metadata outside message_log.
    output = independent_text_opd_processor(
        row, task_data_spec, tokenizer, max_seq_length, idx
    )
    processed = output["extra_env_info"]
    if processed is None:
        raise RewardContractError("Reward processor lost private metadata")
    processed["source_id"] = source_id
    processed["reward_tools"] = copy.deepcopy(tools)
    processed["_terminal_reward_contract"] = {"schema_version": 1}
    return output


def register_terminal_text_reward() -> None:
    """Explicit opt-in registration; the existing pure_text_opd is unchanged."""
    # Registry modules import Ray/processor machinery only in the training path.
    from nemo_rl.data.processors import register_processor
    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
    )
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
    from nemo_rl.environments.utils import register_env

    actor = "nemo_rl.environments.terminal_text_reward.TerminalTextRewardEnvironment"
    ACTOR_ENVIRONMENT_REGISTRY[actor] = PY_EXECUTABLES.SYSTEM
    register_processor("terminal_text_reward_processor", terminal_text_reward_processor)
    register_env("terminal_text_reward", actor)
