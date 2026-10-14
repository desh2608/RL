# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal real-reward environment with an empty observation for text OPD parity."""

import copy
import json
import logging
from collections.abc import Mapping
from typing import Any, Literal

import ray
import torch
from pydantic import BaseModel

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.terminal_text_verifier import (
    RewardContractError,
    score_terminal_response,
    validate_reward_spec,
)


class TerminalTextRewardConfig(BaseModel, extra="allow"):
    """Only final-after-think scoring is supported by this text contract."""

    response_scope: Literal["final_after_think"] = "final_after_think"
    require_think_close: Literal[True] = True
    reject_truncated: Literal[True] = True


class TerminalTextRewardCore(EnvironmentInterface):
    """Local implementation so the exact Ray actor logic is CPU-testable."""

    def __init__(self, config: dict[str, Any]):
        self.config = TerminalTextRewardConfig.model_validate(config)
        self.logger = logging.getLogger(__name__)

    def shutdown(self) -> None:
        """There are no child services to release."""

    def step(
        self, message_log_batch: list[LLMMessageLogType], metadata: list[dict[str, Any]]
    ) -> EnvironmentReturn:
        if len(message_log_batch) != len(metadata):
            raise RewardContractError("Response and metadata batch sizes differ")
        updated, rewards, answers = [], [], []
        for messages, original_metadata in zip(message_log_batch, metadata):
            if not isinstance(original_metadata, Mapping):
                raise RewardContractError("Reward metadata must be a mapping")
            if original_metadata.get("_terminal_reward_contract") != {
                "schema_version": 1
            }:
                raise RewardContractError("Missing terminal reward processor contract")
            runtime = original_metadata.get("_terminal_reward_runtime")
            if (
                not isinstance(runtime, Mapping)
                or type(runtime.get("response_truncated")) is not bool
            ):
                raise RewardContractError("Missing trusted rollout truncation metadata")
            if "_terminal_reward" in original_metadata:
                raise RewardContractError("Terminal reward may only be generated once")
            if not messages or messages[-1].get("role") != "assistant":
                raise RewardContractError(
                    "A newly generated final assistant message is required"
                )
            response = messages[-1].get("content")
            if not isinstance(response, str):
                raise RewardContractError("Assistant response must be decoded text")
            spec = validate_reward_spec(
                original_metadata, tools=original_metadata["reward_tools"]
            )
            result = score_terminal_response(
                response,
                spec=spec,
                source_id=original_metadata["source_id"],
                domain=original_metadata["domain"],
                response_truncated=runtime["response_truncated"],
            )
            row = copy.deepcopy(original_metadata)
            row["_terminal_reward"] = result.to_dict()
            updated.append(row)
            rewards.append(result.score)
            answers.append(None)
            # No answer/reference/prompt text is logged; hashes bind the result.
            self.logger.info(
                "TERMINAL_REWARD_RESULT %s",
                json.dumps(result.to_dict(), sort_keys=True),
            )
        count = len(updated)
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": ""} for _ in updated],
            metadata=updated,
            next_stop_strings=[None] * count,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.ones(count, dtype=torch.bool),
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict[str, float]]:
        # Learner-side opt-in validation verifies the returned receipts against
        # total_reward. This hook does not infer correctness from missing data.
        return batch, {"verification/terminal_real_reward": 1.0}


@ray.remote  # pragma: no cover
class TerminalTextRewardEnvironment(TerminalTextRewardCore):
    """Ray actor using the same interpreter as the sealed training container."""
