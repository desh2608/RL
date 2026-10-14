# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contracts for native asynchronous terminal-reward training."""

import hashlib
import math

import torch
from pydantic import BaseModel, StrictBool

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def validate_optimizer_progress(
    *,
    loss: float,
    grad_norm: float,
    global_valid_tokens: float,
    nonzero_reward_advantage_tokens: float | None,
) -> bool | None:
    """Validate a completed optimizer call without inventing reward signal.

    None selects the legacy OPD gate, which requires a positive gradient. For
    reward training an all-equal group batch may have zero advantage and zero
    gradient. That optimizer call still happened (moments can move parameters);
    returning False means no outcome signal, not a skipped or no-op update.
    Nonzero outcome advantages with zero gradient remain a failure.
    """
    if not all(math.isfinite(value) for value in (loss, grad_norm, global_valid_tokens)):
        raise RuntimeError("Nonfinite optimizer progress metrics")
    if global_valid_tokens <= 0 or grad_norm < 0:
        raise RuntimeError("Optimizer progress requires valid tokens and a nonnegative gradient norm")
    if nonzero_reward_advantage_tokens is None:
        if grad_norm <= 0:
            raise RuntimeError("No effective OPD update: gradient norm is zero")
        return None
    if not math.isfinite(nonzero_reward_advantage_tokens) or nonzero_reward_advantage_tokens < 0:
        raise RuntimeError("Invalid count of nonzero reward advantage tokens")
    outcome_signal_present = nonzero_reward_advantage_tokens > 0
    if outcome_signal_present and grad_norm == 0:
        raise RuntimeError("Nonzero reward advantages produced zero gradient")
    return outcome_signal_present


class RewardTrainingConfig(BaseModel, extra="forbid"):
    """Opt into verified terminal rewards on the native async GRPO path.

    Disabled by default to preserve existing OPD recipes. Enabling this contract
    requires complete prompt groups, finite terminal rewards, and retaining
    capped trajectories for the reward objective. It does not enable hybrid OPD.
    """

    enabled: StrictBool = False


def validate_reward_values(
    batch: BatchedDataDict, *, num_generations_per_prompt: int
) -> None:
    """Reject incomplete groups or invalid reward values before any baseline.

    Verifier infrastructure failures must raise in the environment. They cannot
    be represented as a zero reward, even if the sample is later loss-masked:
    its reward could otherwise change another response's group baseline.
    """
    rewards = batch["total_reward"]
    if not isinstance(rewards, torch.Tensor) or rewards.ndim != 1:
        raise ValueError("Terminal rewards must be a one-dimensional tensor")
    if rewards.numel() != batch.size or rewards.numel() == 0:
        raise ValueError("Terminal rewards must cover every generated response")
    if num_generations_per_prompt < 2 or rewards.numel() % num_generations_per_prompt:
        raise ValueError("Reward training requires complete prompt groups with K >= 2")
    if not torch.isfinite(rewards).all():
        raise ValueError("Non-finite terminal reward; refusing to form a group baseline")
    reward_valid = batch.get("reward_valid")
    if reward_valid is not None:
        if (
            not isinstance(reward_valid, torch.Tensor)
            or reward_valid.shape != rewards.shape
            or reward_valid.dtype != torch.bool
            or not reward_valid.all()
        ):
            raise ValueError("Invalid terminal reward; infrastructure failures are not task failures")


def validate_terminal_rewards(
    batch: BatchedDataDict, *, num_generations_per_prompt: int
) -> None:
    """Bind each raw reward to the environment's valid, same-source receipt."""
    validate_reward_values(batch, num_generations_per_prompt=num_generations_per_prompt)
    metadata_rows = batch.get("extra_env_info")
    if not isinstance(metadata_rows, list) or len(metadata_rows) != batch.size:
        raise ValueError("Terminal rewards require per-response verifier metadata")
    message_logs = batch.get("message_log")
    if not isinstance(message_logs, list) or len(message_logs) != batch.size:
        raise ValueError("Terminal rewards require the exact generated message logs")
    for index, metadata in enumerate(metadata_rows):
        if not isinstance(metadata, dict):
            raise ValueError("Terminal reward metadata must be a mapping")
        receipt = metadata.get("_terminal_reward")
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != 1
            or receipt.get("status") != "valid"
            or receipt.get("score") != batch["total_reward"][index].item()
        ):
            raise ValueError("Missing, invalid, or mismatched terminal reward receipt")
        for field in ("source_id", "domain"):
            if field not in metadata or receipt.get(field) != metadata[field]:
                raise ValueError(f"Terminal reward receipt has a different {field}")
        generated = [message for message in message_logs[index]
                     if message["role"] == "assistant" and "generation_logprobs" in message]
        if len(generated) != 1 or not isinstance(generated[0].get("content"), str):
            raise ValueError("Terminal reward requires exactly one generated assistant response")
        response_hash = hashlib.sha256(generated[0]["content"].encode()).hexdigest()
        if receipt.get("response_sha256") != response_hash:
            raise ValueError("Terminal reward receipt belongs to a different response")
        group_start = index - index % num_generations_per_prompt
        group_identity = metadata_rows[group_start]
        if any(metadata[field] != group_identity[field] for field in ("source_id", "domain")):
            raise ValueError("Terminal reward prompt group mixes different sources")


def reward_signal_metrics(
    *, rewards: torch.Tensor, group_size: int, advantages: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    """Measure actual retained group diversity and post-clip policy signal."""
    valid_responses = mask.ne(0).any(dim=-1)
    mixed_groups = 0
    valid_groups = 0
    for start in range(0, rewards.numel(), group_size):
        group_rewards = rewards[start : start + group_size][valid_responses[start : start + group_size]]
        valid_groups += int(group_rewards.numel() >= 2)
        mixed_groups += int(group_rewards.numel() > 1 and torch.unique(group_rewards).numel() > 1)
    valid_tokens = mask.ne(0)
    nonzero_tokens = (advantages.ne(0) & valid_tokens).sum().item()
    total_valid_tokens = valid_tokens.sum().item()
    return {
        "reward_training/mixed_group_fraction": mixed_groups / (rewards.numel() // group_size),
        "reward_training/mixed_group_count": float(mixed_groups),
        "reward_training/valid_group_count": float(valid_groups),
        "reward_training/total_group_count": float(rewards.numel() // group_size),
        "reward_training/nonzero_advantage_token_fraction": nonzero_tokens / max(total_valid_tokens, 1),
        "reward_training/valid_responses": float(valid_responses.sum().item()),
        "reward_training/nonzero_advantage_tokens": float(nonzero_tokens),
    }
