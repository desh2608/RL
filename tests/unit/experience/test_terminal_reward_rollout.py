# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.environments.interfaces import EnvironmentReturn
from nemo_rl.experience import rollouts


@pytest.fixture(scope="session", autouse=True)
def init_ray_cluster():
    yield


@pytest.fixture(scope="session", autouse=True)
def ray_gpu_monitor():
    yield None


@pytest.fixture(scope="session", autouse=True)
def session_data():
    yield {}


def _run(monkeypatch, *, enabled: bool, cap: bool | None, generator_fails: bool = False):
    seen = []

    async def generate(*args, **kwargs):
        if generator_fails:
            raise RuntimeError("generation backend unavailable")
        messages = copy.deepcopy(args[1])
        messages.append({
            "role": "assistant", "content": "answer", "token_ids": torch.tensor([8, 9]),
            "generation_logprobs": torch.tensor([-0.3, -0.4]),
        })
        metrics = {} if cap is None else {"_response_truncated": [cap]}
        return messages, torch.tensor([8, 9]), torch.tensor(2), metrics

    def score(batch, task_to_env):
        metadata = copy.deepcopy(batch["extra_env_info"][0])
        seen.append(copy.deepcopy(metadata))
        metadata["_terminal_reward"] = {"status": "valid", "score": 0.0 if cap else 1.0}
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": ""}],
            metadata=[metadata], next_stop_strings=[None],
            rewards=torch.tensor([0.0 if cap else 1.0]),
            terminateds=torch.tensor([True]), answers=[None],
        )

    monkeypatch.setattr(rollouts, "async_generate_response_for_sample_turn", generate)
    monkeypatch.setattr(rollouts, "calculate_rewards", score)
    metadata = {"source_id": "source-1", "domain": "if"}
    if enabled:
        metadata["_terminal_reward_contract"] = {"schema_version": 1}
    initial = {
        "message_log": [{"role": "user", "content": "prompt", "token_ids": torch.tensor([1, 2])}],
        "extra_env_info": metadata, "task_name": "if",
    }
    tokenizer = lambda *args, **kwargs: SimpleNamespace(input_ids=torch.empty((1, 0), dtype=torch.long))
    result, metrics = asyncio.run(rollouts.run_sample_multi_turn_rollout(
        0, initial, object(), tokenizer, {}, max_seq_len=64, max_rollout_turns=1,
    ))
    return result, metrics, seen, initial


@pytest.mark.parametrize("cap", [True, False])
def test_actual_truncation_reaches_verifier_and_terminal_receipt_survives(monkeypatch, cap):
    result, metrics, seen, initial = _run(monkeypatch, enabled=True, cap=cap)
    assert seen[0]["_terminal_reward_runtime"] == {"response_truncated": cap}
    assert "_terminal_reward_runtime" not in initial["extra_env_info"]
    assert result["extra_env_info"]["_terminal_reward"]["status"] == "valid"
    assert result["total_reward"].item() == (0.0 if cap else 1.0)
    # Only two generated tokens: true must come from the backend, not length 64.
    assert len(result["message_log"][1]["token_ids"]) == 2
    assert result["extra_env_info"]["source_id"] == "source-1"


def test_missing_backend_truncation_is_not_assumed_success(monkeypatch):
    with pytest.raises(ValueError, match="actual generation truncation"):
        _run(monkeypatch, enabled=True, cap=None)


def test_terminal_reward_generation_failure_propagates(monkeypatch):
    with pytest.raises(RuntimeError, match="backend unavailable"):
        _run(monkeypatch, enabled=True, cap=False, generator_fails=True)


def test_legacy_rollout_does_not_require_terminal_reward_metadata(monkeypatch):
    result, _, seen, _ = _run(monkeypatch, enabled=False, cap=None)
    assert "_terminal_reward_runtime" not in seen[0]
    assert "_terminal_reward" not in result["extra_env_info"]
    assert result["total_reward"].item() == 1.0
