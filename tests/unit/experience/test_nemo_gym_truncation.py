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
"""``truncated`` on the NeMo-Gym paths means "a response hit max_new_tokens"."""

import pytest
import torch

import nemo_rl.experience.rollouts as rollouts_mod
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollout_manager import AsyncNemoGymRolloutImpl
from nemo_rl.experience.rollouts import (
    nemo_gym_response_truncated,
    nemo_gym_row_output_cap,
)


def _assistant(n_tokens, generated=True):
    message = {"role": "assistant", "content": "a", "token_ids": torch.tensor([2] * n_tokens)}
    if generated:
        message["generation_logprobs"] = torch.tensor([-0.1] * n_tokens)
    return message


def _user(n_tokens):
    return {"role": "user", "content": "u", "token_ids": torch.tensor([1] * n_tokens)}


def test_row_output_cap_prefers_row_then_configured():
    assert nemo_gym_row_output_cap({"responses_create_params": {"max_output_tokens": 2048}}, 8192) == 2048
    assert nemo_gym_row_output_cap({"responses_create_params": {"max_output_tokens": None}}, 8192) == 8192
    assert nemo_gym_row_output_cap({"responses_create_params": {}}, 8192) == 8192
    assert nemo_gym_row_output_cap({}, None) is None
    assert nemo_gym_row_output_cap(None, 4096) == 4096


def test_per_turn_output_cap_detects_truncation_without_full_context():
    # 3 assistant turns of 4 tokens against a 4-token budget: truncated even though the
    # total (16) is far below the 128-token context window.
    log = [_user(2), _assistant(4), _user(1), _assistant(3), _user(1), _assistant(4)]
    output_cap_hit, context_cap_hit = nemo_gym_response_truncated(log, output_cap=4, context_cap=128)
    assert output_cap_hit is True
    assert context_cap_hit is False


def test_sum_of_turns_never_counts_against_the_per_turn_cap():
    log = [_user(1), _assistant(3), _user(1), _assistant(3)]
    output_cap_hit, _ = nemo_gym_response_truncated(log, output_cap=4, context_cap=128)
    assert output_cap_hit is False


def test_environment_seeded_assistant_turns_are_ignored():
    # Tau2 injects the opening "Hi! How can I help you today?" without generation_logprobs.
    log = [_assistant(4, generated=False), _user(1), _assistant(2)]
    output_cap_hit, _ = nemo_gym_response_truncated(log, output_cap=4, context_cap=128)
    assert output_cap_hit is False


def test_context_window_signal_is_kept_as_fallback():
    log = [_user(2), _assistant(2)]
    assert nemo_gym_response_truncated(log, output_cap=None, context_cap=4) == (False, True)
    assert nemo_gym_response_truncated(log, output_cap=8, context_cap=4) == (False, True)
    assert nemo_gym_response_truncated(log, output_cap=8, context_cap=128) == (False, False)


def _postprocess(rows, results, cfg):
    return rollouts_mod._postprocess_single_nemo_gym_group(
        nemo_gym_rows=rows,
        results=results,
        timer=rollouts_mod.Timer(),
        timer_prefix="timing/rollout",
        policy_generation=type("_PolicyGeneration", (), {"cfg": cfg})(),
        input_batch=BatchedDataDict({"loss_multiplier": torch.ones(len(results))}),
        tokenizer=type("_Tokenizer", (), {"pad_token_id": 0})(),
        log_full_result_tables=False,
    )


def _result(message_log, reward=1.0):
    return {"input_message_log": message_log[:1], "message_log": message_log, "full_result": {"reward": reward}}


def test_postprocess_group_truncated_means_max_new_tokens_hit():
    rows = [
        {"agent_ref": {"name": "tool"}, "responses_create_params": {"max_output_tokens": 4}},
        {"agent_ref": {"name": "tool"}, "responses_create_params": {"max_output_tokens": 4}},
        {"agent_ref": {"name": "search"}, "responses_create_params": {"max_output_tokens": 2}},
        {"agent_ref": {"name": "search"}, "responses_create_params": {"max_output_tokens": 2}},
    ]
    results = [
        _result([_user(1), _assistant(4)]),  # hit the 4-token budget
        _result([_user(1), _assistant(3), _user(1), _assistant(3)]),  # 8 tokens total, no turn hit
        _result([_user(1), _assistant(2)]),  # hit the row-level 2-token budget
        _result([_user(1), _assistant(1)]),
    ]
    cfg = {"max_new_tokens": 4, "vllm_cfg": {"max_model_len": 128}}
    rollout_result = _postprocess(rows, results, cfg)
    assert rollout_result.final_batch["truncated"].tolist() == [True, False, True, False]
    metrics = rollout_result.rollout_metrics
    assert metrics["truncation_rate"] == pytest.approx(0.5)
    assert metrics["output_cap_hit_rate"] == pytest.approx(0.5)
    assert metrics["context_cap_hit_rate"] == pytest.approx(0.0)
    assert metrics["natural_termination_rate"] == pytest.approx(0.5)
    assert metrics["tool/truncation_rate"] == pytest.approx(0.5)
    assert metrics["search/truncation_rate"] == pytest.approx(0.5)


def test_postprocess_group_context_window_still_flags():
    rows = [{"agent_ref": {"name": "tool"}}]
    results = [_result([_user(1), _assistant(2)])]
    cfg = {"max_new_tokens": 8, "vllm_cfg": {"max_model_len": 3}}
    rollout_result = _postprocess(rows, results, cfg)
    assert rollout_result.final_batch["truncated"].tolist() == [True]
    assert rollout_result.rollout_metrics["context_cap_hit_rate"] == pytest.approx(1.0)
    assert rollout_result.rollout_metrics["output_cap_hit_rate"] == pytest.approx(0.0)


def _impl(generation_config):
    return AsyncNemoGymRolloutImpl(
        tokenizer=None,
        task_to_env={},
        num_generations_per_prompt=1,
        max_seq_len=100,
        max_rollout_turns=1,
        generation_config=generation_config,
        mask_env_flagged_samples=True,
    )


def test_rollout_manager_completion_truncated_uses_output_budget():
    impl = _impl({"stop_strings": None, "stop_token_ids": None, "top_k": None, "max_new_tokens": 4})
    hit = {
        "message_log": [{"role": "user", "token_ids": [1]}, {"role": "assistant", "token_ids": [1, 2, 3, 4], "generation_logprobs": [0.0] * 4}],
        "full_result": {"reward": 1.0, "responses_create_params": {"max_output_tokens": 4}},
    }
    short = {
        "message_log": [{"role": "user", "token_ids": [1]}, {"role": "assistant", "token_ids": [1, 2], "generation_logprobs": [0.0, 0.0]}],
        "full_result": {"reward": 1.0, "responses_create_params": {"max_output_tokens": 4}},
    }
    row_cap = {
        "message_log": [{"role": "user", "token_ids": [1]}, {"role": "assistant", "token_ids": [1, 2], "generation_logprobs": [0.0, 0.0]}],
        "full_result": {"reward": 1.0, "responses_create_params": {"max_output_tokens": 2}},
    }
    assert impl._result_to_completion(hit).truncated is True
    assert impl._result_to_completion(short).truncated is False
    assert impl._result_to_completion(row_cap).truncated is True


def test_rollout_manager_completion_legacy_rule_without_budget():
    impl = _impl({"stop_strings": None, "stop_token_ids": None, "top_k": None})
    full = {
        "message_log": [{"role": "user", "token_ids": [1] * 96}, {"role": "assistant", "token_ids": [1] * 4, "generation_logprobs": [0.0] * 4}],
        "full_result": {"reward": 1.0},
    }
    assert impl._result_to_completion(full).truncated is True
    full["message_log"][0]["token_ids"] = [1] * 10
    assert impl._result_to_completion(full).truncated is False
