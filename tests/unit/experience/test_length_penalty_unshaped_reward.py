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
"""Length adjustments must not turn an env-uniform K-group into a trainable group.

``apply_group_length_penalties`` rewrites ``full_result["reward"]``; the collector's
zero-variance decision reads ``unshaped_total_reward`` = ``_unshaped_total_reward``,
which must therefore see the verifier's reward, not the length-adjusted one
(static-v4 job 35972778, 2026-10-03: every all-correct group survived the drop and
trained on length alone).
"""

import torch

from nemo_rl.algorithms.async_utils.trajectory_collector import (
    _group_rewards_identical,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollouts import _unshaped_total_reward
from nemo_rl.utils.length_penalty import (
    REWARD_BEFORE_LENGTH_PENALTY_KEY,
    apply_group_length_penalties,
)


def _result(reward, reasoning_words, agent="a"):
    return {
        "agent_ref": {"name": agent},
        "full_result": {
            "reward": reward,
            "response": {
                "output": [
                    {"type": "reasoning", "summary": [{"text": "x " * reasoning_words}]},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    },
                ]
            },
        },
    }


def _config(num_gens):
    return {
        "grpo": {
            "num_generations_per_prompt": num_gens,
            "length_penalty": {
                "default": {
                    "enabled": True,
                    "length_type": "chars",
                    "group_total_length_penalty_coeff": 0.2,
                }
            },
        }
    }


class TestUnshapedRewardSurvivesLengthPenalty:
    def test_all_correct_group_keeps_uniform_env_reward(self):
        results = [_result(1.0, w) for w in (100, 800, 300, 50)]
        apply_group_length_penalties(results, _config(4))
        adjusted = [r["full_result"]["reward"] for r in results]
        assert len(set(adjusted)) > 1, "length penalty should spread the rewards"
        assert all(
            r["full_result"][REWARD_BEFORE_LENGTH_PENALTY_KEY] == 1.0 for r in results
        )
        unshaped = [_unshaped_total_reward(r["full_result"]) for r in results]
        assert unshaped == [1.0, 1.0, 1.0, 1.0]
        batch = BatchedDataDict(
            {
                "total_reward": torch.tensor(adjusted),
                "unshaped_total_reward": torch.tensor(unshaped),
            }
        )
        assert _group_rewards_identical(batch) is True

    def test_mixed_group_is_still_trainable(self):
        results = [_result(r, w) for r, w in ((1.0, 100), (0.0, 800), (1.0, 300), (0.0, 50))]
        apply_group_length_penalties(results, _config(4))
        unshaped = [_unshaped_total_reward(r["full_result"]) for r in results]
        assert unshaped == [1.0, 0.0, 1.0, 0.0]
        batch = BatchedDataDict({"unshaped_total_reward": torch.tensor(unshaped)})
        assert _group_rewards_identical(batch) is False
        # Length only reorders the correct rollouts: wrong ones stay at 0.
        assert results[1]["full_result"]["reward"] == 0.0
        assert results[3]["full_result"]["reward"] == 0.0
        assert results[0]["full_result"]["reward"] > results[2]["full_result"]["reward"]

    def test_rows_without_the_key_fall_back_to_reward(self):
        assert _unshaped_total_reward({"reward": 0.75}) == 0.75

    def test_env_side_shaping_still_wins(self):
        full = {
            "reward": 0.9,
            REWARD_BEFORE_LENGTH_PENALTY_KEY: 1.0,
            "reward_shaping": {"raw_reward": 0.4, "zeroed_on_transfer": False},
        }
        assert _unshaped_total_reward(full) == 0.4

    def test_non_binary_group_is_left_alone_but_still_tagged(self):
        results = [_result(r, w) for r, w in ((0.5, 100), (0.5, 800), (1.0, 300), (0.0, 50))]
        apply_group_length_penalties(results, _config(4))
        assert [r["full_result"]["reward"] for r in results] == [0.5, 0.5, 1.0, 0.0]
        assert [_unshaped_total_reward(r["full_result"]) for r in results] == [0.5, 0.5, 1.0, 0.0]
