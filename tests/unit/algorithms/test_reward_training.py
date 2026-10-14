# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import math

import pytest
import torch
from pydantic import ValidationError

from nemo_rl.algorithms.advantage_estimator import AdvEstimatorConfig, GRPOAdvantageEstimator, OPDAdvantageEstimator
from nemo_rl.algorithms.grpo import (
    AsyncGRPOConfig,
    GRPOConfig,
    MasterConfig,
    _prepare_async_reward_training_batch,
    _validate_reward_training_config,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.opd import OnPolicyDistillationConfig
from nemo_rl.algorithms.reward_functions import RewardShapingConfig
from nemo_rl.algorithms.reward_training import RewardTrainingConfig, reward_signal_metrics, validate_optimizer_progress, validate_terminal_rewards
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.fixture(scope="session", autouse=True)
def init_ray_cluster():
    yield


@pytest.fixture(scope="session", autouse=True)
def ray_gpu_monitor():
    yield None


@pytest.fixture(scope="session", autouse=True)
def session_data():
    yield {}


def _config() -> MasterConfig:
    return MasterConfig.model_construct(
        grpo=GRPOConfig(
            async_grpo=AsyncGRPOConfig(enabled=True),
            reward_training=RewardTrainingConfig(enabled=True),
            num_generations_per_prompt=4,
            overlong_filtering=False,
            adv_estimator=AdvEstimatorConfig(name="grpo", normalize_rewards=False),
        ),
        env={"should_use_nemo_gym": False},
        policy={},
        loss_fn=ClippedPGLossConfig(disable_ppo_ratio=True, reference_policy_kl_penalty=0),
    )


def _batch() -> BatchedDataDict:
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    metadata = []
    for index, reward in enumerate(rewards.tolist()):
        metadata.append({
            "source_id": "train-source-1",
            "domain": "if",
            "_terminal_reward": {
                "schema_version": 1, "status": "valid", "score": reward,
                "source_id": "train-source-1", "domain": "if",
                "response_sha256": hashlib.sha256(f"response {index}".encode()).hexdigest(),
            },
        })
    return BatchedDataDict({
        "total_reward": rewards,
        "extra_env_info": metadata,
        "truncated": torch.tensor([False, True, False, False]),
        "loss_multiplier": torch.ones(4),
        "message_log": [[{"role": "assistant", "content": f"response {index}",
                          "token_ids": torch.tensor([8, 9]),
                          "generation_logprobs": torch.tensor([-0.3, -0.4])}]
                        for index in range(4)],
    })


def _loss_and_gradient(batch: BatchedDataDict, config: MasterConfig):
    mask = torch.tensor([[0.0, 1.0, 1.0]]).expand(4, 3)
    estimator = GRPOAdvantageEstimator(config.grpo.adv_estimator, config.loss_fn)
    advantages = estimator.compute_advantage(
        torch.zeros(4, 1, dtype=torch.long), batch["total_reward"], mask,
        reward_valid_mask=batch["loss_multiplier"],
    )
    current = torch.tensor([[-0.3, -0.4], [-0.6, -0.7], [-0.8, -0.9], [-1.0, -1.1]], requires_grad=True)
    padded = torch.cat((torch.zeros(4, 1), current.detach()), dim=1)
    data = BatchedDataDict({
        "advantages": advantages, "token_mask": mask,
        "sample_mask": batch["loss_multiplier"],
        "prev_logprobs": padded, "generation_logprobs": padded,
    })
    loss, _ = ClippedPGLossFn(config.loss_fn)(current, data, torch.tensor(4.0), torch.tensor(8.0))
    gradient, = torch.autograd.grad(loss, current)
    return loss.detach(), gradient


def test_async_shaping_changes_actual_pg_loss_and_keeps_cap_gradient():
    config = _config()
    _validate_reward_training_config(config)
    original = _prepare_async_reward_training_batch(_batch(), config)
    old_loss, old_gradient = _loss_and_gradient(original, config)
    config.grpo.reward_shaping = RewardShapingConfig(enabled=True, stop_properly_penalty_coef=0)
    shaped = _prepare_async_reward_training_batch(_batch(), config)
    new_loss, new_gradient = _loss_and_gradient(shaped, config)
    assert shaped["total_reward"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert shaped["loss_multiplier"].tolist() == [1.0] * 4
    assert not torch.equal(new_loss, old_loss)
    assert not torch.equal(new_gradient, old_gradient)
    assert (new_gradient[1] > 0).all()  # Capped failure is pushed down, not masked.
    assert (old_gradient[1] < 0).all()


def test_smoke_ppo_response_mean_keeps_capped_failure_gradient():
    """Exercise the smoke's PPO + response mean + ICEPOP path, not REINFORCE."""
    config = _config()
    config.loss_fn = ClippedPGLossConfig(
        disable_ppo_ratio=False,
        token_level_loss=False,
        ratio_clip_min=0.2,
        ratio_clip_max=0.28,
        ratio_clip_c=None,
        reference_policy_kl_penalty=0,
        use_importance_sampling_correction=True,
        truncated_importance_sampling_type="icepop",
        truncated_importance_sampling_ratio_min=0.2,
        truncated_importance_sampling_ratio=5.0,
        use_on_policy_kl_approximation=True,
    )
    _validate_reward_training_config(config)
    batch = _batch()
    # The terminal verifier judges the capped response as a valid task failure.
    batch["total_reward"] = torch.tensor([1.0, 0.0, 0.0, 1.0])
    lengths = [1, 3, 2, 4]
    for index, length in enumerate(lengths):
        batch["extra_env_info"][index]["_terminal_reward"]["score"] = batch["total_reward"][index].item()
        batch["message_log"][index][0]["token_ids"] = torch.arange(length)
        batch["message_log"][index][0]["generation_logprobs"] = torch.full((length,), -0.6)
    batch = _prepare_async_reward_training_batch(batch, config)
    assert batch["truncated"][1] and batch["loss_multiplier"][1] == 1

    token_mask = torch.tensor([
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0],
    ])
    advantage = GRPOAdvantageEstimator(config.grpo.adv_estimator, config.loss_fn).compute_advantage(
        torch.zeros(4, 1), batch["total_reward"], token_mask,
        reward_valid_mask=batch["loss_multiplier"],
    )
    torch.testing.assert_close(advantage[:, 0], torch.tensor([2 / 3, -2 / 3, -2 / 3, 2 / 3]))
    current = torch.full((4, 4), -0.5, requires_grad=True)
    # PPO ratio 1.1 is inside [0.8, 1.28], while actor/generation IS is 1.
    # A non-unit PPO ratio distinguishes this derivative from REINFORCE.
    previous = torch.cat((torch.zeros(4, 1), current.detach() - math.log(1.1)), dim=1)
    data = BatchedDataDict({
        "advantages": advantage, "token_mask": token_mask,
        "sample_mask": batch["loss_multiplier"],
        "prev_logprobs": previous, "generation_logprobs": previous,
    })
    loss, _ = ClippedPGLossFn(config.loss_fn)(current, data, torch.tensor(4.0), torch.tensor(10.0))
    gradient, = torch.autograd.grad(loss, current)
    assert torch.isfinite(loss)
    torch.testing.assert_close(gradient[1, :3], torch.full((3,), 1.1 / 18))
    torch.testing.assert_close(gradient[0, 0], torch.tensor(-1.1 / 6))
    assert not gradient[token_mask[:, 1:] == 0].any()
    # Equal outcome advantages contribute equally per response despite lengths 1/3/2/4.
    torch.testing.assert_close(gradient.abs().sum(dim=1), torch.full((4,), 1.1 / 6))


def test_invalid_masked_reward_cannot_change_other_group_baselines():
    estimator = GRPOAdvantageEstimator(AdvEstimatorConfig(normalize_rewards=False), {})
    valid = torch.tensor([1.0, 1.0, 1.0, 0.0])
    outputs = [estimator.compute_advantage(
        torch.zeros(4, 1), torch.tensor([1.0, 0.0, 0.0, excluded]), torch.ones(4, 3),
        reward_valid_mask=valid,
    ) for excluded in (0.0, 10000.0)]
    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(outputs[0][:, 0], torch.tensor([1.0, -0.5, -0.5, 0.0]))


def test_single_valid_response_has_no_group_signal():
    estimator = GRPOAdvantageEstimator(AdvEstimatorConfig(), {})
    advantage = estimator.compute_advantage(
        torch.zeros(4, 1), torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.ones(4, 3),
        reward_valid_mask=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    assert torch.isfinite(advantage).all() and not advantage.any()


@pytest.mark.parametrize("failure", ["nan", "invalid", "receipt", "score", "source", "domain", "incomplete", "response", "group"])
def test_terminal_reward_failures_are_not_zero_filled(failure):
    batch = _batch()
    if failure == "nan":
        batch["total_reward"][0] = float("nan")
    elif failure == "invalid":
        batch["reward_valid"] = torch.tensor([True, False, True, True])
    elif failure == "receipt":
        batch["extra_env_info"][0].pop("_terminal_reward")
    elif failure == "score":
        batch["extra_env_info"][0]["_terminal_reward"]["score"] = 0.0
    elif failure in ("source", "domain"):
        key = "source_id" if failure == "source" else "domain"
        batch["extra_env_info"][0]["_terminal_reward"][key] = "different"
    elif failure == "response":
        batch["extra_env_info"][0]["_terminal_reward"]["response_sha256"] = batch["extra_env_info"][1]["_terminal_reward"]["response_sha256"]
    elif failure == "group":
        batch["extra_env_info"][1]["source_id"] = "different"
        batch["extra_env_info"][1]["_terminal_reward"]["source_id"] = "different"
    else:
        batch = batch.select_indices([0, 1, 2])
    with pytest.raises(ValueError):
        validate_terminal_rewards(batch, num_generations_per_prompt=4)


@pytest.mark.parametrize("setting", ["cap_mask", "one_sample", "dynamic", "reward_kl", "nll", "teacher", "orm"])
def test_unsupported_reward_config_fails_before_setup(setting):
    config = _config()
    if setting == "cap_mask":
        config.grpo.overlong_filtering = True
    elif setting == "one_sample":
        config.grpo.num_generations_per_prompt = 1
    elif setting == "dynamic":
        config.grpo.use_dynamic_sampling = True
    elif setting == "reward_kl":
        config.loss_fn.use_kl_in_reward = True
    elif setting == "nll":
        config.loss_fn.positive_example_nll_weight = 1
    elif setting == "teacher":
        config.on_policy_distillation = OnPolicyDistillationConfig(enabled=True)
    else:
        config.on_policy_distillation = OnPolicyDistillationConfig(enabled=False, use_orm_advantage=True)
    with pytest.raises(ValueError):
        _validate_reward_training_config(config)


def test_unopted_async_shaping_is_rejected_and_default_opd_is_unchanged():
    config = _config()
    config.grpo.reward_training.enabled = False
    config.grpo.adv_estimator.name = "opd"
    config.on_policy_distillation = OnPolicyDistillationConfig(enabled=True)
    _validate_reward_training_config(config)
    batch = _batch()
    assert _prepare_async_reward_training_batch(batch, config) is batch
    estimator = OPDAdvantageEstimator(config.grpo.adv_estimator, config.loss_fn)
    kwargs = {"teacher_logprobs": torch.full((4, 3), -0.4), "prev_logprobs": torch.full((4, 3), -0.8)}
    expected = estimator.compute_advantage(None, batch["total_reward"], torch.ones(4, 3), **kwargs)
    actual = estimator.compute_advantage(None, 1 - batch["total_reward"], torch.ones(4, 3), **kwargs)
    torch.testing.assert_close(expected, actual)
    config.grpo.reward_shaping.enabled = True
    with pytest.raises(ValueError, match="requires grpo.reward_training"):
        _validate_reward_training_config(config)


def test_reward_training_schema_rejects_unknown_switch():
    with pytest.raises(ValidationError):
        RewardTrainingConfig(enabled=True, hybrid=True)


def test_reward_signal_metrics_exclude_masked_samples():
    result = reward_signal_metrics(
        rewards=torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]),
        group_size=4,
        advantages=torch.tensor([[1.0], [-1.0], [-1.0], [-1.0], [0.0], [0.0], [0.0], [99.0]]),
        mask=torch.tensor([[1.0]] * 7 + [[0.0]]),
    )
    assert result["reward_training/mixed_group_fraction"] == 0.5
    assert result["reward_training/nonzero_advantage_tokens"] == 4
    assert result["reward_training/valid_responses"] == 7


def test_all_same_outcomes_allow_completed_zero_gradient_without_claiming_signal():
    config = _config()
    batch = _batch()
    batch["total_reward"].zero_()
    for metadata in batch["extra_env_info"]:
        metadata["_terminal_reward"]["score"] = 0.0
    batch = _prepare_async_reward_training_batch(batch, config)
    loss, gradient = _loss_and_gradient(batch, config)
    assert not gradient.any()
    assert validate_optimizer_progress(
        loss=loss.item(), grad_norm=gradient.norm().item(), global_valid_tokens=8.0,
        nonzero_reward_advantage_tokens=0.0,
    ) is False


@pytest.mark.parametrize(("gradient", "signal_tokens", "expected"), [
    (0.0, 0.0, False), (0.2, 0.0, False), (0.2, 4.0, True), (0.2, None, None),
])
def test_optimizer_progress_reports_outcome_signal_separately(gradient, signal_tokens, expected):
    assert validate_optimizer_progress(
        loss=0.0, grad_norm=gradient, global_valid_tokens=8.0,
        nonzero_reward_advantage_tokens=signal_tokens,
    ) is expected


@pytest.mark.parametrize(("loss", "gradient", "tokens", "signal_tokens"), [
    (0.0, 0.0, 8.0, None),  # Legacy OPD still requires a positive gradient.
    (0.0, 0.0, 8.0, 4.0),  # A reward gradient disappearing is not an all-equal group.
    (float("nan"), 0.0, 8.0, 0.0),
    (0.0, float("inf"), 8.0, 0.0),
    (0.0, 0.0, float("nan"), 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, -0.1, 8.0, 0.0),
    (0.0, 0.0, 8.0, -1.0),
    (0.0, 0.0, 8.0, float("nan")),
])
def test_optimizer_progress_rejects_invalid_or_unexpected_zero_gradient(loss, gradient, tokens, signal_tokens):
    with pytest.raises(RuntimeError):
        validate_optimizer_progress(
            loss=loss, grad_norm=gradient, global_valid_tokens=tokens,
            nonzero_reward_advantage_tokens=signal_tokens,
        )
