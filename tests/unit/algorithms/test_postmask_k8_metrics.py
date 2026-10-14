import json
import os
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from nemo_rl.algorithms.postmask_k8_metrics import (
    ENV_FLAG, MAX_JSON_BYTES, ROUTES, collect_postmask_k8_metrics,
    postmask_k8_diagnostics, postmask_k8_metrics_enabled,
)


ROUTE = "instruction_following_simple_agent"


def fixture(groups=2, width=3):
    rows = groups * 8
    return dict(rewards=torch.tensor([0., 1.] * (rows // 2)),
                advantages=torch.ones((rows, width)),
                token_mask=torch.ones((rows, width)), sample_mask=torch.ones(rows),
                group_ids=torch.arange(groups).repeat_interleave(8)[:, None],
                domains=[ROUTE] * rows, step=1)


class MetricsTests(unittest.TestCase):
    def test_default_off_and_exact_enablement(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(postmask_k8_metrics_enabled())
        for value, expected in [("0", False), ("1", True), ("true", False), ("", False)]:
            with patch.dict(os.environ, {ENV_FLAG: value}):
                self.assertEqual(postmask_k8_metrics_enabled(), expected)

    def test_mixed_and_homogeneous_reward_dispersion(self):
        data = fixture()
        data["rewards"][8:] = 1
        result = collect_postmask_k8_metrics(**data)["global"]
        self.assertEqual(result["survivors_0_to_8"], [0] * 8 + [2])
        self.assertEqual(result["mixed_reward_groups"], 1)
        self.assertEqual(result["mean_retained_group_reward_std"], 0.25)
        self.assertEqual(result["valid_tokens"], 48)
        self.assertEqual(result["nonzero_fraction_of_finite_valid_tokens"], 1)

    def test_zero_one_two_survivors_and_token_mask(self):
        data = fixture(groups=3)
        data["sample_mask"][:] = 0
        data["sample_mask"][8] = 1
        data["sample_mask"][16:19] = 1
        data["token_mask"][18] = 0  # Empty response is not a survivor.
        data["token_mask"][17, 1:] = 0
        data["advantages"][16] = -20  # Observe final clipped/penalized values.
        data["advantages"][17] = 0
        result = collect_postmask_k8_metrics(**data)["global"]
        self.assertEqual(result["survivors_0_to_8"], [1, 1, 1] + [0] * 6)
        self.assertEqual(result["retained_responses"], 3)
        self.assertEqual(result["dispersion_groups"], 1)
        self.assertEqual(result["valid_tokens"], 7)
        self.assertEqual(result["nonzero_advantage_tokens"], 6)
        self.assertEqual(result["positive_advantage_tokens"], 3)
        self.assertEqual(result["negative_advantage_tokens"], 3)
        self.assertEqual(result["nonzero_fraction_of_finite_valid_tokens"], 6 / 7)

    def test_one_survivor_keeps_observed_advantage_signal(self):
        data = fixture(groups=1)
        data["sample_mask"][1:] = 0
        result = collect_postmask_k8_metrics(**data)["global"]
        self.assertEqual(result["retained_responses"], 1)
        self.assertEqual(result["nonzero_advantage_tokens"], 3)
        self.assertEqual(result["nonzero_fraction_of_finite_valid_tokens"], 1)
        self.assertIsNone(result["mean_retained_group_reward_std"])
        self.assertIsNone(result["mixed_fraction_of_dispersion_groups"])

    def test_all_masked_is_undefined_not_zero(self):
        data = fixture()
        data["sample_mask"][:] = 0
        metrics, payload = postmask_k8_diagnostics(**data)
        result = json.loads(payload)["global"]
        self.assertEqual(result["survivors_0_to_8"][0], 2)
        self.assertIsNone(result["mean_retained_group_reward_std"])
        self.assertIsNone(result["nonzero_fraction_of_finite_valid_tokens"])
        self.assertNotIn("postmask_signal/global/nonzero_fraction_of_finite_valid_tokens", metrics)

    def test_nonfinite_observations_explicit_and_mask_aware(self):
        data = fixture()
        data["rewards"][0] = float("nan")
        data["advantages"][1, 0] = float("inf")
        data["advantages"][2, 0] = float("nan")
        data["sample_mask"][2] = 0
        metrics, payload = postmask_k8_diagnostics(**data)
        result = json.loads(payload)["global"]
        self.assertEqual(result["nonfinite_reward_groups"], 1)
        self.assertEqual(result["dispersion_groups"], 1)
        self.assertEqual(result["nonfinite_advantage_tokens"], 1)
        self.assertEqual(result["finite_valid_tokens"], 44)
        self.assertEqual(metrics["postmask_signal/available"], 1)
        self.assertNotIn("NaN", payload)
        self.assertNotIn("Infinity", payload)

    def test_explicit_group_ids_allow_noncontiguous_order(self):
        data = fixture(groups=2)
        data["rewards"][8:] = 1
        expected = collect_postmask_k8_metrics(**data)
        permutation = torch.tensor([i for pair in zip(range(8), range(8, 16)) for i in pair])
        permuted = {key: value[permutation] if isinstance(value, torch.Tensor) else value
                    for key, value in data.items()}
        self.assertEqual(collect_postmask_k8_metrics(**permuted), expected)

    def test_unknown_and_mixed_routes_have_explicit_buckets(self):
        data = fixture()
        data["domains"][0] = "unrecognized-route-that-must-not-be-logged"
        data["domains"][8:] = [None] * 8
        result = collect_postmask_k8_metrics(**data)
        self.assertEqual(result["unknown_route_rows"], 9)
        self.assertEqual(set(result["routes"]), {"mixed_routes", "unknown"})
        self.assertNotIn("unrecognized", json.dumps(result))

    def test_chunk_invariance(self):
        torch.manual_seed(122)
        data = fixture(groups=4, width=31)
        data["advantages"] = torch.randn(32, 31).clamp(-2, 2)
        data["token_mask"] = torch.rand(32, 31).gt(0.3)
        data["sample_mask"] = torch.rand(32).gt(0.2)
        expected = postmask_k8_diagnostics(**data)
        for rows, tokens in [(1, 1), (7, 11), (16, 9), (64, 4096)]:
            self.assertEqual(postmask_k8_diagnostics(**data, row_chunk=rows, token_chunk=tokens), expected)

    def test_bad_shapes_and_masks_report_unavailable(self):
        mutations = [
            ("rewards", torch.ones(15)),
            ("token_mask", torch.ones(16, 2)),
            ("group_ids", torch.arange(16)),
            ("group_ids", torch.zeros(16, dtype=torch.float32)),
            ("domains", []),
            ("sample_mask", torch.full((16,), float("nan"))),
            ("token_mask", torch.full((16, 3), float("nan"))),
            ("step", 0), ("row_chunk", 65), ("token_chunk", 4097),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                data = fixture();data[key] = value
                metrics, payload = postmask_k8_diagnostics(**data)
                self.assertEqual(metrics, {"postmask_signal/available": 0})
                self.assertEqual(json.loads(payload)["status"], "unavailable")
                self.assertNotIn("global", json.loads(payload))
        data = fixture()
        for key, value in list(data.items()):
            if isinstance(value, torch.Tensor): data[key] = value[:15]
        data["domains"] = data["domains"][:15]
        self.assertEqual(json.loads(postmask_k8_diagnostics(**data)[1])["reason"], "k8_shape")

    def test_all_route_json_budget(self):
        routes = sorted(ROUTES) + ["unknown", "mixed_routes"]
        data = fixture(groups=len(routes))
        data["domains"] = [route for route in routes for _ in range(8)]
        data["domains"][-1] = ROUTE
        _, payload = postmask_k8_diagnostics(**data)
        self.assertEqual(json.loads(payload)["status"], "ok")
        self.assertLessEqual(len(payload.encode()), MAX_JSON_BYTES)

    def test_large_shape_uses_bounded_intermediates(self):
        data = fixture(groups=1, width=98304)

        class Bounded(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                for item in result if isinstance(result, (list, tuple)) else [result]:
                    if isinstance(item, torch.Tensor) and item.numel() > 64 * 4096:
                        raise AssertionError("Unbounded diagnostic intermediate " + str(func))
                return result
        with Bounded():
            result = collect_postmask_k8_metrics(**data)
        self.assertEqual(result["global"]["valid_tokens"], 8 * 98304)

    def test_read_only_inputs_and_rng(self):
        torch.manual_seed(481)
        data = fixture()
        data["advantages"] = torch.randn(16, 3).clamp(-20, 20)
        data["sample_mask"][1] = 0
        inputs = {key: value.clone() for key, value in data.items() if isinstance(value, torch.Tensor)}
        versions = {key: value._version for key, value in data.items() if isinstance(value, torch.Tensor)}
        rng = torch.get_rng_state().clone()
        postmask_k8_diagnostics(**data)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, before in inputs.items():
            self.assertTrue(torch.equal(data[key], before), key)
            self.assertEqual(data[key]._version, versions[key], key)



class NativeLossParityTests(unittest.TestCase):
    def test_native_clipped_pg_loss_on_off_parity(self):
        """Strict native CPU test: actual async hook and actual ClippedPGLossFn.

        Requires the real pinned RL runtime imports. No skip or substitute loss.
        """
        import ast
        import contextlib
        import io
        from pathlib import Path
        from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
        from nemo_rl.utils.training_guardrails import metadata_domain, resolve_monitor_metadata
        # Exact unchanged native Gym loss recipe for this arm.
        config = ClippedPGLossConfig(
            disable_ppo_ratio=True, force_on_policy_ratio=True,
            token_level_loss=True, sequence_level_importance_ratios=False,
            ratio_clip_min=0.2, ratio_clip_max=0.28, ratio_clip_c=None,
            reference_policy_kl_penalty=0, reference_policy_kl_type="k3",
            kl_input_clamp_value=None, kl_output_clamp_value=None,
            use_kl_in_reward=False, use_importance_sampling_correction=True,
            truncated_importance_sampling_type="tis",
            truncated_importance_sampling_ratio=5,
            truncated_importance_sampling_ratio_min=0.2,
            use_on_policy_kl_approximation=True, use_cispo=False,
            positive_example_nll_weight=0,
        )
        native_loss = ClippedPGLossFn(config)
        source = Path(__file__).resolve().parents[3] / "nemo_rl/algorithms/grpo.py"
        tree = ast.parse(source.read_text())
        observers = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name)
                     and node.test.func.id == "postmask_k8_metrics_enabled"]
        self.assertEqual(len(observers), 1)
        hook = compile(ast.Module(body=[observers[0]], type_ignores=[]), str(source), "exec")
        rows, width = 24, 6
        template = {
            "input_ids": torch.zeros(rows, width, dtype=torch.long),
            "advantages": torch.linspace(-3, 3, rows * width).reshape(rows, width),
            "token_mask": torch.ones(rows, width), "sample_mask": torch.ones(rows),
            "prev_logprobs": torch.full((rows, width), -1.0),
            "generation_logprobs": torch.full((rows, width), -1.2),
            "reference_policy_logprobs": torch.full((rows, width), -1.0),
        }
        template["token_mask"][:, 0] = 0
        template["sample_mask"][:15] = 0  # First group:0; second:1; third:8 survivors.
        template["advantages"][15] = 2  # One survivor may retain nonzero native advantage.
        rewards = torch.tensor([0., 1.] * 12)
        group_ids = torch.arange(3).repeat_interleave(8)[:, None]
        initial_rng = torch.get_rng_state().clone()
        class Batch(dict):size = rows
        results = []
        for enabled in ("0", "1"):
            data = {key: value.clone() for key, value in template.items()}
            before = {key: value.clone() for key, value in data.items()}
            versions = {key: value._version for key, value in data.items()}
            namespace = dict(train_data=data, rewards=rewards.clone(), prompt_ids_for_adv=group_ids.clone(),
                             repeated_batch=Batch(extra_env_info=[{"domain":ROUTE}] * rows),
                             step=0, rollout_metrics={},
                             metadata_domain=metadata_domain, resolve_monitor_metadata=resolve_monitor_metadata,
                             postmask_k8_metrics_enabled=postmask_k8_metrics_enabled,
                             postmask_k8_diagnostics=postmask_k8_diagnostics)
            torch.set_rng_state(initial_rng)
            with patch.dict(os.environ, {ENV_FLAG:enabled}), contextlib.redirect_stdout(io.StringIO()) as output:
                exec(hook, namespace)
            for key, original in before.items():
                self.assertTrue(torch.equal(data[key], original), key)
                self.assertEqual(data[key]._version, versions[key], key)
            self.assertTrue(torch.equal(namespace["rewards"], rewards))
            self.assertTrue(torch.equal(torch.get_rng_state(), initial_rng))
            logprobs = (-torch.rand(rows, width - 1) - 0.5).requires_grad_()
            valid = data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
            loss, actual_metrics = native_loss(logprobs, data,
                                               global_valid_seqs=data["sample_mask"].sum(),
                                               global_valid_toks=valid.sum())
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(logprobs.grad.abs().sum().item(), 0)
            self.assertIn("sampling_importance_ratio", actual_metrics)
            self.assertIn("is_oob_ratio", actual_metrics)
            if enabled == "1":
                record = json.loads(output.getvalue().split("] ", 1)[1])
                self.assertEqual(record["status"], "ok")
                self.assertEqual(record["global"]["survivors_0_to_8"], [1, 1] + [0] * 6 + [1])
                self.assertEqual(record["global"]["valid_tokens"], int(valid.sum()))
                self.assertGreater(record["global"]["nonzero_advantage_tokens"], 0)
            results.append((loss.detach(), logprobs.grad.clone(), actual_metrics, torch.get_rng_state().clone()))
        for index in (0, 1, 3): self.assertTrue(torch.equal(results[0][index], results[1][index]))
        self.assertEqual(results[0][2], results[1][2])
        self.assertFalse(torch.cuda.is_initialized(), "Native CPU test initialized CUDA")


if __name__ == "__main__":
    unittest.main()
