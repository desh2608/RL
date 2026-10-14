# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for static OPD's streaming native prompt groups."""

import asyncio

import unittest
from unittest.mock import patch
import torch

import nemo_rl.experience.rollouts as rollouts
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _batch(count=4):
    return BatchedDataDict(
        message_log=[
            [{"role": "user", "token_ids": torch.tensor([100 + i]), "content": ""}]
            for i in range(count)
        ],
        extra_env_info=[
            {
                "source_id": f"task-{i // 2}",
                "_independent_text_opd": {
                    "teacher_alias": "nano",
                    "teacher_view": {"prompt": f"teacher-prefix-{i // 2}"},
                },
            }
            for i in range(count)
        ],
        task_name=["pure_text_opd"] * count,
        idx=list(range(100, 100 + count)),
        loss_multiplier=torch.arange(count),
        target_weight_version=torch.full((count,), 7),
    )


def _sample_result(index, state):
    response = torch.tensor([14, 400 + index, 11])
    final_state = {
        "message_log": state["message_log"]
        + [
            {
                "role": "assistant",
                "token_ids": response,
                "generation_logprobs": torch.tensor([-0.1, -0.2, -0.3]),
                "content": f"answer-{index}",
            }
        ],
        "extra_env_info": state["extra_env_info"],
        "task_name": state["task_name"],
        "idx": state["idx"],
        "total_reward": torch.tensor(float(index)),
    }
    metrics = {
        "turn_count": 1,
        "total_tokens": 4,
        "assistant_tokens": 3,
        "env_tokens": 0,
        "terminated": index != 3,
        "truncated": index == 3,
        "max_turns_reached": False,
        "total_reward": float(index),
        "turn_gen_tokens": [3],
        "turn_input_tokens": [1],
        "turn_total_tokens": [4],
        "max_gen_tokens_per_turn": 3,
        "per_worker_token_counts": {f"worker-{index % 2}": 3},
    }
    return final_state, metrics


def _kwargs(batch):
    return {
        "policy_generation": None,
        "input_batch": batch,
        "tokenizer": None,
        "task_to_env": {},
        "max_seq_len": 98304,
        "num_generations": 2,
        "max_rollout_turns": 1,
        "stream_prompt_groups": True,
    }


class NativeOPDStreamingTests(unittest.TestCase):
    def _patch_generation(self, generate):
        patcher = patch.object(rollouts, "run_sample_multi_turn_rollout", generate)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_k2_streams_complete_group_before_slow_group_preserving_exact_rows(self):
        """A fast group's teacher can start without one unfinished K row leaking out."""

        async def scenario():
            gates = [asyncio.Event() for _ in range(4)]
            returned = [asyncio.Event() for _ in range(4)]
            calls = []

            async def generate(sample_idx, initial_sample_state, **kwargs):
                calls.append(sample_idx)
                await gates[sample_idx].wait()
                result = _sample_result(sample_idx, initial_sample_state)
                returned[sample_idx].set()
                return result

            self._patch_generation(generate)
            batch = _batch()
            stream = rollouts.run_async_multi_turn_rollout_groups(**_kwargs(batch))
            first_result = asyncio.create_task(anext(stream))
            gates[3].set()
            await asyncio.wait_for(returned[3].wait(), 5)
            assert not first_result.done()  # K=2: sample 2 is still missing.
            gates[2].set()
            fast = await asyncio.wait_for(first_result, 5)
            assert fast.group_index == 1
            assert not returned[0].is_set() and not returned[1].is_set()
            assert fast.final_batch["idx"] == [102, 103]
            assert fast.final_batch["loss_multiplier"].tolist() == [2, 3]
            assert fast.final_batch["target_weight_version"].tolist() == [7, 7]
            assert fast.final_batch["truncated"].tolist() == [False, True]
            assert fast.rollout_metrics["truncation_rate"] == 0.5
            for local, original in enumerate((2, 3)):
                message = fast.final_batch["message_log"][local][-1]
                assert message["token_ids"].tolist() == [14, 400 + original, 11]
                assert torch.equal(
                    message["generation_logprobs"], torch.tensor([-0.1, -0.2, -0.3])
                )
                assert fast.final_batch["extra_env_info"][local] == batch["extra_env_info"][original]
            gates[1].set()
            gates[0].set()
            slow = await asyncio.wait_for(anext(stream), 5)
            assert slow.group_index == 0
            assert slow.final_batch["idx"] == [100, 101]
            with self.assertRaises(StopAsyncIteration):
                await anext(stream)
            assert sorted(calls) == [0, 1, 2, 3]
            assert {fast.group_index, slow.group_index} == {0, 1}

        asyncio.run(scenario())


    def test_streaming_matches_whole_batch_values_and_metrics(self):
        async def generate(sample_idx, initial_sample_state, **kwargs):
            return _sample_result(sample_idx, initial_sample_state)

        self._patch_generation(generate)

        async def collect(streaming):
            kwargs = _kwargs(_batch())
            kwargs["stream_prompt_groups"] = streaming
            return sorted(
                [g async for g in rollouts.run_async_multi_turn_rollout_groups(**kwargs)],
                key=lambda group: group.group_index,
            )

        expected = asyncio.run(collect(False))
        actual = asyncio.run(collect(True))
        for left, right in zip(expected, actual, strict=True):
            assert left.group_index == right.group_index
            assert left.rollout_metrics == right.rollout_metrics
            assert left.final_batch["idx"] == right.final_batch["idx"]
            assert left.final_batch["extra_env_info"] == right.final_batch["extra_env_info"]
            for key in ("total_reward", "truncated", "loss_multiplier", "target_weight_version"):
                assert torch.equal(left.final_batch[key], right.final_batch[key])
            for left_log, right_log in zip(
                left.final_batch["message_log"], right.final_batch["message_log"], strict=True
            ):
                assert torch.equal(left_log[-1]["token_ids"], right_log[-1]["token_ids"])
                assert torch.equal(
                    left_log[-1]["generation_logprobs"], right_log[-1]["generation_logprobs"]
                )


    def test_failed_sample_propagates_and_cancels_all_other_group_rows(self):
        async def scenario():
            started = [asyncio.Event() for _ in range(4)]
            cancelled = set()

            async def generate(sample_idx, initial_sample_state, **kwargs):
                started[sample_idx].set()
                if sample_idx == 2:
                    await asyncio.gather(*(event.wait() for event in started))
                    raise ValueError("injected generation failure")
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.add(sample_idx)
                    raise

            self._patch_generation(generate)
            stream = rollouts.run_async_multi_turn_rollout_groups(**_kwargs(_batch()))
            with self.assertRaisesRegex(RuntimeError, "Error in sample 2 rollout"):
                await asyncio.wait_for(anext(stream), 5)
            assert cancelled == {0, 1, 3}

        asyncio.run(scenario())


    def test_consumer_close_cancels_unfinished_group_without_partial_emission(self):
        async def scenario():
            started = [asyncio.Event() for _ in range(4)]
            cancelled = set()

            async def generate(sample_idx, initial_sample_state, **kwargs):
                started[sample_idx].set()
                if sample_idx >= 2:
                    await asyncio.gather(*(event.wait() for event in started))
                    return _sample_result(sample_idx, initial_sample_state)
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.add(sample_idx)
                    raise

            self._patch_generation(generate)
            stream = rollouts.run_async_multi_turn_rollout_groups(**_kwargs(_batch()))
            first = await asyncio.wait_for(anext(stream), 5)
            assert first.group_index == 1 and first.final_batch.size == 2
            await asyncio.wait_for(stream.aclose(), 5)
            assert cancelled == {0, 1}

        asyncio.run(scenario())

    def test_late_failure_keeps_completed_group_and_never_emits_partial_group(self):
        async def scenario():
            started = [asyncio.Event() for _ in range(4)]
            fail = asyncio.Event()
            cancelled = set()

            async def generate(sample_idx, initial_sample_state, **kwargs):
                started[sample_idx].set()
                if sample_idx < 2:
                    await asyncio.gather(*(event.wait() for event in started))
                    return _sample_result(sample_idx, initial_sample_state)
                if sample_idx == 2:
                    await fail.wait()
                    raise ValueError("late failure")
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.add(sample_idx)
                    raise

            self._patch_generation(generate)
            stream = rollouts.run_async_multi_turn_rollout_groups(**_kwargs(_batch()))
            first = await asyncio.wait_for(anext(stream), 5)
            assert first.group_index == 0 and first.final_batch["idx"] == [100, 101]
            fail.set()
            with self.assertRaisesRegex(RuntimeError, "Error in sample 2 rollout"):
                await asyncio.wait_for(anext(stream), 5)
            assert cancelled == {3}
            with self.assertRaises(StopAsyncIteration):
                await anext(stream)

        asyncio.run(scenario())


    def test_invalid_group_shape_fails_before_generation(self):
        for count, k in ((4, 0), (3, 2)):
            with self.subTest(count=count, k=k):
                calls = []

                async def generate(**kwargs):
                    calls.append(kwargs)
                    raise AssertionError("invalid groups must not generate")

                self._patch_generation(generate)

                async def scenario():
                    kwargs = _kwargs(_batch(count))
                    kwargs["num_generations"] = k
                    stream = rollouts.run_async_multi_turn_rollout_groups(**kwargs)
                    with self.assertRaises(ValueError):
                        await anext(stream)

                asyncio.run(scenario())
                assert calls == []


if __name__ == "__main__":
    unittest.main(verbosity=2)
