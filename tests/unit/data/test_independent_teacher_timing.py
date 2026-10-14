# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU scoring regression: queue/inference timing must not change score alignment."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from nemo_rl.data.independent_teacher import score_independent_text_batch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class IndependentTeacherTimingTest(unittest.TestCase):
    def score(self, dp_size):
        clock = [0.0]

        class Tokenizer:
            pad_token_id = 0

            def apply_chat_template(self, *args, **kwargs):
                return "prefix"

            def encode(self, *args, **kwargs):
                clock[0] += 1.0
                return [101, 102]

        class QueuedLock:
            def __enter__(self):
                clock[0] += 5.0

            def __exit__(self, *args):
                return False

        class Teacher:
            tokenizer = Tokenizer()
            tokenizer_config = {"chat_template_kwargs": {}, "required_rendered_prompt_suffix": "prefix"}
            sequence_length_pad_multiple = 4
            cfg = {"generation": {"_shared_response_excluded_token_ids": list(range(18, 30))},
                   "max_total_sequence_length": 64}
            sharding_annotations = SimpleNamespace(get_axis_size=lambda axis: dp_size)

            def get_logprobs(self, data):
                self.received = data
                clock[0] += 7.0
                return {"reference_logprobs": torch.arange(data["input_ids"].shape[1], dtype=torch.float32).expand_as(data["input_ids"])}

        teacher = Teacher()
        view = {"messages": [{"role": "user", "content": "hello"}], "tools": [],
                "modality": "text", "multimodal_payload": {}}
        batch = BatchedDataDict({
            "message_log": [[{"role": "user", "token_ids": torch.tensor([101, 102])},
                             {"role": "assistant", "token_ids": torch.tensor(response)}]
                            for response in ([41, 42], [51, 52])],
            "extra_env_info": [{"_independent_text_opd": {"teacher_alias": "nano", "teacher_view": view}}
                               for _ in range(2)],
        })
        output = io.StringIO()
        with patch("nemo_rl.data.independent_teacher.time.perf_counter", side_effect=lambda: clock[0]), contextlib.redirect_stdout(output):
            result = score_independent_text_batch(batch, {"nano": teacher}, {"nano": "nano"},
                                                   {"nano": QueuedLock()}, 4)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        scores, total = result
        torch.testing.assert_close(scores, torch.tensor([[0., 0., 2., 3.], [0., 0., 2., 3.]]))
        self.assertEqual(total, 14.0)  # Two renders + queue wait + inference.
        lines = output.getvalue().splitlines()
        self.assertEqual(sum(line.startswith("[INDEPENDENT_TEACHER_SCORE_OK]") for line in lines), 1)
        timing = next(line for line in lines if line.startswith("[INDEPENDENT_TEACHER_TIMING]"))
        fields = dict(field.split("=", 1) for field in timing.split()[1:])
        self.assertEqual(float(fields["render_prepare_s"]), 2.0)
        self.assertEqual(float(fields["lock_wait_s"]), 5.0)
        self.assertEqual(float(fields["inference_s"]), 7.0)
        self.assertEqual(float(fields["post_score_s"]), 0.0)
        self.assertEqual(float(fields["group_total_s"]), 14.0)
        self.assertEqual(int(fields["rows"]), 2)
        self.assertEqual(int(fields["dp_size"]), dp_size)
        return teacher, fields

    def test_k2_dp2_uses_both_real_samples_and_separates_queue_wait(self):
        teacher, fields = self.score(2)
        self.assertEqual(int(fields["padding_rows"]), 0)
        self.assertEqual(teacher.received["input_ids"].shape[0], 2)

    def test_padding_remains_trimmed_and_does_not_change_return_contract(self):
        teacher, fields = self.score(4)
        self.assertEqual(int(fields["padding_rows"]), 2)
        self.assertEqual(teacher.received["input_ids"].shape[0], 4)
        torch.testing.assert_close(teacher.received["input_ids"][2], teacher.received["input_ids"][1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
