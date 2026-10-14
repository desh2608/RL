# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow text-only, single-response bridge for independently rendered teachers."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from typing import Any

import torch
from transformers import AutoTokenizer

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.data.text_tool_prompt import text_tool_render_messages, validate_text_tool_schemas
from nemo_rl.algorithms.logits_sampling_utils import canonicalize_excluded_token_ids


def load_teacher_tokenizer(config: dict[str, Any], student_tokenizer: Any, excluded_token_ids: tuple[int, ...]) -> Any:
    """Load the explicit teacher contract and reject unequal token-ID maps."""
    if config["chat_template"] != "default":
        raise ValueError("The legacy text bridge requires the original teacher template")
    if config["use_fastokens"]:
        raise ValueError("The legacy text bridge requires the original HF tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        config["name"],
        trust_remote_code=True,
        fix_mistral_regex=config["fix_mistral_regex"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    student = getattr(student_tokenizer, "tokenizer", student_tokenizer)
    excluded = set(canonicalize_excluded_token_ids(excluded_token_ids))
    if len(tokenizer) != len(student):
        raise ValueError("Student and teacher vocabulary sizes differ")
    if any(token_id >= len(tokenizer) for token_id in excluded):
        raise ValueError("Excluded token ID exceeds tokenizer vocabulary")
    teacher_support = {token: token_id for token, token_id in tokenizer.get_vocab().items() if token_id not in excluded}
    student_support = {token: token_id for token, token_id in student.get_vocab().items() if token_id not in excluded}
    if teacher_support != student_support:
        raise ValueError("Independent text OPD requires identical token-ID maps on the shared response support")
    fingerprint = hashlib.sha256(
        json.dumps(teacher_support, sort_keys=True).encode()
    ).hexdigest()
    print(f"[INDEPENDENT_TEACHER_TOKENIZER_OK] vocab_sha256={fingerprint}", flush=True)
    return tokenizer


def _response_positions(message_log: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact generated token IDs and their flattened student positions."""
    offset = 0
    responses = []
    positions = []
    for message in message_log:
        tokens = message["token_ids"].detach().cpu().to(torch.long)
        if tokens.ndim != 1:
            raise ValueError("Message token IDs must be one-dimensional")
        if message["role"] == "assistant":
            responses.append(tokens)
            positions.append(torch.arange(offset, offset + tokens.numel()))
        elif message["role"] == "environment" and tokens.numel():
            raise ValueError("Independent text teacher cannot score tool/environment continuations")
        offset += tokens.numel()
    if len(responses) != 1 or responses[0].numel() == 0:
        raise ValueError("Independent text teacher requires exactly one non-empty generated response")
    return responses[0], positions[0]


def score_independent_text_batch(
    batch: BatchedDataDict,
    teacher_worker_groups: dict[str, Any],
    alias_to_group_alias: dict[str, str],
    teacher_locks: dict[str, Any],
    student_sequence_length: int,
) -> tuple[torch.Tensor, float]:
    """Score teacher-prompt + exact response, then align to student positions.

    Megatron LogprobsPostProcessor prepends the dummy first-token probability,
    so slot j is log p(token[j] | token[:j]). No additional shift is needed.
    """
    started = time.perf_counter()
    count = len(batch["message_log"])
    result = torch.zeros(count, student_sequence_length, dtype=torch.float32)
    grouped = defaultdict(list)
    for index, metadata in enumerate(batch["extra_env_info"]):
        contract = metadata["_independent_text_opd"]
        alias = contract["teacher_alias"]
        group = alias_to_group_alias[alias]
        grouped[group].append((index, contract))

    for group, rows in grouped.items():
        group_started = time.perf_counter()
        teacher = teacher_worker_groups[group]
        tokenizer = teacher.tokenizer
        tokenizer_config = teacher.tokenizer_config
        if tokenizer_config is None:
            raise ValueError("Independent teacher has no tokenizer configuration")
        sequences, prompt_lengths, responses, student_positions = [], [], [], []
        for index, contract in rows:
            view = contract["teacher_view"]
            if view["modality"] != "text" or view["multimodal_payload"] != {}:
                raise ValueError("Teacher view must be text-only")
            prompt = tokenizer.apply_chat_template(
                text_tool_render_messages(view["messages"], owner="teacher"),
                tools=validate_text_tool_schemas(view["tools"], owner="teacher"),
                tokenize=False,
                add_generation_prompt=True,
                **tokenizer_config["chat_template_kwargs"],
            )
            if not prompt.endswith(tokenizer_config["required_rendered_prompt_suffix"]):
                raise ValueError("Teacher prompt does not end in the required generation prefix")
            prompt_ids = torch.tensor(
                tokenizer.encode(prompt, add_special_tokens=False), dtype=torch.long
            )
            completion, positions = _response_positions(batch["message_log"][index])
            excluded = teacher.cfg["generation"]["_shared_response_excluded_token_ids"]
            if torch.isin(completion, torch.tensor(excluded, dtype=torch.long)).any():
                raise ValueError("Text OPD response contains reserved modality/tool-control IDs")
            sequence = torch.cat((prompt_ids, completion))
            if sequence.numel() > teacher.cfg["max_total_sequence_length"]:
                raise ValueError("Independently rendered teacher input exceeds configured context")
            if positions[-1] >= student_sequence_length:
                raise ValueError("Student response positions exceed flattened batch shape")
            sequences.append(sequence)
            prompt_lengths.append(prompt_ids.numel())
            responses.append(completion)
            student_positions.append(positions)

        multiple = teacher.sequence_length_pad_multiple
        sequence_length = ((max(s.numel() for s in sequences) + multiple - 1) // multiple) * multiple
        lengths = torch.tensor([s.numel() for s in sequences], dtype=torch.long)
        token_ids = torch.full((len(rows), sequence_length), tokenizer.pad_token_id, dtype=torch.long)
        token_mask = torch.zeros_like(token_ids, dtype=torch.float32)
        for local_index, sequence in enumerate(sequences):
            token_ids[local_index, : sequence.numel()] = sequence
            token_mask[local_index, prompt_lengths[local_index] : sequence.numel()] = 1
        teacher_data = BatchedDataDict(
            input_ids=token_ids,
            input_lengths=lengths,
            token_mask=token_mask,
            sample_mask=torch.ones(len(rows)),
        )
        dp_size = teacher.sharding_annotations.get_axis_size("data_parallel")
        padding = (-len(rows)) % dp_size
        if padding:
            teacher_data = teacher_data.select_indices(list(range(len(rows))) + [len(rows)-1] * padding)
        lock_started = time.perf_counter()
        with teacher_locks[group]:
            inference_started = time.perf_counter()
            scores = teacher.get_logprobs(teacher_data)["reference_logprobs"].detach().cpu()
            inference_finished = time.perf_counter()
        if scores.shape != teacher_data["input_ids"].shape:
            raise ValueError("Teacher scores do not align with its input token shape")
        for local_index, (index, contract) in enumerate(rows):
            prompt_length = prompt_lengths[local_index]
            completion = responses[local_index]
            scored_ids = token_ids[local_index, prompt_length : prompt_length + completion.numel()]
            if not torch.equal(scored_ids, completion):
                raise ValueError("Teacher scored token IDs differ from the student rollout")
            response_scores = scores[local_index, prompt_length : prompt_length + completion.numel()]
            if not torch.isfinite(response_scores).all():
                raise ValueError("Teacher returned non-finite response logprobs")
            result[index, student_positions[local_index]] = response_scores
        digest = hashlib.sha256(
            json.dumps([tokens.tolist() for tokens in responses]).encode()
        ).hexdigest()
        print(
            f"[INDEPENDENT_TEACHER_SCORE_OK] group={group} rows={len(rows)} "
            f"response_tokens={sum(t.numel() for t in responses)} "
            f"teacher_prompt_lengths={prompt_lengths} response_sha256={digest}",
            flush=True,
        )
        # Wall times, not CUDA kernel times. Inference includes the blocking Ray
        # call and result transfer. Keep queue wait separate so a larger rollout
        # fleet cannot make a saturated teacher appear to have slower kernels.
        group_finished = time.perf_counter()
        print(
            f"[INDEPENDENT_TEACHER_TIMING] group={group} rows={len(rows)} "
            f"dp_size={dp_size} padding_rows={padding} "
            f"render_prepare_s={lock_started - group_started:.6f} "
            f"lock_wait_s={inference_started - lock_started:.6f} "
            f"inference_s={inference_finished - inference_started:.6f} "
            f"post_score_s={group_finished - inference_finished:.6f} "
            f"group_total_s={group_finished - group_started:.6f}",
            flush=True,
        )
    return result, time.perf_counter() - started
