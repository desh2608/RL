# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import contextlib
import logging
import multiprocessing as mp
from io import StringIO
from typing import Any, ClassVar, Dict, List, Optional, Union

from fastapi import FastAPI
from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from pydantic import BaseModel, PositiveFloat, PositiveInt

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics


class LibraryJudgeMathResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    should_use_judge: bool = True
    fail_on_judge_error: bool = False
    fail_on_library_error: bool = False
    library_verifier_timeout_seconds: PositiveFloat = 10.0
    library_verifier_max_concurrency: PositiveInt = 32


class LibraryJudgeMathRunRequest(BaseRunRequest):
    question: str
    expected_answer: str


class LibraryJudgeMathVerifyRequest(LibraryJudgeMathRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class LibraryJudgeMathVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]
    library_reward: float
    judge_evaluations: Optional[list[JudgeEvaluation]]


def _extract_last_boxed_answer(response: str) -> Optional[str]:
    r"""Extract the exact contents of the last complete ``\boxed{...}``."""
    boxed_start = response.rfind("\\boxed{")
    if boxed_start < 0:
        return None

    content_start = boxed_start + len("\\boxed{")
    brace_depth = 1
    for index in range(content_start, len(response)):
        if response[index] == "{":
            brace_depth += 1
        elif response[index] == "}":
            brace_depth -= 1
            if brace_depth == 0:
                return response[content_start:index]

    return None


def _run_math_verify(
    library_verifier: Any, expected_answer: str, generated_answer: str
) -> tuple[float, Optional[str]]:
    # This functionality is migrated from Nemo RL.
    # https://github.com/NVIDIA-NeMo/RL/blob/e1f56c42ae175d3863ccaf4e21b7de7e9c46c2e1/nemo_rl/environments/math_environment.py
    try:
        stripped = LibraryJudgeMathResourcesServer._strip_math_delimiters(expected_answer)
        ground_truth_parsable = "\\boxed{" + stripped + "}"
        with LibraryJudgeMathResourcesServer._mute_output():
            ret_score, extracted_answer = library_verifier([ground_truth_parsable], [generated_answer])

        reward = float(ret_score)

        if extracted_answer is not None:
            # Make sure the extracted answer has two elements.
            assert len(extracted_answer) == 2

            extracted_gold, extracted_prediction = extracted_answer

            # Get the extracted answer.
            for pred in extracted_prediction:
                if any(grader.verify(gold, pred) for gold in extracted_gold):
                    extracted_answer = pred
                    break
            else:
                # If no match is found, that means all the answers are
                # incorrect.  The first prediction is used as the extracted
                # answer.
                extracted_answer = extracted_prediction[0] if extracted_prediction else None

        return reward, extracted_answer

    # It's possible to emit a TimeoutException and that wouldn't be caught since
    # it actually subclasses from BaseException and math-verify itself does not
    # catch it.
    except (Exception, TimeoutException):
        return 0.0, None


def _run_math_verify_in_subprocess(expected_answer: str, generated_answer: str, result_connection: Any) -> None:
    # Keep math_verify construction inside the child process. A wedged SymPy call
    # can then be killed by terminating this process without poisoning the server.
    library_verifier = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(
            ExprExtractionConfig(),
            LatexExtractionConfig(),
        ),
    )
    try:
        result_connection.send(_run_math_verify(library_verifier, expected_answer, generated_answer))
    finally:
        result_connection.close()


class LibraryVerificationError(RuntimeError):
    """A verifier failure, never a model wrong-answer reward."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__("Math library verification failed: " + reason)


def _strict_library_packet(library_verifier: Any, expected_answer: str, generated_answer: str) -> dict:
    """Keep native metric semantics while exposing errors that escape the library.

    Nonempty native gold extraction is support evidence, not reference truth.
    No-box paths additionally use a reference self-check; absent pairs fail.
    math_verify may itself suppress parser/grader exceptions; this wrapper does
    not claim to recover exceptions that the dependency does not expose.
    """
    try:
        stripped = LibraryJudgeMathResourcesServer._strip_math_delimiters(expected_answer)
        if not stripped:
            raise LibraryVerificationError("reference_unsupported")
        gold = "\\boxed{" + stripped + "}"
        boxed = _extract_last_boxed_answer(generated_answer)
        format_invalid = boxed is None or not boxed.strip()
        with LibraryJudgeMathResourcesServer._mute_output():
            ret_score, extracted = library_verifier([gold], [gold] if format_invalid else [generated_answer])
        reward = float(ret_score)
        if reward not in (0.0, 1.0):
            raise LibraryVerificationError("invalid_result")
        if format_invalid:
            if (reward != 1.0 or not isinstance(extracted, (tuple, list))
                    or len(extracted) != 2 or not extracted[0] or not extracted[1]):
                raise LibraryVerificationError("reference_unsupported")
            return {"status": "ok", "outcome": "candidate_format_invalid", "reward": 0.0, "extracted_answer": None}
        if extracted is None:
            # The dependency may suppress exceptions into an absent pair.
            # A separate gold check cannot prove this was model bad formatting.
            raise LibraryVerificationError("unresolved_library_result")
        selected = None
        if extracted is not None:
            if not isinstance(extracted, (tuple, list)) or len(extracted) != 2:
                raise LibraryVerificationError("invalid_result")
            extracted_gold, predictions = extracted
            if not isinstance(extracted_gold, (list, tuple)) or not isinstance(predictions, (list, tuple)):
                raise LibraryVerificationError("invalid_result")
            if not extracted_gold:
                raise LibraryVerificationError("reference_unsupported")
            # Preserve the native first-correct/otherwise-first extraction rule.
            for pred in predictions:
                if any(grader.verify(g, pred) for g in extracted_gold):
                    selected = pred
                    break
            else:
                selected = predictions[0] if predictions else None
        if reward == 1.0 and selected is None:
            raise LibraryVerificationError("invalid_result")
        outcome = "equivalent" if reward else ("wrong" if selected is not None else "candidate_format_invalid")
        # The public response requires Optional[str]; keep IPC scalar and explicit.
        selected_text = None if selected is None else str(selected)
        return {"status": "ok", "outcome": outcome, "reward": reward, "extracted_answer": selected_text}
    except LibraryVerificationError as exc:
        return {"status": "error", "reason": exc.reason}
    except TimeoutException:
        return {"status": "error", "reason": "library_timeout"}
    except Exception:
        return {"status": "error", "reason": "library_exception"}


def _strict_math_verify_in_subprocess(expected_answer: str, generated_answer: str, result_connection: Any) -> None:
    try:
        try:
            library_verifier = math_metric(
                gold_extraction_target=(LatexExtractionConfig(),),
                pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
            )
            packet = _strict_library_packet(library_verifier, expected_answer, generated_answer)
        except TimeoutException:
            packet = {"status": "error", "reason": "library_timeout"}
        except Exception:
            packet = {"status": "error", "reason": "library_exception"}
        result_connection.send(packet)
    finally:
        result_connection.close()


def _decode_strict_library_packet(packet: Any) -> tuple[float, Optional[str]]:
    if isinstance(packet, dict) and set(packet) == {"status", "reason"} and packet["status"] == "error":
        if isinstance(packet["reason"], str) and packet["reason"] in {"reference_unsupported", "invalid_result", "library_timeout", "library_exception", "unresolved_library_result"}:
            raise LibraryVerificationError(packet["reason"])
    if not isinstance(packet, dict) or set(packet) != {"status", "outcome", "reward", "extracted_answer"}:
        raise LibraryVerificationError("invalid_packet")
    if packet["status"] != "ok" or type(packet["reward"]) is not float:
        raise LibraryVerificationError("invalid_packet")
    expected = {"equivalent": 1.0, "wrong": 0.0, "candidate_format_invalid": 0.0}
    if not isinstance(packet["outcome"], str) or packet["outcome"] not in expected or packet["reward"] != expected[packet["outcome"]]:
        raise LibraryVerificationError("invalid_packet")
    if packet["extracted_answer"] is not None and not isinstance(packet["extracted_answer"], str):
        raise LibraryVerificationError("invalid_packet")
    if (packet["outcome"] == "candidate_format_invalid") != (packet["extracted_answer"] is None):
        raise LibraryVerificationError("invalid_packet")
    return packet["reward"], packet["extracted_answer"]


class LibraryJudgeMathResourcesServer(SimpleResourcesServer):
    # These judge messages are adapted from ones used in Arena Hard.
    # https://github.com/lmarena/arena-hard-auto/blob/196f6b826783b3da7310e361a805fa36f0be83f3/utils/judge_utils.py
    # They are intended to serve as example messages for an LLM judge, and have not
    # been customized for a specific judge model.
    JUDGE_SYSTEM_MESSAGE: ClassVar[
        str
    ] = """Please act as an impartial judge and evaluate the equivalence of the solutions given by two AI assistants to the mathematical problem displayed below. You will be given AI assistant A's answer and AI assistant B's answer. Your job is to evaluate whether assistant A's answer is equivalent to assistant B's answer.

Consider the mathematical equivalence of the AI assistants' answers above all other considerations. If the problem requests special formatting instructions, you may disregard any formatting considerations when evaluating the answers -- consider only mathematical equivalence.

After evaluating both answers for equivalence, you must output only one of the following choices as your final verdict with a label:

1.  The AI assistants' answers are equivalent: [[A=B]]
2.  The AI assistants' answers are different: [[A!=B]]

Example output: "My final verdict is different [[A!=B]]"."""

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = (
        "<|Problem|>\n{question}\n\n<|Start of Assistant A's Answer|>\n{first_answer}\n<|End of Assistant A's Answer|>\n\n<|Start of Assistant B's Answer|>\n{second_answer}\n<|End of Assistant B's Answer|>"
    )

    JUDGE_EQUAL_LABEL: ClassVar[str] = "[[A=B]]"
    JUDGE_NOT_EQUAL_LABEL: ClassVar[str] = "[[A!=B]]"

    config: LibraryJudgeMathResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        # The async path no longer blocks the event loop while SymPy runs, so
        # cap child-process fanout explicitly.
        self._library_verifier_semaphore = asyncio.Semaphore(value=self.config.library_verifier_max_concurrency)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(self, body: LibraryJudgeMathVerifyRequest) -> LibraryJudgeMathVerifyResponse:
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue

            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue

                assistant_responses.append(content_item.text)

        combined_response = "".join(assistant_responses)
        reward, extracted_answer, library_reward, judge_evaluations = await self._verify_answer(
            body.question, body.expected_answer, combined_response
        )
        return LibraryJudgeMathVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=extracted_answer,
            library_reward=library_reward,
            judge_evaluations=judge_evaluations,
        )

    async def _verify_answer(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str], float, Optional[list[JudgeEvaluation]]]:
        """Verify the correctness of a generated answer.

        Verify the correctness of the specified model-generated answer to the
        specified question in comparison with the specified expected answer.
        """

        if getattr(self.config, "fail_on_library_error", False):
            if self.config.should_use_judge:
                raise LibraryVerificationError("strict_mode_requires_rules_only")
            library_reward, extracted_answer = await self._verify_answer_with_library_async(
                expected_answer, generated_answer
            )
            return library_reward, extracted_answer, library_reward, None

        boxed_answer = _extract_last_boxed_answer(generated_answer)
        if boxed_answer is None or not boxed_answer.strip():
            return 0.0, None, 0.0, None

        library_reward, extracted_answer = await self._verify_answer_with_library_async(
            expected_answer, generated_answer
        )
        if not self.config.should_use_judge or library_reward > 0.5:
            return library_reward, extracted_answer, library_reward, None

        judge_reward, judge_evaluations = await self._verify_answer_with_judge(question, expected_answer, boxed_answer)
        return judge_reward, extracted_answer, library_reward, judge_evaluations

    @classmethod
    @contextlib.contextmanager
    def _mute_output(cls):
        devnull_out, devnull_err = StringIO(), StringIO()
        with (
            contextlib.redirect_stdout(devnull_out),
            contextlib.redirect_stderr(devnull_err),
        ):
            yield

    @staticmethod
    def _strip_math_delimiters(s: str) -> str:
        """Strip outer math delimiters from expected answers.

        Many expected_answer values are wrapped in \\(...\\) or $...$,
        which causes the math_verify parser to fail when we wrap them
        in \\boxed{}.  Removing these outer delimiters fixes parsing.
        """
        s = s.strip()
        if s.startswith("\\(") and s.endswith("\\)"):
            s = s[2:-2].strip()
        if s.startswith("$") and s.endswith("$") and len(s) > 1:
            s = s[1:-1].strip()
        return s

    async def _verify_answer_with_library_async(
        self, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str]]:
        async with self._library_verifier_semaphore:
            # The production rollout workers run on Linux. Pin fork so the
            # verifier child is cheap to start and can be killed independently
            # if SymPy wedges.
            ctx = mp.get_context("fork")
            result_connection, child_connection = ctx.Pipe(duplex=False)
            strict = getattr(self.config, "fail_on_library_error", False)
            process = ctx.Process(
                target=_strict_math_verify_in_subprocess if strict else _run_math_verify_in_subprocess,
                args=(expected_answer, generated_answer, child_connection),
            )
            if strict:
                try:
                    process.start()
                except Exception:
                    child_connection.close()
                    result_connection.close()
                    raise LibraryVerificationError("worker_start") from None
            else:
                process.start()
            child_connection.close()
            if strict:
                return await self._wait_for_library_verifier_process(
                    process, result_connection, self.config.library_verifier_timeout_seconds,
                    fail_on_library_error=True,
                )
            return await self._wait_for_library_verifier_process(
                process,
                result_connection,
                self.config.library_verifier_timeout_seconds,
            )

    @staticmethod
    async def _wait_for_library_verifier_process(
        process: Any, result_connection: Any, timeout_seconds: float, fail_on_library_error: bool = False
    ) -> tuple[float, Optional[str]]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        try:
            while process.is_alive():
                remaining = deadline - loop.time()
                if remaining <= 0:
                    process.terminate()
                    terminate_deadline = loop.time() + 1.0
                    while process.is_alive() and loop.time() < terminate_deadline:
                        await asyncio.sleep(0.05)
                    if process.is_alive():
                        process.kill()
                    while process.is_alive():
                        await asyncio.sleep(0.05)
                    process.join(timeout=0)
                    if fail_on_library_error:
                        raise LibraryVerificationError("worker_timeout")
                    return 0.0, None
                await asyncio.sleep(min(0.05, remaining))

            process.join(timeout=0)
            if process.exitcode != 0:
                if fail_on_library_error:
                    raise LibraryVerificationError("worker_exit")
                return 0.0, None
            try:
                if not result_connection.poll():
                    if fail_on_library_error:
                        raise LibraryVerificationError("pipe_missing")
                    return 0.0, None
                packet = result_connection.recv()
                return _decode_strict_library_packet(packet) if fail_on_library_error else packet
            except EOFError:
                if fail_on_library_error:
                    raise LibraryVerificationError("pipe_eof") from None
                return 0.0, None
            except OSError:
                if fail_on_library_error:
                    raise LibraryVerificationError("pipe_error") from None
                raise
        finally:
            with contextlib.suppress(OSError, AssertionError):
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1.0)
            result_connection.close()

    async def _verify_answer_with_judge(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, list[JudgeEvaluation]]:
        # The judge is asked to evaluate whether the answers are equal using both
        # orders of the answers, in case there is any positional bias in terms of
        # the order in which the answers are presented to the judge model.
        (
            first_order_equal,
            first_judge_evaluation,
        ) = await self._generate_judge_evaluation(question, expected_answer, generated_answer)
        if not first_order_equal:
            return 0.0, [first_judge_evaluation]

        (
            second_order_equal,
            second_judge_evaluation,
        ) = await self._generate_judge_evaluation(question, generated_answer, expected_answer)
        if second_order_equal:
            reward = 1.0
        else:
            reward = 0.0
        return reward, [first_judge_evaluation, second_judge_evaluation]

    async def _generate_judge_evaluation(
        self, question: str, first_answer: str, second_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        config = self.config
        responses_create_params = config.judge_responses_create_params.model_copy(deep=True)

        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question, first_answer=first_answer, second_answer=second_answer
        )
        responses_create_params.input = [
            NeMoGymEasyInputMessage(
                role="system",
                content=self.JUDGE_SYSTEM_MESSAGE,
            ),
            NeMoGymEasyInputMessage(
                role="user",
                content=judge_prompt,
            ),
        ]

        judge_response = await call_judge(
            self.server_client,
            server_name=config.judge_model_server.name,
            url_path="/v1/responses",
            json=responses_create_params,
            response_model=NeMoGymResponse,
        )
        judge_evaluation = JudgeEvaluation(responses_create_params=responses_create_params, response=judge_response)

        if self.config.fail_on_judge_error:
            if getattr(judge_response, "status", "completed") != "completed":
                raise JudgeError("Math judge response is incomplete")
            try:
                final = judge_response.output[-1]
                content = final.content[-1]
                text = content.text
                valid = final.type == "message" and content.type == "output_text"
            except (AttributeError, IndexError, TypeError):
                raise JudgeError("Math judge output is empty or malformed") from None
            if not valid or not isinstance(text, str):
                raise JudgeError("Math judge output is not final text")
            labels = (self.JUDGE_EQUAL_LABEL in text, self.JUDGE_NOT_EQUAL_LABEL in text)
            if sum(labels) != 1:
                raise JudgeError("Math judge verdict is missing or conflicting")

        # Currently, for all the cases in which the response from the LLM judge
        # does not conform to the expected format, the judge's evaluation is
        # treated as if the answers are not equal.  This may not be ideal, but it
        # is intended to minimize the number of failures for verify requests.
        last_output = judge_response.output[-1]
        if last_output.type != "message":
            return False, judge_evaluation

        last_content = last_output.content[-1]
        if last_content.type != "output_text":
            return False, judge_evaluation

        output_text = last_content.text
        equal_choice_position = output_text.find(self.JUDGE_EQUAL_LABEL)
        not_equal_choice_position = output_text.find(self.JUDGE_NOT_EQUAL_LABEL)

        # The first label that appears in the text is used for the evaluation.
        if equal_choice_position < 0:
            if not_equal_choice_position < 0:
                return False, judge_evaluation
            else:
                return False, judge_evaluation
        else:
            if not_equal_choice_position < 0:
                return True, judge_evaluation
            elif equal_choice_position < not_equal_choice_position:
                return True, judge_evaluation
            else:
                return False, judge_evaluation

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics overrides
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _math_score_fn(r: dict) -> Dict[str, Union[float, bool]]:
        scores: Dict[str, Union[float, bool]] = {}
        if "library_reward" in r:
            scores["symbolic_accuracy"] = r["library_reward"]
        if "judge_evaluations" in r and r["judge_evaluations"] is not None:
            scores["judge_accuracy"] = r["reward"]
        return scores

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute math-specific metrics: pass@k, majority@k, per-sample statistics."""
        return compute_pass_majority_metrics(
            tasks,
            score_fn=self._math_score_fn,
            answer_key="extracted_answer",
        )[0]

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Select headline metrics for this math benchmark."""
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", exclude_names=["no_answer"]))

        return key


if __name__ == "__main__":
    LibraryJudgeMathResourcesServer.run_webserver()
