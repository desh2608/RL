"""Controlled scheduling of the actual GenRM method ASTs, without model calls.

Only schemas, judge I/O and the aggregation-input spy are substituted. This
tests completion, exception and cancellation semantics, not numerical reward
math, HTTP or Pydantic. Normal pytest and unittest discovery are supported.
GENRM_JIT_SOURCE may point at an isolated directory containing app.py/utils.py.
"""
from __future__ import annotations

import ast
import asyncio
import itertools
import os
import random
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(os.environ.get("GENRM_JIT_SOURCE", Path(__file__).resolve().parents[1]))
METHODS = {
    "verify", "_verify_impl", "_abort_cohort", "_get_verify_cohort_key",
    "_run_jit_compare_using_most_recent_response_obj",
}


async def settled(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 3)


class Harness:
    def __init__(self, k=8, *, strict=True, strategy="circular", judges=1):
        tree = ast.parse((SOURCE / "app.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "GenRMCompareResourcesServer")
        methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name in METHODS]
        assert {n.name for n in methods} == METHODS
        utility = ast.parse((SOURCE / "utils.py").read_text())
        pairs = next(n for n in utility.body if isinstance(n, ast.FunctionDef)
                     and n.name == "generate_comparison_pairs")
        self.calls = []
        self.events = {}
        self.behavior = {}
        self.active = set()
        self.finished = set()
        self.cancelled = set()
        self.tasks = []
        self.aggregations = []
        self.bad_reward_count = False
        self.aggregate_error = False
        self.buffers = defaultdict(list)
        self.jit = defaultdict(list)
        self.failures = {}
        self.ns = {
            "asyncio": asyncio, "itertools": itertools,
            "SUPPORTED_COMPARISON_STRATEGIES": {"circular", "all_pairs"},
            "_cohort_failures": self.failures, "_cohort_buffers": self.buffers,
            "_cohort_jit_tasks": self.jit,
            "BaseVerifyResponse": lambda **kw: SimpleNamespace(**kw),
            "_input_to_conversation_history": lambda messages: messages,
            "get_prompt_key_from_input": lambda messages, principle: "same-rendered-prompt",
            "aggregate_scores": self.aggregate,
        }
        module = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            pairs,
            ast.ClassDef(name="ActualServer", bases=[], keywords=[], body=methods, decorator_list=[]),
        ], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(SOURCE / "app.py"), "exec"), self.ns)
        self.server = self.ns["ActualServer"]()
        self.server.config = SimpleNamespace(
            fail_on_error=strict, num_rollouts_per_prompt=k,
            comparison_strategy=strategy, num_judges_per_comparison=judges,
            default_score=3., aggregator_method="simple_tiebreaker", reasoning_bonus=.5,
            answer_bonus=.5, top_percentile=.2, group_reasoning_length_penalty_coeff=.12,
            group_answer_length_penalty_coeff=.12, group_style_penalty_coeff=0.,
        )
        self.server._run_single_comparison = self.judge
        self.k = k
        self.expected_pairs = self.ns["generate_comparison_pairs"](strategy, k) if k > 1 else []
        self.judges = judges

    def aggregate(self, **kwargs):
        self.aggregations.append(kwargs)
        if self.aggregate_error:
            raise RuntimeError("controlled aggregate failure")
        n = len(kwargs["response_objs"])
        if self.bad_reward_count:
            n -= 1
        # Observable, index-specific sentinel, deliberately not a numeric oracle.
        return [100. + i for i in range(n)], {}, {}, {}

    async def judge(self, history, a, b, *, pair_idx, principle):
        self.assert_body_pair(a, b, pair_idx)
        cohort = a["cohort"]
        serial = sum(1 for c in self.calls if c[:2] == (cohort, pair_idx))
        key = (cohort, pair_idx, serial)
        self.calls.append(key)
        self.active.add(key)
        self.events[key] = asyncio.Event()
        try:
            await self.events[key].wait()
            action = self.behavior.get(key)
            if action == "error":
                raise RuntimeError("controlled pair failure")
            if action == "cancel":
                raise asyncio.CancelledError()
            if action == "legacy_default":
                return (3., 3., 3.5)
            return (float(pair_idx[0] + 1), float(pair_idx[1] + 1), 3.5)
        except asyncio.CancelledError:
            self.cancelled.add(key)
            # Cancellation must allow child cleanup to run, not double-cancel it.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            raise
        finally:
            self.active.remove(key)
            self.finished.add(key)

    @staticmethod
    def assert_body_pair(a, b, pair):
        assert a["cohort"] == b["cohort"]
        assert (a["index"], b["index"]) == pair

    def body(self, cohort, index):
        return SimpleNamespace(
            task_index=cohort, prompt_id=None, principle="same",
            responses_create_params=SimpleNamespace(input=[{"role": "user", "content": "same"}]),
            response={"cohort": cohort, "index": index},
        )

    @staticmethod
    def key(cohort):
        return f"task_idx::{cohort}::same-rendered-prompt"

    async def arrive(self, cohort, index):
        task = asyncio.create_task(self.server.verify(self.body(cohort, index)))
        self.tasks.append(task)
        await settled(lambda: len(self.buffers.get(self.key(cohort), [])) == index + 1 or task.done())
        if index:
            expected = sum(max(pair) <= index for pair in self.expected_pairs) * self.judges
            await settled(lambda: sum(c[0] == cohort for c in self.calls) == expected or task.done())
        return task

    async def start(self, cohort=0):
        return [await self.arrive(cohort, i) for i in range(self.k)]

    def release(self, cohort=None, pair=None):
        for key, event in self.events.items():
            if (cohort is None or key[0] == cohort) and (pair is None or key[1] == pair):
                event.set()

    async def values(self, tasks=None):
        return await asyncio.wait_for(asyncio.gather(*(tasks or self.tasks), return_exceptions=True), 3)

    async def close(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        # A failing test must not strand its controlled I/O.
        self.release()
        if self.active:
            await settled(lambda: not self.active)


class JitCohortTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.harnesses = []
        self.loop_errors = []
        self.loop = asyncio.get_running_loop()
        self.old_handler = self.loop.get_exception_handler()
        self.loop.set_exception_handler(lambda loop, ctx: self.loop_errors.append(ctx))

    async def asyncTearDown(self):
        try:
            for harness in self.harnesses:
                await harness.close()
            await asyncio.sleep(0)
            self.assertEqual(self.loop_errors, [], "Unretrieved task/future error")
        finally:
            self.loop.set_exception_handler(self.old_handler)

    def harness(self, *args, **kwargs):
        h = Harness(*args, **kwargs)
        self.harnesses.append(h)
        return h

    def assert_success(self, h, values, cohort=0):
        self.assertEqual([v.reward for v in values], [100. + i for i in range(h.k)])
        calls = [c for c in h.aggregations if c["response_objs"][0]["cohort"] == cohort]
        self.assertEqual(len(calls), 1)
        expected = sorted([(i, j, judge) for i, j in h.expected_pairs for judge in range(h.judges)],
                          key=lambda x: (x[2], x[0], x[1]))
        self.assertEqual(list(calls[0]["comparison_metadata"]), expected)
        self.assertEqual(len(calls[0]["comparison_results"]), len(expected))
        self.assertNotIn(h.key(cohort), h.buffers)
        self.assertNotIn(h.key(cohort), h.jit)
        self.assertNotIn(h.key(cohort), h.failures)
        self.assertFalse(any(key[0] == cohort for key in h.active))

    async def assert_failed_and_drained(self, h):
        values = await h.values()
        self.assertTrue(all(isinstance(v, BaseException) for v in values))
        self.assertEqual(h.aggregations, [])
        self.assertEqual(h.buffers, {})
        self.assertEqual(h.jit, {})
        self.assertEqual(h.active, set())
        self.assertIn(h.key(0), h.failures)
        with self.assertRaisesRegex(RuntimeError, "previously failed"):
            await h.server.verify(h.body(0, 0))

    async def test_ordered_completion_k4_and_k8(self):
        for k in (4, 8):
            with self.subTest(k=k):
                h = self.harness(k)
                for i in range(k):
                    await h.arrive(0, i)
                    h.release()
                    expected = sum(max(p) <= i for p in h.expected_pairs)
                    await settled(lambda: len(h.finished) == expected)
                self.assert_success(h, await h.values())

    async def test_last_two_first_never_aggregate_early_k4_and_k8(self):
        for k in (4, 8):
            with self.subTest(k=k):
                h = self.harness(k)
                await h.start()
                futures = [f for _, f in h.buffers[h.key(0)]]
                h.release(pair=(k - 2, k - 1))
                h.release(pair=(k - 1, 0))
                await settled(lambda: len(h.finished) == 2)
                for _ in range(12):
                    await asyncio.sleep(0)
                self.assertEqual(h.aggregations, [])
                self.assertFalse(any(f.done() for f in futures))
                self.assertFalse(any(t.done() for t in h.tasks))
                h.release()
                self.assert_success(h, await h.values())

    async def test_random_completion_orders(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                h = self.harness(8)
                await h.start()
                order = list(h.events)
                random.Random(seed).shuffle(order)
                for count, key in enumerate(order, 1):
                    h.events[key].set()
                    await settled(lambda: len(h.finished) == count)
                    if count < 8:
                        self.assertEqual(h.aggregations, [])
                self.assert_success(h, await h.values())

    async def test_cohorts_progress_independently(self):
        h = self.harness(8)
        a = await h.start(0)
        b = await h.start(1)
        h.release(cohort=1)
        self.assert_success(h, await h.values(b), cohort=1)
        self.assertFalse(any(t.done() for t in a))
        self.assertEqual(sum(key[0] == 0 for key in h.active), 8)
        h.release(cohort=0)
        self.assert_success(h, await h.values(a))

    async def test_pair_error_before_last_arrival_wakes_waiters(self):
        h = self.harness(8)
        await h.arrive(0, 0)
        await h.arrive(0, 1)
        h.behavior[(0, (0, 1), 0)] = "error"
        h.release()
        await self.assert_failed_and_drained(h)

    async def test_final_pair_error_cancels_and_joins_its_sibling_and_prior_pairs(self):
        h = self.harness(8)
        await h.start()
        h.behavior[(0, (6, 7), 0)] = "error"
        h.release(pair=(6, 7))
        await self.assert_failed_and_drained(h)
        self.assertIn((0, (7, 0), 0), h.cancelled)
        self.assertEqual(len(h.finished), 8)

    async def test_pair_cancellation_aborts_all_waiters(self):
        h = self.harness(8)
        await h.start()
        h.behavior[(0, (6, 7), 0)] = "cancel"
        h.release(pair=(6, 7))
        await self.assert_failed_and_drained(h)
        self.assertEqual(len(h.finished), 8)

    async def test_waiter_cancellation_aborts_partial_cohort(self):
        h = self.harness(8)
        await h.arrive(0, 0)
        await h.arrive(0, 1)
        h.tasks[0].cancel()
        await self.assert_failed_and_drained(h)

    async def test_final_caller_cancellation_cancels_registered_tasks(self):
        h = self.harness(8)
        await h.start()
        h.release(pair=(6, 7))
        h.release(pair=(7, 0))
        await settled(lambda: len(h.finished) == 2)
        for _ in range(12):
            await asyncio.sleep(0)
        self.assertEqual(h.aggregations, [])
        h.tasks[-1].cancel()
        await self.assert_failed_and_drained(h)
        self.assertEqual(len(h.finished), 8)

    async def test_failed_cohort_does_not_cancel_an_independent_cohort(self):
        h = self.harness(8)
        a = await h.start(0)
        b = await h.start(1)
        h.behavior[(0, (6, 7), 0)] = "error"
        h.release(cohort=0, pair=(6, 7))
        self.assertTrue(all(isinstance(v, BaseException) for v in await h.values(a)))
        self.assertFalse(any(t.done() for t in b))
        self.assertNotIn(h.key(1), h.failures)
        self.assertEqual(sum(key[0] == 1 for key in h.active), 8)
        h.release(cohort=1)
        self.assert_success(h, await h.values(b), cohort=1)

    async def test_all_pairs_multi_judge_keeps_pair_schedule(self):
        h = self.harness(4, strategy="all_pairs", judges=2)
        await h.start()
        self.assertEqual(len(h.calls), 12)
        h.release()
        self.assert_success(h, await h.values())

    async def test_legacy_default_tuples_and_single_response_remain_usable(self):
        h = self.harness(4, strict=False)
        await h.start()
        h.behavior = {key: "legacy_default" for key in h.events}
        h.release()
        self.assert_success(h, await h.values())
        self.assertEqual(list(h.aggregations[0]["comparison_results"]), [(3., 3., 3.5)] * 4)
        single = self.harness(1, strict=False)
        self.assertEqual((await single.server.verify(single.body(0, 0))).reward, 3.)
        strict_single = self.harness(1)
        with self.assertRaises(ValueError):
            await strict_single.server.verify(strict_single.body(0, 0))

    async def test_legacy_unexpected_failure_does_not_strand_peers(self):
        h = self.harness(4, strict=False)
        await h.start()
        h.behavior[(0, (2, 3), 0)] = "error"
        h.release(pair=(2, 3))
        await self.assert_failed_and_drained(h)

    async def test_missing_or_duplicate_pair_metadata_fails_closed(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                h = self.harness(4)
                original = h.server._run_jit_compare_using_most_recent_response_obj
                async def corrupt(history, responses, seen, principle):
                    results, metadata = await original(history, responses, seen, principle)
                    if len(responses) == 4:
                        if missing:
                            return results[:-1], metadata[:-1]
                        return results + results[:1], metadata + metadata[:1]
                    return results, metadata
                h.server._run_jit_compare_using_most_recent_response_obj = corrupt
                await h.start()
                h.release()
                await self.assert_failed_and_drained(h)

    async def test_stale_abort_cannot_cancel_a_new_cohort_with_same_key(self):
        h = self.harness(4)
        old = await h.start()
        old_buffer = h.buffers[h.key(0)]
        h.release()
        self.assert_success(h, await h.values(old))
        # A completed caller can only own the old list identity.
        fresh = asyncio.create_task(h.server.verify(h.body(0, 0)))
        h.tasks.append(fresh)
        await settled(lambda: h.key(0) in h.buffers)
        await h.server._abort_cohort(h.key(0), old_buffer)
        self.assertNotIn(h.key(0), h.failures)
        self.assertFalse(fresh.done())
        fresh.cancel()
        await asyncio.gather(fresh, return_exceptions=True)

    async def test_aggregate_failure_or_bad_count_wakes_every_waiter(self):
        for bad_count in (True, False):
            with self.subTest(bad_count=bad_count):
                h = self.harness(4)
                h.bad_reward_count = bad_count
                h.aggregate_error = not bad_count
                await h.start()
                h.release()
                values = await h.values()
                self.assertTrue(all(isinstance(v, RuntimeError) for v in values))
                self.assertEqual(h.buffers, {})
                self.assertEqual(h.jit, {})
                self.assertEqual(h.active, set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
