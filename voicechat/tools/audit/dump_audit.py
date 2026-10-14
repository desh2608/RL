#!/usr/bin/env python3
"""Audit closed_trajectories/stepN.jsonl dumps (pure python): loss-mask layout, reward/advantage values, and the per-token
train-vs-rollout logprob mismatch (prev_logprobs = Megatron, generation_logprobs = vLLM) decomposed by agent, position,
think/answer region and token id; lists the worst tokens/samples.
Usage: dump_audit.py STEP.jsonl [...]"""
import collections
import json
import math
import sys

THINK_OPEN, THINK_CLOSE = 12, 13


def short(agent):
    return agent.replace("_single_step_tool_use_with_argument_comparison_agent", "").replace("single_step_tool_use_with_argument_comparison_agent", "single_step")


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def audit(path):
    out = {"file": path}
    layout = collections.Counter()
    rew = collections.defaultdict(collections.Counter)
    shaped_diff = collections.Counter()
    adv_const = collections.Counter()
    by_agent = collections.defaultdict(lambda: [0.0, 0.0, 0])  # sum|d|, sum exp|d|, n
    by_pos = collections.defaultdict(lambda: [0.0, 0.0, 0])
    by_region = collections.defaultdict(lambda: [0.0, 0.0, 0])
    by_tok = collections.defaultdict(lambda: [0.0, 0.0, 0])
    absd_all, worst_tokens, sample_err = [], [], []
    tot_abs = tot_exp = 0.0
    ntok = 0
    signed = 0.0
    for line in open(path):
        r = json.loads(line)
        ids, m, adv = r["token_ids"], r["token_loss_mask"], r["advantages"]
        g, p = r["generation_logprobs"], r["prev_logprobs"]
        L, IL = r.get("length"), r.get("input_length")
        idx = [i for i, v in enumerate(m) if v]
        agent = short(r["agent_ref"]["name"])
        # mask layout
        if not idx:
            layout["no_trainable_tokens"] += 1
        else:
            contiguous = idx[-1] - idx[0] + 1 == len(idx)
            layout["contiguous" if contiguous else "fragmented"] += 1
            layout["mask_ends_at_length" if idx[-1] + 1 == L else "mask_end!=length(%s)" % ("<" if idx[-1] + 1 < L else ">")] += 1
            layout["seq_len==input_length" if len(ids) == IL else "seq_len!=input_length"] += 1
            # first trainable token should be right after the prompt; prompt = tokens before idx[0]
            layout["gen_lp_zero_on_prompt" if all(g[i] == 0.0 for i in range(idx[0])) else "gen_lp_nonzero_on_prompt"] += 1
            layout["gen_lp_zero_inside_mask"] += sum(1 for i in idx if g[i] == 0.0)
        rew[agent][(r.get("unshaped_total_reward"), r.get("total_reward"), r.get("reward"))] += 1
        if r.get("unshaped_total_reward") != r.get("total_reward"):
            shaped_diff[agent] += 1
        a_vals = set(round(adv[i], 6) for i in idx)
        adv_const["const" if len(a_vals) <= 1 else "varies"] += 1
        # mismatch
        s_abs = s_exp = 0.0
        think_state = None
        open_pos = None
        for k, i in enumerate(idx):
            d = p[i] - g[i]
            ad = abs(d)
            e = math.exp(min(ad, 50))
            tot_abs += ad
            tot_exp += e
            signed += d
            ntok += 1
            s_abs += ad
            s_exp += e
            absd_all.append(ad)
            for key, acc in ((agent, by_agent), (min(k, 10) if k < 10 else ("10-99" if k < 100 else ("100-999" if k < 1000 else "1000+")), by_pos)):
                acc[key][0] += ad
                acc[key][1] += e
                acc[key][2] += 1
            tok = ids[i]
            # region: inside think (after <think> before </think>) vs answer
            if tok == THINK_OPEN:
                think_state = "think"
            region = think_state or "pre"
            if tok == THINK_CLOSE:
                think_state = "answer"
                region = "think_close"
            by_region[region][0] += ad
            by_region[region][1] += e
            by_region[region][2] += 1
            by_tok[tok][0] += ad
            by_tok[tok][1] += e
            by_tok[tok][2] += 1
            if ad > 2:
                worst_tokens.append((round(ad, 3), r["idx"], k, tok, round(g[i], 3), round(p[i], 3), agent))
        if idx:
            sample_err.append((s_exp / len(idx), s_abs / len(idx), len(idx), r["idx"], agent, r.get("truncated"), r.get("total_reward")))
    out["layout"] = dict(layout)
    out["rewards_by_agent"] = {a: {str(k): v for k, v in c.most_common(6)} for a, c in rew.items()}
    out["shaped_reward_differs"] = dict(shaped_diff)
    out["advantage_within_sample"] = dict(adv_const)
    out["mismatch"] = {"tokens": ntok, "mean_exp_absdiff": tot_exp / ntok, "exp_mean_absdiff": math.exp(tot_abs / ntok),
                       "mean_absdiff": tot_abs / ntok, "mean_signed(train-gen)": signed / ntok,
                       "p50": pct(absd_all, .5), "p90": pct(absd_all, .9), "p99": pct(absd_all, .99), "p999": pct(absd_all, .999),
                       "max": max(absd_all), "frac>0.5": sum(x > .5 for x in absd_all) / ntok, "frac>2": sum(x > 2 for x in absd_all) / ntok}
    # share of the excess (mean_exp - 1) coming from the tail
    excess = [math.exp(min(x, 50)) - 1 for x in absd_all]
    tot_ex = sum(excess)
    out["excess_share"] = {thr: sum(e for x, e in zip(absd_all, excess) if x > thr) / tot_ex for thr in (0.1, 0.5, 1, 2, 5)}
    f = lambda acc: {str(k): {"n": v[2], "mean_abs": round(v[0] / v[2], 4), "mean_exp": round(v[1] / v[2], 4)} for k, v in acc.items()}
    out["by_agent"] = f(by_agent)
    out["by_position"] = f(by_pos)
    out["by_region"] = f(by_region)
    top_tok = sorted(by_tok.items(), key=lambda kv: -(kv[1][1] - kv[1][2]))[:15]
    out["tokens_by_excess"] = [{"tok": t, "n": v[2], "mean_abs": round(v[0] / v[2], 3), "excess_sum": round(v[1] - v[2], 1)} for t, v in top_tok]
    worst_tokens.sort(reverse=True)
    out["worst_tokens(abs,idx,pos,tok,gen,train,agent)"] = worst_tokens[:15]
    sample_err.sort(reverse=True)
    out["worst_samples(mean_exp,mean_abs,ntok,idx,agent,trunc,reward)"] = [tuple(round(x, 3) if isinstance(x, float) else x for x in s) for s in sample_err[:10]]
    return out


if __name__ == "__main__":
    for pth in sys.argv[1:]:
        print(json.dumps(audit(pth), indent=1, default=str))
