#!/usr/bin/env python3
"""Summarize the pivot-pool index written by scripts/pivot_data/index_pivot_pools.py.

Outputs (under --out):
  PIVOT_POOLS.json      per-pool counts (gold class, pivot position, domains, difficulty priors, prompt size),
                        cross-pool identity overlap (cond_sha), overlap with an existing bundle, HF399 stage-0 join
  candidates.jsonl      one line per unique prompt (cond_sha) with the chosen source row key, pool, tier and features
"""
import argparse, collections, json, os, re, statistics, sys

TAU2_DOMAIN_RE = re.compile(r'(airline|flight|aviation|travel|hotel|resort|lodging|hospitality|booking|reservation|retail|e-?commerce|'
                            r'online store|order|shipping|delivery|courier|logistics|parcel|telecom|mobile|wireless|cellular|internet|isp|'
                            r'broadband|phone|sim\b|roaming|subscription|bank|insurance|utility|utilities|marketplace|brokerage|'
                            r'rental|car rental|ticket|cruise|rail|train|bus)', re.I)
TAU2_ACTION_RE = re.compile(r'(reservation|booking|book_|flight|baggage|passenger|certificate|order|return|exchange|refund|cancel|'
                            r'modify|update_.*address|payment|roaming|refuel|data|line|plan|sim|transfer_to_human|calculate)', re.I)


def pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def quant(xs):
    if not xs:
        return None
    xs = sorted(xs)
    return {'p50': xs[len(xs) // 2], 'p90': xs[int(len(xs) * 0.9)], 'max': xs[-1], 'mean': round(statistics.mean(xs), 1)}


def hist_prior(xs):
    h = collections.Counter()
    for x in xs:
        if x is None:
            h['missing'] += 1
        elif x <= 0:
            h['0'] += 1
        elif x >= 1:
            h['1'] += 1
        else:
            h['(0,1)'] += 1
    return dict(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default='data/tau2-src/pivot_index')
    ap.add_argument('--out', default='analysis/tau2-data-20261004')
    ap.add_argument('--bundle', default='data/static-v5/train.jsonl', help='existing bundle to test overlap against (cond_sha of [input, tools])')
    ap.add_argument('--stage0', default='analysis/passrate-profile-20260930/per_prompt.jsonl')
    ap.add_argument('--max-prompt-chars', type=int, default=160000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for pool in ('super35', 'union', 'public'):
        p = os.path.join(a.index, f'{pool}.jsonl')
        with open(p) as f:
            for line in f:
                rows.append(json.loads(line))
    print(f'loaded {len(rows)} index rows', file=sys.stderr)

    # bundle overlap
    bundle_sha = set()
    bundle_route_by_sha = {}
    if a.bundle and os.path.exists(a.bundle):
        import hashlib
        with open(a.bundle) as f:
            for line in f:
                r = json.loads(line)
                rcp = r.get('responses_create_params') or {}
                s = hashlib.sha256(json.dumps([rcp.get('input') or [], rcp.get('tools') or []], sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
                bundle_sha.add(s)
                bundle_route_by_sha[s] = (r.get('agent_ref') or {}).get('name')
        print(f'bundle prompts {len(bundle_sha)}', file=sys.stderr)

    # stage-0 HF399 profile join (super35 raw line sha)
    stage0 = {}
    if a.stage0 and os.path.exists(a.stage0):
        for line in open(a.stage0):
            r = json.loads(line)
            if r.get('agent') == 'single_step_tool_use_with_argument_comparison_agent' and r.get('raw_line_sha256'):
                stage0[r['raw_line_sha256']] = r

    # family of each row
    def family(r):
        if r['domain'].startswith('tau_mms:'):
            return 'tau_mms'
        sid = str(r['source_id'])
        if 'Function-Calling-Pivot' in sid:
            return 'fc_pivot'
        if 'Conversational-Tool-Use-Pivot' in sid or sid == 'hf-ctu-pivot-v1' or sid.startswith('single_step_tool_use') or sid.startswith('tau_pivot'):
            return 'ctu'
        if 'evaluator-mixed-rewards-pivot' in sid:
            return 'tau_mms'
        return 'other:' + sid[:40]

    for r in rows:
        r['family'] = family(r)
        r['in_bundle'] = r['cond_sha'] in bundle_sha
        r['tau2_domain_hit'] = bool(TAU2_DOMAIN_RE.search(r['domain'])) if not r['domain'].startswith('tau_mms') else True
        r['tau2_action_hit'] = bool(r['gold_name'] and TAU2_ACTION_RE.search(r['gold_name']))
        r['oversize'] = r['prompt_chars'] > a.max_prompt_chars
        st = stage0.get(r.get('raw_line_sha256')) if r['pool'] == 'super35' else None
        r['hf399_pass'] = st['pass_rate'] if st else None

    out = {'index_rows': len(rows), 'bundle': a.bundle, 'bundle_prompts': len(bundle_sha), 'pools': {}, 'families': {}}

    def summarize(sub, with_domains=True):
        d = {
            'rows': len(sub),
            'unique_prompts': len({r['cond_sha'] for r in sub}),
            'gold_type': dict(collections.Counter(r['gold_type'] for r in sub)),
            'gold_class': dict(collections.Counter(r['gold_class'] for r in sub)),
            'pivot_after': dict(collections.Counter(r['last_type'] for r in sub)),
            'gold_class_x_after': {f"{r['gold_class']}|{r['last_type']}": c for (r, c) in []},
            'prev_call_error_rows': sum(r['prev_call_error'] for r in sub),
            'in_bundle_rows': sum(r['in_bundle'] for r in sub),
            'oversize_rows': sum(r['oversize'] for r in sub),
            'prompt_chars': quant([r['prompt_chars'] for r in sub]),
            'nmsg': quant([r['nmsg'] for r in sub]),
            'n_tools': quant([r['n_tools'] for r in sub]),
            'n_prior_calls': quant([r['n_fc'] for r in sub]),
            'pass_rate_prior': hist_prior([r['pass_rate'] for r in sub]),
            'qwen_prior': hist_prior([r['qwen_mean'] for r in sub]),
            'tau2_domain_hit_rows': sum(r['tau2_domain_hit'] for r in sub),
            'tau2_action_hit_rows': sum(r['tau2_action_hit'] for r in sub),
            'source_ids': dict(collections.Counter(str(r['source_id'])[:70] for r in sub).most_common(12)),
        }
        cx = collections.Counter(f"{r['gold_class']}|{r['last_type']}" for r in sub)
        d['gold_class_x_after'] = dict(cx)
        if with_domains:
            d['domains_top'] = dict(collections.Counter(r['domain'] for r in sub).most_common(40))
            d['domains_unique'] = len({r['domain'] for r in sub})
            d['policies_unique'] = len({r['sys_sha'] for r in sub})
        hp = [r['hf399_pass'] for r in sub if r['hf399_pass'] is not None]
        if hp:
            d['hf399_stage0'] = {'prompts': len(hp), 'mean': round(statistics.mean(hp), 3), 'hist': hist_prior(hp)}
            for key in ('gold_type', 'gold_class', 'last_type'):
                g = collections.defaultdict(list)
                for r in sub:
                    if r['hf399_pass'] is not None:
                        g[r[key]].append(r['hf399_pass'])
                d['hf399_stage0'][key] = {k: {'n': len(v), 'mean': round(statistics.mean(v), 3), 'zero_var': sum(1 for x in v if x in (0.0, 1.0))} for k, v in g.items()}
            g = collections.defaultdict(list)
            for r in sub:
                if r['hf399_pass'] is not None:
                    g[r['domain'] if r['family'] == 'tau_mms' else r['family']].append(r['hf399_pass'])
            d['hf399_stage0']['domain'] = {k: {'n': len(v), 'mean': round(statistics.mean(v), 3)} for k, v in sorted(g.items())}
        return d

    for pool in ('super35', 'union', 'public'):
        sub = [r for r in rows if r['pool'] == pool]
        out['pools'][pool] = summarize(sub)
        out['pools'][pool]['by_family'] = {fam: summarize([r for r in sub if r['family'] == fam], with_domains=(fam == 'tau_mms'))
                                           for fam in sorted({r['family'] for r in sub})}

    # cross-pool identity
    by_sha = collections.defaultdict(list)
    for r in rows:
        by_sha[r['cond_sha']].append(r)
    combo = collections.Counter(tuple(sorted({r['pool'] for r in v})) for v in by_sha.values())
    out['cross_pool_identity'] = {'unique_prompts_all_pools': len(by_sha), 'pool_combinations': {'+'.join(k): v for k, v in combo.items()}}
    fam_combo = collections.Counter((v[0]['family'], tuple(sorted({r['pool'] for r in v}))) for v in by_sha.values())
    out['cross_pool_identity']['by_family'] = {f"{k[0]}|{'+'.join(k[1])}": v for k, v in sorted(fam_combo.items())}

    # unique-prompt candidates: pick the richest source row per cond_sha (prefer one with pass_rate, then public K32)
    def rank(r):
        return (r['pass_rate'] is not None, (r['pass_rate_total'] or 0), r['qwen_mean'] is not None, r['pool'] == 'super35')
    cands = []
    for sha, v in by_sha.items():
        best = max(v, key=rank)
        pr = [r['pass_rate'] for r in v if r['pass_rate'] is not None]
        qm = [r['qwen_mean'] for r in v if r['qwen_mean'] is not None]
        c = {k: best[k] for k in ('pool', 'key', 'source_id', 'cond_sha', 'sys_sha', 'domain', 'family', 'gold_type', 'gold_name', 'gold_class',
                                   'last_type', 'prev_call_error', 'nmsg', 'n_user', 'n_fc', 'n_tools', 'prompt_chars', 'trajectory_id', 'task_id',
                                   'pivot_assistant_index', 'traj_actions', 'in_bundle', 'tau2_domain_hit', 'tau2_action_hit', 'oversize', 'hf399_pass')}
        c['pass_rate'] = pr[0] if pr else None
        c['pass_rate_total'] = best['pass_rate_total']
        c['qwen_mean'] = qm[0] if qm else None
        c['pools'] = sorted({r['pool'] for r in v})
        c['n_copies'] = len(v)
        cands.append(c)
    with open(os.path.join(a.out, 'candidates.jsonl'), 'w') as w:
        for c in cands:
            w.write(json.dumps(c, ensure_ascii=False) + '\n')

    # family-level summaries over unique prompts
    for fam in sorted({c['family'] for c in cands}):
        sub = [c for c in cands if c['family'] == fam]
        d = summarize(sub, with_domains=(fam == 'tau_mms'))
        fresh = [c for c in sub if not c['in_bundle'] and not c['oversize']]
        d['fresh_not_in_bundle_not_oversize'] = {
            'unique_prompts': len(fresh),
            'gold_class': dict(collections.Counter(c['gold_class'] for c in fresh)),
            'gold_class_x_after': dict(collections.Counter(f"{c['gold_class']}|{c['last_type']}" for c in fresh)),
            'prior_available': sum(1 for c in fresh if c['pass_rate'] is not None or c['qwen_mean'] is not None),
            'prior_in_open_interval': sum(1 for c in fresh if (c['pass_rate'] is not None and 0 < c['pass_rate'] < 1) or (c['qwen_mean'] is not None and 0 < c['qwen_mean'] < 1)),
            'prior_all_zero_both': sum(1 for c in fresh if c['pass_rate'] == 0 and (c['qwen_mean'] in (0, None))),
            'prior_all_one_both': sum(1 for c in fresh if c['pass_rate'] == 1 and (c['qwen_mean'] in (1, None))),
            'tau2_domain_hit': sum(c['tau2_domain_hit'] for c in fresh),
            'tau2_action_hit': sum(c['tau2_action_hit'] for c in fresh),
            'write_after_tool_output': sum(1 for c in fresh if c['gold_class'] == 'write' and c['last_type'] == 'function_call_output'),
            'write_after_user': sum(1 for c in fresh if c['gold_class'] == 'write' and c['last_type'] == 'user'),
            'message_after_tool_output': sum(1 for c in fresh if c['gold_type'] == 'message' and c['last_type'] == 'function_call_output'),
            'message_after_user': sum(1 for c in fresh if c['gold_type'] == 'message' and c['last_type'] == 'user'),
            'transfer_gold': sum(1 for c in fresh if c['gold_name'] and 'transfer_to_human' in c['gold_name']),
        }
        if fam == 'tau_mms':
            d['fresh_by_domain'] = dict(collections.Counter(c['domain'] for c in fresh))
            d['tasks_unique'] = len({c['task_id'] for c in sub if c['task_id']})
        else:
            d['fresh_tau2_domain_top'] = dict(collections.Counter(c['domain'] for c in fresh if c['tau2_domain_hit']).most_common(50))
            d['fresh_gold_names_top'] = dict(collections.Counter(c['gold_name'] for c in fresh if c['gold_name']).most_common(50))
        out['families'][fam] = d

    with open(os.path.join(a.out, 'PIVOT_POOLS.json'), 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ('index_rows', 'bundle_prompts', 'cross_pool_identity')}, indent=1), file=sys.stderr)


if __name__ == '__main__':
    main()
