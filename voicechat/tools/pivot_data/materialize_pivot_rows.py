#!/usr/bin/env python3
"""Materialize pivot rows selected from the index/candidates (scripts/pivot_data/index_pivot_pools.py + summarize_pivot_index.py)
into library-format JSONL pools that the legacy static-bundle builder (read-only reference: /lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/scripts/build_static_bundle.py; prefer pivot_data/clean_candidates.py + build_bundle.py) can take via --extra-jsonl.

  --select NAME=EXPR   python expression over the candidate dict `c` (one unique prompt per candidate); may repeat.
Each selected candidate is read from its source pool (super35 route gz / union tier gz / public jsonl) in one streaming
pass per pool. Output row = native record (responses_create_params, expected_action, agent_ref, uuid, provenance fields)
minus profiler arrays, plus `pivot_meta` (pool, key, family, domain, gold_class, pivot position, priors, in_bundle).
No model is executed; rows are byte-for-byte the source request/gold.
"""
import argparse, collections, gzip, hashlib, json, os, sys, time, uuid

from index_pivot_pools import POOLS  # same pool paths

DROP_KEYS = {'raw_rewards_and_lengths', '_ng_task_index', '_ng_rollout_index', 'input_tokens/mean', 'input_tokens/std', 'input_tokens/max',
             'input_tokens/min', 'output_tokens/mean', 'output_tokens/std', 'output_tokens/max', 'output_tokens/min', 'total_tokens/mean',
             'total_tokens/std', 'total_tokens/max', 'total_tokens/min'}
NS = uuid.NAMESPACE_URL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', default='analysis/tau2-data-20261004/candidates.jsonl')
    ap.add_argument('--select', action='append', required=True)
    ap.add_argument('--out', default='data/tau2-src/pools')
    ap.add_argument('--limit', type=int, default=None, help='debug: stop after this many source rows per pool')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    selects = []
    for s in a.select:
        name, expr = s.split('=', 1)
        selects.append((name, expr, compile(expr, name, 'eval')))
    want = collections.defaultdict(dict)  # pool -> key -> (cand, [names])
    counts = collections.Counter()
    with open(a.candidates) as f:
        for line in f:
            c = json.loads(line)
            names = [n for n, _, code in selects if eval(code, {}, {'c': c})]
            if names:
                want[c['pool']][c['key']] = (c, names)
                for n in names:
                    counts[n] += 1
    print('selected', dict(counts), 'by pool', {p: len(v) for p, v in want.items()}, file=sys.stderr)
    writers = {n: open(os.path.join(a.out, n + '.jsonl.tmp'), 'w') for n, _, _ in selects}
    written = collections.Counter(); stats = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    t0 = time.time()
    for pool, keys in want.items():
        path = POOLS[pool]
        opener = gzip.open if path.endswith('.gz') else open
        found = 0; done = set()
        with opener(path, 'rt') as f:
            for i, line in enumerate(f):
                if a.limit and i >= a.limit:
                    break
                if pool == 'public':
                    key = f'line:{i}'
                    if key not in keys:
                        continue
                    rec = json.loads(line)
                elif pool == 'super35':
                    rec = json.loads(line)
                    key = rec.get('uuid')
                    if key not in keys:
                        continue
                else:
                    rec0 = json.loads(line)
                    key = rec0.get('candidate_id')
                    if key not in keys:
                        continue
                    rec = dict(rec0.get('source_record') or {})
                    rec.setdefault('responses_create_params', rec0.get('model_request'))
                    rec['union_candidate_id'] = key
                    rec['union_source_id'] = (rec0.get('selected_source') or {}).get('source_id') if isinstance(rec0.get('selected_source'), dict) else rec0.get('selected_source')
                c, names = keys[key]
                rcp0 = rec.get('responses_create_params') or {}
                if hashlib.sha256(json.dumps([rcp0.get('input') or [], rcp0.get('tools') or []], sort_keys=True, ensure_ascii=False,
                                             separators=(',', ':')).encode()).hexdigest() != c['cond_sha']:
                    continue  # same key (Super35 uuids repeat across datasets) but a different prompt
                if key in done:
                    continue
                done.add(key)
                row = {k: v for k, v in rec.items() if k not in DROP_KEYS}
                row['agent_ref'] = {'type': 'responses_api_agents', 'name': 'single_step_tool_use_with_argument_comparison_agent'}
                if not row.get('uuid'):
                    row['uuid'] = (row.get('metadata') or {}).get('uuid') or str(uuid.uuid5(NS, 'tau2-pivot:' + c['cond_sha']))
                row['pivot_meta'] = {k: c.get(k) for k in ('pool', 'key', 'source_id', 'cond_sha', 'family', 'domain', 'gold_class', 'last_type', 'prev_call_error',
                                                            'pass_rate', 'pass_rate_total', 'qwen_mean', 'in_bundle', 'hf399_pass', 'tau2_domain_hit', 'tau2_action_hit', 'pools')}
                s = json.dumps(row, ensure_ascii=False)
                for n in names:
                    writers[n].write(s + '\n'); written[n] += 1
                    st = stats[n]
                    st['gold_type'][row['expected_action'].get('type', '?')] += 1
                    st['gold_class'][c['gold_class']] += 1
                    st['domain'][c['domain'] if c['family'] == 'tau_mms' else c['family']] += 1
                    st['pool'][pool] += 1
                    st['gold_class_x_after'][f"{c['gold_class']}|{c['last_type']}"] += 1
                found += 1
        print(f'{pool}: found {found}/{len(keys)} in {time.time() - t0:.0f}s', file=sys.stderr)
    for n, w in writers.items():
        w.close()
        final = os.path.join(a.out, n + '.jsonl'); os.replace(final + '.tmp', final)
        side = {'schema': 'pivot-pool-source-v1', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'name': n,
                'select': [e for nn, e, _ in selects if nn == n][0], 'candidates': a.candidates, 'rows': written[n],
                'sha256': hashlib.sha256(open(final, 'rb').read()).hexdigest(), 'bytes': os.path.getsize(final),
                'stats': {k: dict(v.most_common(60)) for k, v in stats[n].items()},
                'row_format': 'native library record minus profiler arrays; agent_ref reset to single_step_tool_use_with_argument_comparison_agent; pivot_meta added',
                'limits': ['pass_rate / qwen_mean are priors from other models (K32 / K16 / K4), not HF399 measurements',
                           'in_bundle = prompt identity ([input, tools]) already in data/static-v5/train.jsonl']}
        json.dump(side, open(os.path.join(a.out, n + '.SOURCE.json'), 'w'), indent=1, ensure_ascii=False)
        print(n, written[n], file=sys.stderr)


if __name__ == '__main__':
    main()
