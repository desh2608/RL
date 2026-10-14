#!/usr/bin/env python3
"""Build a compact per-row index of the "history given, generate one step" (pivot) pools.

Pools (all read-only):
  super35   merged_20260928_super35_v43/sources/super35_v43/routes/single_step_tool_use_with_argument_comparison_agent.jsonl.gz
  union     union_20260917/no_env_needs_verifier_or_judge/agentic_static.jsonl.gz  (rows carry model_request + source_record)
  public    public/nemotron-rl-agentic-ctu-pivot-v1/train.jsonl                    (HF nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1)

One output line per source row with: pool, key, source_id, cond_sha (sha256 of canonical [input, tools] -> cross-pool
identity of the prompt), sys_sha (policy identity), domain, gold type/name, history shape, difficulty priors
(pass_rate / qwen reward mean), prompt size, write-like flag, and whether the prompt is already in a given bundle.
No model or tokenizer is executed; prompt size is in characters.
"""
import argparse, collections, gzip, hashlib, json, os, re, sys, time
from multiprocessing import Pool

LIB = '/lustre/fsw/portfolios/llmservice/users/cchen1/data/library/rl'
POOLS = {
    'super35': f'{LIB}/merged_20260928_super35_v43/sources/super35_v43/routes/single_step_tool_use_with_argument_comparison_agent.jsonl.gz',
    'union': f'{LIB}/union_20260917/no_env_needs_verifier_or_judge/agentic_static.jsonl.gz',
    'public': f'{LIB}/public/nemotron-rl-agentic-ctu-pivot-v1/train.jsonl',
}
WRITE_RE = re.compile(r'(cancel|modify|update|book|create|schedule|reschedule|process|refund|transfer|submit|place|change|exchange|return|issue|send|reset|enroll|withdraw|redeem|pay|add|remove|delete|set_|toggle|grant|revoke|file_|apply|request|register|activate|deactivate|suspend|renew|upgrade|downgrade|order|assign|approve|reject|escalate|dispute|close|open|initiate|replace|reserve|release|extend|waive|charge|credit|debit|lock|unlock|freeze|block|unblock|report|flag|mark|record|log_|post|generate|confirm|complete|finalize|terminate|stop|start|restart|disable|enable|configure|install|dispatch|ship|deliver|pickup|reorder|resend|reissue|rebook|rename|move|merge|split|convert|link|unlink|subscribe|unsubscribe|opt_|verify_|authenticate|validate)', re.I)
READ_RE = re.compile(r'^(get|find|search|list|lookup|look_up|check|retrieve|fetch|query|view|show|calculate|estimate|compare|describe|read|browse|count|summarize|track|locate)', re.I)
HDR_RE = re.compile(r'#+\s*\**([^\n*]{3,80}?)\**\s*$', re.M)
TIME_RE = re.compile(r'current (?:date and )?time', re.I)


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def text_of(c):
    if isinstance(c, list):
        return ' '.join(x.get('text', '') for x in c if isinstance(x, dict))
    return c or ''


GENERIC_HDR = {'domain basics', 'core rules', 'authentication requirements', 'agent capabilities', 'key policies', 'agent limitations',
               'general rules', 'policy', 'overview', 'introduction', 'core requirements', 'response boundaries', 'data handling',
               'tool usage', 'tools', 'rules', 'guidelines', 'scope', 'capabilities', 'key rules', 'authentication', 'basics'}


def domain_of(sys_text, rec):
    t = rec.get('tau_mms_info') or {}
    if t.get('domain'):
        return 'tau_mms:' + t['domain']
    m = re.search(r'<policy>\s*(.*?)<', sys_text, re.S)
    body = m.group(1) if m else sys_text
    for line in body.split('\n'):
        line = line.strip().strip('#').strip().strip('*').strip()
        if not line:
            continue
        if re.search(r'policy', line, re.I):
            d = re.sub(r'\s*(agent|chatbot|assistant|support|customer service|service)?\s*policy\s*:?\s*$', '', line, flags=re.I).strip(' :*-')
            if d:
                return d[:60]
        break
    for h in HDR_RE.finditer(sys_text[:3000]):
        d = re.sub(r'\s+', ' ', h.group(1)).strip().rstrip(':')
        d2 = re.sub(r'\s*(agent|chatbot|assistant|support|customer service)?\s*policy\s*$', '', d, flags=re.I).strip()
        if d2 and d2.lower() not in GENERIC_HDR and not TIME_RE.search(d2):
            return d2[:60]
    return 'unknown'


def classify(gold_name):
    if not gold_name:
        return 'none'
    if READ_RE.search(gold_name):
        return 'read'
    if WRITE_RE.search(gold_name):
        return 'write'
    return 'other'


def index_row(pool, key, rcp, rec, source_id, extra_cond=None):
    inp = rcp.get('input') or []
    tools = rcp.get('tools') or []
    types = []
    for m in inp:
        if not isinstance(m, dict):
            types.append('?'); continue
        if m.get('type') == 'function_call':
            types.append('function_call')
        elif m.get('type') == 'function_call_output':
            types.append('function_call_output')
        elif m.get('type') == 'reasoning':
            types.append('reasoning')
        elif m.get('role') == 'user':
            types.append('user')
        elif m.get('role') == 'assistant':
            types.append('assistant')
        elif m.get('role') == 'system' or m.get('role') == 'developer':
            types.append('system')
        else:
            types.append(m.get('type') or m.get('role') or '?')
    sys_text = ''
    for m in inp:
        if isinstance(m, dict) and m.get('role') in ('system', 'developer'):
            sys_text = text_of(m.get('content')); break
    if not sys_text and rcp.get('instructions'):
        sys_text = rcp['instructions']
    ea = rec.get('expected_action') or {}
    gold_type = ea.get('type') or 'absent'
    gold_name = ea.get('name') if gold_type == 'function_call' else None
    gold_args = ea.get('arguments') if gold_type == 'function_call' else None
    if isinstance(gold_args, dict):
        gold_args = json.dumps(gold_args, sort_keys=True)
    q = rec.get('qwen_235b_info') or {}
    tau = rec.get('tau_mms_info') or {}
    prev_call_error = False
    if types and types[-1] == 'function_call_output':
        out = str(inp[-1].get('output', ''))[:200].lower()
        prev_call_error = ('error' in out) or ('not found' in out) or ('invalid' in out)
    row = {
        'pool': pool, 'key': key, 'source_id': source_id,
        'cond_sha': sha([inp, tools]),
        'sys_sha': hashlib.sha256(sys_text.encode()).hexdigest()[:16],
        'domain': domain_of(sys_text, rec),
        'gold_type': gold_type, 'gold_name': gold_name, 'gold_class': classify(gold_name),
        'gold_args_chars': len(gold_args) if gold_args else 0,
        'nmsg': len(inp), 'n_user': types.count('user'), 'n_fc': types.count('function_call'),
        'n_reasoning': types.count('reasoning'), 'n_assistant_msgs': types.count('assistant'),
        'last_type': types[-1] if types else 'none', 'prev_call_error': prev_call_error,
        'n_tools': len(tools), 'prompt_chars': len(json.dumps(inp, ensure_ascii=False)) + len(json.dumps(tools, ensure_ascii=False)),
        'sys_chars': len(sys_text),
        'pass_rate': rec.get('pass_rate'), 'pass_rate_total': rec.get('pass_rate_total'),
        'qwen_mean': q.get('reward_mean'), 'qwen_n': len(q.get('rewards') or []) if q else None,
        'trajectory_id': rec.get('trajectory_id'), 'task_id': tau.get('task_id'),
        'pivot_assistant_index': tau.get('pivot_assistant_index'), 'traj_actions': tau.get('trajectory_num_assistant_actions'),
        'num_unique_actions': rec.get('num_unique_actions'),
        'dataset': (rec.get('dataset') or '').split('_profiled')[0] or None,
        'uuid': rec.get('uuid') or (rec.get('metadata') or {}).get('uuid'),
        'parallel_tool_calls': rcp.get('parallel_tool_calls'),
    }
    if extra_cond:
        row.update(extra_cond)
    return row


def iter_pool(pool, path, limit=None):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            if pool == 'union':
                sr = r.get('source_record') or {}
                rcp = sr.get('responses_create_params') or r.get('model_request') or {}
                src = r.get('selected_source') or {}
                sid = src.get('source_id') if isinstance(src, dict) else str(src)
                yield index_row(pool, r.get('candidate_id'), rcp, sr, sid, {'messages_tools_sha256': r.get('messages_tools_sha256'), 'tier': r.get('tier')})
            elif pool == 'super35':
                yield index_row(pool, r.get('uuid'), r.get('responses_create_params') or {}, r, r.get('dataset'),
                                {'ordinal_1based': i + 1, 'raw_line_sha256': hashlib.sha256(line.rstrip('\n').encode()).hexdigest()})
            else:
                yield index_row(pool, f'line:{i}', r.get('responses_create_params') or {}, r, 'hf-ctu-pivot-v1')


def run_pool(args):
    pool, path, out_dir, limit = args
    t0 = time.time(); n = 0
    out = os.path.join(out_dir, f'{pool}.jsonl')
    summ = collections.defaultdict(collections.Counter)
    with open(out + '.tmp', 'w') as w:
        for row in iter_pool(pool, path, limit):
            n += 1
            w.write(json.dumps(row, ensure_ascii=False) + '\n')
            summ['source_id'][str(row['source_id'])] += 1
            summ['gold_type'][row['gold_type']] += 1
            summ['gold_class'][row['gold_class']] += 1
            summ['last_type'][row['last_type']] += 1
            summ['domain_top'][row['domain']] += 1
    os.replace(out + '.tmp', out)
    return pool, n, time.time() - t0, {k: dict(v.most_common(60)) for k, v in summ.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pools', default='super35,union,public')
    ap.add_argument('--out', default='data/tau2-src/pivot_index')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pools = [p for p in a.pools.split(',') if p]
    jobs = [(p, POOLS[p], a.out, a.limit) for p in pools]
    with Pool(len(jobs)) as pp:
        results = pp.map(run_pool, jobs)
    summary = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'limit': a.limit, 'pools': {}}
    for pool, n, dt, s in results:
        summary['pools'][pool] = {'path': POOLS[pool], 'rows': n, 'seconds': round(dt, 1), 'counts': s}
        print(f'{pool}: {n} rows in {dt:.0f}s', file=sys.stderr)
    with open(os.path.join(a.out, 'SUMMARY.json' if not a.limit else f'SUMMARY.limit{a.limit}.json'), 'w') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
