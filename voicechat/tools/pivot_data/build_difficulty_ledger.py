#!/usr/bin/env python3
"""Per-prompt difficulty ledger from training runs (no model executed).

For each train run dir (runs/train-v1-*): the async collector logs every zero-variance group it drops as
  "Dropped zero-variance prompt group (agent=A, group_index=G, _ng_task_index=I, reward=R, valid_rows=K, ...)"
where I is the line index of the run's data copy (<run>/data.train.jsonl, data.shuffle false, consumed in order).
Groups that were NOT dropped had mixed rewards (informative). So for every consumed line we know one of:
  all_correct (reward 1.0 for all K rollouts)  |  all_wrong (0.0 for all K)  |  mixed
The consumed range is [min, max] of the dropped indices (a resumed segment starts mid-file). The policy that produced
the observation is identified by the approximate training step (index position / prompts_per_step + first step).

Outputs under --out:
  <run>.ledger.jsonl        one line per consumed data line: uuid, cond_sha, route, task_index, label, approx_step, run
  prompt_difficulty.jsonl   aggregate per uuid over all runs: observations, counts, first/last label, pass-rate prior
  easy_uuids.txt            uuid<TAB>reason for prompts observed all_correct (consumed by build_static_bundle --exclude-uuids)
  impossible_uuids.txt      uuid<TAB>reason for prompts observed all_wrong (kept separate: unlearnable now or gold/verifier mismatch)
  SUMMARY.json              per run / per route counts
"""
import argparse, collections, hashlib, json, os, re, sys, time

DROP_RE = re.compile(r'Dropped zero-variance prompt group \(agent=(\S+), group_index=(\d+), _ng_task_index=(\d+), reward=([0-9.]+), valid_rows=(\d+)')
STEP_RE = re.compile(r'Step (\d+)/(\d+)')


def cond_sha(rcp):
    return hashlib.sha256(json.dumps([rcp.get('input') or [], rcp.get('tools') or []], sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def short(route):
    return route.split('_single_step')[0].split('_simple')[0]


def scan_run(run, prompts_per_step, first_step, limit=None):
    drops = {}
    steps = []
    with open(os.path.join(run, 'driver.log'), errors='ignore') as f:
        for line in f:
            m = DROP_RE.search(line)
            if m:
                drops[int(m.group(3))] = (m.group(1), float(m.group(4)), int(m.group(5)))
            else:
                s = STEP_RE.search(line)
                if s:
                    steps.append(int(s.group(1)))
    if not drops:
        return [], {'drops': 0, 'note': 'no drop lines'}
    idx = sorted(drops)
    lo, hi = idx[0], idx[-1]
    if limit:
        hi = min(hi, lo + limit)
    rows = []
    with open(os.path.join(run, 'data.train.jsonl')) as f:
        for i, line in enumerate(f):
            if i < lo:
                continue
            if i > hi:
                break
            r = json.loads(line)
            route = (r.get('agent_ref') or {}).get('name')
            d = drops.get(i)
            if d:
                label = 'all_correct' if d[1] >= 1.0 else ('all_wrong' if d[1] <= 0.0 else 'uniform_%s' % d[1])
                if d[0] != route:
                    label += '|AGENT_MISMATCH'
            else:
                label = 'mixed'
            rows.append({'run': os.path.basename(run), 'task_index': i, 'uuid': r.get('uuid'), 'cond_sha': cond_sha(r.get('responses_create_params') or {}),
                         'route': route, 'gold_type': (r.get('expected_action') or {}).get('type', 'absent'), 'label': label,
                         'approx_step': first_step + (i - lo) // prompts_per_step, 'k': d[2] if d else None})
    info = {'drops': len(drops), 'consumed_range': [lo, hi], 'consumed_lines': len(rows), 'max_step_logged': max(steps) if steps else None,
            'all_correct': sum(1 for x in rows if x['label'] == 'all_correct'), 'all_wrong': sum(1 for x in rows if x['label'] == 'all_wrong'),
            'mixed': sum(1 for x in rows if x['label'] == 'mixed'), 'agent_mismatch': sum(1 for x in rows if 'MISMATCH' in x['label'])}
    return rows, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, metavar='DIR[:first_step]', help='train run dir; :first_step for resumed segments (default 1)')
    ap.add_argument('--prompts-per-step', type=int, default=512)
    ap.add_argument('--out', default='data/difficulty')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'runs': {}, 'by_route': {}}
    agg = collections.defaultdict(lambda: {'obs': [], 'route': None, 'cond_sha': None, 'gold_type': None})
    by_route = collections.defaultdict(collections.Counter)
    for spec in a.run:
        run, _, fs = spec.partition(':')
        first_step = int(fs) if fs else 1
        rows, info = scan_run(run.rstrip('/'), a.prompts_per_step, first_step, a.limit)
        summary['runs'][os.path.basename(run.rstrip('/'))] = info
        with open(os.path.join(a.out, os.path.basename(run.rstrip('/')) + '.ledger.jsonl'), 'w') as w:
            for x in rows:
                w.write(json.dumps(x) + '\n')
                by_route[short(x['route'] or '?')][x['label']] += 1
                g = agg[x['uuid']]
                g['obs'].append({'run': x['run'], 'step': x['approx_step'], 'label': x['label']})
                g['route'] = g['route'] or x['route']; g['cond_sha'] = g['cond_sha'] or x['cond_sha']; g['gold_type'] = g['gold_type'] or x['gold_type']
        print(os.path.basename(run), json.dumps(info), file=sys.stderr)
    summary['by_route'] = {r: dict(c) for r, c in by_route.items()}
    n_easy = n_imp = 0
    with open(os.path.join(a.out, 'prompt_difficulty.jsonl'), 'w') as w, open(os.path.join(a.out, 'easy_uuids.txt'), 'w') as we, open(os.path.join(a.out, 'impossible_uuids.txt'), 'w') as wi:
        we.write('# uuid\treason — observed all-correct at K16 in training (every rollout right; no gradient); exclude from new bundles (user 2026-10-05)\n')
        wi.write('# uuid\treason — observed all-wrong at K16 in training (no gradient now); review gold/verifier before reuse\n')
        for uid, g in agg.items():
            labels = collections.Counter(o['label'] for o in g['obs'])
            steps_ac = [o['step'] for o in g['obs'] if o['label'] == 'all_correct']
            steps_aw = [o['step'] for o in g['obs'] if o['label'] == 'all_wrong']
            rec = {'uuid': uid, 'route': g['route'], 'gold_type': g['gold_type'], 'cond_sha': g['cond_sha'], 'n_obs': len(g['obs']),
                   'all_correct': labels['all_correct'], 'all_wrong': labels['all_wrong'], 'mixed': labels['mixed'],
                   'first_label': g['obs'][0]['label'], 'last_label': g['obs'][-1]['label'],
                   'earliest_all_correct_step': min(steps_ac) if steps_ac else None, 'observations': g['obs']}
            w.write(json.dumps(rec) + '\n')
            if steps_ac:
                n_easy += 1
                we.write('%s\t%s\tall_correct@%s step%d%s\n' % (uid, short(g['route'] or '?'), g['obs'][0]['run'], min(steps_ac), ' x%d' % len(steps_ac) if len(steps_ac) > 1 else ''))
            elif steps_aw and not labels['mixed']:
                n_imp += 1
                wi.write('%s\t%s\tall_wrong@%s step%d%s\n' % (uid, short(g['route'] or '?'), g['obs'][0]['run'], min(steps_aw), ' x%d' % len(steps_aw) if len(steps_aw) > 1 else ''))
    summary['unique_prompts'] = len(agg); summary['easy_uuids'] = n_easy; summary['impossible_uuids'] = n_imp
    json.dump(summary, open(os.path.join(a.out, 'SUMMARY.json'), 'w'), indent=1)
    print(json.dumps({k: summary[k] for k in ('unique_prompts', 'easy_uuids', 'impossible_uuids')}), file=sys.stderr)


if __name__ == '__main__':
    main()
