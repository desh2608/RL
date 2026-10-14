#!/usr/bin/env python3
"""Aggregate stack-B tau2 rollouts: per agent (user-simulator variant) reward, per-task pass rate,
termination reasons, agent steps, tokens, duration. Stdlib only. Usage: aggregate_tau2.py --out DIR/stackB"""
import argparse, json, statistics, collections, os
ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
report={}
for agent in sorted(os.listdir(a.out)):
    f=f'{a.out}/{agent}/rollouts.jsonl'
    if not os.path.isfile(f): continue
    rows=[json.loads(l) for l in open(f) if l.strip()]
    if not rows: continue
    by_task=collections.defaultdict(list); term=collections.Counter(); steps=[]; ctoks=[]; ptoks=[]; dur=[]; agent_steps=[]
    for r in rows:
        tid=(r.get('task') or {}).get('id') or r.get('_ng_task_index'); by_task[tid].append(float(r.get('reward') or 0))
        res=r.get('result') or {}
        term[str(res.get('termination_reason'))]+=1
        for k,lst in (('num_steps',steps),('mean_completion_tokens',ctoks),('mean_prompt_tokens',ptoks),('duration',dur),('agent_steps',agent_steps)):
            v=r.get(k); 
            if isinstance(v,(int,float)): lst.append(v)
    prs=[statistics.mean(v) for v in by_task.values()]
    hist=collections.Counter('0' if p==0 else '1' if p==1 else '(0,1)' for p in prs)
    med=lambda x: round(statistics.median(x),1) if x else None
    report[agent]={'rollouts':len(rows),'tasks':len(by_task),'reward_mean':round(statistics.mean(r for v in by_task.values() for r in v),4),
        'task_passrate_hist':dict(hist),'pass_at_k_any':round(sum(1 for v in by_task.values() if max(v)>0)/len(by_task),3),
        'termination_reasons':dict(term.most_common()),'median_num_messages':med(steps),'median_agent_steps':med(agent_steps),
        'median_completion_tokens_per_call':med(ctoks),'median_prompt_tokens_per_call':med(ptoks),'median_duration_s':med(dur)}
    fails=f'{a.out}/{agent}/rollouts_failures.jsonl'
    report[agent]['failure_rows']=sum(1 for _ in open(fails)) if os.path.isfile(fails) else 0
json.dump(report,open(f'{a.out}/SUMMARY.json','w'),indent=1)
lines=['# tau2 telecom profile (stack B)','',"| agent (user sim) | rollouts | tasks | reward | pass-rate hist | any-pass | median agent steps | median completion tok/call | median s | failures | termination |",'|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|']
for k,v in report.items():
    lines.append(f"| {k} | {v['rollouts']} | {v['tasks']} | {v['reward_mean']} | {v['task_passrate_hist']} | {v['pass_at_k_any']} | {v['median_agent_steps']} | {v['median_completion_tokens_per_call']} | {v['median_duration_s']} | {v['failure_rows']} | {v['termination_reasons']} |")
open(f'{a.out}/SUMMARY.md','w').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
