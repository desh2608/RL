#!/usr/bin/env python3
"""Sample profiling prompts per route from the v43 library (raw Gym rows, unchanged) and
the tau2 telecom non-eval task subset. Login-node safe (JSON only)."""
import gzip, json, random, re, sys, os
LIB='/lustre/fsw/portfolios/llmservice/users/cchen1/data/library/rl/merged_20260928_super35_v43/sources/super35_v43/routes'
TAU='/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_eval/artifacts/extensions/prepared-v1/tau3_banking/upstream/data/tau2/domains/telecom'
out=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 256; seed=20260930
routes=['single_step_tool_use_with_argument_comparison_agent','toolcall_schema_single_step_tool_use_with_argument_comparison_agent',
        'search_pivot_single_step_tool_use_with_argument_comparison_agent','droid_pivot_single_step_tool_use_with_argument_comparison_agent',
        'conversational_tool_use_agent','workplace_assistant_simple_agent']
os.makedirs(out, exist_ok=True); manifest={}
for r in routes:
    rows=[json.loads(l) for l in gzip.open(f'{LIB}/{r}.jsonl.gz','rt')]
    rnd=random.Random(seed); rnd.shuffle(rows); pick=rows[:n]
    with open(f'{out}/{r}.jsonl','w') as f:
        for x in pick: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    manifest[r]={'total':len(rows),'sampled':len(pick)}
# tau2 telecom: tasks outside the eval 'base' split
sp=json.load(open(f'{TAU}/split_tasks.json')); base=set(map(str,sp['base']))
full=json.load(open(f'{TAU}/tasks_full.json')); pool=[t for t in full if str(t['id']) not in base]
rnd=random.Random(seed); rnd.shuffle(pool)
json.dump(pool[:n], open(f'{out}/tau2_telecom_tasks.json','w'), ensure_ascii=False)
json.dump(pool, open(f'{out}/tau2_telecom_tasks_all_nonbase.json','w'), ensure_ascii=False)
manifest['tau2_telecom']={'total_nonbase':len(pool),'sampled':min(n,len(pool)),'eval_base_excluded':len(base)}
json.dump({'seed':seed,'n_per_route':n,'routes':manifest}, open(f'{out}/SAMPLE_MANIFEST.json','w'), indent=1)
print(json.dumps(manifest,indent=1))
