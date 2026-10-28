"""INBOX67 deterministic identity, preference and joint limit calculations."""
import collections,copy
from v3_reference import canonical_prefix,digest,rank
PRIORITY={'v4':0,'v3':1,'v2':2}

def prepare_legacy(row):
 r=copy.deepcopy(row);v=r['version'];r.update(source_version=v,source_record_id=r['record_id'],original_record_id=r['record_id'],source_bundle='v3-pool-20261025',source_in_default_pool=r['in_default_pool'],source_duplicate_of=r.get('duplicate_of'),source_not_default_reason=r.get('not_default_reason'))
 r['policy_model']='t35b-step500' if v=='v3' else r.get('policy_model') or r.get('agent_model')
 assert r['policy_model'],('Missing source model',r['record_id'])
 if r['audio_only']:
  r['source_holds_preserved']=not r['passes_eligibility'];return r
 # Old dedupe/default flags are recomputed; semantic holds remain intact.
 keep=r['audit62_decision']['keep'] and (r.get('audit59_decision') or {}).get('keep',True)
 if not r['passes_eligibility'] and not r.get('duplicate_of') and r.get('not_default_reason') not in ['cross_version_duplicate']:keep=False
 r['source_holds_preserved']=not keep;r['passes_eligibility']=keep;r['duplicate_of']=None
 if not r['audio_only']:r.update(in_default_pool=False,not_default_reason=None if keep else r.get('not_default_reason') or 'source_hold')
 r['prefix_hash']=digest(canonical_prefix(r['public_prefix']));return r

def dedupe(rows):
 candidates=[r for r in rows if r['passes_eligibility'] and not r['audio_only']];indices={};drops=[];kept=[]
 for r in sorted(candidates,key=lambda r:(PRIORITY[r['source_version']],rank(r,'dedupe'))):
  base=(r['task_id'],r['category']);keys=[base+('prefix',r['prefix_hash'])]
  if r.get('native_boundary_state_hash'):keys.append((r['dialogue_key'],r['category'],'native_boundary_state_hash',r['native_boundary_state_hash']))
  matches=[(key,indices[key]) for key in keys if key in indices]
  match=min(matches,key=lambda pair:(PRIORITY[pair[1]['source_version']],rank(pair[1],'dedupe'),pair[0][2]!='prefix')) if matches else None
  if match:
   key,winner=match;r.update(duplicate_of=winner['record_id'],not_default_reason='exact_duplicate')
   drops.append(dict(dropped=r['record_id'],kept=winner['record_id'],source_version=r['source_version'],kept_version=winner['source_version'],task_id=r['task_id'],category=r['category'],basis=key[2],key=list(key)))
  else:
   kept.append(r)
   for key in keys:indices[key]=r
 # A dropped record always directly matches its kept anchor. No transitive
 # collapse through a removed record with a different hash namespace.
 near=collections.defaultdict(list)
 for r in candidates:near[(r['task_id'],r['category'],r['pivot_turn'])].append(r)
 near_groups=near_pairs=0
 for group in near.values():
  count=0
  for i,left in enumerate(group):
   for right in group[i+1:]:
    if left['prefix_hash']==right['prefix_hash']:continue
    if left['dialogue_key']==right['dialogue_key'] and left.get('native_boundary_state_hash') and left.get('native_boundary_state_hash')==right.get('native_boundary_state_hash'):continue
    count+=1
  near_groups+=count>0;near_pairs+=count
 state_groups=collections.defaultdict(list)
 for r in candidates:
  if r.get('native_boundary_state_hash'):state_groups[(r['task_id'],r['category'],r['native_boundary_state_hash'])].append(r)
 cross=same_prefix=overlap=groups=0
 for group in state_groups.values():
  count=0
  for i,left in enumerate(group):
   for right in group[i+1:]:
    if left['dialogue_key']==right['dialogue_key']:continue
    count+=1;same_prefix+=left['prefix_hash']==right['prefix_hash'];overlap+=left['prefix_hash']!=right['prefix_hash'] and left['pivot_turn']==right['pivot_turn']
  groups+=count>0;cross+=count
 near_counts=dict(cross_dialogue_same_state_groups=groups,cross_dialogue_same_state_pairs=cross,cross_dialogue_same_state_same_prefix_pairs=same_prefix,cross_dialogue_same_state_different_prefix_pairs=cross-same_prefix,overlap_with_pivot_near_pairs=overlap,groups=near_groups,record_pairs=near_pairs,definition='eligible source inventory pairs with same task/category/pivot index and different canonical prefix; excludes exact within-dialogue shared native-state matches; cross-dialogue same-state counts are separately reported and can overlap pivot-near pairs; no cross-dialogue state-only dedupe; count only, no drop is based on near identity')
 return kept,drops,near_counts

def limits(rows):
 freq=collections.Counter((r['source_version'],r['category']) for r in rows)
 ordered=sorted(rows,key=lambda r:(PRIORITY[r['source_version']],freq[(r['source_version'],r['category'])],rank(r,'limits')))
 dialogues=collections.Counter();tasks=collections.Counter();scenarios=collections.Counter();diverse=[];drops=[]
 for r in ordered:
  reasons=[]
  if dialogues[r['dialogue_key']]>=3:reasons.append('max_3_per_dialogue')
  if tasks[(r['task_id'],r['category'])]>=3:reasons.append('max_3_per_task_category')
  if scenarios[r['scenario_key']]>=50:reasons.append('max_50_per_scenario')
  if reasons:r['not_default_reason']=','.join(reasons);drops.append(dict(record_id=r['record_id'],source_version=r['source_version'],task_id=r['task_id'],category=r['category'],rules=reasons));continue
  dialogues[r['dialogue_key']]+=1;tasks[(r['task_id'],r['category'])]+=1;scenarios[r['scenario_key']]+=1;diverse.append(r)
 counts=collections.Counter(r['category'] for r in diverse);capacities={}
 for pct in [30,35,40]:
  n=max(n for n in range(len(diverse)+1) if sum(min(v,pct*n//100) for v in counts.values())>=n);ct=collections.Counter();selected=[]
  for r in sorted(diverse,key=lambda r:(PRIORITY[r['source_version']],rank(r,'category_cap'))):
   if ct[r['category']]<pct*n//100:ct[r['category']]+=1;selected.append(r['record_id'])
   if len(selected)==n:break
  assert len(selected)==n
  capacities[str(pct)]=dict(records=n,ids=selected,by_category=dict(ct),max_per_category=pct*n//100)
 default=set(capacities['30']['ids'])
 for r in diverse:
  r['in_default_pool']=r['record_id'] in default;r['not_default_reason']=None if r['in_default_pool'] else 'category_30pct_cap'
  if not r['in_default_pool']:drops.append(dict(record_id=r['record_id'],source_version=r['source_version'],task_id=r['task_id'],category=r['category'],rules=['category_30pct_cap']))
 return capacities,drops
