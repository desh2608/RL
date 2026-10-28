"""CPU package v2 plus the audited v3 Stage A pool; no replay or model calls."""
import argparse,ast,collections,copy,csv,hashlib,json,shutil
from pathlib import Path

SEED=20261025
AUDIT_SEED=20261026
AUDIT62={}
GROUPS={'wrong_action':'write','wrong_arguments':'write','consent_missing':'consent','reask_after_yes':'consent','missing_lookup_or_login':'lookup','unneeded_transfer_or_premature_end':'flow','empty_turn':'flow','infeasible_or_rejected_write':'write','missed_write':'write','wrong_identifier':'lookup','forbidden_write_should_refuse':'consent'}
KINDS={'K1':'wrong_action','K2':'wrong_arguments','K4':'consent_missing','K5':'reask_after_yes','K6':'missing_lookup_or_login','K7':'unneeded_transfer_or_premature_end'}
ROOT=Path('/lustre/fsw/portfolios/nemotron/projects/nemotron_speech_chat/users/deshr/results/tau-voice-rl/pivot-dataset')
def read(p):return json.loads(Path(p).read_text())
def jlread(p):return [json.loads(l) for l in Path(p).open()]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
SHA_CACHE={}
def sha(p):
 p=Path(p);st=p.stat();key=(str(p),st.st_size,st.st_mtime_ns)
 if key in SHA_CACHE:return SHA_CACHE[key]
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 result=h.hexdigest();SHA_CACHE[key]=result;return result
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
def jl(p,rs):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in rs))
def shape(out):
 xs=[x for x in out if x.get('type')!='reasoning']
 if len(xs)!=1 or xs[0].get('type')!='function_call':return None
 x=xs[0];return dict(name=x['name'],arguments=json.loads(x['arguments']) if isinstance(x['arguments'],str) else x['arguments'])
def canonical_prefix(prefix):
 calls={};items=[]
 for raw in prefix:
  x=copy.deepcopy(raw);x.pop('id',None);x.pop('status',None)
  if x.get('type')=='function_call':calls.setdefault(x.get('call_id'),str(len(calls)));x['call_id']=calls[x.get('call_id')]
  elif x.get('type')=='function_call_output':x['call_id']=calls.get(x.get('call_id'),x.get('call_id'))
  for key in ['arguments','output']:
   if isinstance(x.get(key),str):
    try:x[key]=json.loads(x[key])
    except ValueError:pass
  items.append(x)
 return items

def rank(r,tag):return digest([SEED,tag,r['record_id']])
def normalize(r,version):
 x=copy.deepcopy(r);oldsource=x['source'];oldcategory=x.get('category');kind=x.get('k_code',x.get('error_kind',x.get('kind')))
 cat=KINDS[kind] if version=='v2' else {'afterquery_wrong_action':'wrong_action','afterquery_wrong_arguments':'wrong_arguments'}.get(oldcategory,oldcategory)
 assert cat in GROUPS,(kind,cat)
 native=x.get('native_task_id') or x['task_id'].removeprefix(x['domain']+':');task=x['domain']+':'+native
 source=('B0' if oldsource=='b0' else 'SDG') if version=='v2' else ('T35b repair' if oldsource=='repair' else 'T35b original')
 prior_default=x.get('in_default_pool',True);audio=bool(x.get('audio_only',False));scenario=x.get('scenario_key')
 if not scenario:scenario=task
 x.update(version=version,legacy_source=oldsource,legacy_category=oldcategory,legacy_kind=kind,legacy_task_id=x['task_id'],legacy_dialogue_key=x['dialogue_key'],legacy_in_default_pool=prior_default,legacy_runtime_key=x.get('runtime_key'),source=source,category=cat,group=GROUPS[cat],category_group={'write':'write_errors','consent':'consent_errors','lookup':'lookup_errors','flow':'conversation_flow_errors'}[GROUPS[cat]],task_id=task,native_task_id=native,scenario_key=scenario,dialogue_key=version+':'+x['dialogue_key'],audio_only=audio,training_selected=False,in_default_pool=False,not_default_reason=None,kind=kind)
 x['positive_output']=x.get('positive_output') or x['controls']['positive']['output']
 x['failing_output']=x.get('failed_output',x.get('failing_output',x.get('output')))
 x['source_eligibility_contract']=x.get('eligibility_contract');x['eligibility_contract']='INBOX62-F1-F5'
 x['audit62_decision']=AUDIT62[x['record_id']];x['pre_f3_positive_output']=copy.deepcopy(x['positive_output']);x['positive_output']=copy.deepcopy(x['audit62_decision']['positive_output'])
 x['prefix_hash']=digest(canonical_prefix(x['public_prefix']));x['pivot_turn']=sum(m.get('type')=='function_call' or m.get('type')=='message' and m.get('role')=='assistant' for m in x['public_prefix']);x['positive_call']=shape(x['positive_output'])
 return x

def inventory(v2,v3,v2_defaults_only):
 rows=[normalize(r,'v2') for r in v2]+[normalize(r,'v3') for r in v3];assert len({r['record_id'] for r in rows})==len(rows)
 newindex=collections.defaultdict(list)
 for r in rows:
  if r['version']!='v3' or not r['audit62_decision']['keep']:continue
  key=(r['audio_only'],r['task_id']);newindex[(key,'prefix',r['prefix_hash'])].append(r)
  if r['positive_call']:newindex[(key,'call',r['pivot_turn'],digest(r['positive_call']))].append(r)
 dedup=[]
 for r in rows:
  r['duplicate_of']=None;r['passes_eligibility']=r['audit62_decision']['keep']
  if not r['passes_eligibility']:r['not_default_reason']=','.join(r['audit62_decision']['reasons']);continue
  if r['version']!='v2':continue
  key=(r['audio_only'],r['task_id']);match=newindex.get((key,'prefix',r['prefix_hash']),[]);basis='same_public_prefix_hash'
  if not match and r['positive_call']:match=newindex.get((key,'call',r['pivot_turn'],digest(r['positive_call'])),[]);basis='same_gold_positive_call_at_same_assistant_turn'
  if match:
   target=min(match,key=lambda x:rank(x,'dedupe'));r['duplicate_of']=target['record_id'];r['passes_eligibility']=False;r['not_default_reason']='cross_version_duplicate';dedup.append(dict(dropped=r['record_id'],kept=target['record_id'],basis=basis,task_id=r['task_id'],pivot_turn=r['pivot_turn']))
 candidates=[]
 for r in rows:
  if not r['passes_eligibility']:continue
  if r['audio_only']:r['not_default_reason']='audio_only_outside_text_default';continue
  if r['duplicate_of']:continue
  if v2_defaults_only and r['version']=='v2' and not r['legacy_in_default_pool']:r['not_default_reason']='outside_requested_v2_default';continue
  candidates.append(r)
 freq=collections.Counter(r['category'] for r in candidates);dialogues=collections.Counter();tasks=collections.Counter();scenarios=collections.Counter();diverse=[];trim=[]
 for r in sorted(candidates,key=lambda r:(freq[r['category']],r['version']!='v3',rank(r,'limits'))):
  why=[]
  if dialogues[r['dialogue_key']]>=3:why.append('max_3_per_dialogue')
  if tasks[(r['task_id'],r['category'])]>=3:why.append('max_3_per_task_category')
  if scenarios[r['scenario_key']]>=50:why.append('max_50_per_scenario')
  if why:r['not_default_reason']=','.join(why);trim.append(dict(record_id=r['record_id'],scenario_key=r['scenario_key'],reasons=why,version=r['version'],category=r['category']));continue
  dialogues[r['dialogue_key']]+=1;tasks[(r['task_id'],r['category'])]+=1;scenarios[r['scenario_key']]+=1;diverse.append(r)
 counts=collections.Counter(r['category'] for r in diverse);capacities={}
 for pct in [30,35,40]:
  n=max(n for n in range(len(diverse)+1) if sum(min(v,pct*n//100) for v in counts.values())>=n);ct=collections.Counter();selected=[]
  for r in sorted(diverse,key=lambda r:rank(r,'category_cap')):
   if ct[r['category']]<pct*n//100:ct[r['category']]+=1;selected.append(r['record_id'])
   if len(selected)==n:break
  assert len(selected)==n;capacities[str(pct)]=dict(records=n,ids=selected,by_category=dict(ct),max_per_category=pct*n//100)
 default=set(capacities['30']['ids'])
 for r in diverse:
  if r['record_id'] in default:r['in_default_pool']=True;r['not_default_reason']=None
  else:r['not_default_reason']='category_30pct_cap'
 return rows,dedup,trim,capacities

def sample_prefix(prefix):
 items=[]
 for raw in prefix:
  x=copy.deepcopy(raw)
  if x.get('type')=='message':items.append(dict(role=x['role'],content=x['content']));continue
  for k in ['arguments','output']:
   if k in x and not isinstance(x[k],str):x[k]=json.dumps(x[k],ensure_ascii=False)
  items.append(x)
 return items

def public_tools(tools):
 return [dict(type='function',name=(x.get('function') or x)['name'],description=(x.get('function') or x).get('description',(x.get('function') or x).get('doc','')),parameters=(x.get('function') or x).get('parameters',(x.get('function') or x).get('params',{})),strict=False) for x in tools]

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--v2-defaults-only',action='store_true');p.add_argument('--reuse-assets',type=Path);a=p.parse_args();
 if a.reuse_assets:
  import os
  original_copy=shutil.copyfile
  def immutable_copy(src,dest,*args,**kwargs):
   dest=Path(dest);name=dest.relative_to(a.bundle).as_posix();cached=a.reuse_assets/name
   if name.startswith('verifier/archives/') or name.startswith('verifier/cache/payloads/'):
    assert cached.is_file() and sha(cached)==sha(src),(name,'cached asset mismatch');os.link(cached,dest);return str(dest)
   return original_copy(src,dest,*args,**kwargs)
  shutil.copyfile=immutable_copy
 ops=a.root;b=a.bundle;b.mkdir(exist_ok=False,parents=True);v2=ROOT/'v2-areal-pool-20261023';v3=ROOT/'g6-turn72/review/build';old=jlread(v2/'text/records.jsonl')+jlread(v2/'audio_only/records.jsonl');new=jlread(v3/'DIVERSITY_CHRONOLOGICAL.jsonl');assert len(old)==2118 and len(new)==2618
 global AUDIT62
 AUDIT62={d['record_id']:d for d in jlread(ops/'recount-a004/DECISIONS.jsonl')}
 rows,dedup,trim,capacities=inventory(old,new,a.v2_defaults_only);ver=b/'verifier';ver.mkdir();(ver/'cache/entries').mkdir(parents=True);(ver/'cache/payloads').mkdir();(ver/'cache/field_tables').mkdir();(ver/'archives').mkdir()
 for f in ['CODE.tar.gz','CODE.json']:shutil.copyfile(ops/'assets'/f,ver/f)
 shutil.copytree(ops/'assets/helpers',ver,dirs_exist_ok=True);shutil.copytree(v2/'verifier/tokenizer',ver/'tokenizer');(ver/'v2-baseline').mkdir()
 for f in ['CODE.tar.gz','CODE.json']:shutil.copyfile(v2/'verifier'/f,ver/'v2-baseline'/f)
 # Compile only the exact pure prompt function from the frozen evaluation adapter.
 adapter=ops/'assets/tau_voice_simple_agent.py';assert sha(adapter)=='2ce9fe756eba13edb8684709f245f9118c4c10f5be5b16930d4816a85bd71428'
 node=next(n for n in ast.parse(adapter.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='build_tau_voice_simple_system_prompt');scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_eval_prompt','exec'),scope);render=scope[node.name]
 context=ROOT/'g6-turn72-context-v2';ctx={r['rollout_id']:r for r in jlread(context/'CONTEXT.jsonl')};policies=read(context/'POLICIES.json');toolsets=read(context/'TOOLSETS.json')
 # Pin every field table to its own original wave instead of merging versions.
 wanted={r['record_id'] for r in new if r['verification'].get('kind')=='afterquery_v21_step'};table_for={}
 for source in read(ROOT/'g6-turn70/stage-a-final-review/build/SOURCES.json'):
  root=Path(source['manifest']).parent;ids={r['record_id'] for r in jlread(source['manifest'])}&wanted
  if not ids:continue
  native=root if (root/'mining').exists() else Path(read(root/'FINALIZE_PLAN.json')['source']);table=native/'mining/afterquery/FIELD_TABLE.json'
  if not table.exists():
   recovery=read(native/'RECOVERY_RECEIPT.json');table=Path(recovery['source'])/'mining/afterquery/FIELD_TABLE.json'
  assert table.exists();h=sha(table);target=ver/'cache/field_tables'/(h+'.json')
  if not target.exists():shutil.copyfile(table,target)
  for ident in ids:table_for[ident]=dict(path='field_tables/'+target.name,sha256=h)
 assert set(table_for)==wanted
 # Normalize stored controls without running a reward function.
 import sys
 sys.path.insert(0,str(Path(__file__).resolve().parent));from stored_controls import controls as expanded_controls
 entries={};runtime_counts=collections.Counter();allrecords=[];v2_cache=read(v2/'verifier/CACHE_MANIFEST.json')['records']
 for r in sorted(rows,key=lambda r:(r['audio_only'],r['version'],r['record_id'])):
  ident=r['record_id']
  if r['version']=='v3':
   c=ctx[r['rollout_id']];r['system']=render(policies[c['policy_key']]);r['tools']=public_tools(toolsets[c['tools_key']])
  if r['version']=='v2':
   entry_info=v2_cache[ident];ep=v2/'verifier'/entry_info['entry'];assert sha(ep)==entry_info['entry_sha256'];entry=read(ep);blob=v2/'verifier/cache'/entry['payload'];key=r['runtime_key'];r['source_verification_ref']=copy.deepcopy(r['verification_ref'])
  elif r.get('cache_item'):
   ep=Path(r['cache_item']['entry']);assert sha(ep)==r['cache_item']['entry_sha256'];entry=read(ep);blob=ep.parent/entry['payload'];key='areal--f761e21628a2a007';r['source_verification_ref']=copy.deepcopy(r['verification_ref']);entry.update(record_id=ident,dataset='v3-stage-a-areal');r['verification_ref']={k:entry[k] for k in ['record_id','dataset','verifier_version','expected_turn_kind']}
  else:
   spec=r['verification'];evaluator='afterquery_v21' if spec.get('kind')=='afterquery_v21_step' else 'g6_refusal' if spec.get('contract')=='G6-structural-v1' else 'g6_error_kinds';version=spec.get('verifier_version') or 'g6-static:'+digest(spec);expected='call' if r['positive_call'] else 'message';entry=dict(record_id=ident,dataset='v3-stage-a-static',verifier_version=version,expected_turn_kind=expected,mode='combined_static',evaluator=evaluator,verification=spec,state_hash_at_boundary=r.get('native_boundary_state_hash') or spec.get('state_hash_at_boundary'))
   if evaluator=='afterquery_v21':entry.update(v2_state=r['v2_state'],field_table=table_for[ident])
   key='combined-static';r['verification_ref']={k:entry[k] for k in ['record_id','dataset','verifier_version','expected_turn_kind']};blob=None
  if blob is not None:
   assert sha(blob)==entry['payload_sha256'];name='payloads/'+entry['payload_sha256']+'.pkl';dest=ver/'cache'/name
   if not dest.exists():shutil.copyfile(blob,dest)
   entry['payload']=name
  entry['schema_contract']='INBOX62-schema-fields-v1';entry['schema_tools']=r['tools']
  ep_new=ver/'cache/entries'/(hashlib.sha256(ident.encode()).hexdigest()+'.json');write(ep_new,entry);info=dict(entry='cache/entries/'+ep_new.name,entry_sha256=sha(ep_new),runtime_key=key)
  if entry.get('mode')=='combined_static':info['mode']='combined_static'
  entries[ident]=info;runtime_counts[key]+=1;r['runtime_key']=key
  if r['version']=='v3':
   legacy_r=dict(r,positive_output=r['pre_f3_positive_output']);r['controls']={str(i)+':'+c['name']:c for i,c in enumerate(expanded_controls(legacy_r))};c=ctx[r['rollout_id']];r['system']=render(policies[c['policy_key']]);r['tools']=public_tools(toolsets[c['tools_key']]);r['prompt_source']='Frozen T35b eval voice system builder + exact policy/public prefix/tools';r['agent_model']='T35b@500'
  from schema_compare import tree as project_calls
  old_controls=copy.deepcopy(r['controls']);r['pre_f3_controls']=old_controls
  def control_key(output):
   kind=shape(output)
   return digest(kind if kind is not None else output)
  positives={control_key(project_calls(c['output'],r['tools'])) for c in old_controls.values() if c['expected']==1}
  r['controls']={name:dict(name=c.get('name',name),output=c['output'],expected=1 if control_key(project_calls(c['output'],r['tools'])) in positives else c['expected'],source_expected=c['expected'],source_reward=c.get('reward'),source_control_name=name) for name,c in old_controls.items()}
  if r['audit62_decision']['positive_changed']:
   r['controls']['F3/schema_correct_positive']=dict(name='F3/schema_correct_positive',output=r['positive_output'],expected=1,rule='schema projection of existing certified positive')
  sample=dict(uuid=hashlib.sha256(ident.encode()).hexdigest(),agent_ref=dict(type='responses_api_agents',name='tau_pivot_stateful_agent'),responses_create_params=dict(input=[dict(role='system',content=r['system'])]+sample_prefix(r['public_prefix']),tools=r['tools'],parallel_tool_calls=True),verification_ref=r['verification_ref'])
  r['_sample']=sample;allrecords.append(r)
  if len(allrecords)%500==0:print(json.dumps(dict(packaged=len(allrecords),total=len(rows))),flush=True)
 # Ship all existing pinned runtimes, with no state replay/cache rebuild.
 staging=read(v2/'verifier/STAGING.json');archives=[]
 for item in staging['archives']:
  if item['runtime_key'] not in runtime_counts:continue
  src=v2/'verifier'/item['bundle_path'];dest=ver/item['bundle_path'];shutil.copyfile(src,dest);assert sha(dest)==item['sha256'];archives.append(item);print(json.dumps(dict(copied_archive=item['runtime_key'],bytes=dest.stat().st_size)),flush=True)
 write(ver/'CACHE_MANIFEST.json',dict(records=entries));write(ver/'STAGING.json',dict(runtime_keys=sorted(runtime_counts),archives=archives,records=dict(runtime_counts),code_sha256=sha(ver/'CODE.tar.gz'),cache_rebuilt=False,native_prefix_replays=0))
 for pool in ['text','audio_only']:
  items=[r for r in allrecords if r['audio_only']==(pool=='audio_only')];root=b/pool;samples=[];bycat=collections.defaultdict(list)
  for i,r in enumerate(items,1):
   r['sample_file']=pool+'/all/sample.jsonl';r['sample_line']=i;sample=r.pop('_sample');samples.append(sample);bycat[r['category']].append(sample)
  jl(root/'all/sample.jsonl',samples);jl(root/'records.jsonl',items);routes=[]
  for cat,s in sorted(bycat.items()):jl(root/'by_category'/cat/'sample.jsonl',s);routes.append(dict(route_key=cat,agent='tau_pivot_stateful_agent',sample_path='by_category/'+cat+'/sample.jsonl',n_prompts=len(s)))
  write(root/'routes.json',routes);write(root/'all/routes.json',[dict(route_key='all',agent='tau_pivot_stateful_agent',sample_path='sample.jsonl',n_prompts=len(samples))]);default=[s for s,r in zip(samples,items) if r['in_default_pool']];jl(root/'default/sample.jsonl',default);write(root/'default/routes.json',[dict(route_key='default',agent='tau_pivot_stateful_agent',sample_path='sample.jsonl',n_prompts=len(default))])
 jl(b/'MANIFEST.jsonl',allrecords);write(b/'DEDUPE.json',dedup);write(b/'LIMIT_DROPS.json',trim);write(b/'CAPACITIES.json',capacities)
 for pct,cap in capacities.items():write(b/('CAPACITY-'+pct+'.json'),cap)
 groups=[]
 for pool in ['text','audio_only']:
  for view in ['inventory','eligible','default']:
   groupset=collections.defaultdict(list)
   for r in allrecords:
    if r['audio_only']!=(pool=='audio_only') or view=='eligible' and not r['passes_eligibility'] or view=='default' and not r['in_default_pool']:continue
    groupset[(r['category'],r['group'],r['version'],r['source'],r['domain'])].append(r)
   groups.extend(dict(pool=pool,view=view,category=k[0],group=k[1],version=k[2],source=k[3],domain=k[4],count=len(rs),unique_tasks=len({r['task_id'] for r in rs})) for k,rs in sorted(groupset.items()))
 counts=dict(inventory=len(rows),text_inventory=sum(not r['audio_only'] for r in rows),audio_inventory=sum(r['audio_only'] for r in rows),eligible=sum(r['passes_eligibility'] for r in rows),text_default=sum(r['in_default_pool'] for r in rows),v2_text_all=1848,v2_prior_text_default=1627,v2_audio_all=270,v3_clean=2618,v3_eligible=sum(r['version']=='v3' and r['passes_eligibility'] for r in rows),audit62=read(ops/'recount-a004/RECOUNT.json'),dedupe_drops=len(dedup),limit_drops=len(trim),capacity={p:cap['records'] for p,cap in capacities.items()},groups=groups,seed=SEED,v2_default_candidates_only=a.v2_defaults_only)
 write(b/'COUNTS.json',counts)
 with (b/'COUNTS.tsv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['pool','view','category','group','version','source','domain','count','unique_tasks'],delimiter='\t');writer.writeheader();writer.writerows(groups)
 # Stratified audit: both versions represented when both occur in a category.
 audits=[];draw={}
 for cat in sorted({r['category'] for r in allrecords if r['in_default_pool']}):
  rs=[r for r in allrecords if r['in_default_pool'] and r['category']==cat];chosen=[]
  for version in ['v2','v3']:
   subset=[r for r in rs if r['version']==version]
   if subset:chosen.append(min(subset,key=lambda r:digest([AUDIT_SEED,'audit',r['record_id']])))
  seen={r['record_id'] for r in chosen}
  for r in sorted(rs,key=lambda r:digest([AUDIT_SEED,'audit',r['record_id']])):
   if len(chosen)>=5:break
   if r['record_id'] not in seen:chosen.append(r);seen.add(r['record_id'])
  jl(b/'audit'/('AUDIT-'+cat+'.jsonl'),chosen);audits.extend(chosen);draw[cat]=dict(pool=len(rs),drawn=len(chosen),versions=sorted({r['version'] for r in chosen}),record_ids=[r['record_id'] for r in chosen])
 write(b/'audit/AUDIT_DRAW.json',dict(seed=AUDIT_SEED,categories=draw,records=len(audits)))
 changed=[r for r in allrecords if r['audit62_decision']['positive_changed']]
 fix=sorted(changed,key=lambda r:digest([AUDIT_SEED,'F3',r['record_id']]))[:10]
 jl(b/'audit/AUDIT-FIX-F3.jsonl',fix)
 write(b/'audit/FIX_DRAW.json',dict(seed=AUDIT_SEED,rule='F3',changed=len(changed),drawn=len(fix),record_ids=[r['record_id'] for r in fix]))
 shutil.copytree(ops/'recount-a004',b/'audit62')
 write(b/'BUILD.json',dict(status='BUILT_AWAIT_HTTP_AND_V2_PARITY',code_commit=read(ver/'CODE.json')['commit'],base_commit='103d56002d3d2d000923a6a128da442454c79f93',v2_source=str(v2),v3_source=str(v3),v2_manifest_sha256=sha(v2/'MANIFEST.jsonl'),v3_clean_sha256=sha(v3/'DIVERSITY_CHRONOLOGICAL.jsonl'),membership_contract='INBOX62-F1-F5',source_previous_manifest_sha256=read(ops/'recount-a004/RECOUNT.json')['source_manifest_sha256'],records=len(rows),controls=sum(len(r['controls']) for r in allrecords),default=counts['text_default'],runtime_keys=len(runtime_counts),model_calls=0,native_prefix_replays=0,cache_rebuilt=False,seed=SEED))
 print(json.dumps(read(b/'BUILD.json')),flush=True)
if __name__=='__main__':main()
