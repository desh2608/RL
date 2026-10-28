import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import copy
from select_pool import dedupe,limits
from v3_reference import digest,canonical_prefix

def row(i,v='v4',prefix=None,state=None,cat='a',task='task',dialogue=None):
 p=prefix or [dict(type='message',role='user',content='help')]
 return dict(record_id=str(i),source_version=v,passes_eligibility=True,audio_only=False,task_id=task,category=cat,public_prefix=p,prefix_hash=digest(canonical_prefix(p)),native_boundary_state_hash=state,pivot_turn=0,dialogue_key=dialogue or str(i),scenario_key=task,in_default_pool=False)
a=row(1,'v3');b=row(2,'v4');kept,dropped,_=dedupe([a,b]);assert [r['record_id'] for r in kept]==['2'] and dropped[0]['kept']=='2'
a=row(1,'v2',state='x');b=row(2,'v4',state='x',prefix=[dict(type='message',role='user',content='different')]);assert len(dedupe([a,b])[0])==2
assert dedupe([a,b])[2]['cross_dialogue_same_state_different_prefix_pairs']==1
b['dialogue_key']=a['dialogue_key'];assert len(dedupe([a,b])[0])==1
b['native_boundary_state_hash']=None;kept,_,near=dedupe([a,b]);assert len(kept)==2 and near['groups']==1
b['category']='b';assert len(dedupe([a,b])[0])==2
x=[dict(type='function_call',name='f',arguments='{"b":2,"a":1}',call_id='old',id='x'),dict(type='function_call_output',call_id='old',output='{"ok":true}')];y=[dict(type='function_call',name='f',arguments={'a':1,'b':2},call_id='new'),dict(type='function_call_output',call_id='new',output={'ok':True})];assert digest(canonical_prefix(x))==digest(canonical_prefix(y))
rs=[row(f'{v}-{cat}-{i}',v,cat=cat,task=f'{cat}-{i}',dialogue=f'{cat}-{i}') for v in ['v2','v3','v4'] for cat in 'abcd' for i in range(4)];caps,drops=limits(rs);selected={r['record_id'] for r in rs if r['in_default_pool']};assert caps['30']['records']==len(selected);assert all(100*n<=30*len(selected) for n in caps['30']['by_category'].values());assert max(__import__('collections').Counter((r['task_id'],r['category']) for r in rs if r['in_default_pool']).values())<=3
# Structural bound always chooses v4 before lower-priority sources.
rs=[row(i,'v2' if i<3 else 'v4',task='same') for i in range(6)];caps,drops=limits(rs);structural=[r for r in rs if r['not_default_reason']!='max_3_per_task_category'];assert all(r['source_version']=='v4' for r in structural)
print('Selection identity/preference/near-duplicate/limit checks PASS')

# When prefix and state match different kept records, source preference wins.
a=row('A','v4',state='x');b=row('B','v3',state='y',prefix=[dict(type='message',role='user',content='other')]);c=row('C','v2',state='x',prefix=b['public_prefix'],dialogue=a['dialogue_key']);kept,drop,_=dedupe([a,b,c]);assert len(kept)==2 and drop[0]['dropped']=='C' and drop[0]['kept']=='A'
from select_pool import prepare_legacy
legacy=dict(record_id='old',version='v2',source='SDG',agent_model='teacher',in_default_pool=False,duplicate_of=None,not_default_reason='upstream_offer_divergence',passes_eligibility=False,audio_only=False,audit62_decision={'keep':False},public_prefix=[dict(type='message',role='user',content='hello')])
assert not prepare_legacy(legacy)['passes_eligibility'] and 'source_version' not in legacy
olddup=copy.deepcopy(legacy);olddup.update(duplicate_of='old-winner',not_default_reason='cross_version_duplicate',audit62_decision={'keep':True});assert prepare_legacy(olddup)['passes_eligibility'] and prepare_legacy(olddup)['duplicate_of'] is None
olddup['audit62_decision']['keep']=False;assert not prepare_legacy(olddup)['passes_eligibility']
audio=copy.deepcopy(olddup);audio['audio_only']=True;audio['prefix_hash']='source-hash';result=prepare_legacy(audio);assert all(result[k]==v for k,v in audio.items())
print('Source holds, exact-rule dedupe recomputation and audio identity checks PASS')

# INBOX69: state matches are strictly scoped to dialogue, not task.
a=row('one',state='shared',dialogue='d1');b=row('two',state='shared',dialogue='d2',prefix=[dict(type='message',role='user',content='second request')]);assert len(dedupe([a,b])[0])==2
b['dialogue_key']='d1';assert len(dedupe([a,b])[0])==1
b['task_id']='other-task';assert len(dedupe([a,b])[0])==1
print('INBOX69 different-dialogue retained / same-dialogue dropped PASS')
