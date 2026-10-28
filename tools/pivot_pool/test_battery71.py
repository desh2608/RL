"""Read-only, exact INBOX71 gate. No selection before PASS."""
import argparse,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from holds70 import h1,h4,h4_analysis,decide,last_user,call
from v3_reference import jlread,write
p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
rows=jlread(a.source/'MANIFEST.jsonl');byid={r['record_id']:r for r in rows};results=[]
def check(r,rule,expected):
 actual=bool({'H1':h1,'H4':h4}[rule](r));results.append(dict(record_id=r['record_id'],rule=rule,expected=expected,actual=actual,pass_=actual==expected,evidence=decide(r),last_user_message=last_user(r)[1],positive_output=r['positive_output']))
oldh1=jlread(a.source/'H1_LAST_USER_MESSAGES.jsonl');assert len(oldh1)==135
for old in oldh1:
 r=byid[old['record_id']];check(r,'H1',r['record_id'].startswith('d2b5c028'))
# Audit table's first two OTP/membership examples identify the old held sample exactly.
heldsamples=jlread(a.source/'H4_SAMPLE_HELD.jsonl')
for pref,expected,suffix in [('cba73e76cab7',True,None),('85cfc9ffce9a',True,None),('809ebb03',True,':5:AQ_wrong_action'),('a6ba37bd',True,':8:AQ_wrong_action'),('712a557cc8f1',False,None),('02e1ced0dfe0',False,None),('e84c822a5326',False,None),('14fa0516b793',False,None)]:
 candidates=[x['record'] for x in heldsamples if x['record']['record_id'].startswith(pref)] if suffix is None else [r for r in rows if r['record_id'].startswith(pref) and r['record_id'].endswith(suffix)]
 assert len(candidates)==1,(pref,len(candidates));check(candidates[0],'H4',expected)
# Preserve H2/H3 decisions exactly for every row, not just examples.
olddec={x['record_id']:x['holds'] for x in jlread(a.source/'HOLD_DECISIONS.jsonl')}
h23=[]
for r in rows:
 now=decide(r)
 for k in ['H2','H3']:
  assert now.get(k)==olddec[r['record_id']].get(k),(r['record_id'],k)
kept=[]
for x in jlread(a.source/'H4_SAMPLE_KEPT.jsonl'):
 r=x['record'];kept.append(dict(record_id=r['record_id'],previous_H4_hold=False,new_H4_hold=bool(h4(r)),flip=bool(h4(r)),analysis=h4_analysis(r),last_user_message=last_user(r)[1],positive_output=r['positive_output']))
unlabelled=[]
for pref in ['f337858bc9b4','cf51679eb127','f40b2e9684b7','816209f1dbbc']:
 matches=[x['record'] for x in heldsamples if x['record']['record_id'].startswith(pref)];assert len(matches)==1,(pref,len(matches))
 r=matches[0];unlabelled.append(dict(record_id=r['record_id'],H4_hold=bool(h4(r)),analysis=h4_analysis(r),last_user_message=last_user(r)[1],positive_output=r['positive_output']))
failed=[x for x in results if not x['pass_']]
security=next(r for r in rows if r['record_id'].startswith('bcebc7703f3e') and r['record_id'].endswith(':11:AQ_wrong_action'));assert not h4(security),'caller-requested security enrollment must not be commerce'
write(a.out,dict(security_enrollment_scope_guard=dict(record_id=security['record_id'],H4_hold=False,analysis=h4_analysis(security)),status='FAIL' if failed else 'PASS',contract='INBOX71',labelled=len(results),matched=len(results)-len(failed),results=results,H2_H3_identical_on_records=len(rows),previously_kept_H4=kept,previously_kept_H4_flips=sum(x['flip'] for x in kept),unlabelled=unlabelled))
print(json.dumps(dict(status='FAIL' if failed else 'PASS',labelled=len(results),matched=len(results)-len(failed),failed=failed,kept_flips=[x for x in kept if x['flip']],unlabelled=unlabelled)),flush=True)
assert not failed
