"""Inspectable cached F1-F5 membership decisions; no environment/model calls."""
import copy,json,re,sys,calendar,os,collections
from pathlib import Path
GYM=Path(os.environ.get('PIVOT_AUDIT_GYM_ROOT','/home/deshr/code/repos/NeMo_RL_voicechat_pivot_combined/3rdparty/Gym-workspace/Gym'))
sys.path.insert(0,str(GYM));sys.path.insert(0,str(GYM/'resources_servers/tau_pivot_stateful/reference'))
from resources_servers.tau_pivot_stateful import schema_compare as schema
from recipes.tau_voice_rl_pivot import areal_v2,confirmed_write as cw,consent_given
READ=re.compile(r'^(get|find|search|list|lookup|read|check|quote|compute|scan|calculate)_')
AUTH=re.compile(r'^(log_verification|send_verification_code|confirm_identity_verification|confirm_otp)(?:_\d+)?$')
WRITE=re.compile(r'^(add|set|update|modify|change|cancel|delete|remove|create|open|close|submit|record|send|issue|book|assign|redeem|enroll|accept|authorize|deny|initiate|process|take|apply|release|waive|recalc|replace|flag|file|convert|reinstate|reissue|request|place|extend|waitlist|resolve|return|exchange|link|give|enable|refuel|execute)_')

def shape(output):
 kind,value=areal_v2.one_turn(output)
 if kind=='call' and value['name']=='call_discoverable_agent_tool':
  args=value['arguments'];inner=args.get('arguments',{});inner=json.loads(inner) if isinstance(inner,str) else inner;value=dict(name=args.get('agent_tool_name',value['name']),arguments=inner)
 return kind,value

def normal(call,tools):
 if not call:return None
 x=schema.call(call,tools);args=x['arguments'];return dict(name=x['name'],arguments=json.loads(args) if isinstance(args,str) else args)

def key(output,tools):
 kind,value=shape(output)
 return json.dumps([kind,normal(value,tools) if kind=='call' else value],sort_keys=True,ensure_ascii=False)

def speech(prefix):
 ui=next((i for i in range(len(prefix)-1,-1,-1) if prefix[i].get('role')=='user'),None)
 if ui is None:return '', '',None
 latest=consent_given.text(prefix[ui]);start=next((i for i in range(ui-1,-1,-1) if prefix[i].get('role')=='user'),-1)
 offer='\n'.join(consent_given.text(m) for m in prefix[start+1:ui] if m.get('role')=='assistant')
 affirmative=bool(consent_given.G_AFFIRM.search(consent_given.assent_clause(latest)) or re.search(r'\b(confirm(?:ed)?|proceed|go ahead|that.s correct|exactly right|do it|place the order)\b',latest,re.I))
 return latest, cw.active_text(offer+'\n'+latest) if affirmative else latest,ui

def leaves(value,path=()):
 if isinstance(value,dict):
  for k,v in value.items():yield from leaves(v,path+(k,))
 elif isinstance(value,list):
  for i,v in enumerate(value):yield from leaves(v,path+(str(i),))
 else:yield path,value

def aliases(value,field):
 vals=[str(value)]
 if field=='state':
  names=['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming','District of Columbia']
  codes='AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC'.split()
  vals += [name for code,name in zip(codes,names) if cw.norm(value) in [cw.norm(code),cw.norm(name)]]
 if field=='country' and cw.norm(value) in ['us','usa','united states','united states of america']:vals+=['United States','USA','United States of America']
 if field=='duration':
  vals+= {'gtc':['good till canceled','good till cancelled','good until canceled','good until cancelled'],'day':['day order','day duration'],'ioc':['immediate or cancel'],'fok':['fill or kill']}.get(str(value).lower(),[])
 return vals

def stated(text,value,field):
 if isinstance(value,str):return any(len(alias)>2 and cw.mentions(text,alias) for alias in aliases(value,field))
 return False

def conflicts(r,p,f):
 prefix=r['public_prefix'];latest,window,ui=speech(prefix);bad=[]
 if not p or not WRITE.match(p['name']) or AUTH.match(p['name']):return bad
 def reject(field,stated_value,positive):bad.append(dict(field=field,stated=stated_value,positive=positive,latest_user_index=ui,latest_user=latest,confirmation_window=window))
 if re.search(r'\btransfer (?:me|my call)|\b(?:connect|speak) (?:me )?to (?:a |an )?(?:human|supervisor|agent)',latest,re.I) and not p['name'].startswith(('transfer_','escalate_')):reject('action','caller_requested_human_transfer',p['name'])
 if f and READ.match(f['name']) and re.search(r'\b(?:check|search|find|look up|look into|tell me|show me)\b',latest,re.I):
  topics=[s for s in f['name'].split('_')[1:] if len(s)>3 and s not in {'details','direct','status','current','available'}]
  if any(re.search(r'\b'+re.escape(w)+r's?\b',latest,re.I) for w in topics):reject('action',f['name'],p['name'])
 if f and READ.match(f['name']) and window!=latest and re.search(r'\b(?:would you like me to|shall I|can I) (?:check|search|look up|find)\b',window,re.I):reject('action','caller_accepted_information_lookup',p['name'])
 tools=cw.offered_tools(window,prefix)
 if f and f['name'] in tools and p['name'] not in tools:reject('action',sorted(tools),p['name'])
 # Compare explicit selected values, including failed/gold values at corresponding
 # nested schema paths. Numeric identifiers remain exact strings; monetary values
 # and quantities use their own units below, not unrelated dates/phone digits.
 if f and f['name']==p['name']:
  other=dict(leaves(f['arguments']))
  for path,value in leaves(p['arguments']):
   fv=other.get(path)
   if type(fv)==type(value) and fv!=value and stated(window,fv,path[-1]) and not stated(window,value,path[-1]):reject('.'.join(path),fv,value)
 objects=cw.public_objects(prefix)
 for field in ['reservation_id','order_id','user_id','customer_id','line_id','policy_id','account_id','confirmation_number']:
  if field not in p['arguments']:continue
  known=cw.known_ids(prefix,field)|{str(p['arguments'][field])}
  ids=cw.target_ids(window,field,known)
  if ids and f and str(f['arguments'].get(field)) in ids and f['arguments'].get(field)!=p['arguments'][field] and str(p['arguments'][field]) not in ids:reject(field,sorted(ids),p['arguments'][field])
 for field in ['payment_id','payment_method_id']:
  if field not in p['arguments']:continue
  ident=str(p['arguments'][field]);types={kind for kind,pat in {'credit_card':r'credit[ _]card|Visa|Mastercard|Amex','gift_card':r'gift[ _]card','paypal':r'paypal'}.items() if re.search(pat,window,re.I)}
  expected=next((k for k in ['credit_card','gift_card','paypal'] if ident.startswith(k)),None)
  if len(types)==1 and expected and expected not in types:reject(field,sorted(types),ident)
  digits=re.findall(r'\bending(?: in)?\s*[.…]*\s*(\d{4})\b',window,re.I)
  for tail in digits:
   matches={str(o['payment_id']) for o in objects if o.get('payment_id') and (any(str(o.get(k,''))==tail for k in ['last_four','last4','last_four_digits']) or str(o['payment_id']).startswith('gift_card_') and str(o['payment_id']).endswith(tail))}
   if matches and ident not in matches:reject(field,sorted(matches),ident)
 # Prefer explicit caller amounts over amounts in a preceding multi-policy offer.
 currency=lambda text:{float(re.sub('[,\u202f\u00a0 ]','',s)) for s in re.findall(r'\$\s*(\d[\d,\u202f\u00a0 ]*(?:\.\d{1,2})?)',text)}
 amounts=currency(latest) or currency(window)
 for field in ['amount','payment_amount','limit_price','stop_price']:
  val=p['arguments'].get(field)
  if type(val) in [int,float] and amounts and float(val) not in amounts:reject(field,sorted(amounts),val)
 # Bind explicit monetary allocations per method, not just the total quote.
 methods={'gift_card':r'gift[ -]card','credit_card':r'credit[ -]card','certificate':r'(?:travel[ -])?certificate','paypal':r'paypal'}
 payments=p['arguments'].get('payment_methods')
 if isinstance(payments,list):
  offered=collections.defaultdict(list)
  for kind,pattern in methods.items():
   for raw in re.findall(r'\$\s*(\d[\d,\u202f\u00a0 ]*(?:\.\d{1,2})?)\s*(?:(?:to|from|on|of)\s+)?(?:(?:your|my|the)\s+)?'+pattern,window,re.I):offered[kind].append(float(re.sub('[,\u202f\u00a0 ]','',raw)))
  for kind,values in offered.items():
   actual=sum(float(x.get('amount',0)) for x in payments if str(x.get('payment_id','')).startswith(kind+'_'))
   if len(set(values))==1 and actual!=values[0]:reject('payment_methods.'+kind+'.amount',values[0],actual)
 # An explicitly ordered first split transaction cannot be replaced by its
 # later, differently funded transaction.
 if p['name']=='update_reservation_baggages' and re.search(r'\b(?:split|two (?:separate )?transactions)\b',latest,re.I) and re.search(r'\bthen\b',latest,re.I):
  first=latest.split('then',1)[0]
  first_methods={kind for kind,pat in methods.items() if re.search(pat,first,re.I)}
  payment=str(p['arguments'].get('payment_id',''))
  if len(first_methods)==1 and not any(payment.startswith(kind+'_') for kind in first_methods):reject('payment_id.first_split_transaction',sorted(first_methods),payment)
 for field,pattern in {'quantity':r'(\d+(?:\.\d+)?)\s*(?:shares?|units?|contracts?)\b','gb_amount':r'(\d+(?:\.\d+)?)\s*GB\b','duration':r'(\d+)\s*(?:days?|months?|years?|nights?)\b'}.items():
  value=p['arguments'].get(field);numbers={float(v) for v in re.findall(pattern,latest,re.I)} or {float(v) for v in re.findall(pattern,window,re.I)}
  if type(value) in (int,float) and numbers and value not in numbers and f and f['name']==p['name'] and f['arguments'].get(field) in numbers:reject(field,sorted(numbers),value)
 tool=schema.schemas(r['tools']).get(p['name'],{});properties=schema.resolve(tool,tool).get('properties',{})
 for field,decl in properties.items():
  value=p['arguments'].get(field);enums=decl.get('enum',[])
  if field=='duration':enums=list(set(enums)|{'gtc','day','ioc','fok'})
  selected={str(e) for e in enums if str(e).lower() not in {'yes','no','true','false','on','off'} and stated(window,e,field)}
  if field=='insurance':
   if re.search(r'\b(?:no|without|decline|do not want|don.t want)\b.{0,22}\binsurance\b',window,re.I):selected={'no'}
   elif re.search(r'\b(?:add|include|want|with)\b.{0,22}\binsurance\b',window,re.I):selected={'yes'}
  if len(selected)==1 and value is not None and not any(cw.norm(value)==cw.norm(e) for e in selected):reject(field,sorted(selected),value)
 return bad

def read_success(turn):
 result=turn.get('native_tool_result') if turn else None
 if result is None:return False
 if isinstance(result,str):
  if re.match(r'^\s*(?:error\b|exception\b)',result,re.I):return False
  try:result=json.loads(result)
  except ValueError:return True
 return not isinstance(result,dict) or not (result.get('error') or result.get('success') is False)

def grounded_passengers(r,f):
 if not f or f['name']!='update_reservation_passengers':return None
 objects=cw.public_objects(r['public_prefix']);proof=[]
 for p in f['arguments'].get('passengers',[]):
  if any(all(o.get(k)==v for k,v in p.items()) for o in objects):proof.append(dict(passenger=p,source='prior_structured_tool_result'));continue
  found=False
  for i,m in enumerate(r['public_prefix']):
   if m.get('role')!='user':continue
   text=consent_given.text(m)
   for month in range(1,13):
    for day,year in re.findall(r'\b'+calendar.month_name[month]+r'\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{4})\b',text,re.I):text+=' '+f'{int(year):04d}-{month:02d}-{int(day):02d}'
   if all(cw.mentions(text,v) for v in p.values()):proof.append(dict(passenger=p,source='caller',index=i));found=True;break
  if not found:return None
 return proof if proof else None

def decide(r,context=None):
 tools=r['tools'];_,p=shape(r['positive_output']);_,f=shape(r['failing_output']);p=normal(p,tools) if p and isinstance(p,dict) else None;f=normal(f,tools) if f and isinstance(f,dict) else None
 reasons=[];e={};cat=r['category']
 if cat=='forbidden_write_should_refuse' and f and f['name']=='update_reservation_flights' and f['arguments'].get('cabin') not in [None,'basic_economy']:
  policy=re.sub(r'[\s*_]+',' ',r['system']).lower()
  if re.search(r'all reservations.{0,60}including basic economy.{0,60}can change cabin|any reservation.{0,60}change.{0,20}cabin',policy):reasons.append('excluded_upgrade_path');e['F1']=dict(bucket='excluded_upgrade_path',failing_cabin=f['arguments']['cabin'],retargeted=False,reason='Allowed cabin path; no newly invented positive')
 if p:
  bad=conflicts(r,p,f)
  if bad:reasons.append('upstream_offer_divergence');e['F2']=bad
 if p and f and areal_v2.typed_equal(p,f):
  reasons.append('equivalent_after_schema_normalization');e['F3']=dict(normalized_positive=p,normalized_failing=f)
 if cat=='missed_write' and f and READ.match(f['name']) and context:
  turn=next((t for t in context['turns'] if t['turn_index']==r['turn']),None)
  if read_success(turn):
   later=[]
   for t in context['turns']:
    if t['turn_index']<=r['turn']:continue
    kind,value=shape(t.get('output',[]))
    if kind=='call' and areal_v2.typed_equal(normal(value,tools),p):later.append(dict(turn=t['turn_index'],call=value,native_result=t.get('native_tool_result')))
   e['F4']=dict(successful_read=True,later_matching_writes=later,raw_path=context['raw_path'],raw_sha256=context['raw_sha256'])
   if later:reasons.append('deferred_not_missed')
 if cat=='missing_lookup_or_login' and f:
  proof=grounded_passengers(r,f)
  if proof:reasons.append('grounded_passenger_write');e['F5']=proof
 projected=schema.tree(r['positive_output'],tools)
 return dict(record_id=r['record_id'],keep=not reasons,reasons=reasons,evidence=e,version=r['version'],category=cat,task_id=r['task_id'],positive_output=projected,positive_changed=key(projected,[])!=key(r['positive_output'],[]))
