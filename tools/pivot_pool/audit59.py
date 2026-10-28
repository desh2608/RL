"""CPU cached INBOX59 eligibility rules; frozen reward functions are untouched."""
import json,re
from recipes.tau_voice_rl_pivot import areal_v2,consent_given,confirmed_write,g6_error_kinds

READ=re.compile(r'^(?:get|find|search|list|lookup|read|check|quote|compute|scan|calculate)_')
WRITE=re.compile(r'^(?:add|set|update|modify|change|cancel|delete|remove|create|open|close|submit|record|log|send|issue|book|assign|redeem|enroll|accept|authorize|deny|confirm|verify_external_bank|initiate|process|take|apply|release|waive|recalc|replace|flag|file|convert|reinstate|reissue|request|place|extend|waitlist|resolve|return|exchange|link|give|enable)_')
TERMINAL=re.compile(r'^(?:end_call|transfer_to_|escalate_to_)')

def shape(output):
 kind,value=areal_v2.one_turn(output)
 if kind=='call' and value['name']=='call_discoverable_agent_tool':
  args=value['arguments'];name=args.get('agent_tool_name');inner=args.get('arguments',{})
  try:inner=json.loads(inner) if isinstance(inner,str) else inner
  except ValueError:inner={}
  value=dict(name=name or value['name'],arguments=inner)
 return kind,value

def verify_boundary_binding(c,context):
 turn=next(t for t in context['turns'] if t['turn_index']==c['turn'])
 if c.get('native_boundary_state_hash') is not None:
  assert c['native_boundary_state_hash']==turn['state_hash'],c['record_id']
  scope='native_environment_hash'
 elif c['verification'].get('contract')=='G6-structural-v1':
  assert c['verification']['state_hash_at_boundary']==turn['state_hash'],c['record_id']
  assert c['verification']['prefix_replayed'] is True
  scope='structural_native_environment_hash'
 else:
  proof=c['verification']['eligibility']
  # Legacy AR certificates hash Boundary.state(), which includes its ledger;
  # raw state_hash hashes the environment alone. They are different namespaces.
  assert c['state_hash_at_boundary']==proof['state_hash_at_boundary'],c['record_id']
  assert proof['prefix_replayed'] is True and proof['proven'] is True
  scope='legacy_boundary_plus_ledger_certificate'
 assert c['prefix_verified_before'] is True and c['controls_matched'] is True
 return scope

def action_type(output,turn):
 kind,call=shape(output)
 if kind=='message':return 'speech'
 if kind!='call':return 'unresolved'
 name=call['name']
 if TERMINAL.match(name):return 'terminal'
 is_read=bool(READ.match(name)) or name in {'verify_claim_coverage','list_discoverable_user_tool'}
 if is_read:
  result=turn.get('native_tool_result');err=False
  if isinstance(result,str):
   err=bool(re.match(r'^\s*(?:error\b|exception\b)',result,re.I))
   try:
    obj=json.loads(result)
    if isinstance(obj,dict):err=err or bool(obj.get('error')) or obj.get('success') is False
   except ValueError:pass
  elif isinstance(result,dict):err=bool(result.get('error')) or result.get('success') is False
  return 'errored_read' if err else ('benign_read' if result is not None else 'unresolved_read')
 if WRITE.match(name):return 'write'
 return 'unresolved'

def explicit_read_gate(candidate,call,policy,native_result=None):
 """Require exact tool and explicit log-before-call policy, not gold ordering.

 General 'verify before client information' prose is insufficient: identity
 lookup reads are explicitly necessary to perform verification. Missing a gold
 log step alone never proves this read was forbidden.
 """
 prefix=candidate['public_prefix']
 if any(m.get('type')=='function_call' and m.get('name')=='log_verification' for m in prefix):return None
 name=call['name'];base=re.sub(r'_\d+[a-z]?$', '',name)
 for line in str(policy).splitlines():
  if len(line)>2000:continue
  if not any(re.search(r'\b'+re.escape(n)+r'\b',line) for n in {name,base}):continue
  if not re.search(r'\b(?:must not|do not|cannot|never)\b',line,re.I):continue
  if re.search(r'\b(?:before|until)\b.{0,150}(?:log_verification|verification (?:is|has been) logged)',line,re.I):
   return dict(policy_quote=line,rule='Explicit exact-tool prohibition until verification is logged',logged_in_prefix=False)
 # One explicit business-read class in the policy: account balances/positions
 # before authentication. Do not confuse a missing log with absent verification:
 # any plausible supplied factor or completed-auth statement makes this a hold.
 policy_lines=str(policy).splitlines()
 quote=next((line for line in policy_lines if re.search(r'before you read an account balance, list positions',line,re.I) and re.search(r'authenticate',line,re.I)),None)
 positions=bool(re.match(r'^get_positions(?:_by_user|_\d+)?$',call['name']))
 balance=bool(re.match(r'^get_account_(?:summary|overview)(?:_\d+)?$',call['name']) and re.search(r'cash_balance|settled_cash|buying_power|account balance',str(native_result),re.I))
 spoken=[consent_given.text(m) for m in prefix if m.get('role')=='user']
 factors=any(re.search(r'\b(?:birth|born|SSN|social security|phone|address|password|passcode|OTP|verified|verification|authenticated)\b|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{4,6}\b',text,re.I) for text in spoken)
 factors=factors or any(re.search(r'\b(?:password|passcode|OTP|verification code|date of birth|last (?:four|4)|phone number|street address)\b',consent_given.text(m),re.I) for m in prefix if m.get('role')=='assistant')
 claimed=any(re.search(r'(?:verified|authenticated) your (?:identity|account)|identity (?:is|has been) verified',consent_given.text(m),re.I) for m in prefix if m.get('role')=='assistant')
 if quote and (positions or balance) and not factors and not claimed:
  return dict(policy_quote=quote,rule='Explicit balance/position-read prohibition before authentication; no supplied factor/auth-completion evidence in public prefix',logged_in_prefix=False,possible_authentication_evidence=False)
 return None

def gold(c):return c.get('verdict',{}).get('gold') or c['verification'].get('witness',{}).get('missing_gold')

def spoken_target(prefix,g):
 # A later database lookup never changes which order the caller is discussing.
 key=next((k for k in ['order_id','reservation_id','line_id','customer_id','user_id'] if k in g['arguments']),None)
 if key is None:return dict(ok=True,scope='no_target_key')
 known=confirmed_write.known_ids(prefix,key)|{str(g['arguments'][key])}
 for i in range(len(prefix)-1,-1,-1):
  m=prefix[i]
  if m.get('type')!='message' or m.get('role') not in {'assistant','user'}:continue
  ids=confirmed_write.target_ids(consent_given.text(m),key,known)
  if ids:return dict(ok=str(g['arguments'][key]) in ids,key=key,targets=sorted(ids),speech_index=i)
 # Preserve existing unique public identity proof when no spoken entity exists.
 return confirmed_write.discussion_target([m for m in prefix if m.get('type')!='function_call'],g)

def payment_binding(prefix,g,consent):
 key=next((k for k in ['payment_method_id','payment_id'] if k in g['arguments']),None)
 if key is None:return dict(ok=True)
 ident=str(g['arguments'][key]);expected=next((x for x in ['gift_card','credit_card','paypal'] if ident.startswith(x)),None)
 if not expected:return dict(ok=False,reason='payment_type_unresolved')
 witness=consent.get('witness')
 if witness:text=witness['offer']+' '+witness['reply']
 else:
  assents=consent.get('write_segment',{}).get('assents',[]);last=assents[-1] if assents else {};grounding=last.get('agent_grounding') or {};text=grounding.get('text','')+' '+last.get('reply','')
 active=confirmed_write.active_text(text)
 found={k for k,pattern in {'gift_card':r'gift[ _]card','credit_card':r'credit[ _]card|Visa|Mastercard|Amex','paypal':r'paypal'}.items() if re.search(pattern,active,re.I)}
 return dict(ok=not found or found=={expected},expected=expected,stated=sorted(found),confirmation=text,reason='payment_type_matches' if not found or found=={expected} else 'confirmed_payment_type_mismatch')

def k4_target(c):
 g=gold(c)
 if not g:return False,dict(reason='K4_gold_missing')
 target=spoken_target(c['public_prefix'],g)
 if not target['ok']:return False,dict(reason='K4_discussion_target_unproven_or_mismatch',target=target,gold=g)
 # Current caller cancellation withdrawal cannot authorize an obsolete question.
 users=[consent_given.text(m) for m in c['public_prefix'] if m.get('role')=='user']
 if users and re.search(r"\b(?:keep (?:the |my |these |those )?(?:orders?|reservations?|flights?) (?:as they are|as (?:is|they were))|do not cancel|don['’]t cancel|hold off|leave (?:it|them|everything) (?:as is|unchanged))\b",users[-1],re.I):
  return False,dict(reason='K4_latest_caller_withdrawal',target=target,latest_caller=users[-1])
 return True,dict(reason='K4_target_bound',target=target)

def aq_write_consent(prefix,g,consent):
 """A business-write offer/assent is distinct from identity verification assent."""
 witnesses=[]
 if consent.get('witness'):witnesses.append(consent['witness'])
 for assent in consent.get('write_segment',{}).get('assents',[]):
  grounding=assent.get('agent_grounding')
  if grounding:witnesses.append(dict(offer=grounding['text'],offer_index=grounding['index'],reply_index=assent['reply_index']))
 words=[w for w in re.sub(r'_\d+[a-z]?$', '',g['name']).split('_') if w not in {'set','update','modify','change','record','initiate','take','add','process','apply','submit','request','send','issue','to','in','out','one','time','user','account','by','and','of'} and len(w)>2]
 # Explicit native concepts use common public word forms without ID speech.
 stems={'guarantee':'guarant','delivery':'deliver','dividend':'dividend','beneficiary':'beneficiar','membership':'member','ancillary':'ancillar','occupancy':'occupan','wire':'wire','deposit':'deposit','payment':'pay','two':'two','contact':'contact'}
 anchors=[stems.get(w,w) for w in words]
 attempts=[]
 for w in witnesses:
  oi,ri=w['offer_index'],w['reply_index'];offer=consent_given.text(prefix[oi]);reply=consent_given.text(prefix[ri]);clause=consent_given.assent_clause(reply)
  if any(m.get('role') in {'assistant','user'} for m in prefix[oi+1:ri]):continue
  if any(reset.get('index',-1)>ri for reset in consent.get('resets',[])):continue
  identity_question=bool(re.search(r'\b(?:verify (?:your )?identity|verification (?:code|purposes)|date of birth|birth date|last (?:four|4)|full name|social security|SSN|security question|email address on file)\b',offer,re.I) and ('?' in offer or re.search(r'\b(?:please|provide|confirm|tell me)\b',offer,re.I)))
  business=bool(anchors and any(re.search(r'\b'+re.escape(a),offer,re.I) for a in anchors))
  affirmative=bool(consent_given.G_AFFIRM.search(clause) or consent_given.explicit_instruction(clause)) and '?' not in reply and not consent_given.G_VETO.search(clause)
  # Choice answers and acknowledgements of a completion claim are not a new offer.
  offer_request=bool(consent_given.G_OFFER.search(offer)) and not consent_given.G_CHOICE.search(offer)
  conflict=confirmed_write.conflicts(prefix,offer+' '+reply,g)
  evidence=dict(offer_index=oi,reply_index=ri,offer=offer,reply=reply,business_action_anchors=anchors,identity_question=identity_question,business_action_grounded=business,affirmative=affirmative,offer_request=offer_request,conflicts=conflict);attempts.append(evidence)
  if not identity_question and business and affirmative and offer_request and not conflict:return True,dict(reason='AQ_specific_business_offer_and_assent',witness=evidence)
 return False,dict(reason='AQ_specific_write_consent_unproven',attempts=attempts)

def bound_write_consent(c,g):
 consent=consent_given.detect(c['public_prefix'],g);evidence=dict(consent=consent,gold=g)
 if consent['consent_label']!='given':return False,dict(reason='lenient_consent_not_given',**evidence)
 if c['environment_source']=='areal':
  binding=g6_error_kinds.lenient_write_binding(c['public_prefix'],g,consent);target=spoken_target(c['public_prefix'],g);payment=payment_binding(c['public_prefix'],g,consent);keep=binding['ok'] and target['ok'] and payment['ok'];evidence.update(binding=binding,target=target,payment=payment);reason='AR_'+(binding['reason'] if not binding['ok'] else ('spoken_target_mismatch' if not target['ok'] else ('payment_mismatch' if not payment['ok'] else 'specific_consent_bound')))
 else:keep,binding=aq_write_consent(c['public_prefix'],g,consent);evidence['binding']=binding;reason=binding['reason']
 return keep,dict(reason=reason,**evidence)

def decide(c,context,policies):
 cat=c['category'];evidence={};reason='unchanged_eligible_category';keep=True
 if cat=='afterquery_wrong_action':
  turn=next(t for t in context['turns'] if t['turn_index']==c['turn']);typ=action_type(c['failed_output'],turn);evidence=dict(failing_action_type=typ,native_result=turn.get('native_tool_result'),raw_path=context['raw_path'],raw_sha256=context['raw_sha256']);keep=typ in {'write','terminal','errored_read'};reason='AQ_'+typ
  if typ=='benign_read':
   _,call=shape(c['failed_output']);gate=explicit_read_gate(c,call,policies[context['policy_key']],turn.get('native_tool_result'));evidence['policy_key']=context['policy_key'];evidence['explicit_preverification_prohibition']=gate
   if gate:keep=True;reason='AQ_read_explicitly_forbidden_before_logged_verification'
 elif cat in {'empty_turn','infeasible_or_rejected_write'}:
  typ,p=shape(c['positive_output']);approved=c['verification'].get('approved_calls',[])
  keep=typ=='call' and any(areal_v2.typed_equal(p,a) for a in approved)
  if cat=='infeasible_or_rejected_write':keep=keep and bool(READ.match(p['name'])) and bool(re.search(r'flight|seat|availability|eligible|eligibility|order_details|product_details',p['name'],re.I))
  reason=('K0_gold_call_positive' if cat=='empty_turn' else 'K9b_gold_availability_read_positive') if keep else ('K0_speech_or_unproven_due_call' if cat=='empty_turn' else 'K9b_speech_or_unproven_availability_read')
  evidence=dict(positive_shape=typ,approved_calls=approved)
  if keep and cat=='empty_turn' and action_type(c['positive_output'],{})=='write' and not re.match(r'^(?:log_verification|send_verification_code|confirm_identity_verification|confirm_otp)(?:_\d+)?$',p['name']):
   consent_ok,proof=bound_write_consent(c,p);evidence['write_consent']=proof
   if not consent_ok:keep=False;reason='K0_write_consent_unproven'
 elif cat=='consent_missing':keep,evidence=k4_target(c);reason=evidence['reason']
 elif cat=='missed_write':
  keep,evidence=bound_write_consent(c,gold(c));reason='K10_'+evidence['reason']
 elif cat=='unneeded_transfer_or_premature_end':
  # Retain the published K7 handoff/closing veto and all its stored controls.
  reason='K7_existing_handoff_closing_veto_retained'
 return dict(record_id=c['record_id'],keep=bool(keep),reason=reason,evidence=evidence,category=cat,domain=c['domain'],source=c['source'])
