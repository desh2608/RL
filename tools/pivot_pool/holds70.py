"""INBOX71 narrowed admission holds. Pure public-prefix/record tests; no scorer calls."""
import collections,json,os,pathlib,re,sys
REF=pathlib.Path(os.environ.get('PIVOT_HOLD_REFERENCE',str(pathlib.Path(__file__).resolve().parents[1]/'selection_reference')))
sys.path.insert(0,str(REF))
# Old CPU-node Python can parse PEP604 annotations but cannot evaluate them.
# Defer annotations only in the selection helper imports; preserve pinned bytes
# and all function bodies. Native scorer execution is never invoked here.
import __future__,importlib.abc,importlib.machinery,importlib.util
class _ReferenceLoader(importlib.machinery.SourceFileLoader):
    def get_code(self,fullname):
        return compile(self.get_data(self.path),self.path,'exec',flags=__future__.annotations.compiler_flag,dont_inherit=True)
class _ReferenceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if not fullname.startswith('recipes.tau_voice_rl_pivot'):return None
        file=REF.joinpath(*fullname.split('.')).with_suffix('.py')
        if file.is_file():return importlib.util.spec_from_loader(fullname,_ReferenceLoader(fullname,str(file)))
        return None
sys.meta_path.insert(0,_ReferenceFinder())
from recipes.tau_voice_rl_pivot import consent_given as cg,confirmed_write as cw
from audit59 import shape
READ_REGISTRY=json.loads((REF/'READ_ONLY_TOOLS.json').read_text())['domains']
READ=re.compile(r'^(?:compute_|get_|calculate|search_|list_|find_|lookup_|read_|check_|quote_|scan_)',re.I)
DECLINE=re.compile(r"\b(?:pass on|no (?:more |further |any )?changes|don['’]?t want|do not want|not going to|never mind|nevermind|no thanks|I['’]ll keep|I will keep|cancel that|forget it|leave (?:it|them|everything) as (?:it is|is)|hold off|do not cancel|don['’]?t cancel)\b",re.I)
PAYMENT=re.compile(r'balance|payment|certificate|gift[ _-]?card',re.I)
AUTH=re.compile(r'^(?:log_verification|record_verification|confirm_identity_verification|confirm_otp|verify_otp|verify_identity)(?:_\d+[a-z]?)?$',re.I)
CODE=re.compile(r'\b(?:code|OTP|passcode|one.time.password)\b',re.I)

def messages(r):return r['public_prefix']
def last_user(r):
    return next(((i,cg.text(m)) for i,m in reversed(list(enumerate(messages(r)))) if m.get('role')=='user'),(None,''))
def clauses(text):return [s.strip() for s in re.split(r'(?<=[.!?;])\s+|\b(?:but|however|instead)\b',text,flags=re.I) if s.strip()]
def affirmative(text):
    for part in clauses(text):
        c=cg.assent_clause(part)
        if (cg.G_AFFIRM.search(c) or cg.G_SEGMENT_ASSENT.search(c) or cg.explicit_instruction(c)) and not cg.G_VETO.search(c) and not DECLINE.search(c) and '?' not in c:return True
    return False
def call(r):
    kind,value=shape(r.get('positive_output',[]))
    return value if kind=='call' else None
def read_tool(r,c):
    if not c:return False
    name=c['name'];base=re.sub(r'_\d+[a-z]?$','',name)
    if READ.match(base) or name in READ_REGISTRY.get(r['domain'],[]) or base in READ_REGISTRY.get(r['domain'],[]):return True
    for tool in r.get('tools',[]):
        t=tool.get('function',tool)
        if t.get('name')==name and (t.get('annotations',{}).get('readOnlyHint') is True or t.get('read_only') is True):return True
    return False
def native_error(r):
    found=[]
    def walk(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in {'native_error','native_tool_error'} and v:found.append(str(v))
                elif isinstance(v,(dict,list)):walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    walk(r.get('verification',{}))
    for x in [r.get('native_error'),r.get('native_tool_error')]:
        if x:found.append(str(x))
    if not found:
        result=r.get('audit59_decision',{}).get('evidence',{}).get('native_result')
        if isinstance(result,dict) and result.get('error'):found.append(str(result['error']))
        elif isinstance(result,str) and re.match(r'\s*(?:error|exception)',result,re.I):found.append(result)
    text='\n'.join(dict.fromkeys(found))
    cls='payment' if PAYMENT.search(text) else 'seats' if re.search(r'\bseats?\b|sold out|flight.*(?:full|unavailable)',text,re.I) else 'other'
    return text,cls
def afterquery(r):
    env=r.get('environment_source') or r.get('stage_a',{}).get('source','')
    return env=='afterquery' or r.get('verification',{}).get('kind','').startswith('afterquery_') or r['domain'].endswith('_knowledge')
def nonnegated(part):
    return not re.search(r"\b(?:not|never|no|don['’]?t|do not|can['’]?t|cannot|won['’]?t|wouldn['’]?t|unless|only if)\b|pass on|hold off",part,re.I)
def positive_text(r):
    return '\n'.join(cg.text(m) for m in r.get('positive_output',[]) if m.get('role')=='assistant')
def h1(r):
    if r['category']!='consent_missing':return None
    idx,text=last_user(r);parts=clauses(text)
    declines=[part for part in parts if DECLINE.search(part)]
    if not declines:return None
    positive=positive_text(r)
    families=[(r'refuel|add.{0,12}(?:data|GB)|data.{0,12}(?:GB|add)',r'\brefuel\w*|\badd\w*.{0,15}(?:data|GB)'),(r'cancel',r'\bcancel\w*'),(r'change|modify|update',r'\b(?:chang|modif|updat)\w*'),(r'return',r'\breturn\w*'),(r'exchange',r'\bexchang\w*')]
    verbs=[verb for marker,verb in families if re.search(marker,positive,re.I)]
    # Target IDs and numeric quantities from the positive, with numeric boundaries.
    tokens=set(re.findall(r'\b(?=[A-Za-z0-9_]*\d)[A-Za-z0-9_]{2,}\b|\b\d+(?:\.\d+)?\b',positive))
    # Item names in quoted spans or directly after the positive action are
    # literal targets too; do not use unrelated words from the conversation.
    for phrase in re.findall(r'[“\"]([^”\"]+)[”\"]',positive):
        tokens.update(re.findall(r'[A-Za-z]{3,}',phrase))
    for phrase in re.findall(r'\b(?:return(?:ing)?|exchang(?:e|ing)|purchas(?:e|ing))\s+(?:the\s+)?([A-Za-z ]{3,60})',positive,re.I):
        phrase=re.split(r'\b(?:for|with|from|to|and|do|please)\b',phrase,flags=re.I)[0]
        tokens.update(w for w in re.findall(r'[A-Za-z]{3,}',phrase) if w.lower() not in {'item','items','this','that'})
    for part in parts:
        if not nonnegated(part):continue
        hits=[v for v in verbs if re.search(v,part,re.I)]+[t for t in tokens if re.search(r'(?<![\w.])'+re.escape(t)+r'(?![\w.])',part,re.I)]
        if hits:return None
    return dict(reason='latest_caller_declines_without_positive_target_request',last_user_index=idx,last_user_message=text,decline_clauses=declines,positive_output=r.get('positive_output'),target_tokens=sorted(tokens),action_family_patterns=verbs)
def h2(r):
    if r['category']!='infeasible_or_rejected_write':return None
    error,cls=native_error(r);c=call(r)
    if cls=='payment' and c and read_tool(r,c) and not PAYMENT.search(c['name']):return dict(reason='payment_error_with_nonpayment_read_positive',native_error=error,native_error_class=cls,positive_call=c)
    return None
def h3(r):
    c=call(r)
    if r['category']=='missed_write' and read_tool(r,c):return dict(reason='missed_write_positive_is_read_only',positive_call=c,domain_read_registry=READ_REGISTRY.get(r['domain'],[]))
    return None
def all_calls(prefix):
    for i,m in enumerate(prefix):
        if m.get('type')!='function_call':continue
        kind,c=shape([m])
        if kind=='call':yield i,m,c
def code_evidence(r,c):
    prefix=messages(r);sent=[];outputs={m.get('call_id'):m.get('output') for m in prefix if m.get('type')=='function_call_output'}
    for i,m,k in all_calls(prefix):
        if not re.search(r'(?:send|deliver|issue|request).*(?:code|otp|passcode)',k['name'],re.I):continue
        result=outputs.get(m.get('call_id'))
        if result is None:continue
        if isinstance(result,str):
            try:result=json.loads(result)
            except ValueError:
                if re.match(r'\s*(error|exception)',result,re.I):continue
        if isinstance(result,dict) and (result.get('error') or result.get('success') is False):continue
        if isinstance(result,dict):
            status=str(result.get('status','')).lower()
            if status in {'failed','failure','error','pending'}:continue
        expected=set()
        def codes(x):
            if isinstance(x,dict):
                for key,value in x.items():
                    if re.fullmatch(r'(?:verification_)?(?:code|otp|passcode)',key,re.I) and re.fullmatch(r'\d{4,8}',str(value)):expected.add(str(value))
                    elif isinstance(value,(dict,list)):codes(value)
            elif isinstance(x,list):
                for v in x:codes(v)
        codes(result);sent.append((i,expected,'successful_send_tool',k['name']))
    target=cw.discussion_target(prefix,c)
    if not target['ok']:return dict(ok=False,reason='verification_target_unbound',target=target)
    for i,expected,kind,proof in reversed(sent):
        for j in range(i+1,len(prefix)):
            m=prefix[j]
            if m.get('role')!='user':continue
            t=cg.text(m);given=re.findall(r'\b(?:code|OTP|passcode)\b(?:\s+(?:is|was))?\s*[:=]?\s*["\']?(\d{4,8})\b',t,re.I)
            if re.fullmatch(r'\s*\d{4,8}[.!]?\s*',t):given.append(re.sub(r'\D','',t))
            if given and (not expected or expected.intersection(given)):
                return dict(ok=True,reason='code_sent_then_caller_supplied',sent_index=i,caller_index=j,send_proof_type=kind,send_proof=proof,caller_message=t,code_matches_public_send_result=bool(expected))
    return dict(ok=False,reason='no_sent_and_caller_supplied_code',send_events=len(sent))
def action_pattern(c):
    name=re.sub(r'_\d+[a-z]?$','',c['name']).lower()
    if 'cancell' in name or name.startswith('cancel'):return r'cancel|terminat' if 'membership' not in name else r'(?:cancel|terminat|stop|end).{0,45}(?:membership|renewal|subscription|annual|Plus)'
    if name.startswith('enroll'):return r'enroll|join|sign up|subscrib|put me on|start.{0,30}(?:Plus|membership)'
    if name.startswith('redeem'):return r'redeem|apply.{0,25}gift.card'
    if name.startswith('waive'):return r'waiv|remov.{0,20}fee'
    if name.startswith('send') and re.search('code|otp',name):return r'(?:send|email|text).{0,35}(?:code|otp|passcode)'
    if name.startswith('log'):return r'log|record'
    words=[w for w in name.split('_') if len(w)>2 and w not in {'user','account','policy','record','request','submit','process','update','modify','change','set','add','the','and','by'}]
    objects='|'.join(re.escape(w.rstrip('s')) for w in words[1:] if len(w)>3) or '|'.join(re.escape(w) for w in words)
    verb=name.split('_')[0]
    if verb in {'update','modify','change','set','add'}:verb='update|modify|chang|switch|set|add'
    return r'(?:'+verb+r').{0,100}(?:'+objects+r')' if objects else re.escape(name)
_BINDING=None
def argument_conflicts(r,c,reply_index):
    # Reuse the frozen INBOX62 binding functions, without its runtime-path loader.
    global _BINDING
    if _BINDING is None:
        import ast,calendar,schema_compare
        from audit59 import READ as read_pattern,WRITE as write_pattern
        module=ast.parse((pathlib.Path(__file__).parent/'audit62_pinned.py').read_text())
        nodes=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in {'speech','leaves','aliases','stated','conflicts'}]
        scope=dict(re=re,calendar=calendar,collections=collections,schema=schema_compare,cw=cw,consent_given=cg,READ=read_pattern,WRITE=write_pattern,AUTH=AUTH)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen_INBOX62_binding','exec',flags=__future__.annotations.compiler_flag),scope)
        _BINDING=scope
    scope=_BINDING
    f=shape(r.get('failing_output',[]))[1]
    if not isinstance(f,dict) or 'name' not in f:f=None
    return scope['conflicts'](dict(r,public_prefix=r['public_prefix'][:reply_index+1]),c,f)
def business_binding(r,c):
    prefix=messages(r);pattern=r'\b(?:'+action_pattern(c)+r')';offers=[]
    for i,m in enumerate(prefix):
        t=cg.text(m)
        if m.get('role')!='assistant' or not re.search(pattern,t,re.I):continue
        direction=r"(?:\b(?:may|can|shall|could|should) (?:I|we)\b|\b(?:I|we)(?:['’]ll| will| can)\b|\b(?:would|do) you (?:like|want|authorize|approve)\b)"
        proposal=bool(re.search(direction+r'.{0,180}(?:'+pattern+r')',t,re.I|re.S) or re.search(r'(?:'+pattern+r').{0,180}(?:do you confirm|shall I proceed|would you like to proceed|please confirm)',t,re.I|re.S))
        info=bool(re.search(r'\b(?:may|can|shall|could) I (?:check|search|look up|verify)\b',t,re.I))
        if proposal and not info:offers.append((i,t))
    if not offers:return dict(ok=False,reason='no_offer_of_positive_action_family',action_family_pattern=pattern)
    oi,offer=offers[-1];consent=cg.detect(prefix,c)
    send_code=bool(re.match(r'^send_.*(?:code|otp)',c['name'],re.I))
    conditional_code=bool(re.search(r'\b(?:need|require|once|after|before|must)\b.{0,90}\b(?:code|OTP|passcode)\b|\b(?:code|OTP|passcode)\b.{0,90}\bbefore\b',offer,re.I|re.S))
    if conditional_code and not send_code:
        code=code_evidence(r,c)
        if not code['ok']:return dict(ok=False,reason='conditional_write_requires_code_not_yet_supplied',offer_index=oi,offer=offer,verification_evidence=code)

    committed=max((i for i,m,k in all_calls(prefix) if k==c),default=-1)
    if committed>oi:return dict(ok=False,reason='same_positive_already_called_after_offer',offer_index=oi,call_index=committed)
    attempts=[]
    for ri in range(oi+1,len(prefix)):
        m=prefix[ri]
        if m.get('role')!='user':continue
        reply=cg.text(m)
        if not affirmative(reply):continue
        identity_question=bool(re.search(r'\b(?:verify (?:your )?identity|verification|date of birth|SSN|social security|email|phone|actively enrolled|enrolled this semester)\b',offer,re.I) and ('?' in offer or re.search(r'\b(?:please|provide|confirm|tell me)\b',offer,re.I)))
        business_instruction=bool(re.search(pattern,reply,re.I) and re.search(r"\b(?:please|go ahead|proceed|put me on|enroll me|file (?:the|that)|waive (?:the|that)|I want|I['’]d like)\b|(?:^|[.!;]\s*|\b(?:and|then)\s+)(?:cancel|start|enroll|join|redeem|issue|update|change|set|switch|file|waive|record|send|reissue|apply|execute|place|add)\b",reply,re.I))
        send_code=bool(re.match(r'^send_.*(?:code|otp)',c['name'],re.I))
        if identity_question and not business_instruction and not send_code:continue
        staged=bool(re.search(r'\b(?:first|once|then|after)\b',offer,re.I))
        factual_reply=bool(re.search(r'\b(?:gift.card|identity|birth|phone|email|enrolled|semester|orders?)\b',reply,re.I))
        if staged and factual_reply and not business_instruction and not re.search(pattern,reply,re.I) and not send_code:continue
        q=offer.rfind('?')
        if q>=0 and not re.search(pattern,offer[:q+1],re.I) and not business_instruction and not send_code:continue

        # A code/identity acknowledgement cannot authorize an unrelated write.
        between=[cg.text(x) for x in prefix[oi+1:ri] if x.get('role')=='assistant' and cg.text(x)]
        if between and re.search(r'\b(?:verify|verification|identity|code|OTP|date of birth|phone|email)\b',between[-1],re.I):continue
        if any(x.get('index',-1)>ri for x in consent.get('resets',[])):continue
        if any(DECLINE.search(cg.text(x)) for x in prefix[ri+1:] if x.get('role')=='user'):continue
        choice=bool(re.search(r'\bor\b',offer,re.I) and ('?' in offer or re.search(r'\b(?:please|provide|confirm|tell me)\b',offer,re.I)))
        if choice and not re.search(pattern,reply,re.I):continue
        conflicts=cw.conflicts(prefix,offer+' '+reply,c)+argument_conflicts(r,c,ri)
        target=cw.discussion_target(prefix[:ri+1],c)
        evidence=dict(offer_index=oi,caller_index=ri,offer=offer,caller_message=reply,action_family_pattern=pattern,conflicts=conflicts,target_binding=target,lenient_consent_label=consent['consent_label'])
        attempts.append(evidence)
        if not conflicts and target['ok']:return dict(ok=True,reason='latest_family_offer_with_lenient_assent_and_binding',witness=evidence)
    return dict(ok=False,reason='no_bound_caller_assent_since_latest_family_offer',offer_index=oi,offer=offer,action_family_pattern=pattern,attempts=attempts)
def otp_proof(r,c):
    events=[]
    for i,m in enumerate(messages(r)):
        t=cg.text(m)
        if m.get('role')=='assistant' and CODE.search(t) and re.search(r'\b(?:send|sent|provide|share|enter|give|read|tell|request)\w*\b',t,re.I):events.append((i,t))
    for i,m,k in all_calls(messages(r)):
        if re.search(r'(?:send|deliver|issue|request).*(?:code|otp|passcode)',k['name'],re.I):events.append((i,k['name']))
    boundary=max(events,default=(-1,''))
    for i,m in enumerate(messages(r)):
        if i>boundary[0] and m.get('role')=='user' and re.search(r'(?<!\d)\d{4,8}(?!\d)',cg.text(m)):
            return dict(ok=True,reason='user_code_after_latest_send_or_request',boundary=boundary,caller_index=i,caller_message=cg.text(m))
    return dict(ok=False,reason='no_user_code_after_latest_send_or_request',boundary=boundary)
def commerce(c):
    n=re.sub(r'_\d+[a-z]?$','',c['name']).lower()
    if re.search(r'two_factor|2fa|authenticat',n):return False # Security enrollment is not commerce.
    return bool(re.match(r'^(?:enroll_|subscribe|purchase)',n) or re.match(r'^upgrade_.*plan',n) or ('membership' in n and re.search(r'change|update|set|upgrade|tier',n)))
def product_terms(c):
    args=c.get('arguments') or {}
    return {k:str(v).lower() for k,v in args.items() if re.search(r'plan|tier|product|membership',k,re.I) and v is not None}
def plan_conflicts(t,c):
    terms=product_terms(c);errors=[]
    for k,v in terms.items():
        norm=v.replace('_',' ')
        if 'plan' in k:
            for word in ['annual','monthly','quarterly','yearly']:
                if re.search(r'\b'+word+r'\b',t,re.I) and word not in norm and not (word=='yearly' and 'annual' in norm):errors.append([k,word,v])
        if 'tier' in k:
            if re.search(r'\bstudent\b',t,re.I) and 'student' not in norm:errors.append([k,'student',v])
            if re.search(r'\b(?:basic|premium|plus|gold|silver|platinum)\b',t,re.I):
                for word in re.findall(r'\b(?:basic|premium|plus|gold|silver|platinum)\b',t,re.I):
                    if word.lower() not in norm:errors.append([k,word,v])
    return errors
def commerce_proof(r,c):
    prefix=messages(r);terms=product_terms(c)
    nouns=set(re.findall(r'[a-z]{3,}',c['name'].lower()))-{'enroll','subscribe','purchase','upgrade','update','change','set','plan','tier'}
    nouns.update(w for v in terms.values() for w in re.findall(r'[a-z]{3,}',v) if w not in {'monthly','annual'})
    product=r'\b(?:'+ '|'.join(re.escape(w.rstrip('s'))+r'\w*' for w in sorted(nouns or {'membership','subscription','plan'}))+r')\b'
    verb=r'\b(?:enroll|join|subscribe|purchase|buy|upgrade|sign up|put me on|start)\b'
    requests=[]
    for i,m in enumerate(prefix):
        if m.get('role')!='user':continue
        for part in re.split(r'(?<=[.!?;])\s+|\b(?:but|however|instead|and then|and)\b',cg.text(m),flags=re.I):
            if nonnegated(part) and re.search(verb,part,re.I) and re.search(product,part,re.I) and not plan_conflicts(part,c):requests.append(dict(index=i,clause=part))
    if requests:return dict(ok=True,reason='caller_requested_product_write',requests=requests,product_terms=terms)
    explanations=[(i,cg.text(m)) for i,m in enumerate(prefix) if m.get('role')=='assistant' and re.search(product,cg.text(m),re.I)]
    boundary=max(explanations,default=(-1,''))
    attempts=[]
    for i,m in enumerate(prefix):
        if i<=boundary[0] or m.get('role')!='user':continue
        t=cg.text(m);assent_parts=[part for part in re.split(r'(?<=[.!?;])\s+|\b(?:but|however|instead|and then|and)\b',t,flags=re.I) if nonnegated(part) and (affirmative(part) or re.search(verb,part,re.I))];conflicts=[error for part in assent_parts for error in plan_conflicts(part,c)]
        # A bare yes can bind only to a product-specific offer, not an identity/details question.
        previous=next((cg.text(x) for x in reversed(prefix[:i]) if x.get('role')=='assistant'),'')
        direct_offer=bool(re.search(r'(?:would|do) you (?:like|want)|(?:shall|may|can) I|do you confirm|proceed|enroll',previous,re.I) and re.search(product,previous,re.I))
        factual=bool(re.search(r'gift.?card|email|@|phone|date of birth|identifier|verification|code|ZIP',t,re.I));identity_question=bool(re.search(r'verify|identity|provide.{0,30}(?:email|phone|ID)|confirm.{0,30}(?:phone|ZIP|birth)',previous,re.I))
        bound=bool(re.search(product,t,re.I) or ((direct_offer or re.search(r'that sounds good|sounds good|that works',t,re.I)) and not factual and not identity_question))
        ok=any(nonnegated(part) and affirmative(part) for part in clauses(t)) and bound and not conflicts
        attempts.append(dict(index=i,message=t,product_bound=bound,plan_tier_conflicts=conflicts,assent=ok))
        if ok:return dict(ok=True,reason='product_bound_assent_after_explanation',boundary=boundary,attempts=attempts,product_terms=terms)
    return dict(ok=False,reason='no_product_bound_assent_after_latest_explanation',boundary=boundary,attempts=attempts,product_terms=terms)
def h4_analysis(r):
    c=call(r)
    if not afterquery(r) or not c:return dict(ok=True,reason='outside_afterquery_call_scope')
    n=c['name'].lower();method=str(c.get('arguments',{}).get('method','')).lower().replace('-','_').replace(' ','_')
    verification=bool(re.search(r'verif|otp|2fa',n) and re.search(r'log|record|confirm|verify',n))
    if verification and method in {'one_time_code','2fa','two_factor','two_factor_authentication','otp'}:return otp_proof(r,c)
    if re.search(r'concern|flag|report|note',n) or (n.startswith('log_') and not verification):return dict(ok=True,reason='internal_compliance_exempt')
    if not commerce(c):return dict(ok=True,reason='outside_opt_in_commerce_and_otp_scope')
    return commerce_proof(r,c)
def h4(r):
    proof=h4_analysis(r)
    return None if proof['ok'] else dict(reason='afterquery_narrow_authorization_hold',positive_call=call(r),authorization=proof)
def decide(r):
    return {name:e for name,f in [('H1',h1),('H2',h2),('H3',h3),('H4',h4)] if (e:=f(r)) is not None}
