#!/usr/bin/env python3
"""Build config JSONs + row files for the tau2-telecom profile on the pinned eval Gym runtime
(cb588bb1: has tau2 + openai_model + vllm_model venvs). One config per user-simulator variant.
Usage: stackB_tau2_config.py --out DIR --rows tau2_telecom.jsonl --model-url http://127.0.0.1:PORT/v1 --model hf399 [--n 128] [--k 4]"""
import argparse, json, os
E='/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_eval'
GYM=f'{E}/.runtime/gym-cs/cb588bb1'
HUB='https://inference-api.nvidia.com/v1'
SIMS={'hf399user': None, 'qwen': 'nvidia/qwen/qwen3.8-flash-next', 'gptoss': 'nvidia/openai/gpt-oss-120b'}
ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rows',required=True)
ap.add_argument('--model-url',required=True); ap.add_argument('--model',default='hf399'); ap.add_argument('--n',type=int,default=128)
ap.add_argument('--k',type=int,default=4); ap.add_argument('--top-p',type=float,default=1.0); ap.add_argument('--max-output-tokens',type=int,default=8192); ap.add_argument('--chat-template-kwargs',default=None,help='JSON for the policy vllm_model chat_template_kwargs (default Nemotron: enable_thinking true, truncate_history_thinking false)'); ap.add_argument('--policy-extra-body',default=None,help='JSON merged into the policy vllm_model extra_body (default skip_special_tokens false)'); ap.add_argument('--num-workers',type=int,default=32); ap.add_argument('--sim-base-url',default=None,help='override the user-simulator base URL (default NVIDIA hub); e.g. a local vLLM http://127.0.0.1:18100/v1'); ap.add_argument('--sim-api-key',default=None,help='api key for --sim-base-url (default: hub key from env NVIDIA_API_KEY; use dummy for local vLLM)'); ap.add_argument('--sims',default=None,help='comma list key=hub_model_id (key hf399user => policy as user)'); ap.add_argument('--max-agent-steps',type=int,default=40); ap.add_argument('--head-port',type=int,default=23000)
a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
if a.sims:
    SIMS={}
    for kv in a.sims.split(','):
        k,v=kv.split('=',1); SIMS[k]=(None if v in ('policy','none') else v)
rows=[json.loads(l) for l in open(a.rows)][:a.n]
for i,(key,sim) in enumerate(SIMS.items()):
    agent=f'tau2_{key}'; d=f'{a.out}/{agent}'; os.makedirs(d,exist_ok=True)
    with open(f'{d}/TASKS.jsonl','w') as f:
        for r in rows:
            r=dict(r); r['agent_ref']={'type':'responses_api_agents','name':agent}; f.write(json.dumps(r,ensure_ascii=False)+'\n')
    user_server='policy_model' if sim is None else f'sim_{key}'
    cfg={'config_paths':['responses_api_agents/tau2/configs/tau2.yaml','responses_api_models/vllm_model/configs/vllm_model.yaml'],
         'policy_base_url':a.model_url,'policy_model_name':a.model,'policy_api_key':'dummy',
         'responses_create_params':{'temperature':1.0,'top_p':a.top_p,'max_output_tokens':a.max_output_tokens},
         'head_server':{'host':'127.0.0.1','port':a.head_port+i*10},
         'port_range_low':33000+i*1000,'port_range_high':33999+i*1000,
         'skip_venv_if_present':True,'overwrite_metrics_conflicts':True,
         'results_dir':f'{d}/native_results','cache_dir':f'{E}/.runtime/gym-cs/cache','uv_cache_dir':f'{E}/.runtime/gym-cs/uv-cache',
         'nemo_gym_log_dir':f'{d}/gym_logs',
         'policy_model':{'responses_api_models':{'vllm_model':{'chat_template_kwargs':(json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else {'enable_thinking':True,'truncate_history_thinking':False}),
                          'extra_body':(dict({'skip_special_tokens':False}, **(json.loads(a.policy_extra_body) if a.policy_extra_body else {}))),'sequential_reasoning_allowed':False,'return_token_id_information':False}}},
         agent:{'responses_api_agents':{'tau2':{'_inherit_from':'tau2_agent',
                'user_model_server':{'type':'responses_api_models','name':user_server},
                'user_llm_args':{'temperature':0.7},
                'num_workers':a.num_workers,'max_steps':200,'max_agent_steps':a.max_agent_steps,'turns_remaining_interval':1,'print_step_counts':True,'debug':False,
                'datasets':[{'name':agent,'type':'train','jsonl_fpath':f'{d}/TASKS.jsonl','prompt_config':None,'num_repeats':a.k,'license':'MIT'}]}}}}
    # Gym's _inherit_from lives at the top level of the agent block in benchmark configs
    cfg[agent]={'_inherit_from':'tau2_agent','responses_api_agents':{'tau2':cfg[agent]['responses_api_agents']['tau2']}}
    cfg[agent]['responses_api_agents']['tau2'].pop('_inherit_from',None)
    if sim is not None:
        cfg[f'sim_{key}']={'responses_api_models':{'openai_model':{'entrypoint':'app.py','openai_base_url':(a.sim_base_url or HUB),'openai_model':sim,
                            'openai_api_key':(a.sim_api_key if a.sim_base_url else '${oc.env:NVIDIA_API_KEY}'),'openai_default_headers':{},'drop_input_reasoning_items':True}}}
    json.dump(cfg,open(f'{d}/config.json','w'),indent=1)
    print(agent, len(rows), 'rows x K', a.k, '->', d)
