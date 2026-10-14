#!/usr/bin/env python3
"""Each training node verifies the allocation's shared sandbox is reachable."""
import json,os,socket,sys,urllib.request
from pathlib import Path
run=Path(sys.argv[1]);job=sys.argv[2]
value=json.loads((run/'sandbox'/job/'READY.json').read_text())
assert value['job_id']==job==os.environ['NVC_GYM_SANDBOX_JOB_ID']
assert value['address']==os.environ['NEMO_SKILLS_SANDBOX_HOST']
assert str(value['port'])==os.environ['NEMO_SKILLS_SANDBOX_PORT']
assert value['stateful_execution_canary'] is True and value['shell_execution_canary'] is True
with urllib.request.urlopen(f"http://{value['address']}:{value['port']}/health",timeout=15) as response:
 assert response.status==200
 health=json.load(response)
assert health['status']=='healthy' and int(health['worker']) in range(value['workers'])
p=run/'sandbox'/job/('CLIENT_'+socket.gethostname()+'.json')
p.write_text(json.dumps({'job_id':job,'host':socket.gethostname(),'endpoint_address':value['address'],'endpoint_port':value['port'],'status':'reachable','worker':health['worker']},indent=2)+'\n')
