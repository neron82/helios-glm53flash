#!/usr/bin/env python3
"""Read-only end-state audit; never starts, stops, or writes production configuration."""
import argparse,datetime,hashlib,json,os,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent/'results/determinism'
def call(args): return subprocess.run(args,check=True,capture_output=True,text=True).stdout.strip()
def get(port,path): return json.loads(call(['curl','-fsS','--max-time','5',f'http://127.0.0.1:{port}/{path}']))
parser=argparse.ArgumentParser();parser.add_argument('--no-write',action='store_true');args=parser.parse_args()
pid=Path(os.environ.get('XDG_RUNTIME_DIR','/run/user/1000')+'/helios-glm53flash-abl-8080.pid').read_text().split()[0]
cgroup=Path('/proc/'+pid+'/cgroup').read_text().strip()
unit=cgroup.rsplit('/',1)[1]
result=dict(time=datetime.datetime.now().astimezone().isoformat(),pid=int(pid),cgroup=cgroup,
            binary_sha256=hashlib.file_digest(open('/proc/'+pid+'/exe','rb'),'sha256').hexdigest(),
            health=get(8080,'health'),models=get(8080,'v1/models'),
            omnivoice=call(['systemctl','--user','is-active','omnivoice.service']),omnivoice_health=get(8880,'health'),
            bge_pid=26170,bge_health=get(7777,'health'),bge_listener=call(['ss','-H','-ltnp','sport = :7777']),
            restore_properties=call(['systemctl','--user','show',unit,'-p','Type','-p','RemainAfterExit','-p','KillMode','-p','MemoryMax','-p','ActiveState']))
assert result['binary_sha256']=='1a9cf7dc0557207a717746d796d224a5a94d4e99bf837c95f839061752c7a813'
assert result['health']=={'status':'ok'}
assert any(x['id']=='glm-5.3-flash-exl3' and x['context_length']==262144 for x in result['models']['data'])
assert result['omnivoice']=='active' and result['omnivoice_health']['ready']
assert result['bge_health']=={'status':'ok'} and 'pid=26170,' in result['bge_listener']
assert 'helios-prod-restore-det-' in cgroup
assert all(x in result['restore_properties'] for x in ['Type=oneshot','RemainAfterExit=yes','KillMode=none','MemoryMax=infinity','ActiveState=active'])
if not args.no_write: (OUT/'final-production-state.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
