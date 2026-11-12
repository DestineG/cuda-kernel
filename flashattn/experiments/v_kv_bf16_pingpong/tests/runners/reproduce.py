"""Rebuild a recorded configuration and run selected saved input cases."""
from pathlib import Path
import argparse,json,os,re,subprocess,sys,shutil
p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--output',required=True);p.add_argument('--config',required=True);p.add_argument('--phase',choices=['correctness','performance'],default='correctness');p.add_argument('--case',action='append');a=p.parse_args()
artifact=Path(a.artifact).resolve();out=Path(a.output).resolve()
if artifact==out:p.error('Use a separate output directory')
for folder in ['build','adapters','raw','logs','jobs']:(out/folder).mkdir(parents=True,exist_ok=True)
(out/'configs').mkdir(exist_ok=True)
shutil.copy2(artifact/'configs/settings.json',out/'configs/settings.json')
cid=a.config
if cid in ['official_fa3','torch_flash_perseq','torch_aten_varlen']:cfg=dict(id=cid,impl=cid,br=64,bc=64)
else:
 meta=json.loads((artifact/'build'/(cid+'.json')).read_text());cfg=meta['config']
 text=(artifact/'adapters'/(cid+'.cu')).read_text()
 text=re.sub(r'#include "[^\n"]*/sources/([^\n"]+)"',lambda m:'#include "'+str(artifact/'sources'/m[1])+'"',text)
 src=out/'adapters'/(cid+'.cu');src.write_text(text)
 cmd=['/usr/local/cuda-12.4/bin/nvcc','-std=c++17','-O3','-gencode=arch=compute_90a,code=sm_90a','-lineinfo','--shared','-Xcompiler=-fPIC','--ptxas-options=-v',str(src),'-o',str(out/'build'/(cid+'.so')),'-lcuda']
 with (out/'build'/(cid+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
cases=json.loads((artifact/'configs'/(a.phase+'.json')).read_text())
if a.case:cases=[c for c in cases if c['id'] in a.case]
if not cases:p.error('No matching cases')
job=dict(id='reproduce',phase=a.phase,configs=[cfg],cases=cases,output=str(out/'raw/results.jsonl'))
path=out/'jobs/reproduce.json';path.write_text(json.dumps(job,indent=2))
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
subprocess.run([sys.executable,str(artifact/'harness/runners/worker.py'),'--run',str(out),'--job',str(path)],env=env,check=True,timeout=600)
