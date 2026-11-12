"""Build/validation/search orchestration; GPU work is serialized under a host lock."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import collections,datetime,fcntl,hashlib,itertools,json,math,os,shutil,signal,statistics,subprocess,sys,time,traceback,zipfile
TESTS=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def inspect_environment():
 import torch,importlib.metadata,flash_attn_3.flash_attn_config as flags
 result=dict(time=now(),python=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),fa3_version=importlib.metadata.version('flash_attn_3'),fa3_build_flags=flags.CONFIG,baselines=['official_fa3','torch_aten_varlen','torch_flash_perseq'])
 if not result['cuda_available']:raise RuntimeError('CUDA is unavailable')
 for name,cmd in [('gpu',['nvidia-smi','-q']),('nvcc',['/usr/local/cuda-12.4/bin/nvcc','--version']),('gcc',['gcc','--version'])]:
  p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);result[name]=dict(command=cmd,exit_code=p.returncode,output=p.stdout)
 q=torch.randn((65,4,64),device='cuda',dtype=torch.bfloat16);k=torch.randn((130,1,64),device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k);cq=torch.tensor([0,65],device='cuda',dtype=torch.int32);ck=torch.tensor([0,130],device='cuda',dtype=torch.int32)
 from flash_attn_3.flash_attn_interface import flash_attn_varlen_func
 o=flash_attn_varlen_func(q,k,v,cq,ck,65,130,causal=True,num_splits=0)
 native=torch.ops.aten._flash_attention_forward(q,k,v,cq,ck,65,130,0.,True,False)[0]
 torch.cuda.synchronize();result['baseline_smoke']=dict(finite=bool(torch.isfinite(o).all() and torch.isfinite(native).all()),max_abs_difference=(o.float()-native.float()).abs().max().item(),aten_schema=str(torch.ops.aten._flash_attention_forward.default._schema))
 if not result['baseline_smoke']['finite']:raise RuntimeError('Baseline nonfinite output')
 return result
def create_run(config_path,run_root=None,name=None):
 config_path=config_path.resolve();settings=read(config_path);kernels=read(config_path.parent/settings['kernels']);source=(config_path.parent/settings['source_root']).resolve()
 if len(kernels)!=10:raise RuntimeError('Expected all ten FA3 kernel entries')
 root=run_root or TESTS/'runs';root.mkdir(parents=True,exist_ok=True)
 label=name or settings['name']
 if not label.replace('-','').replace('_','').isalnum():raise ValueError('Invalid run name')
 stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
 run=root/(stamp+'-'+label);run.mkdir()
 for folder in ['sources','configs','build','adapters','raw','logs','jobs','state']:(run/folder).mkdir()
 shutil.copytree(TESTS,run/'harness',ignore=shutil.ignore_patterns('runs','__pycache__','*.pyc'))
 required=[v['source'] for v in kernels]+['mbarrier.cuh','tma.cuh','wgmma.cuh']
 for filename in required:shutil.copy2(source/filename,run/'sources'/filename)
 write(run/'configs/settings.json',settings);write(run/'configs/kernels.json',kernels)
 for key in ['correctness','performance']:shutil.copy2((config_path.parent/settings[key]).resolve(),run/'configs'/(key+'.json'))
 manifest=dict(status='CREATED',created_at=now(),name=label,source_directory=str(source),source_sha256={n:hashlib.sha256((run/'sources'/n).read_bytes()).hexdigest() for n in required},settings_sha256=hashlib.sha256((run/'configs/settings.json').read_bytes()).hexdigest(),kernels=kernels,notes=['All ten kernels freshly compiled; no old performance results reused.','No kernel code changes inside test adapters.','Public SDPA per-sequence and private ATen varlen baselines are separate.'])
 write(run/'manifest.json',manifest);(root/'latest-run.txt').write_text(str(run));return run
class Engine:
 def __init__(self,run):
  self.run=run;self.s=read(run/'configs/settings.json');self.kernels=read(run/'configs/kernels.json');self.byname={k['id']:k for k in self.kernels}
  self.env=dict(os.environ,PATH='/root/miniconda3/bin:/usr/local/cuda-12.4/bin:'+os.environ.get('PATH',''),CUDA_HOME='/usr/local/cuda-12.4',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
 def configs(self,n,br=64,bc=64,rr=False,pr=40,cr=232,qb=64,kb=64,ob=64):
  k=self.byname[n];cid=f'{n}_r{br}_c{bc}_rr{int(rr)}_p{pr}_a{cr}_q{qb}_kv{kb}_o{ob}'
  return dict(id=cid,impl=n,br=br,bc=bc,rr=rr,producer_regs=pr,consumer_regs=cr,qbox=qb,kvbox=kb,obox=ob,k_stages=k['k_stages'],v_stages=k['v_stages'],barrier_bytes=k['barrier_bytes'])
 def logrecord(self,kind,row):
  with (self.run/'raw'/(kind+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
 def build(self,c):
  cid=c['id'];meta=self.run/'build'/(cid+'.json')
  if meta.exists():return read(meta)['exit_code']==0
  s=(self.run/'harness/adapters/template.cu').read_text();k=self.byname[c['impl']]
  s=s.replace('../sources/initial/fa3_bf16_sm90.cu',str(self.run/'sources'/k['source'])).replace('fa3_bf16_forward',k['symbol'])
  s=s.replace('constexpr int bytes=(R+2*C)*D*2+16;',f'constexpr int bytes=(R+{c["k_stages"]+c["v_stages"]}*C)*D*2+{c["barrier_bytes"]};')
  params=f'R,C,D,CA,{c["qbox"]},64,{c["kvbox"]},64,{c["obox"]},32,64,64,16,{str(c["rr"]).lower()},{c["producer_regs"]},{c["consumer_regs"]}'
  s=s.replace('<R,C,D,CA,64,64,64,64,64,32,64,64,16>','<'+params+'>').replace('_launcher<R,C,D,CA>','_launcher<'+params+'>')
  for tensor,field in [('q','qbox'),('k','kvbox'),('v','kvbox')]:s=s.replace('swizzle128B<64,64>('+tensor+',',f'swizzle128B<{c[field]},64>('+tensor+',')
  s=s.replace('swizzle128B<64,32>(o,',f'swizzle128B<{c["obox"]},32>(o,')
  s='\n'.join(l for l in s.splitlines() if 'return dispatch<' not in l)
  dispatch='\n'.join(f'if(x->br=={c["br"]} && x->bc=={c["bc"]} && x->d=={d} && x->causal=={ca}) return dispatch<{c["br"]},{c["bc"]},{d},{str(bool(ca)).lower()}>(x,stream,mode);' for d in [64,128] for ca in [0,1])
  s=s.replace('int select(Context* x,cudaStream_t stream,int mode) {','int select(Context* x,cudaStream_t stream,int mode) {\n'+dispatch)
  source=self.run/'adapters'/(cid+'.cu');source.write_text(s)
  cmd=[self.s['nvcc'],'-std=c++17','-O3','-gencode=arch=compute_90a,code='+self.s['architecture'],'-lineinfo','--shared','-Xcompiler=-fPIC','--ptxas-options=-v',str(source),'-o',str(self.run/'build'/(cid+'.so')),'-lcuda']
  log=self.run/'build'/(cid+'.log');start=time.monotonic()
  code=self.process(cmd,log,self.s['build_timeout_s'])
  row=dict(config=c,exit_code=code,command=cmd,log=str(log),elapsed_s=time.monotonic()-start)
  if code==0:row['binary_sha256']=hashlib.sha256((self.run/'build'/(cid+'.so')).read_bytes()).hexdigest()
  write(meta,row);print('BUILD',cid,code,flush=True);return code==0
 def builds(self,configs):
  unique=list({c['id']:c for c in configs}.values())
  with ThreadPoolExecutor(max_workers=self.s['build_jobs']) as pool:results=list(pool.map(self.build,unique))
  return [c for c,ok in zip(unique,results) if ok]
 def process(self,cmd,log,timeout):
  with Path(log).open('w') as f:
   p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=self.env,start_new_session=True)
   try:return p.wait(timeout=timeout)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();return 124
 def task(self,phase,configs,cases,label):
  # Each attempt has separate raw/log files. A successful task is reusable only
  # within this run, whose source/settings hashes are checked before resume.
  payload=dict(phase=phase,configs=list({c['id']:c for c in configs}.values()),cases=cases)
  digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
  state=self.run/'state'/(label+'.json')
  previous=read(state) if state.exists() else {}
  if previous.get('status')=='PASS' and previous.get('payload_sha256')==digest:return
  attempt=previous.get('attempt',0)+1;jobid=label+f'.attempt{attempt}'
  job=dict(id=jobid,**payload,output=str(self.run/'raw'/(jobid+'.jsonl')))
  path=self.run/'jobs'/(jobid+'.json');write(path,job);log=self.run/'logs'/(jobid+'.log')
  row=dict(label=label,phase=phase,attempt=attempt,status='RUNNING',payload_sha256=digest,job=str(path),raw=job['output'],log=str(log),started_at=now());write(state,row)
  print('TASK',label,'configs',len(configs),'cases',len(cases),flush=True)
  cmd=[sys.executable,str(self.run/'harness/runners/worker.py'),'--run',str(self.run),'--job',str(path)]
  code=self.process(cmd,log,self.s['worker_timeout_s']);row.update(exit_code=code,status='PASS' if code==0 else ('TIMEOUT' if code==124 else 'FAIL'),completed_at=now());write(state,row)
  print('TASK_EXIT',label,code,flush=True)
  if code:raise RuntimeError('Task stopped for review: '+label+'; '+str(log))
 def records(self,phase):
  records=[]
  for f in (self.run/'state').glob('*.json'):
   state=read(f)
   if state.get('phase')==phase and state.get('status')=='PASS':
    p=Path(state['raw']);records += [json.loads(l) for l in p.read_text().splitlines() if l]
  return records
 def ranking(self,phases,pool,n,d,ca,filter_fn=lambda c:True):
  latest={}
  for phase in phases:
   for v in self.records(phase):
    if v.get('status')=='PASS' and 'median_us' in v:latest[(v['config_id'],v['case_id'])]=v
  values=[]
  for c in {c['id']:c for c in pool}.values():
   if c['impl']!=n or not filter_fn(c):continue
   rows=[v for (cid,_),v in latest.items() if cid==c['id'] and v['input']['d']==d and v['input']['causal']==ca]
   if len(rows)==3:values.append((math.exp(statistics.mean(math.log(v['median_us']) for v in rows)),c))
  return sorted(values,key=lambda v:v[0])
 def gate(self,c,correct):
  rejected=self.run/'state'/('rejected-'+c['id']+'.json')
  if rejected.exists():return False
  self.task('selected_correctness',[c],correct,'selected-'+c['id'])
  import gate
  result=gate.check(self.run,c,self.s,self.env)
  if not result:write(rejected,dict(status='REJECTED',config=c,reason='sanitizer/application output gate',time=now()))
  return result
 def stages(self):
  correct=read(self.run/'configs/correctness.json');perf=read(self.run/'configs/performance.json');search=self.s['search'];names=list(self.byname)
  reps=[c for c in perf if (c['ql']==[2048] and c['hk']==16) or (c['ql']==[128]*8 and c['kl']==[8192]*8 and c['hk']==4) or (c['ql']==[257,385,513,769,1025,1281,1537,1793] and c['hk']==1)]
  assert len(reps)==12
  baselines=[dict(id=n,impl=n,br=None,bc=None) for n in self.s['baselines']]
  defaults=self.builds([self.configs(n,rr=rr) for n in names for rr in [False,True]])
  assert len(defaults)==20
  for n in self.s['baselines']+names:self.task('correctness',[c for c in baselines+defaults if c['impl']==n],correct,'correctness-'+n)
  # New buffering changes synchronization. Gate all common controls before tuning.
  for c in defaults:
   if not self.gate(c,correct):raise RuntimeError('Common baseline gate failed: '+c['id'])
  coarse=self.builds([self.configs(n,br,bc,rr) for n in names for br,bc in search['tiles'] for rr in [False,True]])
  for n in names:self.task('tile_search',[c for c in coarse if c['impl']==n],reps,'tile-'+n)
  top_tiles={};register=[]
  for n in names:
   for d,ca in itertools.product([64,128],[False,True]):
    ranked=self.ranking(['tile_search'],coarse,n,d,ca);assert ranked,(n,d,ca)
    tiles=list(dict.fromkeys((c['br'],c['bc']) for _,c in ranked));top_tiles[f'{n}:{d}:{int(ca)}']=tiles[:search['joint_tiles_per_group']]
    for br,bc in tiles[:search['register_tiles_per_group']]:register += [self.configs(n,br,bc,True,pr,cr) for pr,cr in itertools.product(search['producer_regs'],search['consumer_regs'])]
  write(self.run/'configs/top_tiles.json',top_tiles)
  register=self.builds(register)
  for n in names:
   group=[c for c in register if c['impl']==n]
   for i in range(0,len(group),28):self.task('register_search',group[i:i+28],reps,f'register-{n}-{i}')
  joint=[]
  for n in names:
   for d,ca in itertools.product([64,128],[False,True]):
    ranked=self.ranking(['register_search'],register,n,d,ca);assert ranked
    pairs=[(c['producer_regs'],c['consumer_regs']) for _,c in ranked[:search['joint_register_pairs']]]
    for br,bc in top_tiles[f'{n}:{d}:{int(ca)}']:
     for pr,cr in pairs:joint.append(self.configs(n,br,bc,True,pr,cr))
     pr,cr=pairs[0]
     for qb,kb,ob in itertools.product(*[[v for v in search['box_rows'] if br%v==0],[v for v in search['box_rows'] if bc%v==0],[v for v in search['box_rows'] if br%v==0]]):joint.append(self.configs(n,br,bc,True,pr,cr,qb,kb,ob))
  joint=self.builds(joint)
  for n in names:
   group=[c for c in joint if c['impl']==n]
   for i in range(0,len(group),24):self.task('joint_search',group[i:i+24],reps,f'joint-{n}-{i}')
  pool=coarse+register+joint;selected={};off={};default_best={}
  phases=['tile_search','register_search','joint_search']
  for n in names:
   for d,ca in itertools.product([64,128],[False,True]):
    key=f'{n}:{d}:{int(ca)}'
    for role,target,filter_fn in [('tuned',selected,lambda c:True),('best_off',off,lambda c:not c['rr']),('best_default',default_best,lambda c:c['rr'] and c['producer_regs']==40 and c['consumer_regs']==232)]:
     for _,candidate in self.ranking(phases,pool,n,d,ca,filter_fn):
      if self.gate(candidate,correct):target[key]=candidate;break
     else:raise RuntimeError('No fully validated candidate for '+key+' '+role)
     print('SELECT',key,role,target[key]['id'],flush=True)
  # A config rejected under another D is conservatively invalid for all roles.
  rejected={read(f)['config']['id'] for f in (self.run/'state').glob('rejected-*.json')}
  for target,filter_fn in [(selected,lambda c:True),(off,lambda c:not c['rr']),(default_best,lambda c:c['rr'] and c['producer_regs']==40 and c['consumer_regs']==232)]:
   for key,c in list(target.items()):
    if c['id'] not in rejected:continue
    n,d,ca=key.split(':')
    for _,candidate in self.ranking(phases,pool,n,int(d),bool(int(ca)),filter_fn):
     if self.gate(candidate,correct):target[key]=candidate;break
    else:raise RuntimeError('No replacement: '+key)
  for name,value in [('selected',selected),('selected_off',off),('selected_default',default_best)]:write(self.run/'configs'/(name+'.json'),value)
  def final_configs(c):
   cs=baselines.copy();roles={}
   for n in names:
    key=f'{n}:{c["d"]}:{int(c["causal"])}'
    for role,config in [('common_off',self.configs(n)),('common_default',self.configs(n,rr=True)),('best_off',off[key]),('best_default',default_best[key]),('tuned',selected[key])]:cs.append(config);roles[n+':'+role]=config['id']
   write(self.run/'configs'/(c['id']+'-roles.json'),roles);return cs
  for c in perf:self.task('performance',final_configs(c),[c],'performance-'+c['id'])
  for c in reps:self.task('confirmation',list(reversed(final_configs(c))),[c],'confirmation-'+c['id'])
def execute(run):
 run=Path(run).resolve();manifest=read(run/'manifest.json')
 for n,digest in manifest['source_sha256'].items():
  if hashlib.sha256((run/'sources'/n).read_bytes()).hexdigest()!=digest:raise RuntimeError('Snapshot changed: '+n)
 if hashlib.sha256((run/'configs/settings.json').read_bytes()).hexdigest()!=manifest['settings_sha256']:raise RuntimeError('Run settings changed; create new run')
 with open('/tmp/fa3-test-gpu.lock','w') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('Another FA3 test process holds the GPU lock')
  try:
   if 'environment' not in manifest:manifest['environment']=inspect_environment()
   manifest.update(status='RUNNING',resumed_at=now());write(run/'manifest.json',manifest)
   e=Engine(run);e.stages()
   rejected=[read(f) for f in (run/'state').glob('rejected-*.json')]
   manifest.update(status='COMPLETE_WITH_REJECTED_CONFIGS' if rejected else 'COMPLETE',rejected_config_ids=[v['config']['id'] for v in rejected],completed_at=now())
  except Exception as error:
   manifest.update(status='STOPPED_FOR_REVIEW',error=str(error),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
  finally:write(run/'manifest.json',manifest);report(run)
 if manifest['status'].startswith('COMPLETE'):archive(run)
def report(run):
 subprocess.run([sys.executable,str(Path(run)/'harness/reporting/generate.py'),'--run',str(run)],check=True)
def archive(run):
 run=Path(run);report(run)
 from verify import verify
 print('VERIFY',verify(run),flush=True)
 # Preserve the exact installed official package, since its original git commit
 # may not be recoverable from a wheel. Do not bundle all of PyTorch.
 import importlib.metadata
 dist=importlib.metadata.distribution('flash_attn_3');package=Path(dist.locate_file('flash_attn_3'))
 shutil.copytree(package,run/'baseline_environment/flash_attn_3',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 files=[p for p in run.rglob('*') if p.is_file() and not {'reference-cache','__pycache__','toolchain'}&set(p.parts) and not (p.suffix in ['.so','.o'] and 'build' in p.parts) and p.name not in ['SHA256.json','sanitizer_runner']]
 write(run/'SHA256.json',{str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files});files.append(run/'SHA256.json')
 target=run.parent/(run.name+'.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,run.name+'/'+str(p.relative_to(run)))
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
 print('ARCHIVE',target,target.stat().st_size,hashlib.sha256(target.read_bytes()).hexdigest(),flush=True)
