"""CUDA-only sanitizer/application gate, persistent per config and per tool."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,tarfile,urllib.request
def setup(run,settings,env):
 tool=run/'toolchain';tool.mkdir(exist_ok=True)
 checker=tool/'cuda_sanitizer_api-linux-x86_64-12.8.93-archive/bin/compute-sanitizer'
 if not checker.exists():
  archive=tool/'sanitizer.tar.xz'
  if not archive.exists():urllib.request.urlretrieve('https://developer.download.nvidia.com/compute/cuda/redist/cuda_sanitizer_api/linux-x86_64/cuda_sanitizer_api-linux-x86_64-12.8.93-archive.tar.xz',archive)
  assert hashlib.sha256(archive.read_bytes()).hexdigest()=='ae3574f052c0e06c95305962668eb1fe6ab571dfbb58b305fdb14d523bb1b240'
  with tarfile.open(archive) as t:t.extractall(tool,filter='data')
 runner=run/'build/sanitizer_runner'
 if not runner.exists():subprocess.run([settings['nvcc'],'-std=c++17','-O2','-lineinfo',str(run/'harness/runners/sanitizer.cu'),'-o',str(runner),'-ldl'],check=True,env=env)
 return checker,runner
def check(run,c,settings,env):
 checker,runner=setup(run,settings,env)
 for tool in ['memcheck','racecheck','synccheck','initcheck']:
  state=run/'state'/('gate-'+tool+'-'+c['id']+'.json')
  if state.exists():
   if json.loads(state.read_text())['status']=='PASS':continue
   return False
  log=run/'logs'/('gate-'+tool+'-'+c['id']+'.log')
  cmd=[str(checker),'--tool',tool,'--error-exitcode','99',str(runner),str(run/'build'/(c['id']+'.so')),str(c['br']),str(c['bc'])]
  with log.open('w') as f:
   p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   try:code=p.wait(timeout=settings['sanitizer_timeout_s'])
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();code=124
  text=log.read_text(errors='replace');completed=text.count('PASS D=')
  ok=code==0 and completed==4 and ('ERROR SUMMARY: 0 errors' in text or 'RACECHECK SUMMARY: 0 hazards' in text)
  row=dict(config_id=c['id'],config=c,tool=tool,status='PASS' if ok else 'FAIL',exit_code=code,completed=completed,expected=4,command=cmd,log=str(log),checker='Compute Sanitizer 2025.1.0.0')
  state.write_text(json.dumps(row,indent=2))
  with (run/'raw/gates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print('GATE',c['id'],tool,row['status'],flush=True)
  if not ok:return False
 return True
