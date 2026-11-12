"""Check archive completeness independently of the presentation layer."""
from pathlib import Path
import json,hashlib,collections
def verify(run):
 run=Path(run);m=json.loads((run/'manifest.json').read_text());settings=json.loads((run/'configs/settings.json').read_text())
 for name,digest in m['source_sha256'].items():assert hashlib.sha256((run/'sources'/name).read_bytes()).hexdigest()==digest,name
 builds=[json.loads(f.read_text()) for f in (run/'build').glob('*.json')]
 for b in builds:
  if b['exit_code']==0:assert hashlib.sha256((run/'build'/(b['config']['id']+'.so')).read_bytes()).hexdigest()==b['binary_sha256']
 tasks=[json.loads(f.read_text()) for f in (run/'state').glob('*.json')]
 for task in tasks:
  if 'job' not in task:continue
  job=json.loads(Path(task['job']).read_text());payload={key:job[key] for key in ['phase','configs','cases']}
  assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==task['payload_sha256']
  assert Path(task['raw']).exists() and Path(task['log']).exists()
  if task['status']=='PASS':
   values=[json.loads(l) for l in Path(task['raw']).read_text().splitlines()]
   ended={v['case_id'] for v in values if v.get('record_type')=='case_end'}
   assert ended=={c['id'] for c in job['cases']},task['label']
 complete=m['status'].startswith('COMPLETE')
 rows=json.loads((run/'results.json').read_text());perf=[v for v in rows if v['phase']=='performance' and v.get('current_successful_attempt')]
 if complete:
  cases=json.loads((run/'configs/performance.json').read_text());ids={c['id'] for c in cases}
  assert {v['case_id'] for v in perf}==ids
  assert all(v['status']=='PASS' and len(v['round_us'])==settings['timing']['rounds'] and v['gpu_median_us']>0 for v in perf)
  rejected={json.loads(f.read_text())['config']['id'] for f in (run/'state').glob('rejected-*.json')}
  assert not rejected&{v['config_id'] for v in perf}
  gate={}
  for f in (run/'state').glob('gate-*.json'):
   v=json.loads(f.read_text())
   if v['status']=='PASS':gate.setdefault(v['config_id'],set()).add(v['tool'])
  for v in perf:
   if v['kernel_id'] in {k['id'] for k in m['kernels']}:assert gate[v['config_id']]=={'memcheck','racecheck','synccheck','initcheck'}
  for case_id in ids:
   values=[v for v in perf if v['case_id']==case_id]
   assert {v['kernel_id'] for v in values}>={k['id'] for k in m['kernels']}|set(settings['baselines'])
   roles=json.loads((run/'configs'/(case_id+'-roles.json')).read_text())
   assert set(roles.values())<={v['config_id'] for v in values}
 result=dict(status='PASS',run_status=m['status'],complete_matrix_verified=complete,source_hashes=True,binary_hashes=True,task_payload_hashes=True,performance_records=len(perf),performance_cases=len({v['case_id'] for v in perf}),notes='Verification of recorded artifacts, not an additional numerical proof.')
 (run/'artifact_verification.json').write_text(json.dumps(result,indent=2));return result
