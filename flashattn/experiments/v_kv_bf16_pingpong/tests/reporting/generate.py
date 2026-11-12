"""Convert immutable raw attempts and task state to one canonical result dataset."""
from pathlib import Path
import argparse,json,csv,collections,re,hashlib
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
def read(p):return json.loads(Path(p).read_text())
m=read(run/'manifest.json');labels={v['id']:v['label'] for v in m['kernels']};labels.update(official_fa3='官方FA3',torch_aten_varlen='PyTorch ATen varlen',torch_flash_perseq='PyTorch逐序列Flash')
resources={};buildrows=[]
for p in (run/'build').glob('*.json'):
 b=read(p);c=b['config'];text=Path(b['log']).read_text(errors='replace');buildrows.append(b)
 for match in re.finditer(r"Compiling entry function '([^']+)'(.*?)(?=ptxas info    : Compiling entry function|\Z)",text,re.S):
  sym,body=match.groups();dims=re.search(r'ILm(\d+)ELm(\d+)ELm(\d+)ELb([01])E',sym)
  if not dims:continue
  br,bc,d,ca=map(int,dims.groups());reg=re.search(r'Used (\d+) registers',body);spill=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',body)
  resource=dict(static_registers=int(reg[1]) if reg else None,smem_bytes=(c['consumer_wgs']*br+(c['k_stages']+c['v_stages'])*bc)*d*2+c['barrier_bytes'],compiler_warnings=[l for l in text.splitlines() if 'C751' in l])
  if spill:resource.update(stack_bytes=int(spill[1]),spill_stores=int(spill[2]),spill_loads=int(spill[3]))
  resources[(c['id'],d,bool(ca))]=resource
roles={p.name.split('-roles')[0]:read(p) for p in (run/'configs').glob('*-roles.json')}
selected=read(run/'configs/selected.json') if (run/'configs/selected.json').exists() else {}
taskstates={};rows=[]
for p in (run/'state').glob('*.json'):
 state=read(p)
 if 'raw' in state:taskstates[Path(state['raw']).name]=state
for p in sorted((run/'raw').glob('*.jsonl')):
 state=taskstates.get(p.name);latest=bool(state and state['status']=='PASS')
 for line in p.read_text().splitlines():
  if not line:continue
  v=json.loads(line)
  if v.get('record_type')=='case_end':continue
  if 'status' not in v:continue
  c=v.get('config',{});i=v.get('input',{});phase=v.get('phase',state.get('phase') if state else 'sanitizer');cid=v.get('config_id',c.get('id'))
  n=c.get('impl');case=v.get('case_id');views=[]
  if phase=='performance':
   if n in ['official_fa3','torch_aten_varlen','torch_flash_perseq']:views=['common_off','common_default','best_off','best_default','tuned']
   else:views=[role.split(':')[-1] for role,target in roles.get(case,{}).items() if target==cid and role.startswith(n+':')]
  row=dict(run_id=run.name,row_id=len(rows),phase=phase,attempt=state.get('attempt') if state else None,current_successful_attempt=latest,case_id=case,config_id=cid,kernel=labels.get(n,n),kernel_id=n,family=case.split('_',1)[1] if case and '_' in case else None,batch=len(i['ql']) if 'ql' in i else None,q_lengths=i.get('ql'),kv_lengths=i.get('kl'),hq=i.get('h'),hkv=i.get('hk'),d=i.get('d'),causal=i.get('causal'),br=c.get('br'),bc=c.get('bc'),k_stages=c.get('k_stages'),v_stages=c.get('v_stages'),consumer_wgs=c.get('consumer_wgs'),cta_rows=c.get('br',0)*c.get('consumer_wgs',1) if c.get('br') else None,pingpong=c.get('pingpong'),reg_realloc=c.get('rr'),producer_regs=c.get('producer_regs') if c.get('rr') else None,consumer_regs=c.get('consumer_regs') if c.get('rr') else None,qbox=c.get('qbox'),kvbox=c.get('kvbox'),obox=c.get('obox'),status=v['status'],gpu_median_us=v.get('median_us'),gpu_min_us=v.get('min_us'),gpu_max_us=v.get('max_us'),tflops=v.get('tflops'),launcher_wall_us=v.get('launcher_wall_us'),max_abs_error=v.get('max_abs_error'),rms_error=v.get('rms_error'),nonfinite_count=v.get('nonfinite_count'),round_us=v.get('round_us'),output_dtype=v.get('dtype_output'),rounded_fp32_equal=v.get('rounded_fp32_equal'),rounded_fp32_mismatch_count=v.get('rounded_fp32_mismatch_count'),rounded_fp32_max_abs=v.get('rounded_fp32_max_abs'),views=views,tool=v.get('tool'),raw_file=str(p.relative_to(run)),log=state.get('log') if state else v.get('log'),reason=v.get('reason',v.get('message')),source_sha256=m['source_sha256'].get(n+'.cu') if n else None)
  row.update(resources.get((cid,i.get('d'),i.get('causal')),{}));rows.append(row)
  if n in ['official_fa3','torch_aten_varlen','torch_flash_perseq']:
   for field in ['br','bc','k_stages','v_stages','reg_realloc','producer_regs','consumer_regs','qbox','kvbox','obox']:row[field]=None
# Build failures and rejected candidates have no fabricated input dimensions.
for b in buildrows:
 if b['exit_code']:
  c=b['config'];rows.append(dict(run_id=run.name,row_id=len(rows),phase='build',kernel=labels[c['impl']],kernel_id=c['impl'],config_id=c['id'],br=c['br'],bc=c['bc'],status='BUILD_FAIL',reason='Compiler exit '+str(b['exit_code']),log=b['log'],views=[]))
for p in (run/'state').glob('rejected-*.json'):
 v=read(p);c=v['config'];rows.append(dict(run_id=run.name,row_id=len(rows),phase='candidate_gate',kernel=labels[c['impl']],kernel_id=c['impl'],config_id=c['id'],br=c['br'],bc=c['bc'],reg_realloc=c['rr'],producer_regs=c['producer_regs'],consumer_regs=c['consumer_regs'],status='REJECTED',reason=v['reason'],views=[]))
def export(name,values):
 fields=list(dict.fromkeys(k for v in values for k in v)) or ['status']
 with (run/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for v in values:w.writerow({k:json.dumps(x,ensure_ascii=False) if isinstance(x,(dict,list)) else x for k,x in v.items()})
export('results.csv',rows);(run/'results.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
export('resources.csv',[dict(config_id=key[0],d=key[1],causal=key[2],**v) for key,v in resources.items()])
export('failures.csv',[v for v in rows if v['status'] not in ['PASS','EXPECTED_NO_WRITE','UNSUPPORTED']])
export('unsupported.csv',[v for v in rows if v['status']=='UNSUPPORTED'])
export('selected.csv',[dict(group=key,**v) for key,v in selected.items()])
export('rounding.csv',[v for v in rows if v.get('rounded_fp32_equal') is not None])
performance=[v for v in rows if v['phase']=='performance' and v.get('current_successful_attempt') and v['status']=='PASS']
counts=dict(collections.Counter(v['status'] for v in rows))
summary=dict(status=m['status'],kernel_count=len(m['kernels']),performance_cases=len({v['case_id'] for v in performance}),performance_records=len(performance),all_record_status=counts,successful_tasks=sum(v['status']=='PASS' for v in taskstates.values()),selected_configs=len({v['id'] for v in selected.values()}))
(run/'summary.json').write_text(json.dumps(summary,indent=2))
template=(run/'harness/reporting/template.html').read_text(encoding='utf-8')
page=template.replace('__RESULT_DATA__',json.dumps(rows,ensure_ascii=False).replace('</',r'<\/')).replace('__RUN_META__',json.dumps(dict(run_id=run.name,**summary),ensure_ascii=False).replace('</',r'<\/'))
(run/'report.html').write_text(page,encoding='utf-8')
(run/'summary.md').write_text('# FA3 测试结果\n\n状态：'+m['status']+'\n\n[打开可筛选报告](report.html) · [完整结果CSV](results.csv) · [选用参数](selected.csv) · [失败记录](failures.csv) · [不支持配置](unsupported.csv)\n\n输入条件在左侧，kernel与超参数在中间，实际结果在右侧。不显示相对值。所有记录均来自原始JSONL；旧尝试和失败不覆盖。\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
print('REPORT',run/'report.html',json.dumps(summary),flush=True)
