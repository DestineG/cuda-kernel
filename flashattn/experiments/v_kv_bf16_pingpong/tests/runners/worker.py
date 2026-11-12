import argparse,json,sys,time,statistics,math,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'references'))
import common
from common import *
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--job',required=True);a=p.parse_args()
run=Path(a.run);common.ROOT=run;job=json.loads(Path(a.job).read_text());dest=Path(job['output'])
settings=json.loads((run/'configs/settings.json').read_text());timing=settings['timing'];validation=settings['validation']
is_cpu=job['phase'] in ('correctness','selected_correctness','output_rounding')
validated_metrics=False
def write(v):
 with dest.open('a') as f:f.write(json.dumps(v)+'\n')
def measure(fn):
 for _ in range(timing['warmup']):fn()
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph();hold=[]
 with torch.cuda.graph(g):
  for _ in range(timing['graph_nodes']):hold.append(fn())
 g.replay();torch.cuda.synchronize()
 s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 s.record();g.replay();e.record();e.synchronize();reps=max(1,min(timing['max_graph_replays'],math.ceil(timing['target_round_ms']/max(s.elapsed_time(e),.001))))
 return g,hold,reps
def timed(g,reps):
 s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 s.record()
 for _ in range(reps):g.replay()
 e.record();e.synchronize();return s.elapsed_time(e)*1000/(timing['graph_nodes']*reps)
def gpu_metrics(out,ref,atol=.008,rtol=.02):
 # The reference computation is unchanged. Reduce errors on GPU, outside timing,
 # instead of repeatedly copying large attention outputs to the host.
 aa=out.detach().double();bb=ref.detach().to(device=out.device,dtype=torch.float64)
 finite=torch.isfinite(aa);err=(aa-bb).abs();bad=(~finite)|(err>atol+rtol*bb.abs())
 count=int(bad.sum());coords=bad.nonzero()[0].cpu().tolist() if count else None
 return dict(status='PASS' if not count else 'FAIL',max_abs_error=err.max().item() if err.numel() else 0,rms_error=err.square().mean().sqrt().item() if err.numel() else 0,max_rel_error=(err/bb.abs().clamp_min(1e-6)).max().item() if err.numel() else 0,nonfinite_count=int((~finite).sum()),mismatch_count=count,first_mismatch=coords,atol=atol,rtol=rtol)
for c in job['cases']:
 print('START',c['id'],flush=True);(run/'logs/current_worker.json').write_text(json.dumps(dict(job=job['id'],case=c)))
 x=Inputs(c)
 cache=run/'reference-cache';cache.mkdir(exist_ok=True)
 cached=cache/(c['id']+'.pt')
 if is_cpu:ref=cpu_reference(x)
 elif job['phase']=='performance':ref=gpu_reference(x)
 elif cached.exists():ref=torch.load(cached,map_location='cuda',weights_only=True)
 else:
  ref=gpu_reference(x);torch.save(ref.cpu(),cached)
 rounded_outputs={};objects=[];valid=[];graphs={};rows={};samples={};functions={};wall_functions={}
 try:
  for cfg in job['configs']:
   print('CONFIG_START',cfg['id'],flush=True)
   cid=cfg['id'];baseline=cfg['impl'] in ['official_fa3','torch_flash_perseq','torch_aten_varlen']
   if not baseline:
    required=(cfg['consumer_wgs']*cfg['br']+(cfg['k_stages']+cfg['v_stages'])*cfg['bc'])*x.d*2+cfg['barrier_bytes']
    capacity=shared_memory_limit()
    if required>capacity:
     write(dict(phase=job['phase'],case_id=c['id'],config_id=cid,config=cfg,input=c,status='UNSUPPORTED',reason='dynamic shared memory exceeds device limit',required_smem_bytes=required,device_limit_bytes=capacity));continue
   obj=None if baseline else Own(cid,x,cfg['br'],cfg['bc'])
   if obj:objects.append(obj)
   fn=TorchFlash(x) if cfg['impl']=='torch_flash_perseq' else ((lambda:torch_varlen(x)) if cfg['impl']=='torch_aten_varlen' else ((lambda:official(x,0)) if obj is None else obj))
   functions[cid]=fn;wall_functions[cid]=fn if obj is None else lambda obj=obj:obj(1)
   row=dict(phase=job['phase'],case_id=c['id'],config_id=cid,config=cfg,input=c,reference='CPU_FP64' if is_cpu else 'GPU_FP32',dtype_output='bf16' if baseline else cfg['output_dtype'],sdpa_backend='FLASH_ATTENTION' if cfg['impl']=='torch_flash_perseq' else None)
   worst=None;first=None;equal=True;guard=True
   check_repeats=validation['repeats'] if is_cpu else 1
   row['correctness_repeats']=check_repeats
   for rep in range(check_repeats):
    x.o.fill_(float('nan'));x.o_bf16.fill_(float('nan'));out=fn() if obj is None else obj(rep%2)
    torch.cuda.synchronize()
    guard &= bool((x.storage[:256]==12345).all() and (x.storage[-256:]==12345).all() and (x.storage_bf16[:256]==123).all() and (x.storage_bf16[-256:]==123).all())
    if first is None:first=out.clone()
    else:equal &= torch.equal(first,out)
    if any(m==0 and n>0 for n,m in zip(c['ql'],c['kl'])) and obj is not None:
     empty=[b for b,(n,m) in enumerate(zip(c['ql'],c['kl'])) if m==0 and n>0]
     untouched=all(bool(torch.isnan(out[x.cq[b]:x.cq[b+1]]).all()) for b in empty)
     row.update(status='EXPECTED_NO_WRITE' if untouched else 'FAIL_ZERO_KV');continue
    compare=metrics if is_cpu else gpu_metrics
    m=compare(out,ref,atol=validation['sharp_atol'] if c.get('pattern')=='sharp' else validation['atol'],rtol=validation['rtol'])
    if not is_cpu and not validated_metrics:
     host=metrics(out,ref,atol=m['atol'],rtol=m['rtol'])
     assert all(host[k]==m[k] for k in ['status','mismatch_count','nonfinite_count','first_mismatch']), (host,m)
     hm,gm=host['max_abs_error'],m['max_abs_error']
     assert hm==gm or (math.isnan(hm) and math.isnan(gm)) or abs(hm-gm)<1e-12
     validated_metrics=True
     print('METRIC_REDUCTION_CROSSCHECK_PASS',flush=True)
    if worst is None or (m['mismatch_count'],m['max_abs_error'])>(worst['mismatch_count'],worst['max_abs_error']):worst=m
   if worst:row.update(worst)
   row.update(guard_ok=guard,repeat_equal=equal)
   if not guard:row['status']='GUARD_FAIL'
   if not equal and row['status']!='EXPECTED_NO_WRITE':row['status']='NONDETERMINISTIC'
   if job['phase']=='output_rounding' and row['status']=='PASS':
    pair=tuple(cfg.get(k) for k in ['br','bc','rr','producer_regs','consumer_regs','qbox','kvbox','obox'])
    if cfg['output_dtype']=='fp32':rounded_outputs[pair]=out.to(torch.bfloat16).clone()
    else:
     expected=rounded_outputs[pair];different=(out!=expected)
     row.update(rounded_fp32_equal=bool(not different.any()),rounded_fp32_mismatch_count=int(different.sum()),rounded_fp32_max_abs=(out.float()-expected.float()).abs().max().item() if out.numel() else 0)
   rows[cid]=row
   if row['status'] not in ['PASS','EXPECTED_NO_WRITE']:
    write(row)
    raise RuntimeError('Numerical/guard failure: '+cid+' '+c['id']+'; stop before further timing')
   if is_cpu or row['status']!='PASS':write(row);continue
   graphs[cid]=measure(fn);valid.append(cid);samples[cid]=[]
  for rnd in range(timing['rounds']):
   order=valid[rnd%len(valid):]+valid[:rnd%len(valid)] if valid else []
   for cid in order:samples[cid].append(timed(graphs[cid][0],graphs[cid][2]))
  for cid in valid:
   row=rows[cid];cfg=row['config'];values=samples[cid]
   pair=sum(sum(max(0,min(k,i+k-q+1)) for i in range(q)) if c['causal'] else q*k for q,k in zip(c['ql'],c['kl']))
   flops=4*c['h']*c['d']*pair
   row.update(round_us=values,median_us=statistics.median(values),min_us=min(values),max_us=max(values),effective_flops=flops,tflops=flops/statistics.median(values)/1e6,timing_scope='CUDA_graph_8_calls_GPU',graph_repeats=graphs[cid][2])
   fn=wall_functions[cid]
   torch.cuda.synchronize();start=time.perf_counter()
   for _ in range(timing['wall_repeats']):fn()
   torch.cuda.synchronize();row['launcher_wall_us']=(time.perf_counter()-start)*1e6/timing['wall_repeats']
   write(row)
  write(dict(record_type='case_end',phase=job['phase'],case_id=c['id'],job_id=job['id']))
  print('DONE',c['id'],{k:v['status'] for k,v in rows.items()},flush=True)
 except Exception as e:
  write(dict(phase=job['phase'],case_id=c['id'],status='ERROR',message=str(e),traceback=traceback.format_exc()));raise
 finally:
  graphs.clear()
  functions.clear();wall_functions.clear()
  for o in objects:o.close()
 del x,ref,objects
 torch.cuda.empty_cache()
