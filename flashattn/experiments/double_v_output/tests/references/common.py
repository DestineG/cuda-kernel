"""Test-only adapters and explicit attention references; no kernel changes."""
import ctypes as C
import math
from pathlib import Path
import numpy as np
import torch
from flash_attn_3.flash_attn_interface import flash_attn_varlen_func

ROOT=Path(__file__).resolve().parents[1]
NAMES=['fa3_bf16_sm90','fa3_bf16_sm90_kv_pipeline','fa3_bf16_sm90_kv_pipeline_qk_pv_overlap','fa3_bf16_sm90_kv_pipeline_qk_pv_overlap_v1']
NAMES += [NAMES[-1]+'_'+suffix for suffix in ['compile_opt','mask_opt','compile_mask_opt']]
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def shared_memory_limit():
    lib=C.CDLL('/usr/local/cuda-12.4/lib64/libcudart.so')
    value=C.c_int()
    error=lib.cudaDeviceGetAttribute(C.byref(value),97,torch.cuda.current_device())
    if error:raise RuntimeError(f'cudaDeviceGetAttribute(97): {error}')
    return value.value

class Inputs:
    def __init__(self,c):
        self.c=c; self.h=c['h']; self.hk=c['hk']; self.d=c['d']
        self.ql=c['ql'];self.kl=c['kl']
        self.cq=[0]+list(np.cumsum(self.ql).astype(int));self.ck=[0]+list(np.cumsum(self.kl).astype(int))
        gen=torch.Generator(device='cuda').manual_seed(c.get('seed',123))
        def rand(n,h): return torch.randn((max(n,1),h,self.d),generator=gen,device='cuda',dtype=torch.float32).to(torch.bfloat16)[:n]
        self.q=rand(self.cq[-1],self.h);self.k=rand(self.ck[-1],self.hk);self.v=rand(self.ck[-1],self.hk)
        pattern=c.get('pattern','random')
        if pattern=='zero_qk':self.q.zero_();self.k.zero_()
        if pattern=='constant_v':self.v.fill_(0.75)
        if pattern=='sharp':self.q.mul_(4);self.k.mul_(4)
        if pattern in ('rising','falling'):
            self.q.fill_(1)
            for b,n in enumerate(self.kl):
                ramp=torch.linspace(-2,2,n,device='cuda')
                if pattern=='falling':ramp=ramp.flip(0)
                self.k[self.ck[b]:self.ck[b+1]]=ramp[:,None,None]
        # Guard allocation keeps the output TMA base 128-byte aligned.
        self.storage=torch.full((self.cq[-1]*self.h*self.d+512,),12345.,device='cuda')
        self.o=self.storage[256:-256].view(self.cq[-1],self.h,self.d)
        self.o.fill_(float('nan'))
        self.storage_bf16=torch.full((self.cq[-1]*self.h*self.d+512,),123.,device='cuda',dtype=torch.bfloat16)
        self.o_bf16=self.storage_bf16[256:-256].view(self.cq[-1],self.h,self.d)
        self.o_bf16.fill_(float('nan'))
        self.cq64=torch.tensor(self.cq,dtype=torch.int64,device='cuda')
        self.ck64=torch.tensor(self.ck,dtype=torch.int64,device='cuda')
        self.cq32=self.cq64.to(torch.int32);self.ck32=self.ck64.to(torch.int32)

class Own:
    def __init__(self,impl,x,br=64,bc=64):
        self.x=x;self.out=x.o_bf16 if '_bf16_output_' in impl else x.o;self.lib=C.CDLL(str(ROOT/'build'/f'{impl}.so'))
        self.lib.prepare.argtypes=[C.c_void_p]*6+[C.c_size_t]*6+[C.c_int]*4+[C.POINTER(C.c_int)]
        self.lib.prepare.restype=C.c_void_p
        self.lib.launch.argtypes=[C.c_void_p,C.c_void_p,C.c_int];self.lib.launch.restype=C.c_int
        self.lib.release.argtypes=[C.c_void_p]
        err=C.c_int()
        self.ctx=self.lib.prepare(*[t.data_ptr() for t in (x.q,x.k,x.v,self.out,x.cq64,x.ck64)],len(x.ql),x.h,x.hk,max(x.ql),x.cq[-1],x.ck[-1],br,bc,x.d,int(x.c['causal']),C.byref(err))
        if err.value or not self.ctx: raise RuntimeError(f'prepare failed {err.value}')
    def __call__(self,mode=0):
        err=self.lib.launch(self.ctx,torch.cuda.current_stream().cuda_stream,mode)
        if err:raise RuntimeError(f'launch error {err}')
        return self.out
    def close(self):
        if self.ctx:self.lib.release(self.ctx);self.ctx=None

def official(x,num_splits=1,pack_gqa=None):
    return flash_attn_varlen_func(x.q,x.k,x.v,x.cq32,x.ck32,max(x.ql),max(x.kl),causal=x.c['causal'],num_splits=num_splits,pack_gqa=pack_gqa)

def torch_varlen(x):
    # Version-specific private ATen FlashAttention entry point; no per-sequence loop.
    return torch.ops.aten._flash_attention_forward(
        x.q,x.k,x.v,x.cq32,x.ck32,max(x.ql),max(x.kl),0.0,x.c['causal'],False
    )[0]

class TorchFlash:
    """Per-sequence forced flash SDPA, including packed output assembly."""
    def __init__(self,x):
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from torch.nn.attention.bias import causal_lower_right
        self.x=x;self.items=[];self.sdpa_kernel=sdpa_kernel;self.backend=SDPBackend.FLASH_ATTENTION
        for b,(n,m) in enumerate(zip(x.ql,x.kl)):
            if not n:continue
            q=x.q[x.cq[b]:x.cq[b+1]].transpose(0,1).unsqueeze(0)
            k=x.k[x.ck[b]:x.ck[b+1]].transpose(0,1).unsqueeze(0)
            v=x.v[x.ck[b]:x.ck[b+1]].transpose(0,1).unsqueeze(0)
            bias=causal_lower_right(n,m) if x.c['causal'] and n!=m and m>0 else None
            self.items.append((q,k,v,bias,x.c['causal'] and n==m,n,m))
    def __call__(self):
        outputs=[]
        with self.sdpa_kernel(self.backend):
            for q,k,v,bias,ca,n,m in self.items:
                if not m:out=torch.zeros((1,self.x.h,n,self.x.d),device=q.device,dtype=q.dtype)
                else:
                    out=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=bias,is_causal=ca,enable_gqa=True)
                outputs.append(out.squeeze(0).transpose(0,1))
        return torch.cat(outputs,dim=0) if len(outputs)>1 else outputs[0]

def cpu_reference(x):
    # Independent CPU FP64 calculation from the exact BF16 input values.
    q=x.q.float().cpu().numpy().astype(np.float64)
    k=x.k.float().cpu().numpy().astype(np.float64)
    v=x.v.float().cpu().numpy().astype(np.float64)
    result=np.zeros(q.shape,dtype=np.float64)
    for b,(n,m) in enumerate(zip(x.ql,x.kl)):
        if not n or not m:continue
        for h in range(x.h):
            hk=h//(x.h//x.hk)
            scores=q[x.cq[b]:x.cq[b+1],h]@k[x.ck[b]:x.ck[b+1],hk].T/math.sqrt(x.d)
            if x.c['causal']:
                mask=np.arange(m)[None,:]<=np.arange(n)[:,None]+m-n
                scores=np.where(mask,scores,-np.inf)
            mx=scores.max(-1,keepdims=True)
            p=np.exp(scores-np.where(np.isfinite(mx),mx,0))
            den=p.sum(-1,keepdims=True)
            p=np.divide(p,den,out=np.zeros_like(p),where=den!=0)
            result[x.cq[b]:x.cq[b+1],h]=p@v[x.ck[b]:x.ck[b+1],hk]
    return torch.from_numpy(result)

def gpu_reference(x):
    # Explicit FP32 attention in query chunks; TF32 is disabled above.
    out=torch.empty_like(x.o)
    mapping=torch.arange(x.h,device='cuda')//(x.h//x.hk)
    for b,(n,m) in enumerate(zip(x.ql,x.kl)):
        if not n:continue
        if not m:out[x.cq[b]:x.cq[b+1]].zero_();continue
        k=x.k[x.ck[b]:x.ck[b+1],mapping].transpose(0,1).float()
        v=x.v[x.ck[b]:x.ck[b+1],mapping].transpose(0,1).float()
        for start in range(0,n,128):
            end=min(start+128,n)
            q=x.q[x.cq[b]+start:x.cq[b]+end].transpose(0,1).float()
            s=q@k.transpose(-1,-2)/math.sqrt(x.d)
            if x.c['causal']:
                mask=torch.arange(m,device='cuda')[None,:]<=torch.arange(start,end,device='cuda')[:,None]+m-n
                s.masked_fill_(~mask,-float('inf'))
            p=torch.softmax(s,-1).nan_to_num()
            out[x.cq[b]+start:x.cq[b]+end]=(p@v).transpose(0,1)
    return out

def metrics(out,ref,atol=.008,rtol=.02):
    a=out.detach().double().cpu();b=ref.detach().double().cpu()
    finite=torch.isfinite(a);err=(a-b).abs()
    bad=(~finite)|(err>atol+rtol*b.abs())
    coords=bad.nonzero()
    return dict(status='PASS' if not bad.any() else 'FAIL',max_abs_error=err.max().item() if err.numel() else 0,rms_error=err.square().mean().sqrt().item() if err.numel() else 0,max_rel_error=(err/b.abs().clamp_min(1e-6)).max().item() if err.numel() else 0,nonfinite_count=int((~finite).sum()),mismatch_count=int(bad.sum()),first_mismatch=coords[0].tolist() if len(coords) else None,atol=atol,rtol=rtol)
