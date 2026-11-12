// Test adapter only. Original source is included unchanged, except main is renamed.
#include <cmath>
#include <cstdlib>
#define main original_demo_main
#include "../sources/initial/fa3_bf16_sm90.cu"
#undef main
struct Context {
    alignas(64) CUtensorMap q,k,v,o;
    const __nv_bfloat16 *qp,*kp,*vp;
    float *op;
    const size_t *cq,*ck;
    size_t batch,h,hk,mq,tq,tk;
    int br,bc,d,causal;
};
template<int R,int C,int D,bool CA>
int dispatch(Context* x,cudaStream_t stream,int mode) {
    constexpr int bytes=(R+2*C)*D*2+16;
    auto kernel=fa3_bf16_forward<R,C,D,CA,64,64,64,64,64,32,64,64,16>;
    if(mode==2) return (int)cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(mode==1) {
        fa3_bf16_forward_launcher<R,C,D,CA>(x->qp,x->kp,x->vp,x->op,x->cq,x->ck,x->batch,x->h,x->hk,x->mq,x->tq,x->tk,stream);
    } else {
        kernel<<<dim3((x->mq+R-1)/R,x->h,x->batch),256,bytes,stream>>>(x->q,x->k,x->v,x->o,x->cq,x->ck,x->h,x->hk,1/sqrtf(D),x->op);
    }
    return (int)cudaGetLastError();
}
int select(Context* x,cudaStream_t stream,int mode) {
    if(x->br==64 && x->bc==64 && x->d==64 && x->causal==0) return dispatch<64,64,64,false>(x,stream,mode);
    if(x->br==64 && x->bc==64 && x->d==64 && x->causal==1) return dispatch<64,64,64,true>(x,stream,mode);
    if(x->br==64 && x->bc==64 && x->d==128 && x->causal==0) return dispatch<64,64,128,false>(x,stream,mode);
    if(x->br==64 && x->bc==64 && x->d==128 && x->causal==1) return dispatch<64,64,128,true>(x,stream,mode);
    if(x->br==64 && x->bc==128 && x->d==64 && x->causal==0) return dispatch<64,128,64,false>(x,stream,mode);
    if(x->br==64 && x->bc==128 && x->d==64 && x->causal==1) return dispatch<64,128,64,true>(x,stream,mode);
    if(x->br==64 && x->bc==128 && x->d==128 && x->causal==0) return dispatch<64,128,128,false>(x,stream,mode);
    if(x->br==64 && x->bc==128 && x->d==128 && x->causal==1) return dispatch<64,128,128,true>(x,stream,mode);
    if(x->br==128 && x->bc==64 && x->d==64 && x->causal==0) return dispatch<128,64,64,false>(x,stream,mode);
    if(x->br==128 && x->bc==64 && x->d==64 && x->causal==1) return dispatch<128,64,64,true>(x,stream,mode);
    if(x->br==128 && x->bc==64 && x->d==128 && x->causal==0) return dispatch<128,64,128,false>(x,stream,mode);
    if(x->br==128 && x->bc==64 && x->d==128 && x->causal==1) return dispatch<128,64,128,true>(x,stream,mode);
    if(x->br==128 && x->bc==128 && x->d==64 && x->causal==0) return dispatch<128,128,64,false>(x,stream,mode);
    if(x->br==128 && x->bc==128 && x->d==64 && x->causal==1) return dispatch<128,128,64,true>(x,stream,mode);
    if(x->br==128 && x->bc==128 && x->d==128 && x->causal==0) return dispatch<128,128,128,false>(x,stream,mode);
    if(x->br==128 && x->bc==128 && x->d==128 && x->causal==1) return dispatch<128,128,128,true>(x,stream,mode);
    return -1;
}
extern "C" void* prepare(void* q,void* k,void* v,void* o,void* cq,void* ck,
    size_t batch,size_t h,size_t hk,size_t mq,size_t tq,size_t tk,int br,int bc,int d,int causal,int* err) {
    Context* x=new Context{};
    x->qp=(__nv_bfloat16*)q; x->kp=(__nv_bfloat16*)k; x->vp=(__nv_bfloat16*)v; x->op=(float*)o;
    x->cq=(size_t*)cq; x->ck=(size_t*)ck;
    x->batch=batch;x->h=h;x->hk=hk;x->mq=mq;x->tq=tq;x->tk=tk;x->br=br;x->bc=bc;x->d=d;x->causal=causal;
    bool ok=create_tma_descriptor_swizzle128B<64,64>(q,tq,h*d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,&x->q,h*d*2)
        &&create_tma_descriptor_swizzle128B<64,64>(k,tk,hk*d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,&x->k,hk*d*2)
        &&create_tma_descriptor_swizzle128B<64,64>(v,tk,hk*d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,&x->v,hk*d*2)
        &&create_tma_descriptor_swizzle128B<64,32>(o,tq,h*d,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,&x->o,h*d*4);
    *err=ok?select(x,nullptr,2):-2;
    if(*err) {delete x;return nullptr;}
    return x;
}
extern "C" int launch(void* x,void* stream,int mode) {return select((Context*)x,(cudaStream_t)stream,mode);}
extern "C" void release(void* x) {delete (Context*)x;}
