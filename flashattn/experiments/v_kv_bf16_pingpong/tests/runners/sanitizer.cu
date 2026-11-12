// CUDA-only gate for either output type. The adapter exports output_bytes().
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <dlfcn.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#define CHECK(call) do {cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(2);}} while(0)
using Prepare=void*(*)(void*,void*,void*,void*,void*,void*,size_t,size_t,size_t,size_t,size_t,size_t,int,int,int,int,int*);
using Launch=int(*)(void*,void*,int);
using Release=void(*)(void*);
void* alloc(size_t n){void* p;CHECK(cudaMalloc(&p,n));return p;}

template<class Out>
int run(void* lib,int br,int bc){
 auto prep=(Prepare)dlsym(lib,"prepare");auto launch=(Launch)dlsym(lib,"launch");auto release=(Release)dlsym(lib,"release");
 size_t cq[]={0,65,194},ck[]={0,577,834};
 void* dq=alloc(sizeof(cq));void* dk=alloc(sizeof(ck));
 CHECK(cudaMemcpy(dq,cq,sizeof(cq),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dk,ck,sizeof(ck),cudaMemcpyHostToDevice));
 for(int d:{64,128})for(int causal:{0,1}){
  const size_t nq=194*4*d,nk=834*d;
  std::vector<__nv_bfloat16> hq(nq),hk(nk),hv(nk);unsigned s=123;
  auto rnd=[&](){s^=s<<13;s^=s>>17;s^=s<<5;return float(int(s%2049)-1024)/1024;};
  for(auto& a:hq)a=__float2bfloat16(rnd());for(auto& a:hk)a=__float2bfloat16(rnd());for(auto& a:hv)a=__float2bfloat16(rnd());
  void* q=alloc(nq*2);void* k=alloc(nk*2);void* v=alloc(nk*2);
  Out* storage=(Out*)alloc((nq+512)*sizeof(Out));Out* o=storage+256;
  CHECK(cudaMemcpy(q,hq.data(),nq*2,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(k,hk.data(),nk*2,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(v,hv.data(),nk*2,cudaMemcpyHostToDevice));
  std::vector<Out> ho(nq+512,Out(123.f));int error=0;
  void* ctx=prep(q,k,v,o,dq,dk,2,4,1,129,194,834,br,bc,d,causal,&error);
  if(error||!ctx){fprintf(stderr,"prepare %d\n",error);return 3;}
  for(int repeat=0;repeat<3;repeat++){
   std::fill(ho.begin()+256,ho.end()-256,Out(NAN));
   CHECK(cudaMemcpy(storage,ho.data(),ho.size()*sizeof(Out),cudaMemcpyHostToDevice));
   error=launch(ctx,nullptr,repeat%2);if(error)return 3;
   CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(ho.data(),storage,ho.size()*sizeof(Out),cudaMemcpyDeviceToHost));
   for(size_t i=0;i<ho.size();i++){
    float value=float(ho[i]);bool ok=(i<256||i>=nq+256)?value==123.f:std::isfinite(value);
    if(!ok){fprintf(stderr,"guard/nonfinite at %zu\n",i);return 4;}
   }
  }
  printf("PASS D=%d causal=%d Br=%d Bc=%d repeats=3 output_bytes=%zu\n",d,causal,br,bc,sizeof(Out));fflush(stdout);
  release(ctx);CHECK(cudaFree(q));CHECK(cudaFree(k));CHECK(cudaFree(v));CHECK(cudaFree(storage));
 }
 CHECK(cudaFree(dq));CHECK(cudaFree(dk));return 0;
}
int main(int argc,char** argv){
 if(argc!=4)return 2;
 void* lib=dlopen(argv[1],RTLD_NOW);if(!lib){fprintf(stderr,"%s\n",dlerror());return 2;}
 auto bytes=(int(*)())dlsym(lib,"output_bytes");if(!bytes)return 2;
 if(bytes()==2)return run<__nv_bfloat16>(lib,atoi(argv[2]),atoi(argv[3]));
 if(bytes()==4)return run<float>(lib,atoi(argv[2]),atoi(argv[3]));
 return 2;
}
