// Minimal CUDA-only sanitizer driver: avoids unrelated PyTorch CUDA activity.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <dlfcn.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#define CHECK(call) do {cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(2);}} while(0)
using Prepare=void*(*)(void*,void*,void*,void*,void*,void*,size_t,size_t,size_t,size_t,size_t,size_t,int,int,int,int,int*);
using Launch=int(*)(void*,void*,int);
using Release=void(*)(void*);
void* alloc(size_t bytes){void* p;CHECK(cudaMalloc(&p,bytes));return p;}
int main(int argc,char**argv){
    if(argc!=4)return 2;
    void* lib=dlopen(argv[1],RTLD_NOW);if(!lib){fprintf(stderr,"%s\n",dlerror());return 2;}
    auto prep=(Prepare)dlsym(lib,"prepare");auto launch=(Launch)dlsym(lib,"launch");auto release=(Release)dlsym(lib,"release");
    size_t cq[]={0,65,194},ck[]={0,577,834};
    void* dq=alloc(sizeof(cq));void* dk=alloc(sizeof(ck));
    CHECK(cudaMemcpy(dq,cq,sizeof(cq),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dk,ck,sizeof(ck),cudaMemcpyHostToDevice));
    for(int d:{64,128}) for(int causal:{0,1}) for(int br:{atoi(argv[2])}) for(int bc:{atoi(argv[3])}) {
        const size_t nq=194*4*d,nk=834*d;
        std::vector<__nv_bfloat16> hq(nq),hk(nk),hv(nk);
        unsigned s=123;
        auto rnd=[&](){s^=s<<13;s^=s>>17;s^=s<<5;return float(int(s%2049)-1024)/1024;};
        for(auto&v:hq)v=__float2bfloat16(rnd());for(auto&v:hk)v=__float2bfloat16(rnd());for(auto&v:hv)v=__float2bfloat16(rnd());
        void* q=alloc(nq*2);void*k=alloc(nk*2);void*v=alloc(nk*2);float* storage=(float*)alloc((nq+512)*4);float*o=storage+256;
        CHECK(cudaMemcpy(q,hq.data(),nq*2,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(k,hk.data(),nk*2,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(v,hv.data(),nk*2,cudaMemcpyHostToDevice));
        std::vector<float> ho(nq+512,12345.f);int err=0;
        void* ctx=prep(q,k,v,o,dq,dk,2,4,1,129,194,834,br,bc,d,causal,&err);
        if(err||!ctx){fprintf(stderr,"prepare %d\n",err);return 3;}
        for(int repeat=0;repeat<3;++repeat){
            std::fill(ho.begin()+256,ho.end()-256,NAN);
            CHECK(cudaMemcpy(storage,ho.data(),ho.size()*4,cudaMemcpyHostToDevice));
            err=launch(ctx,nullptr,repeat%2);if(err){fprintf(stderr,"launch %d\n",err);return 3;}
            CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(ho.data(),storage,ho.size()*4,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<ho.size();++i){bool ok=(i<256||i>=nq+256)?ho[i]==12345.f:std::isfinite(ho[i]);if(!ok){fprintf(stderr,"guard/nonfinite at %zu\n",i);return 4;}}
        }
        printf("PASS D=%d causal=%d Br=%d Bc=%d repeats=3\n",d,causal,br,bc);fflush(stdout);
        release(ctx);CHECK(cudaFree(q));CHECK(cudaFree(k));CHECK(cudaFree(v));CHECK(cudaFree(storage));
    }
    CHECK(cudaFree(dq));CHECK(cudaFree(dk));return 0;
}
