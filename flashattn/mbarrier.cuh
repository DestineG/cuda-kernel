#pragma once

#include<stdint.h>
#include <cuda_runtime.h>


__device__ __forceinline__
void barrier_init(uint64_t* barrier, uint32_t count) {
    uint32_t addr = __cvta_generic_to_shared(barrier);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        "fence.mbarrier_init.release.cluster;"
        :
        : "r"(addr), "r"(count)
        : "memory"
    );
}

__device__ __forceinline__
void barrier_arrive(uint64_t* barrier) {
    uint32_t addr = __cvta_generic_to_shared(barrier);
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :
        : "r"(addr)
        : "memory"
    );
}

__device__ __forceinline__
void barrier_expect_tx(
    uint64_t* barrier,
    uint32_t bytes
) {
    uint32_t addr = __cvta_generic_to_shared(barrier);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :
        : "r"(addr), "r"(bytes)
        : "memory"
    );
}

__device__ __forceinline__
bool barrier_try_wait(
    uint64_t* barrier,
    uint32_t phase
) {
    uint32_t addr = __cvta_generic_to_shared(barrier);
    uint32_t ready;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
        "p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t"
        "}\n"
        : "=r"(ready)
        : "r"(addr), "r"(phase)
        : "memory"
    );

    return ready != 0;
}

__device__ __forceinline__
void barrier_wait(
    uint64_t* barrier,
    uint32_t phase
) {
    while (!barrier_try_wait(barrier, phase)) {
    }
}