#pragma once

#include<stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>


__device__ __forceinline__
void tma_load_2d(
    void* smem_ptr,
    const CUtensorMap* desc,
    uint32_t x,
    uint32_t y,
    uint64_t* barrier
) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    uint32_t barrier_addr = __cvta_generic_to_shared(barrier);
    asm volatile(
        "cp.async.bulk.tensor.2d."
        "shared::cluster.global."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :
        : "r"(smem_addr), "l"(desc), "r"(x), "r"(y), "r"(barrier_addr)
        : "memory"
    );
}

__device__ __forceinline__
void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__
void tma_store_2d(
    const CUtensorMap* desc,
    uint32_t x,
    uint32_t y,
    const void* smem_ptr
) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d."
        "global.shared::cta.bulk_group "
        "[%0, {%1, %2}], [%3];"
        :
        : "l"(desc),
          "r"(x),
          "r"(y),
          "r"(smem_addr)
        : "memory"
    );
}

__device__ __forceinline__
void tma_store_commit() {
    asm volatile(
        "cp.async.bulk.commit_group;"
        :
        :
        : "memory"
    );
}

__device__ __forceinline__
void tma_store_wait() {
    asm volatile(
        "cp.async.bulk.wait_group 0;"
        :
        :
        : "memory"
    );
}
