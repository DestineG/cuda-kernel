#pragma once

#include<stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>


// smem_ptr: 0-13; LBO: 16-29; SBO: 32-45; offset: 49-51; swizzle_mode: 62-63
// swizzle_mode: 0 none; 1 128B; 2 64B; 3 32B;
__device__ __forceinline__
uint64_t create_smem_desc(
    const void* smem_ptr,
    uint32_t lbo_bytes,
    uint32_t sbo_bytes,
    uint32_t swizzle_mode,
    uint32_t base_offset = 0
) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);

    // assert(addr % 16 == 0);
    // assert(lbo_bytes % 16 == 0);
    // assert(sbo_bytes % 16 == 0);

    // // 三个字段均为 14 位，单位为 16B
    // assert((addr >> 4) < (1u << 14));
    // assert((lbo_bytes >> 4) < (1u << 14));
    // assert((sbo_bytes >> 4) < (1u << 14));

    // assert(swizzle_mode <= 3);
    // assert(base_offset <= 7);

    return
        (uint64_t((addr      >> 4) & 0x3FFF))       |
        (uint64_t((lbo_bytes >> 4) & 0x3FFF) << 16) |
        (uint64_t((sbo_bytes >> 4) & 0x3FFF) << 32) |
        (uint64_t(base_offset & 0x7)        << 49) |
        (uint64_t(swizzle_mode & 0x3)       << 62);
}

// 保证此前对 WGMMA 操作数寄存器的访问（如打包 regs_p、缩放 regs_o）先于后续 WGMMA 对这些寄存器的访问
__device__ __forceinline__
void wgmma_fence() {
    asm volatile(
        "wgmma.fence.sync.aligned;"
        :
        :
        : "memory"
    );
}

__device__ __forceinline__
void wgmma_m64n64k16_ss(
    float (&d)[32],
    uint64_t desc_a,
    uint64_t desc_b,
    bool accumulate = true
) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %34, 0;\n\t"

        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1, 0, 0;\n\t"

        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(desc_a),
          "l"(desc_b),
          "r"(static_cast<int>(accumulate))
        : "memory"
    );
}

__device__ __forceinline__
void wgmma_m64n64k16_rs(
    float (&d)[32],
    const uint32_t (&a)[4],
    uint64_t desc_b,
    bool accumulate = true
) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %37, 0;\n\t"

        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, %36, p, 1, 1, 1;\n\t"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(desc_b),
          "r"(static_cast<int>(accumulate))
        : "memory"
    );
}

// 打包分组
__device__ __forceinline__
void wgmma_commit() {
    asm volatile(
        "wgmma.commit_group.sync.aligned;"
        :
        :
        : "memory"
    );
}

// 清空队列
template<int N = 0>
__device__ __forceinline__
void wgmma_wait() {
    asm volatile(
        "wgmma.wait_group.sync.aligned %0;"
        :
        : "n"(N)
        : "memory"
    );
}

template<uint32_t RegCount>
__device__ __forceinline__
void warpgroup_reg_alloc() {
    static_assert(RegCount >= 24 && RegCount <= 256);
    static_assert(RegCount % 8 == 0);

    asm volatile(
        "setmaxnreg.inc.sync.aligned.u32 %0;"
        :
        : "n"(RegCount)
    );
}

template<uint32_t RegCount>
__device__ __forceinline__
void warpgroup_reg_dealloc() {
    static_assert(RegCount >= 24 && RegCount <= 256);
    static_assert(RegCount % 8 == 0);

    asm volatile(
        "setmaxnreg.dec.sync.aligned.u32 %0;"
        :
        : "n"(RegCount)
    );
}