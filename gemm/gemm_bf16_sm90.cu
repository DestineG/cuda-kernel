#include<cuda.h>
#include<cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <assert.h>

#define WARP_SIZE 32

// Register limits are template arguments, not -D switches or runtime values.
// setmaxnreg requires an immediate: RegCount must be known at compile time.
// Counts below are ABSOLUTE per-thread limits, not increments.
template<int RegCount>
__device__ __forceinline__
void producer_release_registers() {
    static_assert(RegCount >= 24 && RegCount <= 256 && RegCount % 8 == 0,
                  "invalid producer register target");
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;"
                 : : "n"(RegCount) : "memory");
}

template<int RegCount>
__device__ __forceinline__
void consumer_acquire_registers() {
    static_assert(RegCount >= 24 && RegCount <= 256 && RegCount % 8 == 0,
                  "invalid consumer register target");
    // May block if the CTA register pool cannot satisfy this request.
    // launch_bounds alone does not prove a sufficient initial register budget.
    // Later tuning must check compiler allocation and valid inc/dec directions.
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;"
                 : : "n"(RegCount) : "memory");
}

size_t get_dtype_size(CUtensorMapDataType dtype) {
    switch (dtype) {
        case CU_TENSOR_MAP_DATA_TYPE_UINT8:
            return 1;

        case CU_TENSOR_MAP_DATA_TYPE_UINT16:
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
        case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
            return 2;

        case CU_TENSOR_MAP_DATA_TYPE_UINT32:
        case CU_TENSOR_MAP_DATA_TYPE_INT32:
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
        case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
        case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
            return 4;

        case CU_TENSOR_MAP_DATA_TYPE_UINT64:
        case CU_TENSOR_MAP_DATA_TYPE_INT64:
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
            return 8;

        default:
            return 0;
    }
}

template<size_t BOX_M, size_t BOX_N>
bool create_tma_descriptor(
    const void* matrix,
    size_t M,
    size_t N,
    CUtensorMapDataType dtype,
    CUtensorMap* desc
) {
    size_t dtype_size = get_dtype_size(dtype);
    assert(dtype_size != 0);
    assert(BOX_N * dtype_size == 128);

    // 检查 global 基地址和行跨度的 16B 对齐
    size_t row_stride = N * dtype_size;
    assert((uintptr_t)matrix % 16 == 0);
    assert(row_stride % 16 == 0);

    cuuint64_t globalDim[2] = {N, M};
    cuuint64_t globalStrides[1] = {(cuuint64_t)row_stride};
    cuuint32_t boxDim[2] = {(cuuint32_t)BOX_N, (cuuint32_t)BOX_M};
    cuuint32_t elementStrides[2] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        desc,
        dtype,
        2,
        const_cast<void*>(matrix),
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    assert(res == CUDA_SUCCESS);
    return res == CUDA_SUCCESS;
}

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

__device__ __forceinline__
void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__
uint32_t box_swizzle128B(
    uint32_t row,
    uint32_t col,
    uint32_t dtype_size
) {
    col = ((col / (16 / dtype_size)) ^ (row % 8)) * (16 / dtype_size) + (col % (16 / dtype_size));
    return row * (128 / dtype_size) + col;
}

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

    assert(addr % 16 == 0);
    assert(lbo_bytes % 16 == 0);
    assert(sbo_bytes % 16 == 0);

    // 三个字段均为 14 位，单位为 16B
    assert((addr >> 4) < (1u << 14));
    assert((lbo_bytes >> 4) < (1u << 14));
    assert((sbo_bytes >> 4) < (1u << 14));

    assert(swizzle_mode <= 3);
    assert(base_offset <= 7);

    return
        (uint64_t((addr      >> 4) & 0x3FFF))       |
        (uint64_t((lbo_bytes >> 4) & 0x3FFF) << 16) |
        (uint64_t((sbo_bytes >> 4) & 0x3FFF) << 32) |
        (uint64_t(base_offset & 0x7)        << 49) |
        (uint64_t(swizzle_mode & 0x3)       << 62);
}

__device__ __forceinline__
void wgmma_m64n64k16(
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
void wgmma_fence() {
    asm volatile(
        "wgmma.fence.sync.aligned;"
        :
        :
        : "memory"
    );
}

__device__ __forceinline__
void wgmma_commit() {
    asm volatile(
        "wgmma.commit_group.sync.aligned;"
        :
        :
        : "memory"
    );
}

__device__ __forceinline__
void wgmma_wait() {
    asm volatile(
        "wgmma.wait_group.sync.aligned 0;"
        :
        :
        : "memory"
    );
}

template<
    size_t BM,
    size_t BN,
    size_t BK,

    size_t BOX_A_M,
    size_t BOX_A_K,

    size_t BOX_B_N,
    size_t BOX_B_K,

    size_t BOX_C_M,
    size_t BOX_C_N,

    size_t MMA_M,
    size_t MMA_N,
    size_t MMA_K,
    bool RegRealloc = true,
    int ProducerRegs = 40,
    int ConsumerRegs = 232
>
__global__ __launch_bounds__((RegRealloc ? 8 : 5) * WARP_SIZE, 1)
void gemm_bf16_kernel(
    const __grid_constant__ CUtensorMap A_desc,
    const __grid_constant__ CUtensorMap B_desc,
    const __grid_constant__ CUtensorMap C_desc,
    size_t M,
    size_t N,
    size_t K
){
    // wgmmam64n64k16
    static_assert(MMA_M == 64 && MMA_N == 64 && MMA_K == 16);
    // Defaults40/232 are untested tuning candidates, not measured optima.
    constexpr uint32_t CTA_THREADS = (RegRealloc ? 8 : 5) * WARP_SIZE;
    static_assert(!RegRealloc ||
                  (ProducerRegs >= 24 && ProducerRegs <= 256 && ProducerRegs % 8 == 0),
                  "invalid producer register target");
    static_assert(!RegRealloc ||
                  (ConsumerRegs >= 24 && ConsumerRegs <= 256 && ConsumerRegs % 8 == 0),
                  "invalid consumer register target");
    static_assert(!RegRealloc || ProducerRegs <= ConsumerRegs,
                  "producer target must not exceed consumer target");
    static_assert(!RegRealloc || 128 * (ProducerRegs + ConsumerRegs) <= 65536,
                  "register targets exceed one SM90 register file");

    // index
    size_t row = blockIdx.y * BM;
    size_t col = blockIdx.x * BN;
    uint32_t local_thread_idx = threadIdx.x;
    uint32_t warp_idx = local_thread_idx / WARP_SIZE;
    uint32_t lane_idx = local_thread_idx % WARP_SIZE;
    assert(blockDim.x == CTA_THREADS && blockDim.y == 1 && blockDim.z == 1);

    // shared
    extern __shared__ __align__(1024) char smem[];
    float* smem_tileC = reinterpret_cast<float*>(smem);
    __nv_bfloat16* smem_tileA = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_tileB = reinterpret_cast<__nv_bfloat16*>(smem_tileA + BM * BK);
    uint64_t* barrier = reinterpret_cast<uint64_t*>(smem_tileB + BN * BK);
    uint64_t* barrier_productor = barrier;
    uint64_t* barrier_consumer = barrier_productor + 1;

    // regs
    constexpr uint32_t A_TILE_BYTES = BM * BK * sizeof(__nv_bfloat16);
    constexpr uint32_t B_TILE_BYTES = BK * BN * sizeof(__nv_bfloat16); 
    uint32_t phase_productor = 0;
    uint32_t phase_consumer = 0;

    // mbarrier
    if(local_thread_idx == 0) {
        barrier_init(barrier_productor, 1);
        barrier_init(barrier_consumer, 128);
    }
    __syncthreads();

    if constexpr (RegRealloc) {
        // Full producer warpgroup releases first, including idle warps5..7.
        // Never move dec into warp4/lane0: the instruction is WG-uniform.
        if(warp_idx >= 4) {
            producer_release_registers<ProducerRegs>();
        }
        // All256 threads join BEFORE consumer inc, which can block.
        __syncthreads();
    }

    // consumer-productor
    if(warp_idx < 4) {
        if constexpr (RegRealloc) {
            consumer_acquire_registers<ConsumerRegs>();
        }
        // Initialize newly acquired registers before first use. Keeping this
        // inside the consumer role also avoids a producer accumulator lifetime.
        float regs_d[BM / MMA_M][BN / MMA_N][MMA_M * MMA_N / (4 * WARP_SIZE)] = {};

        // wgmma
        for(uint32_t k_start = 0; k_start < K; k_start += BK) {
            // wait productor
            barrier_wait(barrier_productor, phase_productor);
            phase_productor ^= 1;

            // wgmma
            wgmma_fence();
            for(uint32_t tileC_row_start = 0; tileC_row_start < BM; tileC_row_start += MMA_M) {
                for(uint32_t tileC_col_start = 0; tileC_col_start < BN; tileC_col_start += MMA_N) {
                    for(uint32_t tileC_k_start = 0; tileC_k_start < BK; tileC_k_start += MMA_K) {
                        // create A_desc
                        uint32_t Abox_row = tileC_row_start / BOX_A_M;
                        uint32_t Abox_col = tileC_k_start / BOX_A_K;
                        uint32_t Abox_local_row = tileC_row_start % BOX_A_M;
                        uint32_t Abox_local_col = tileC_k_start % BOX_A_K;
                        uint32_t Abox_offset = Abox_row * BOX_A_M * BK + Abox_col * BOX_A_M * BOX_A_K;
                        Abox_offset += box_swizzle128B(
                            Abox_local_row,
                            Abox_local_col,
                            sizeof(__nv_bfloat16)
                        );
                        uint64_t A_desc = create_smem_desc(
                            smem_tileA + Abox_offset,
                            16,
                            8 * BOX_A_K * sizeof(__nv_bfloat16),
                            1,
                            0
                        );

                        // create B_desc
                        uint32_t Bbox_row = tileC_col_start / BOX_B_N;
                        uint32_t Bbox_col = tileC_k_start / BOX_B_K;
                        uint32_t Bbox_local_row = tileC_col_start % BOX_B_N;
                        uint32_t Bbox_local_col = tileC_k_start % BOX_B_K;
                        uint32_t Bbox_offset = Bbox_row * BOX_B_N * BK + Bbox_col * BOX_B_N * BOX_B_K;
                        Bbox_offset += box_swizzle128B(
                            Bbox_local_row,
                            Bbox_local_col,
                            sizeof(__nv_bfloat16)
                        );
                        uint64_t B_desc = create_smem_desc(
                            smem_tileB + Bbox_offset,
                            16,
                            8 * BOX_B_K * sizeof(__nv_bfloat16),
                            1,
                            0
                        );

                        // wgmma
                        wgmma_m64n64k16(
                            regs_d[tileC_row_start / MMA_M][tileC_col_start / MMA_N],
                            A_desc,
                            B_desc,
                            true
                        );
                    }
                }
            }
            wgmma_commit();
            wgmma_wait();

            // activate productor
            barrier_arrive(barrier_consumer);
        }

        // 握手表示矩阵乘法完毕
        barrier_wait(barrier_productor, phase_productor);
        phase_productor ^= 1;

        // store D to shared: [BM / MMA_M][BN / MMA_N][MMA_M * MMA_N / (4 * WARP_SIZE)]
        for(uint32_t d_row = 0; d_row < BM; d_row += MMA_M) {
            uint32_t mma_row = d_row + warp_idx * 16;
            for(uint32_t d_col = 0; d_col < BN; d_col += MMA_N) {
                for(uint32_t mma_n8_start = 0; mma_n8_start < MMA_N; mma_n8_start += 8) {
                    float d0 = regs_d[d_row / MMA_M][d_col / MMA_N][(mma_n8_start / 8) * 4];
                    float d1 = regs_d[d_row / MMA_M][d_col / MMA_N][1 + (mma_n8_start / 8) * 4];
                    float2 vec_d0 = {d0, d1};
                    uint32_t vec_d0_row = (lane_idx / 4) + mma_row;
                    uint32_t vec_d0_col = (lane_idx % 4) * 2 + d_col + mma_n8_start;
                    uint32_t Cbox_row = vec_d0_row / BOX_C_M;
                    uint32_t Cbox_col = vec_d0_col / BOX_C_N;
                    uint32_t Cbox_local_row = vec_d0_row % BOX_C_M;
                    uint32_t Cbox_local_col = vec_d0_col % BOX_C_N;
                    uint32_t offset = Cbox_row * BOX_C_M * BN + Cbox_col * BOX_C_M * BOX_C_N;
                    offset += box_swizzle128B(
                        Cbox_local_row,
                        Cbox_local_col,
                        sizeof(float)
                    );
                    float2* vec_d0_ptr = reinterpret_cast<float2*>(smem_tileC + offset);
                    *vec_d0_ptr = vec_d0;

                    float d2 = regs_d[d_row / MMA_M][d_col / MMA_N][2 + (mma_n8_start / 8) * 4];
                    float d3 = regs_d[d_row / MMA_M][d_col / MMA_N][3 + (mma_n8_start / 8) * 4];
                    float2 vec_d1 = {d2, d3};
                    uint32_t vec_d1_row = (lane_idx / 4) + 8 + mma_row;
                    uint32_t vec_d1_col = (lane_idx % 4) * 2 + d_col + mma_n8_start;
                    Cbox_row = vec_d1_row / BOX_C_M;
                    Cbox_col = vec_d1_col / BOX_C_N;
                    Cbox_local_row = vec_d1_row % BOX_C_M;
                    Cbox_local_col = vec_d1_col % BOX_C_N;
                    offset = Cbox_row * BOX_C_M * BN + Cbox_col * BOX_C_M * BOX_C_N;
                    offset += box_swizzle128B(
                        Cbox_local_row,
                        Cbox_local_col,
                        sizeof(float)
                    );
                    float2* vec_d1_ptr = reinterpret_cast<float2*>(smem_tileC + offset);
                    *vec_d1_ptr = vec_d1;
                }
            }
        }
        // 提醒生产者 shared 已经写好了
        fence_proxy_async();
        barrier_arrive(barrier_consumer);
    }
    else if(warp_idx == 4) {
        if(lane_idx == 0) {
            for(uint32_t k_start = 0; k_start < K; k_start += BK) {
                // TMA
                barrier_expect_tx(
                    barrier_productor,
                    A_TILE_BYTES + B_TILE_BYTES
                );

                // load tileA
                for(uint32_t box_row = 0; box_row < BM; box_row += BOX_A_M) {
                    for(uint32_t box_col = 0; box_col < BK; box_col += BOX_A_K) {
                        uint32_t offset = box_row * BK + box_col * BOX_A_M;
                        tma_load_2d(
                            smem_tileA + offset,
                            &A_desc,
                            k_start + box_col,
                            row + box_row,
                            barrier_productor
                        );
                    }
                }

                // load tileB
                for(uint32_t box_row = 0; box_row < BN; box_row += BOX_B_N) {
                    for(uint32_t box_col = 0; box_col < BK; box_col += BOX_B_K) {
                        uint32_t offset = box_row * BK + box_col * BOX_B_N;
                        tma_load_2d(
                            smem_tileB + offset,
                            &B_desc,
                            k_start + box_col,
                            col + box_row,
                            barrier_productor
                        );
                    }
                }

                // wait consumer
                barrier_wait(barrier_consumer, phase_consumer);
                phase_consumer ^= 1;
            }

            // 握手表示矩阵乘法完毕
            barrier_arrive(barrier_productor);

            // 等待消费者写完 shared
            barrier_wait(barrier_consumer, phase_consumer);
            phase_consumer ^= 1;

            // store d to global
            for (uint32_t box_row = 0; box_row < BM; box_row += BOX_C_M) {
                for (uint32_t box_col = 0; box_col < BN; box_col += BOX_C_N) {
                    uint32_t offset = box_row * BN + box_col * BOX_C_M;
                    tma_store_2d(
                        &C_desc,
                        col + box_col,
                        row + box_row,
                        smem_tileC + offset
                    );
                }
            }

            tma_store_commit();
            tma_store_wait();
        }
    }
}

// Usage (same signature for both source files):
//   gemm_bf16_launcher<64,64,64,true,40,232>(A,B,C,M,N,K,stream);
//   gemm_bf16_launcher<64,64,64,true,32,128>(A,B,C,M,N,K,stream);
//   gemm_bf16_launcher<64,64,64,false>(A,B,C,M,N,K,stream);
// Register arguments precede the optional BOX/MMA parameters. Existing calls
// supplying custom BOX/MMA arguments must insert the three new arguments.
// Not compiled or tested since register redistribution was added.
template<
    size_t BM,
    size_t BN,
    size_t BK,
    bool RegRealloc = true,
    int ProducerRegs = 40,
    int ConsumerRegs = 232,

    size_t BOX_A_M = 64,
    size_t BOX_A_K = 64,

    size_t BOX_B_N = 64,
    size_t BOX_B_K = 64,

    size_t BOX_C_M = 64,
    size_t BOX_C_N = 32,

    size_t MMA_M = 64,
    size_t MMA_N = 64,
    size_t MMA_K = 16
>
void gemm_bf16_launcher(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float* C,
    size_t M,
    size_t N,
    size_t K,
    cudaStream_t stream = nullptr
) {
    assert(M != 0 && N != 0 && K != 0);

    // SWI128B
    static_assert(BM % 64 == 0);
    static_assert(BN % 64 == 0);
    static_assert(BK % 64 == 0);
    static_assert(BOX_A_K == 64);
    static_assert(BOX_B_K == 64);
    static_assert(BOX_C_N == 32);
    static_assert(BM % BOX_A_M == 0);
    static_assert(BK % BOX_A_K == 0);
    static_assert(BN % BOX_B_N == 0);
    static_assert(BK % BOX_B_K == 0);
    static_assert(BM % BOX_C_M == 0);
    static_assert(BN % BOX_C_N == 0);
    static_assert(BOX_A_M % MMA_M == 0);
    static_assert(BOX_A_K * sizeof(__nv_bfloat16) == 128);
    static_assert(BOX_B_N % MMA_N == 0);
    static_assert(BOX_B_K * sizeof(__nv_bfloat16) == 128);
    static_assert(BOX_C_N * sizeof(float) == 128);

    // SMEM
    constexpr size_t A_BYTES = BM * BK * sizeof(__nv_bfloat16);
    constexpr size_t B_BYTES = BN * BK * sizeof(__nv_bfloat16);
    constexpr size_t C_BYTES = BM * BN * sizeof(float);
    constexpr size_t MBARRIER_BYTES = 2 * sizeof(uint64_t);
    // smem_B 1024bytes 对齐
    static_assert(A_BYTES % 1024 == 0);
    // smem 复用
    static_assert(A_BYTES + B_BYTES >= C_BYTES);
    constexpr size_t SMEM_BYTES = A_BYTES + B_BYTES + MBARRIER_BYTES;

    // TMA desc
    alignas(64) CUtensorMap A_desc, B_desc, C_desc;
    if(
        !create_tma_descriptor<BOX_A_M, BOX_A_K>(A, M, K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, &A_desc)
        || !create_tma_descriptor<BOX_B_N, BOX_B_K>(B, N, K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, &B_desc)
        || !create_tma_descriptor<BOX_C_M, BOX_C_N>(C, M, N, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, &C_desc)
    ) return;

    // shared opt
    auto kernel = gemm_bf16_kernel<
        BM, BN, BK,
        BOX_A_M, BOX_A_K,
        BOX_B_N, BOX_B_K,
        BOX_C_M, BOX_C_N,
        MMA_M, MMA_N, MMA_K,
        RegRealloc, ProducerRegs, ConsumerRegs
    >;
    cudaError_t err = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SMEM_BYTES)
    );
    assert(err == cudaSuccess);
    if (err != cudaSuccess) return;

    // kernel call
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    // Enabled: consumer WG0 + producer WG1 (only warp4/lane0 issues TMA).
    dim3 block((RegRealloc ? 8 : 5) * WARP_SIZE);
    kernel<<<grid, block, SMEM_BYTES, stream>>>(
        A_desc, B_desc, C_desc,
        M, N, K
    );

    err = cudaGetLastError();
    assert(err == cudaSuccess);
}