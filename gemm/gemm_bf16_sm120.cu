#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_bf16.h>

uint64_t get_dtype_size(CUtensorMapDataType dtype) {
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

// Row-major matrix
template<uint32_t BOX_M, uint32_t BOX_N>
bool create_matrix_descriptor(
    void* matrix,
    uint64_t M,
    uint64_t N,
    CUtensorMapDataType dtype,
    CUtensorMap* desc
) {
    static_assert(BOX_M >= 1 && BOX_M <= 256);
    static_assert(BOX_N >= 1 && BOX_N <= 256);

    if (matrix == nullptr || desc == nullptr || M == 0 || N == 0) {
        printf("Invalid arguments\n");
        return false;
    }

    if ((uintptr_t)desc % 64 != 0) {
        printf("Descriptor alignment error\n");
        return false;
    }

    uint64_t dtype_size = get_dtype_size(dtype);

    if (dtype_size == 0) {
        printf("Unsupported tensor data type\n");
        return false;
    }

    uint64_t row_stride = N * dtype_size;

    if ((uintptr_t)matrix % 16 != 0 ||
        row_stride % 16 != 0 ||
        BOX_N * dtype_size % 16 != 0) {
        printf("TMA alignment error\n");
        return false;
    }

    uint64_t globalDim[2] = {N, M};
    uint64_t globalStrides[1] = {row_stride};

    uint32_t boxDim[2] = {
        BOX_N,
        BOX_M
    };

    uint32_t elementStrides[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        desc,
        dtype,
        2,
        matrix,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (result != CUDA_SUCCESS) {
        printf("cuTensorMapEncodeTiled failed: %d\n", result);
        return false;
    }

    return true;
}

// 初始化 barrier
__device__ __forceinline__
void init_barrier(uint64_t* barrier) {
    uint32_t addr = __cvta_generic_to_shared(barrier);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], 1;\n"
        "fence.mbarrier_init.release.cluster;"
        :
        : "r"(addr)
        : "memory"
    );
}

// 设置预期传输字节数量
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

// 等待传输完成
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

// 发起传输
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
        "shared::cta.global."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :
        : "r"(smem_addr), "l"(desc), "r"(x), "r"(y), "r"(barrier_addr)
        : "memory"
    );
}

template<CUtensorMapSwizzle SWIZZLE>
__device__ __forceinline__
uint32_t elem16B_offset(
    uint32_t tile_rows,
    uint32_t tile_cols,
    uint32_t box_rows,
    uint32_t box_cols,
    uint32_t row,
    uint32_t col
) {
    static_assert(SWIZZLE == CU_TENSOR_MAP_SWIZZLE_128B);
    uint32_t box_row = row / box_rows;
    uint32_t box_col = col / box_cols;
    uint32_t local_row = row % box_rows;
    uint32_t local_col = col % box_cols;

    uint32_t box_offset = (box_row * (tile_cols / box_cols) + box_col) * box_rows * box_cols;
    // 128B Swizzle
    uint32_t swizzled_col = local_col ^ (local_row & 7);
    uint32_t local_offset = local_row * box_cols + swizzled_col;

    return box_offset + local_offset;
}


__device__ __forceinline__
void ldmatrix_x4(
    uint32_t& r0,
    uint32_t& r1,
    uint32_t& r2,
    uint32_t& r3,
    const void* smem_ptr
) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

__device__ __forceinline__
void ldmatrix_x2_trans(
    uint32_t& r0,
    uint32_t& r1,
    const void* smem_ptr
) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];"
        : "=r"(r0), "=r"(r1)
        : "r"(addr)
    );
}


__device__ __forceinline__
void mma_m16n8k16(
    float (&c)[4],
    const uint32_t (&a)[4],
    const uint32_t (&b)[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]),
          "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]),
          "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
}

template<CUtensorMapSwizzle SWIZZLE>
__device__ __forceinline__
uint32_t elem16B_Cbox_offset(
    uint32_t box_rows,
    uint32_t box_cols,
    uint32_t row,
    uint32_t col,
    uint32_t half_idx
) {
    static_assert(SWIZZLE == CU_TENSOR_MAP_SWIZZLE_128B);
    col = col ^ (row & 7);
    return (row * box_cols + col) * 2 + half_idx;
}

// Shared -> Global
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

#define WARP_SIZE 32
// mma.m16n8k16
template<
    uint32_t BM,
    uint32_t BN,
    uint32_t BK,

    uint32_t A_BOX_M,
    uint32_t A_BOX_K,
    uint32_t B_BOX_K,
    uint32_t B_BOX_N,
    uint32_t C_BOX_M,
    uint32_t C_BOX_N,

    uint32_t WARPS_M,
    uint32_t WARPS_N
>
__global__ void gemm_bf16_kernel(
    const __grid_constant__ CUtensorMap A_desc,
    const __grid_constant__ CUtensorMap B_desc,
    const __grid_constant__ CUtensorMap C_desc,
    uint32_t M,
    uint32_t N,
    uint32_t K
) {
    // index 初始化
    uint32_t row = blockIdx.y * BM;
    uint32_t col = blockIdx.x * BN;
    uint32_t local_thread_idx = threadIdx.x;
    uint32_t warp_idx = local_thread_idx / WARP_SIZE;
    uint32_t lane_idx = local_thread_idx % WARP_SIZE;
    uint32_t row_warp = warp_idx / WARPS_N;
    uint32_t col_warp = warp_idx % WARPS_N;

    // shared 初始化
    extern __shared__ __align__(128) char smem[];
    __nv_bfloat16* smem_tileA = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_tileB = smem_tileA + BM * BK;
    float* smem_tileC = reinterpret_cast<float*>(smem_tileB + BK * BN);
    uint64_t* barrier = reinterpret_cast<uint64_t*>(smem_tileC + BM * BN);

    // regs 初始化
    uint32_t regs_A[BM / WARPS_M / 16][4];
    uint32_t regs_B[BN / WARPS_N / 8][2];
    float regs_C[BM / WARPS_M / 16][BN / WARPS_N / 8][4] = {};

    // mbarrier 初始化
    if (local_thread_idx == 0) {
        init_barrier(barrier);
    }

    __syncthreads();
    constexpr uint32_t A_TILE_BYTES = BM * BK * sizeof(__nv_bfloat16);
    constexpr uint32_t B_TILE_BYTES = BK * BN * sizeof(__nv_bfloat16);
    uint32_t current_phase = 0;
    // K维度循环
    for(uint32_t k_start = 0; k_start < K; k_start += BK){
        // TMA 传输
        if(local_thread_idx == 0) {
            barrier_expect_tx(barrier, A_TILE_BYTES + B_TILE_BYTES);
            for(uint32_t box_row = 0; box_row < BM / A_BOX_M; box_row++) {
                for(uint32_t box_col = 0; box_col < BK / A_BOX_K; box_col++) {
                    uint32_t x = k_start + box_col * A_BOX_K;
                    uint32_t y = row + box_row * A_BOX_M;
                    __nv_bfloat16* dst = smem_tileA + (box_row * (BK / A_BOX_K) + box_col) * A_BOX_M * A_BOX_K;
                    tma_load_2d(
                        dst,
                        &A_desc,
                        x,
                        y,
                        barrier
                    );
                }
            }
            for(uint32_t box_row = 0; box_row < BK / B_BOX_K; box_row++) {
                for(uint32_t box_col = 0; box_col < BN / B_BOX_N; box_col++) {
                    uint32_t x = col + box_col * B_BOX_N;
                    uint32_t y = k_start + box_row * B_BOX_K;
                    __nv_bfloat16* dst = smem_tileB + (box_row * (BN / B_BOX_N) + box_col) * B_BOX_K * B_BOX_N;
                    tma_load_2d(
                        dst,
                        &B_desc,
                        x,
                        y,
                        barrier
                    );
                }
            }
            barrier_wait(barrier, current_phase);
        }
        __syncthreads();

        for(uint32_t ktile16_start = 0; ktile16_start < BK; ktile16_start += 16) {
            // ldmatrix
            uint32_t row_offset_Atile16x16 = row_warp * (BM / WARPS_M);
            uint32_t col_offset_Atile16x16 = ktile16_start;
            for(uint32_t tile16_start = 0; tile16_start < BM / WARPS_M; tile16_start += 16) {
                // 逻辑地址 -> 物理地址
                uint32_t row_tile16x16 = row_offset_Atile16x16 + tile16_start;
                uint32_t col_tile16x16 = col_offset_Atile16x16 / (16 / sizeof(__nv_bfloat16));
                uint32_t row_elem16B = row_tile16x16 + lane_idx % 16;
                uint32_t col_elem16B = col_tile16x16 + lane_idx / 16;
                uint32_t offset_elem = elem16B_offset<CU_TENSOR_MAP_SWIZZLE_128B>(
                    BM, BK / (16 / sizeof(__nv_bfloat16)),
                    A_BOX_M, A_BOX_K / (16 / sizeof(__nv_bfloat16)),
                    row_elem16B, col_elem16B
                );
                __nv_bfloat16* ptr = smem_tileA + offset_elem * 8;

                // smem -> regs
                ldmatrix_x4(
                    regs_A[tile16_start / 16][0],
                    regs_A[tile16_start / 16][1],
                    regs_A[tile16_start / 16][2],
                    regs_A[tile16_start / 16][3],
                    ptr
                );
            }
            uint32_t row_offset_Btile16x8 = ktile16_start;
            uint32_t col_offset_Btile16x8 = col_warp * (BN / WARPS_N);
            for(uint32_t tile8_start = 0; tile8_start < BN / WARPS_N; tile8_start += 8) {
                // 逻辑地址 -> 物理地址
                uint32_t row_tile16x8 = row_offset_Btile16x8;
                uint32_t col_tile16x8 = (col_offset_Btile16x8 + tile8_start) / (16 / sizeof(__nv_bfloat16));
                uint32_t row_elem16B = row_tile16x8 + lane_idx % 16;
                uint32_t col_elem16B = col_tile16x8 + lane_idx / 16;
                uint32_t offset_elem = elem16B_offset<CU_TENSOR_MAP_SWIZZLE_128B>(
                    BK, BN / (16 / sizeof(__nv_bfloat16)),
                    B_BOX_K, B_BOX_N / (16 / sizeof(__nv_bfloat16)),
                    row_elem16B, col_elem16B
                );
                __nv_bfloat16* ptr = smem_tileB + offset_elem * 8;

                // smem -> regs
                ldmatrix_x2_trans(
                    regs_B[tile8_start / 8][0],
                    regs_B[tile8_start / 8][1],
                    ptr
                );
            }

            // mma
            for(uint32_t tile16_start = 0; tile16_start < BM / WARPS_M / 16; tile16_start++) {
                for(uint32_t tile8_start = 0; tile8_start < BN / WARPS_N / 8; tile8_start++) {
                    mma_m16n8k16(
                        regs_C[tile16_start][tile8_start],
                        regs_A[tile16_start],
                        regs_B[tile8_start]
                    );
                }
            }
        }
        __syncthreads();
        current_phase ^= 1;
    }
    
    // regs -> shared
    float2* smem_c_ptr = reinterpret_cast<float2*>(smem_tileC);
    uint32_t row_offset_ctile = row_warp * (BM / WARPS_M);
    uint32_t col_offset_ctile = col_warp * (BN / WARPS_N);
    for(uint32_t m16_idx = 0; m16_idx < BM / WARPS_M / 16; m16_idx++) {
        uint32_t row_offset_tile16x8 = row_offset_ctile + m16_idx * 16;
        for(uint32_t n8_idx = 0; n8_idx < BN / WARPS_N / 8; n8_idx++) {
            uint32_t col_offset_tile16x8 = (col_offset_ctile + n8_idx * 8) / 2;
            
            uint32_t row_elem0 = row_offset_tile16x8 + lane_idx / 4;
            uint32_t col_elem0 = col_offset_tile16x8 + lane_idx % 4;
            uint32_t row_elem1 = row_offset_tile16x8 + 8 + lane_idx / 4;
            uint32_t col_elem1 = col_offset_tile16x8 + lane_idx % 4;
            
            uint32_t row_box_elem0 = row_elem0 / C_BOX_M;
            uint32_t col_box_elem0 = col_elem0 / (C_BOX_N / 2);
            uint32_t row_box_elem1 = row_elem1 / C_BOX_M;
            uint32_t col_box_elem1 = col_elem1 / (C_BOX_N / 2);

            uint32_t local_row_box_elem0 = row_elem0 % C_BOX_M;
            uint32_t local_col_box_elem0 = col_elem0 % (C_BOX_N / 2);
            uint32_t local_col_box_elem0_16B = local_col_box_elem0 / 2;
            uint32_t local_row_box_elem1 = row_elem1 % C_BOX_M;
            uint32_t local_col_box_elem1 = col_elem1 % (C_BOX_N / 2);
            uint32_t local_col_box_elem1_16B = local_col_box_elem1 / 2;

            uint32_t local_offset_box_elem0 = elem16B_Cbox_offset<CU_TENSOR_MAP_SWIZZLE_128B>(
                C_BOX_M, C_BOX_N / 4,
                local_row_box_elem0, local_col_box_elem0_16B,
                lane_idx % 2
            );
            uint32_t local_offset_box_elem1 = elem16B_Cbox_offset<CU_TENSOR_MAP_SWIZZLE_128B>(
                C_BOX_M, C_BOX_N / 4,
                local_row_box_elem1, local_col_box_elem1_16B,
                lane_idx % 2
            );

            uint32_t offset_elem0 = (row_box_elem0 * (BN / C_BOX_N) + col_box_elem0) * (C_BOX_M * C_BOX_N / 2) + local_offset_box_elem0;
            uint32_t offset_elem1 = (row_box_elem1 * (BN / C_BOX_N) + col_box_elem1) * (C_BOX_M * C_BOX_N / 2) + local_offset_box_elem1;
            
            float2 elem0 = make_float2(regs_C[m16_idx][n8_idx][0], regs_C[m16_idx][n8_idx][1]);
            float2 elem1 = make_float2(regs_C[m16_idx][n8_idx][2], regs_C[m16_idx][n8_idx][3]);

            smem_c_ptr[offset_elem0] = elem0;
            smem_c_ptr[offset_elem1] = elem1;
        }
    }
    __syncthreads();

    // 将普通 Shared Memory 写入同步到 Async Proxy
    asm volatile(
        "fence.proxy.async.shared::cta;"
        :
        :
        : "memory"
    );

    __syncthreads();

    if (local_thread_idx == 0) {
        for (uint32_t box_row = 0; box_row < BM / C_BOX_M; box_row++) {
            for (uint32_t box_col = 0; box_col < BN / C_BOX_N; box_col++) {
                uint32_t x = col + box_col * C_BOX_N;
                uint32_t y = row + box_row * C_BOX_M;

                float* src = smem_tileC + (box_row * (BN / C_BOX_N) + box_col) * C_BOX_M * C_BOX_N;
                tma_store_2d(
                    &C_desc,
                    x,
                    y,
                    src
                );
            }
        }

        // 提交所有 TMA Store
        tma_store_commit();

        // 等待所有 Store 完成
        tma_store_wait();
    }

    __syncthreads();
}

template<
    uint32_t BM = 128,
    uint32_t BN = 128,
    uint32_t BK = 64,
    uint32_t WARPS_M = 2,
    uint32_t WARPS_N = 2
>
void gemm_bf16_launcher(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float* C,
    uint64_t M,
    uint64_t N,
    uint64_t K,
    cudaStream_t stream = nullptr
) {
    // Tile 与 warp 划分必须匹配当前 MMA 和 128B swizzle 布局。
    static_assert(BM > 0 && BM <= 256, "BM must be in [1, 256]");
    static_assert(BN > 0 && BN % 64 == 0, "BN must be a multiple of 64");
    static_assert(BK > 0 && BK <= 256 && BK % 64 == 0,
                  "BK must be a multiple of 64 in [64, 256]");
    static_assert(WARPS_M > 0 && WARPS_N > 0 &&
                  WARPS_M <= 32 / WARPS_N, "Invalid warp count");
    static_assert(BM % (WARPS_M * 16) == 0,
                  "Each warp must cover a multiple of 16 rows");
    static_assert(BN % (WARPS_N * 8) == 0,
                  "Each warp must cover a multiple of 8 columns");

    // TMA box：BF16 每行 64 个元素，FP32 每行 32 个元素，均为 128B。
    constexpr uint32_t A_BOX_M = BM;
    constexpr uint32_t A_BOX_K = 64;
    constexpr uint32_t B_BOX_K = BK;
    constexpr uint32_t B_BOX_N = 64;
    constexpr uint32_t C_BOX_M = BM;
    constexpr uint32_t C_BOX_N = 32;

    constexpr size_t A_BYTES = size_t(BM) * BK * sizeof(__nv_bfloat16);
    constexpr size_t B_BYTES = size_t(BK) * BN * sizeof(__nv_bfloat16);
    constexpr size_t C_BYTES = size_t(BM) * BN * sizeof(float);
    constexpr size_t SMEM_BYTES = A_BYTES + B_BYTES + C_BYTES + sizeof(uint64_t);

    if (M > UINT32_MAX || N > UINT32_MAX || K > UINT32_MAX) {
        printf("Matrix dimensions exceed the kernel's uint32_t range\n");
        return;
    }

    // A[M, K]、B[K, N]、C[M, N] 均为连续行主序矩阵。
    alignas(64) CUtensorMap A_desc, B_desc, C_desc;
    if (!create_matrix_descriptor<A_BOX_M, A_BOX_K>(
            const_cast<__nv_bfloat16*>(A), M, K,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, &A_desc) ||
        !create_matrix_descriptor<B_BOX_K, B_BOX_N>(
            const_cast<__nv_bfloat16*>(B), K, N,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, &B_desc) ||
        !create_matrix_descriptor<C_BOX_M, C_BOX_N>(
            C, M, N, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, &C_desc)) {
        return;
    }

    auto kernel = gemm_bf16_kernel<
        BM, BN, BK,
        A_BOX_M, A_BOX_K,
        B_BOX_K, B_BOX_N,
        C_BOX_M, C_BOX_N,
        WARPS_M, WARPS_N
    >;

    // 动态 shared memory 超过默认上限时需要 opt-in。
    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SMEM_BYTES)
    );
    if (err != cudaSuccess) {
        printf("Shared memory opt-in failed: %s\n", cudaGetErrorString(err));
        return;
    }

    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const dim3 block(WARPS_M * WARPS_N * WARP_SIZE);
    kernel<<<grid, block, SMEM_BYTES, stream>>>(
        A_desc, B_desc, C_desc,
        static_cast<uint32_t>(M),
        static_cast<uint32_t>(N),
        static_cast<uint32_t>(K)
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

int main(void) {
    printf("hello cuda\n");
    return 0;
}