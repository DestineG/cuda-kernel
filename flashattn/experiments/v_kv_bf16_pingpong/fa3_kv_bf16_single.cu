// Isolated KV double-buffer BF16 experiment: single.
// Independent K/V ring slots, derived from the v1 combined baseline.
// Derived from fa3_bf16_sm90_kv_pipeline_qk_pv_overlap_v1.cu; combined compile and causal-mask optimizations.
#include<stdint.h>
#include<assert.h>

#include<cuda.h>
#include<cuda_bf16.h>
#include <cuda_runtime.h>

#include "mbarrier.cuh"
#include "tma.cuh"
#include "wgmma.cuh"


#define WARP_SIZE 32


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
bool create_tma_descriptor_swizzle128B(
    const void* matrix,
    size_t M,
    size_t N,
    CUtensorMapDataType dtype,
    CUtensorMap* desc,
    size_t row_stride
) {
    // global 基地址和行跨度的 16B 对齐
    assert((uintptr_t)matrix % 16 == 0);
    assert(row_stride % 16 == 0);

    // swizzle128B
    size_t dtype_size = get_dtype_size(dtype);
    assert((uintptr_t)matrix % 128 == 0);
    assert(BOX_N * dtype_size == 128);

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

    return res == CUDA_SUCCESS;
}

__device__ __forceinline__
uint32_t tma_box_swizzle128B(
    uint32_t row,
    uint32_t col,
    uint32_t dtype_size
) {
    col = ((col / (16 / dtype_size)) ^ (row % 8)) * (16 / dtype_size) + (col % (16 / dtype_size));
    return row * (128 / dtype_size) + col;
}

template<
    size_t TILE_M,
    size_t TILE_N,
    size_t BOX_M,
    size_t BOX_N
>
__device__ __forceinline__
uint32_t tma_tile_offset_swizzle128B(
    uint32_t row,
    uint32_t col,
    uint32_t dtype_size
) {
    uint32_t box_row = row / BOX_M;
    uint32_t box_col = col / BOX_N;
    uint32_t box_local_row = row % BOX_M;
    uint32_t box_local_col = col % BOX_N;
    uint32_t offset = box_row * BOX_M * TILE_N + box_col * BOX_N * BOX_M;
    return offset + tma_box_swizzle128B(box_local_row, box_local_col, dtype_size);
}

__device__ __forceinline__
uint32_t pack_bf16(float x, float y) {
    uint32_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(x));
    uint32_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(y));
    return lo | (hi << 16);
}

template<
    size_t Br,
    size_t Bc,
    size_t HEAD_DIM,
    bool CAUSAL,

    size_t BOX_Q_M,
    size_t BOX_Q_N,

    size_t BOX_KV_M,
    size_t BOX_KV_N,

    size_t BOX_O_M,
    size_t BOX_O_N,

    size_t MMA_M,
    size_t MMA_N,
    size_t MMA_K,

    bool RegRealloc = false,
    uint32_t ProducerRegs = 40,
    uint32_t ConsumerRegs = 232
>
__global__ __launch_bounds__(8 * WARP_SIZE, 1)
void fa3_kv_bf16_single(
    const __grid_constant__ CUtensorMap tma_desc_Q,
    const __grid_constant__ CUtensorMap tma_desc_K,
    const __grid_constant__ CUtensorMap tma_desc_V,
    const __grid_constant__ CUtensorMap tma_desc_O,

    const size_t* cu_seqlens_q,
    const size_t* cu_seqlens_kv,

    size_t num_heads,
    size_t num_kv_heads,

    float scale,
    __nv_bfloat16* O
) {
    // One consumer warpgroup and one producer warpgroup, 128 threads each.
    static_assert(!RegRealloc ||
                  (ProducerRegs >= 24 && ProducerRegs <= 256 && ProducerRegs % 8 == 0),
                  "ProducerRegs must be a multiple of 8 in [24, 256]");
    static_assert(!RegRealloc ||
                  (ConsumerRegs >= 24 && ConsumerRegs <= 256 && ConsumerRegs % 8 == 0),
                  "ConsumerRegs must be a multiple of 8 in [24, 256]");
    static_assert(!RegRealloc || ProducerRegs <= ConsumerRegs,
                  "ProducerRegs must not exceed ConsumerRegs");
    static_assert(!RegRealloc || 128 * (ProducerRegs + ConsumerRegs) <= 65536,
                  "Register targets exceed the SM90 register-file capacity");

    // index
    size_t seq_idx = blockIdx.z;
    size_t head_idx = blockIdx.y;
    size_t qtile_idx = blockIdx.x;
    size_t local_thread_idx = threadIdx.x;
    size_t warp_idx = local_thread_idx / WARP_SIZE;
    size_t lane_idx = local_thread_idx % WARP_SIZE;

    // Q
    size_t q_start = cu_seqlens_q[seq_idx];
    size_t q_end = cu_seqlens_q[seq_idx + 1];
    size_t q_lens = q_end - q_start;
    size_t qtile_start = q_start + qtile_idx * Br;
    if(qtile_start >= q_end) return;
    size_t num_qtile_row = min(Br, q_end - qtile_start);

    // KV
    size_t kv_start = cu_seqlens_kv[seq_idx];
    size_t kv_end = cu_seqlens_kv[seq_idx + 1];
    size_t kv_lens = kv_end - kv_start;
    if constexpr (CAUSAL) {
        // Avoid an __assertfail device call in the WGMMA kernel.
        if (kv_lens < q_lens) {
            asm volatile("trap;");
            return;
        }
    }
    if (kv_lens == 0) return;

    // Sequence-local Q row range. Both warpgroups use the same KV limit.
    const size_t q_row_begin = qtile_idx * Br;
    size_t kv_limit = kv_lens;
    if constexpr (CAUSAL) {
        // Exclusive K end allowed by the last valid query row (bottom-right mask).
        kv_limit = kv_lens - q_lens + q_row_begin + num_qtile_row;
    }
    size_t num_kvtiles = (kv_limit + Bc - 1) / Bc;
    size_t kv_head_idx = head_idx / (num_heads / num_kv_heads);
    size_t num_last_kvtile_row = kv_limit - (num_kvtiles - 1) * Bc;

    // K and V have independent slots and lifetimes.
    constexpr size_t K_STAGES = 2;
    constexpr size_t V_STAGES = 2;

    // shared
    size_t Q_TILE_BYTES = Br * HEAD_DIM * sizeof(__nv_bfloat16);
    size_t KV_TILE_BYTES = Bc * HEAD_DIM * sizeof(__nv_bfloat16);
    size_t O_TILE_BYTES = Br * HEAD_DIM * sizeof(__nv_bfloat16);
    extern __shared__ __align__(1024) char smem[];
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(
        smem
    );
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(
        smem + Q_TILE_BYTES
    );
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(
        smem + Q_TILE_BYTES + K_STAGES * KV_TILE_BYTES
    );
    __nv_bfloat16* smem_o = reinterpret_cast<__nv_bfloat16*>(smem);
    uint64_t* barrier = reinterpret_cast<uint64_t*>(
        smem + Q_TILE_BYTES + (K_STAGES + V_STAGES) * KV_TILE_BYTES
    );

    // Dedicated Q/epilogue handshake; ring phases always start at zero.
    uint64_t* barrier_q_ready = barrier;
    uint64_t* barrier_q_free = barrier + 1;
    uint64_t* barrier_k_ready = barrier + 2;
    uint64_t* barrier_k_free = barrier_k_ready + K_STAGES;
    uint64_t* barrier_v_ready = barrier_k_free + K_STAGES;
    uint64_t* barrier_v_free = barrier_v_ready + V_STAGES;
    if (local_thread_idx == 0) {
        barrier_init(barrier_q_ready, 1);
        barrier_init(barrier_q_free, 4 * WARP_SIZE);
        for (size_t slot = 0; slot < K_STAGES; ++slot) {
            barrier_init(barrier_k_ready + slot, 1);
            barrier_init(barrier_k_free + slot, 4 * WARP_SIZE);
        }
        for (size_t slot = 0; slot < V_STAGES; ++slot) {
            barrier_init(barrier_v_ready + slot, 1);
            barrier_init(barrier_v_free + slot, 4 * WARP_SIZE);
        }
    }
    __syncthreads();


    // All 128 producer threads release registers, including the idle warps.
    // Do this before the single-thread producer branch below.
    if constexpr (RegRealloc) {
        if (warp_idx >= 4 && warp_idx < 8) {
            warpgroup_reg_dealloc<ProducerRegs>();
        }
    }

    if(warp_idx < 4) {
        // All 128 consumer threads request their target before using MMA registers.
        if constexpr (RegRealloc) {
            warpgroup_reg_alloc<ConsumerRegs>();
        }

        // regs
        float regs_row_max[Br / 64][2];
        float regs_row_factor[Br / 64][2];
        float regs_row_sum[Br / 64][2] = {};
        for (int i = 0; i < Br / 64; ++i) {
            regs_row_max[i][0] = -INFINITY;
            regs_row_max[i][1] = -INFINITY;
        }
        float regs_sp[Br / MMA_M][Bc / MMA_N][MMA_M * MMA_N / (4 * WARP_SIZE)];
        uint32_t regs_p[Br / MMA_M][Bc / MMA_N][MMA_M * MMA_N / (4 * WARP_SIZE) / 2];
        float regs_o[Br / MMA_M][HEAD_DIM / MMA_N][MMA_M * MMA_N / (4 * WARP_SIZE)] = {};
        uint32_t phase_q_ready = 0;
        uint32_t phase_k_ready[K_STAGES] = {};
        uint32_t phase_v_ready[V_STAGES] = {};

        // wait q productor
        barrier_wait(barrier_q_ready, phase_q_ready);
        phase_q_ready ^= 1;

        // arrive q consumer
        barrier_arrive(barrier_q_free);

        // Prologue: no previous P/V exists, so issue only QK_0.
        {
            const size_t kvtile_offset = 0;

            // wait k productor
            const size_t k_slot = (kvtile_offset / Bc) % K_STAGES;
            barrier_wait(barrier_k_ready + k_slot, phase_k_ready[k_slot]);
            phase_k_ready[k_slot] ^= 1;

            // s = q@k
            wgmma_fence();
            for(uint32_t mma_m = 0; mma_m < Br; mma_m += MMA_M) {
                for(uint32_t mma_n = 0; mma_n < Bc; mma_n += MMA_N) {
                    for(uint32_t mma_k = 0; mma_k < HEAD_DIM; mma_k += MMA_K) {
                        // mma_desc_smem_q
                        uint32_t offset = tma_tile_offset_swizzle128B<
                            Br, HEAD_DIM,
                            BOX_Q_M, BOX_Q_N
                        >(mma_m, mma_k, sizeof(__nv_bfloat16));
                        uint64_t mma_desc_smem_Q = create_smem_desc(
                            smem_q + offset,
                            16,
                            8 * BOX_Q_N * sizeof(__nv_bfloat16),
                            1,
                            0
                        );

                        // mma_desc_smem_k
                        offset = tma_tile_offset_swizzle128B<
                            Bc, HEAD_DIM,
                            BOX_KV_M, BOX_KV_N
                        >(mma_n, mma_k, sizeof(__nv_bfloat16));
                        uint64_t mma_desc_smem_K = create_smem_desc(
                            smem_k + k_slot * Bc * HEAD_DIM + offset,
                            16,
                            8 * BOX_KV_N * sizeof(__nv_bfloat16),
                            1,
                            0
                        );

                        // wgmma
                        wgmma_m64n64k16_ss(
                            regs_sp[mma_m / MMA_M][mma_n / MMA_N],
                            mma_desc_smem_Q,
                            mma_desc_smem_K,
                            mma_k != 0
                        );
                    }
                }
            }
            wgmma_commit();


            wgmma_wait<0>();


            // arrive k consumer
            barrier_arrive(barrier_k_free + k_slot);

            // kv 尾块处理
            if(kvtile_offset + Bc > kv_limit) {
                for(uint32_t mma_m = 0; mma_m < Br / MMA_M; mma_m++) {
                    for(uint32_t mma_n = 0; mma_n < Bc / MMA_N; mma_n++) {
                        for(uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; mma_n8++) {
                            uint32_t elem_col = (lane_idx % 4) * 2 + mma_n * MMA_N + mma_n8 * 8;
                            if (elem_col >= num_last_kvtile_row) {
                                regs_sp[mma_m][mma_n][mma_n8 * 4] = -INFINITY;
                                regs_sp[mma_m][mma_n][mma_n8 * 4 + 2] = -INFINITY;
                            }
                            if (elem_col + 1 >= num_last_kvtile_row) {
                                regs_sp[mma_m][mma_n][mma_n8 * 4 + 1] = -INFINITY;
                                regs_sp[mma_m][mma_n][mma_n8 * 4 + 3] = -INFINITY;
                            }
                        }
                    }
                }
            }

            // Only diagonal/boundary tiles need element-wise causal masking.
            if constexpr(CAUSAL) {
                const size_t first_row_k_end = kv_lens - q_lens + q_row_begin + 1;
                if (kvtile_offset + Bc > first_row_k_end) {
                    for (uint32_t mma_m = 0; mma_m < Br / MMA_M; mma_m++) {
                        uint32_t mma_row = mma_m * MMA_M + warp_idx * 16;
                        for (uint32_t mma_n = 0; mma_n < Bc / MMA_N; mma_n++) {
                            for (uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; mma_n8++) {
                                size_t elem_row = (lane_idx / 4) + mma_row + qtile_idx * Br + kv_lens - q_lens;
                                uint32_t elem_col = (lane_idx % 4) * 2 + mma_n * MMA_N + mma_n8 * 8 + kvtile_offset;
                                if(elem_row < elem_col) {
                                    regs_sp[mma_m][mma_n][mma_n8 * 4] = -INFINITY;
                                }
                                if(elem_row < 1 + elem_col) {
                                    regs_sp[mma_m][mma_n][1 + mma_n8 * 4] = -INFINITY;
                                }

                                if(8 + elem_row < elem_col) {
                                    regs_sp[mma_m][mma_n][2 + mma_n8 * 4] = -INFINITY;
                                }
                                if(8 + elem_row < 1 + elem_col) {
                                    regs_sp[mma_m][mma_n][3 + mma_n8 * 4] = -INFINITY;
                                }
                            }
                        }
                    }

                }
            }

            // new_m, factor = rowmax(old_m, s)
            for (uint32_t mma_m = 0; mma_m < Br / MMA_M; mma_m++) {
                float new_max[2] = {-INFINITY, -INFINITY};
                for (uint32_t mma_n = 0; mma_n < Bc / MMA_N; mma_n++) {
                    for (uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; mma_n8++) {
                        regs_sp[mma_m][mma_n][mma_n8 * 4] *= scale;
                        regs_sp[mma_m][mma_n][1 + mma_n8 * 4] *= scale;
                        regs_sp[mma_m][mma_n][2 + mma_n8 * 4] *= scale;
                        regs_sp[mma_m][mma_n][3 + mma_n8 * 4] *= scale;
                        float max0 = fmaxf(regs_sp[mma_m][mma_n][mma_n8 * 4], regs_sp[mma_m][mma_n][1 + mma_n8 * 4]);
                        float max1 = fmaxf(regs_sp[mma_m][mma_n][2 + mma_n8 * 4], regs_sp[mma_m][mma_n][3 + mma_n8 * 4]);
                        new_max[0] = fmaxf(new_max[0], max0);
                        new_max[1] = fmaxf(new_max[1], max1);
                    }
                }
                new_max[0] = fmaxf(new_max[0], __shfl_xor_sync(0xffffffff, new_max[0], 1));
                new_max[0] = fmaxf(new_max[0], __shfl_xor_sync(0xffffffff, new_max[0], 2));
                new_max[1] = fmaxf(new_max[1], __shfl_xor_sync(0xffffffff, new_max[1], 1));
                new_max[1] = fmaxf(new_max[1], __shfl_xor_sync(0xffffffff, new_max[1], 2));
                new_max[0] = fmaxf(new_max[0], regs_row_max[mma_m][0]);
                new_max[1] = fmaxf(new_max[1], regs_row_max[mma_m][1]);
                regs_row_factor[mma_m][0] = expf(regs_row_max[mma_m][0] - new_max[0]);
                regs_row_factor[mma_m][1] = expf(regs_row_max[mma_m][1] - new_max[1]);
                regs_row_max[mma_m][0] = new_max[0];
                regs_row_max[mma_m][1] = new_max[1];
            }

            // new_l, p = apply_max_and_rowsum(old_l, s, new_max, factor)
            for (uint32_t mma_m = 0; mma_m < Br / MMA_M; mma_m++) {
                float new_sum[2] = {0, 0};
                for (uint32_t mma_n = 0; mma_n < Bc / MMA_N; mma_n++) {
                    for (uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; mma_n8++) {
                        regs_sp[mma_m][mma_n][mma_n8 * 4] = expf(regs_sp[mma_m][mma_n][mma_n8 * 4] - regs_row_max[mma_m][0]);
                        regs_sp[mma_m][mma_n][1 + mma_n8 * 4] = expf(regs_sp[mma_m][mma_n][1 + mma_n8 * 4] - regs_row_max[mma_m][0]);
                        regs_sp[mma_m][mma_n][2 + mma_n8 * 4] = expf(regs_sp[mma_m][mma_n][2 + mma_n8 * 4] - regs_row_max[mma_m][1]);
                        regs_sp[mma_m][mma_n][3 + mma_n8 * 4] = expf(regs_sp[mma_m][mma_n][3 + mma_n8 * 4] - regs_row_max[mma_m][1]);
                        float sum0 = regs_sp[mma_m][mma_n][mma_n8 * 4] + regs_sp[mma_m][mma_n][1 + mma_n8 * 4];
                        float sum1 = regs_sp[mma_m][mma_n][2 + mma_n8 * 4] + regs_sp[mma_m][mma_n][3 + mma_n8 * 4];
                        new_sum[0] += sum0;
                        new_sum[1] += sum1;
                    }
                }
                new_sum[0] += __shfl_xor_sync(0xffffffff, new_sum[0], 1);
                new_sum[0] += __shfl_xor_sync(0xffffffff, new_sum[0], 2);
                new_sum[1] += __shfl_xor_sync(0xffffffff, new_sum[1], 1);
                new_sum[1] += __shfl_xor_sync(0xffffffff, new_sum[1], 2);
                regs_row_sum[mma_m][0] = regs_row_sum[mma_m][0] * regs_row_factor[mma_m][0] + new_sum[0];
                regs_row_sum[mma_m][1] = regs_row_sum[mma_m][1] * regs_row_factor[mma_m][1] + new_sum[1];
            }



            // packed(p)
            for (uint32_t mma_m = 0; mma_m < Br / MMA_M; mma_m++) {
                for (uint32_t mma_n = 0; mma_n < Bc / MMA_N; mma_n++) {
                    for (uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; mma_n8++) {
                        regs_p[mma_m][mma_n][mma_n8 * 4 / 2] = pack_bf16(
                            regs_sp[mma_m][mma_n][mma_n8 * 4],
                            regs_sp[mma_m][mma_n][1 + mma_n8 * 4]
                        );
                        regs_p[mma_m][mma_n][1 + mma_n8 * 4 / 2] = pack_bf16(
                            regs_sp[mma_m][mma_n][2 + mma_n8 * 4],
                            regs_sp[mma_m][mma_n][3 + mma_n8 * 4]
                        );
                    }
                }
            }

            // old_o_scale = apply_factor(old_o, factor)
            for(uint32_t mma_m = 0; mma_m < Br / MMA_M; mma_m++) {
                for(uint32_t mma_n = 0; mma_n < HEAD_DIM / MMA_N; mma_n++) {
                    for(uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; mma_n8++) {
                        regs_o[mma_m][mma_n][mma_n8 * 4] *= regs_row_factor[mma_m][0];
                        regs_o[mma_m][mma_n][1 + mma_n8 * 4] *= regs_row_factor[mma_m][0];
                        regs_o[mma_m][mma_n][2 + mma_n8 * 4] *= regs_row_factor[mma_m][1];
                        regs_o[mma_m][mma_n][3 + mma_n8 * 4] *= regs_row_factor[mma_m][1];
                    }
                }
            }
        }

        // Every steady-state iteration has a previous P/V to consume.
        for(size_t kvtile_offset = Bc; kvtile_offset < kv_limit; kvtile_offset += Bc) {

            // wait k productor
            const size_t k_slot = (kvtile_offset / Bc) % K_STAGES;
            barrier_wait(barrier_k_ready + k_slot, phase_k_ready[k_slot]);
            phase_k_ready[k_slot] ^= 1;

            // s = q@k
            wgmma_fence();
            for(uint32_t mma_m = 0; mma_m < Br; mma_m += MMA_M) {
                for(uint32_t mma_n = 0; mma_n < Bc; mma_n += MMA_N) {
                    for(uint32_t mma_k = 0; mma_k < HEAD_DIM; mma_k += MMA_K) {
                        // mma_desc_smem_q
                        uint32_t offset = tma_tile_offset_swizzle128B<
                            Br, HEAD_DIM,
                            BOX_Q_M, BOX_Q_N
                        >(mma_m, mma_k, sizeof(__nv_bfloat16));
                        uint64_t mma_desc_smem_Q = create_smem_desc(
                            smem_q + offset,
                            16,
                            8 * BOX_Q_N * sizeof(__nv_bfloat16),
                            1,
                            0
                        );

                        // mma_desc_smem_k
                        offset = tma_tile_offset_swizzle128B<
                            Bc, HEAD_DIM,
                            BOX_KV_M, BOX_KV_N
                        >(mma_n, mma_k, sizeof(__nv_bfloat16));
                        uint64_t mma_desc_smem_K = create_smem_desc(
                            smem_k + k_slot * Bc * HEAD_DIM + offset,
                            16,
                            8 * BOX_KV_N * sizeof(__nv_bfloat16),
                            1,
                            0
                        );

                        // wgmma
                        wgmma_m64n64k16_ss(
                            regs_sp[mma_m / MMA_M][mma_n / MMA_N],
                            mma_desc_smem_Q,
                            mma_desc_smem_K,
                            mma_k != 0
                        );
                    }
                }
            }
            wgmma_commit();


            // wait v productor
            const size_t v_slot = (kvtile_offset / Bc - 1) % V_STAGES;
            barrier_wait(barrier_v_ready + v_slot, phase_v_ready[v_slot]);
            phase_v_ready[v_slot] ^= 1;

            // new_o = old_o_scale + p@v
            for(uint32_t mma_m = 0; mma_m < Br; mma_m += MMA_M) {
                for(uint32_t mma_n = 0; mma_n < HEAD_DIM; mma_n += MMA_N) {
                    for(uint32_t mma_k = 0; mma_k < Bc; mma_k += MMA_K) {
                        // mma_desc_regs_p
                        uint32_t p_frag[MMA_K / 4];
                        uint32_t dst = 0;

                        for (uint32_t col = mma_k; col < mma_k + MMA_K; col += 8) {
                            uint32_t fragment = col / MMA_N;
                            uint32_t reg = ((col % MMA_N) / 8) * 2;

                            p_frag[dst++] = regs_p[mma_m / MMA_M][fragment][reg];
                            p_frag[dst++] = regs_p[mma_m / MMA_M][fragment][reg + 1];
                        }

                        // mma_desc_smem_v
                        uint32_t offset = tma_tile_offset_swizzle128B<
                            Bc, HEAD_DIM,
                            BOX_KV_M, BOX_KV_N
                        >(mma_k, mma_n, sizeof(__nv_bfloat16));
                        uint64_t mma_desc_smem_V = create_smem_desc(
                            smem_v + v_slot * Bc * HEAD_DIM + offset,
                            16,
                            8 * BOX_KV_N * sizeof(__nv_bfloat16),
                            1,
                            0
                        );

                        // wgmma
                        wgmma_fence();
                        wgmma_m64n64k16_rs(
                            regs_o[mma_m / MMA_M][mma_n / MMA_N],
                            p_frag,
                            mma_desc_smem_V,
                            true
                        );
                    }
                }
            }
            wgmma_commit();
            wgmma_wait<1>();


            // arrive k consumer
            barrier_arrive(barrier_k_free + k_slot);

            // kv 尾块处理
            if(kvtile_offset + Bc > kv_limit) {
                for(uint32_t mma_m = 0; mma_m < Br / MMA_M; mma_m++) {
                    for(uint32_t mma_n = 0; mma_n < Bc / MMA_N; mma_n++) {
                        for(uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; mma_n8++) {
                            uint32_t elem_col = (lane_idx % 4) * 2 + mma_n * MMA_N + mma_n8 * 8;
                            if (elem_col >= num_last_kvtile_row) {
                                regs_sp[mma_m][mma_n][mma_n8 * 4] = -INFINITY;
                                regs_sp[mma_m][mma_n][mma_n8 * 4 + 2] = -INFINITY;
                            }
                            if (elem_col + 1 >= num_last_kvtile_row) {
                                regs_sp[mma_m][mma_n][mma_n8 * 4 + 1] = -INFINITY;
                                regs_sp[mma_m][mma_n][mma_n8 * 4 + 3] = -INFINITY;
                            }
                        }
                    }
                }
            }

            // Only diagonal/boundary tiles need element-wise causal masking.
            if constexpr(CAUSAL) {
                const size_t first_row_k_end = kv_lens - q_lens + q_row_begin + 1;
                if (kvtile_offset + Bc > first_row_k_end) {
                    for (uint32_t mma_m = 0; mma_m < Br / MMA_M; mma_m++) {
                        uint32_t mma_row = mma_m * MMA_M + warp_idx * 16;
                        for (uint32_t mma_n = 0; mma_n < Bc / MMA_N; mma_n++) {
                            for (uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; mma_n8++) {
                                size_t elem_row = (lane_idx / 4) + mma_row + qtile_idx * Br + kv_lens - q_lens;
                                uint32_t elem_col = (lane_idx % 4) * 2 + mma_n * MMA_N + mma_n8 * 8 + kvtile_offset;
                                if(elem_row < elem_col) {
                                    regs_sp[mma_m][mma_n][mma_n8 * 4] = -INFINITY;
                                }
                                if(elem_row < 1 + elem_col) {
                                    regs_sp[mma_m][mma_n][1 + mma_n8 * 4] = -INFINITY;
                                }

                                if(8 + elem_row < elem_col) {
                                    regs_sp[mma_m][mma_n][2 + mma_n8 * 4] = -INFINITY;
                                }
                                if(8 + elem_row < 1 + elem_col) {
                                    regs_sp[mma_m][mma_n][3 + mma_n8 * 4] = -INFINITY;
                                }
                            }
                        }
                    }

                }
            }

            // new_m, factor = rowmax(old_m, s)
            for (uint32_t mma_m = 0; mma_m < Br / MMA_M; mma_m++) {
                float new_max[2] = {-INFINITY, -INFINITY};
                for (uint32_t mma_n = 0; mma_n < Bc / MMA_N; mma_n++) {
                    for (uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; mma_n8++) {
                        regs_sp[mma_m][mma_n][mma_n8 * 4] *= scale;
                        regs_sp[mma_m][mma_n][1 + mma_n8 * 4] *= scale;
                        regs_sp[mma_m][mma_n][2 + mma_n8 * 4] *= scale;
                        regs_sp[mma_m][mma_n][3 + mma_n8 * 4] *= scale;
                        float max0 = fmaxf(regs_sp[mma_m][mma_n][mma_n8 * 4], regs_sp[mma_m][mma_n][1 + mma_n8 * 4]);
                        float max1 = fmaxf(regs_sp[mma_m][mma_n][2 + mma_n8 * 4], regs_sp[mma_m][mma_n][3 + mma_n8 * 4]);
                        new_max[0] = fmaxf(new_max[0], max0);
                        new_max[1] = fmaxf(new_max[1], max1);
                    }
                }
                new_max[0] = fmaxf(new_max[0], __shfl_xor_sync(0xffffffff, new_max[0], 1));
                new_max[0] = fmaxf(new_max[0], __shfl_xor_sync(0xffffffff, new_max[0], 2));
                new_max[1] = fmaxf(new_max[1], __shfl_xor_sync(0xffffffff, new_max[1], 1));
                new_max[1] = fmaxf(new_max[1], __shfl_xor_sync(0xffffffff, new_max[1], 2));
                new_max[0] = fmaxf(new_max[0], regs_row_max[mma_m][0]);
                new_max[1] = fmaxf(new_max[1], regs_row_max[mma_m][1]);
                regs_row_factor[mma_m][0] = expf(regs_row_max[mma_m][0] - new_max[0]);
                regs_row_factor[mma_m][1] = expf(regs_row_max[mma_m][1] - new_max[1]);
                regs_row_max[mma_m][0] = new_max[0];
                regs_row_max[mma_m][1] = new_max[1];
            }

            // new_l, p = apply_max_and_rowsum(old_l, s, new_max, factor)
            for (uint32_t mma_m = 0; mma_m < Br / MMA_M; mma_m++) {
                float new_sum[2] = {0, 0};
                for (uint32_t mma_n = 0; mma_n < Bc / MMA_N; mma_n++) {
                    for (uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; mma_n8++) {
                        regs_sp[mma_m][mma_n][mma_n8 * 4] = expf(regs_sp[mma_m][mma_n][mma_n8 * 4] - regs_row_max[mma_m][0]);
                        regs_sp[mma_m][mma_n][1 + mma_n8 * 4] = expf(regs_sp[mma_m][mma_n][1 + mma_n8 * 4] - regs_row_max[mma_m][0]);
                        regs_sp[mma_m][mma_n][2 + mma_n8 * 4] = expf(regs_sp[mma_m][mma_n][2 + mma_n8 * 4] - regs_row_max[mma_m][1]);
                        regs_sp[mma_m][mma_n][3 + mma_n8 * 4] = expf(regs_sp[mma_m][mma_n][3 + mma_n8 * 4] - regs_row_max[mma_m][1]);
                        float sum0 = regs_sp[mma_m][mma_n][mma_n8 * 4] + regs_sp[mma_m][mma_n][1 + mma_n8 * 4];
                        float sum1 = regs_sp[mma_m][mma_n][2 + mma_n8 * 4] + regs_sp[mma_m][mma_n][3 + mma_n8 * 4];
                        new_sum[0] += sum0;
                        new_sum[1] += sum1;
                    }
                }
                new_sum[0] += __shfl_xor_sync(0xffffffff, new_sum[0], 1);
                new_sum[0] += __shfl_xor_sync(0xffffffff, new_sum[0], 2);
                new_sum[1] += __shfl_xor_sync(0xffffffff, new_sum[1], 1);
                new_sum[1] += __shfl_xor_sync(0xffffffff, new_sum[1], 2);
                regs_row_sum[mma_m][0] = regs_row_sum[mma_m][0] * regs_row_factor[mma_m][0] + new_sum[0];
                regs_row_sum[mma_m][1] = regs_row_sum[mma_m][1] * regs_row_factor[mma_m][1] + new_sum[1];
            }


            // wait pv
            wgmma_wait<0>();

            // arrive v consumer
            barrier_arrive(barrier_v_free + v_slot);


            // packed(p)
            for (uint32_t mma_m = 0; mma_m < Br / MMA_M; mma_m++) {
                for (uint32_t mma_n = 0; mma_n < Bc / MMA_N; mma_n++) {
                    for (uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; mma_n8++) {
                        regs_p[mma_m][mma_n][mma_n8 * 4 / 2] = pack_bf16(
                            regs_sp[mma_m][mma_n][mma_n8 * 4],
                            regs_sp[mma_m][mma_n][1 + mma_n8 * 4]
                        );
                        regs_p[mma_m][mma_n][1 + mma_n8 * 4 / 2] = pack_bf16(
                            regs_sp[mma_m][mma_n][2 + mma_n8 * 4],
                            regs_sp[mma_m][mma_n][3 + mma_n8 * 4]
                        );
                    }
                }
            }

            // old_o_scale = apply_factor(old_o, factor)
            for(uint32_t mma_m = 0; mma_m < Br / MMA_M; mma_m++) {
                for(uint32_t mma_n = 0; mma_n < HEAD_DIM / MMA_N; mma_n++) {
                    for(uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; mma_n8++) {
                        regs_o[mma_m][mma_n][mma_n8 * 4] *= regs_row_factor[mma_m][0];
                        regs_o[mma_m][mma_n][1 + mma_n8 * 4] *= regs_row_factor[mma_m][0];
                        regs_o[mma_m][mma_n][2 + mma_n8 * 4] *= regs_row_factor[mma_m][1];
                        regs_o[mma_m][mma_n][3 + mma_n8 * 4] *= regs_row_factor[mma_m][1];
                    }
                }
            }
        }

        // wait v productor
        const size_t v_slot = (num_kvtiles - 1) % V_STAGES;
        barrier_wait(barrier_v_ready + v_slot, phase_v_ready[v_slot]);
        phase_v_ready[v_slot] ^= 1;

        // new_o = old_o_scale + p@v
        for(uint32_t mma_m = 0; mma_m < Br; mma_m += MMA_M) {
            for(uint32_t mma_n = 0; mma_n < HEAD_DIM; mma_n += MMA_N) {
                for(uint32_t mma_k = 0; mma_k < Bc; mma_k += MMA_K) {
                    // mma_desc_regs_p
                    uint32_t p_frag[MMA_K / 4];
                    uint32_t dst = 0;

                    for (uint32_t col = mma_k; col < mma_k + MMA_K; col += 8) {
                        uint32_t fragment = col / MMA_N;
                        uint32_t reg = ((col % MMA_N) / 8) * 2;

                        p_frag[dst++] = regs_p[mma_m / MMA_M][fragment][reg];
                        p_frag[dst++] = regs_p[mma_m / MMA_M][fragment][reg + 1];
                    }

                    // mma_desc_smem_v
                    uint32_t offset = tma_tile_offset_swizzle128B<
                        Bc, HEAD_DIM,
                        BOX_KV_M, BOX_KV_N
                    >(mma_k, mma_n, sizeof(__nv_bfloat16));
                    uint64_t mma_desc_smem_V = create_smem_desc(
                        smem_v + v_slot * Bc * HEAD_DIM + offset,
                        16,
                        8 * BOX_KV_N * sizeof(__nv_bfloat16),
                        1,
                        0
                    );

                    // wgmma
                    wgmma_fence();
                    wgmma_m64n64k16_rs(
                        regs_o[mma_m / MMA_M][mma_n / MMA_N],
                        p_frag,
                        mma_desc_smem_V,
                        true
                    );
                }
            }
        }
        wgmma_commit();
        wgmma_wait<0>();

        // o scale
        for(uint32_t mma_m = 0; mma_m < Br / MMA_M; mma_m++) {
            uint32_t mma_row = mma_m * MMA_M + warp_idx * 16;
            for(uint32_t mma_n = 0; mma_n < HEAD_DIM / MMA_N; mma_n++) {
                for(uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; mma_n8++) {
                    regs_o[mma_m][mma_n][mma_n8 * 4] /= regs_row_sum[mma_m][0];
                    regs_o[mma_m][mma_n][1 + mma_n8 * 4] /= regs_row_sum[mma_m][0];
                    regs_o[mma_m][mma_n][2 + mma_n8 * 4] /= regs_row_sum[mma_m][1];
                    regs_o[mma_m][mma_n][3 + mma_n8 * 4] /= regs_row_sum[mma_m][1];
                }
            }
        }

        // tma store
        if(num_qtile_row == Br) {
            // Producer has drained every outstanding K slot before sharing Q storage.
            barrier_wait(barrier_q_ready, phase_q_ready);
            phase_q_ready ^= 1;

            // regs_o store shared
            for(uint32_t mma_m = 0; mma_m < Br / MMA_M; mma_m++) {
                uint32_t mma_row = mma_m * MMA_M + warp_idx * 16;
                for(uint32_t mma_n = 0; mma_n < HEAD_DIM / MMA_N; mma_n++) {
                    for(uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; mma_n8++) {
                        float o0 = regs_o[mma_m][mma_n][mma_n8 * 4];
                        float o1 = regs_o[mma_m][mma_n][1 + mma_n8 * 4];
                        uint32_t vec_o = pack_bf16(o0, o1);
                        uint32_t elem_row = (lane_idx / 4) + mma_row;
                        uint32_t elem_col = (lane_idx % 4) * 2 + mma_n * MMA_N + mma_n8 * 8;
                        uint32_t offset = tma_tile_offset_swizzle128B<
                            Br, HEAD_DIM,
                            BOX_O_M, BOX_O_N
                        >(elem_row, elem_col, sizeof(__nv_bfloat16));
                        uint32_t* vec_o_ptr = reinterpret_cast<uint32_t*>(smem_o + offset);
                        *vec_o_ptr = vec_o;

                        o0 = regs_o[mma_m][mma_n][2 + mma_n8 * 4];
                        o1 = regs_o[mma_m][mma_n][3 + mma_n8 * 4];
                        vec_o = pack_bf16(o0, o1);
                        elem_row = (lane_idx / 4) + 8 + mma_row;
                        elem_col = (lane_idx % 4) * 2 + mma_n * MMA_N + mma_n8 * 8;
                        offset = tma_tile_offset_swizzle128B<
                            Br, HEAD_DIM,
                            BOX_O_M, BOX_O_N
                        >(elem_row, elem_col, sizeof(__nv_bfloat16));
                        vec_o_ptr = reinterpret_cast<uint32_t*>(smem_o + offset);
                        *vec_o_ptr = vec_o;
                    }
                }
            }

            // arrive o consumer
            fence_proxy_async();
            barrier_arrive(barrier_q_free);
        }
        // regs store global
        else {
            size_t row_stride = num_heads * HEAD_DIM;
            size_t head_offset = head_idx * HEAD_DIM;
            for (uint32_t mma_m = 0; mma_m < Br / MMA_M; ++mma_m) {
                uint32_t mma_row = mma_m * MMA_M + warp_idx * 16;
                for (uint32_t mma_n = 0; mma_n < HEAD_DIM / MMA_N; ++mma_n) {
                    for (uint32_t mma_n8 = 0; mma_n8 < MMA_N / 8; ++mma_n8) {
                        uint32_t elem_row = (lane_idx / 4) + mma_row;
                        uint32_t elem_col = (lane_idx % 4) * 2 + mma_n * MMA_N + mma_n8 * 8;
                        float2 vec_o;
                        vec_o.x = regs_o[mma_m][mma_n][mma_n8 * 4];
                        vec_o.y = regs_o[mma_m][mma_n][1 + mma_n8 * 4];
                        if (elem_row < num_qtile_row) {
                            size_t offset = (qtile_start + elem_row) * row_stride + head_offset + elem_col;
                            *reinterpret_cast<uint32_t*>(O + offset) = pack_bf16(vec_o.x, vec_o.y);
                        }
                        vec_o.x = regs_o[mma_m][mma_n][2 + mma_n8 * 4];
                        vec_o.y = regs_o[mma_m][mma_n][3 + mma_n8 * 4];
                        if (elem_row + 8 < num_qtile_row) {
                            size_t offset = (qtile_start + elem_row + 8) * row_stride + head_offset + elem_col;
                            *reinterpret_cast<uint32_t*>(O + offset) = pack_bf16(vec_o.x, vec_o.y);
                        }
                    }
                }
            }
        }
    }
    else if(warp_idx >=4 && warp_idx < 8) {
        if(warp_idx == 4 && lane_idx == 0) {
            uint32_t phase_q_free = 0;
            uint32_t phase_k_free[K_STAGES] = {};
            uint32_t phase_v_free[V_STAGES] = {};

            // tma load q
            barrier_expect_tx(
                barrier_q_ready,
                Q_TILE_BYTES
            );
            for(uint32_t box_row = 0; box_row < Br; box_row += BOX_Q_M) {
                for(uint32_t box_col = 0; box_col < HEAD_DIM; box_col += BOX_Q_N) {
                    uint32_t offset_smem = box_row * HEAD_DIM + box_col * BOX_Q_M;
                    tma_load_2d(
                        smem_q + offset_smem,
                        &tma_desc_Q,
                        box_col + head_idx * HEAD_DIM,
                        box_row + qtile_start,
                        barrier_q_ready
                    );
                }
            }

            // Q is resident for every QK; acknowledge before entering the ring.
            barrier_wait(barrier_q_free, phase_q_free);
            phase_q_free ^= 1;

            for(size_t kvtile_offset = 0; kvtile_offset < kv_limit; kvtile_offset += Bc) {
                const size_t tile = kvtile_offset / Bc;
                const size_t k_slot = tile % K_STAGES;
                const size_t v_slot = tile % V_STAGES;

                // First use is free; subsequent uses wait for that slot's QK.
                if (tile >= K_STAGES) {
                    barrier_wait(barrier_k_free + k_slot, phase_k_free[k_slot]);
                    phase_k_free[k_slot] ^= 1;
                }
                // tma load k
                barrier_expect_tx(
                    barrier_k_ready + k_slot,
                    KV_TILE_BYTES
                );
                for(uint32_t box_row = 0; box_row < Bc; box_row += BOX_KV_M) {
                    for(uint32_t box_col = 0; box_col < HEAD_DIM; box_col += BOX_KV_N) {
                        uint32_t offset_smem = box_row * HEAD_DIM + box_col * BOX_KV_M;
                        tma_load_2d(
                            smem_k + k_slot * Bc * HEAD_DIM + offset_smem,
                            &tma_desc_K,
                            box_col + kv_head_idx * HEAD_DIM,
                            box_row + kvtile_offset + kv_start,
                            barrier_k_ready + k_slot
                        );
                    }
                }


                // V remains live until the previous PV has completed.
                if (tile >= V_STAGES) {
                    barrier_wait(barrier_v_free + v_slot, phase_v_free[v_slot]);
                    phase_v_free[v_slot] ^= 1;
                }
                // tma load v
                barrier_expect_tx(
                    barrier_v_ready + v_slot,
                    KV_TILE_BYTES
                );
                for(uint32_t box_row = 0; box_row < Bc; box_row += BOX_KV_M) {
                    for(uint32_t box_col = 0; box_col < HEAD_DIM; box_col += BOX_KV_N) {
                        uint32_t offset_smem = box_row * HEAD_DIM + box_col * BOX_KV_M;
                        tma_load_2d(
                            smem_v + v_slot * Bc * HEAD_DIM + offset_smem,
                            &tma_desc_V,
                            box_col + kv_head_idx * HEAD_DIM,
                            box_row + kvtile_offset + kv_start,
                            barrier_v_ready + v_slot
                        );
                    }
                }

            }

            // Drain exactly the last outstanding generation of each K slot.
            // Earlier generations were waited on before reusing their slots.
            const size_t first_pending_k = num_kvtiles > K_STAGES ? num_kvtiles - K_STAGES : 0;
            for (size_t tile = first_pending_k; tile < num_kvtiles; ++tile) {
                const size_t slot = tile % K_STAGES;
                barrier_wait(barrier_k_free + slot, phase_k_free[slot]);
                phase_k_free[slot] ^= 1;
            }

            // tma store
            if(num_qtile_row == Br) {
                // arrive productor
                barrier_arrive(barrier_q_ready);

                // wait o consumer
                barrier_wait(barrier_q_free, phase_q_free);
                phase_q_free ^= 1;

                // tma store o
                for (uint32_t box_row = 0; box_row < Br; box_row += BOX_O_M) {
                    for (uint32_t box_col = 0; box_col < HEAD_DIM; box_col += BOX_O_N) {
                        uint32_t offset = box_row * HEAD_DIM + box_col * BOX_O_M;
                        tma_store_2d(
                            &tma_desc_O,
                            box_col + head_idx * HEAD_DIM,
                            box_row + qtile_start,
                            smem_o + offset
                        );
                    }
                }
                tma_store_commit();
                tma_store_wait();
            }
        }
    }
}

// Q: [total_tokens_q, num_head * head_dim]
// KV: [total_tokens_kv, num_kv_head * head_dim]
// O: [total_tokens_q, num_head * head_dim]
template<
    size_t Br = 64,
    size_t Bc = 64,
    size_t HEAD_DIM = 128,
    bool CAUSAL = false,

    size_t BOX_Q_M = 64,
    size_t BOX_Q_N = 64,

    size_t BOX_KV_M = 64,
    size_t BOX_KV_N = 64,

    size_t BOX_O_M = 64,
    size_t BOX_O_N = 64,

    size_t MMA_M = 64,
    size_t MMA_N = 64,
    size_t MMA_K = 16,

    // Compile-time targets per thread; ignored when RegRealloc is false.
    bool RegRealloc = false,
    uint32_t ProducerRegs = 40,
    uint32_t ConsumerRegs = 232
>
void fa3_kv_bf16_single_launcher(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,

    const size_t* cu_seqlens_q,
    const size_t* cu_seqlens_kv,

    size_t batch_size,
    size_t num_heads,
    size_t num_kv_heads,
    size_t max_q_len,
    size_t total_q_tokens,
    size_t total_kv_tokens,

    cudaStream_t stream = nullptr
) {
    // tma
    static_assert(Br % BOX_Q_M == 0);
    static_assert(HEAD_DIM % BOX_Q_N == 0);
    static_assert(Bc % BOX_KV_M == 0);
    static_assert(HEAD_DIM % BOX_KV_N == 0);
    static_assert(Br % BOX_O_M == 0);
    static_assert(HEAD_DIM % BOX_O_N == 0);

    // tma swizzle128B
    static_assert(BOX_Q_N * sizeof(__nv_bfloat16) == 128);
    static_assert(BOX_KV_N * sizeof(__nv_bfloat16) == 128);
    static_assert(BOX_O_N * sizeof(__nv_bfloat16) == 128);

    // mma
    static_assert(MMA_M == 64 && MMA_N == 64 && MMA_K == 16);
    static_assert(MMA_N % 8 == 0);

    // q@k & p@v
    static_assert(BOX_Q_M % MMA_M == 0);
    static_assert(BOX_Q_N % MMA_K == 0);
    static_assert(BOX_KV_M % MMA_N == 0);
    static_assert(BOX_KV_N % MMA_K == 0);

    // head
    assert(num_heads > 0 && num_kv_heads > 0);
    assert(num_heads % num_kv_heads == 0);


    // smem
    constexpr size_t Q_BYTES = Br * HEAD_DIM * sizeof(__nv_bfloat16);
    constexpr size_t K_BYTES = Bc * HEAD_DIM * sizeof(__nv_bfloat16);
    constexpr size_t V_BYTES = Bc * HEAD_DIM * sizeof(__nv_bfloat16);
    constexpr size_t O_BYTES = Br * HEAD_DIM * sizeof(__nv_bfloat16);
    constexpr size_t K_STAGES = 2;
    constexpr size_t V_STAGES = 2;
    constexpr size_t MBARRIER_BYTES = (2 + 2 * K_STAGES + 2 * V_STAGES) * sizeof(uint64_t);
    constexpr size_t DATA_BYTES = Q_BYTES + K_STAGES * K_BYTES + V_STAGES * V_BYTES;
    constexpr size_t SMEM_BYTES = DATA_BYTES + MBARRIER_BYTES;
    static_assert(O_BYTES <= DATA_BYTES);

    // tma
    alignas(64) CUtensorMap tma_desc_Q, tma_desc_K, tma_desc_V, tma_desc_O;
    size_t Q_M = total_q_tokens;
    size_t Q_N = num_heads * HEAD_DIM;
    size_t Q_row_stride = Q_N * sizeof(__nv_bfloat16);
    size_t KV_M = total_kv_tokens;
    size_t KV_N = num_kv_heads * HEAD_DIM;
    size_t KV_row_stride = KV_N * sizeof(__nv_bfloat16);
    size_t O_M = total_q_tokens;
    size_t O_N = num_heads * HEAD_DIM;
    size_t O_row_stride = O_N * sizeof(__nv_bfloat16);
    if(
        !create_tma_descriptor_swizzle128B<BOX_Q_M, BOX_Q_N>(
            Q, Q_M, Q_N, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, &tma_desc_Q, Q_row_stride)
        || !create_tma_descriptor_swizzle128B<BOX_KV_M, BOX_KV_N>(
            K, KV_M, KV_N, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, &tma_desc_K, KV_row_stride)
        || !create_tma_descriptor_swizzle128B<BOX_KV_M, BOX_KV_N>(
            V, KV_M, KV_N, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, &tma_desc_V, KV_row_stride)
        || !create_tma_descriptor_swizzle128B<BOX_O_M, BOX_O_N>(
            O, O_M, O_N, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, &tma_desc_O, O_row_stride)
    ) return;

    // shared opt
    auto kernel = fa3_kv_bf16_single<
        Br, Bc, HEAD_DIM, CAUSAL,
        BOX_Q_M, BOX_Q_N,
        BOX_KV_M, BOX_KV_N,
        BOX_O_M, BOX_O_N,
        MMA_M, MMA_N, MMA_K,
        RegRealloc, ProducerRegs, ConsumerRegs
    >;
    cudaError_t err = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SMEM_BYTES)
    );
    if (err != cudaSuccess) return;

    // kernel call
    dim3 grid((max_q_len + Br - 1) / Br, num_heads, batch_size);
    dim3 block((4 + 4) * WARP_SIZE);
    kernel<<<grid, block, SMEM_BYTES, stream>>>(
        tma_desc_Q,
        tma_desc_K,
        tma_desc_V,
        tma_desc_O,
        cu_seqlens_q,
        cu_seqlens_kv,
        num_heads,
        num_kv_heads,
        1 / sqrtf(HEAD_DIM),
        O
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) return;
}

int main(void) {
    return 0;
}
