#include <stdint.h>
#include <assert.h>
#include <cuda_bf16.h>
#include <cfloat>

#define WARP_SIZE 32

// 通用 swizzle
__device__ __forceinline__ int shared_offset_Bank8x128B(
    int element_idx,
    int stride,
    int element_per_row,
    int element_size)
{
    int element_per_128B = 128 / element_size;
    int row = element_idx / stride;
    int col = element_idx % stride;
    int row_8x128B = row % 8;
    int col_8x128B = col % element_per_128B;

    int group_base = element_idx - element_idx % element_per_128B;
    return group_base + (col_8x128B + row_8x128B * element_per_row) % element_per_128B;
}

template <int NUM_ROWS, int HEAD_DIM>
__device__ __forceinline__ void load_tile_from_global_to_shared(
    const __nv_bfloat16 *src,
    int src_stride,
    __nv_bfloat16 *smem,
    int valid_rows)
{
    const uint4 *src_vec = reinterpret_cast<const uint4 *>(src);
    uint4 *smem_vec = reinterpret_cast<uint4 *>(smem);

    constexpr int VEC_PER_ROW = HEAD_DIM / 8;
    int nums_element_vec = valid_rows * VEC_PER_ROW;

    // load valid rows
    for (int element_idx = threadIdx.x; element_idx < nums_element_vec; element_idx += blockDim.x)
    {
        int row = element_idx / VEC_PER_ROW;
        int col = element_idx % VEC_PER_ROW;
        smem_vec[shared_offset_Bank8x128B(element_idx, VEC_PER_ROW, 1, sizeof(uint4))] = src_vec[row * (src_stride / 8) + col];
    }

    // padding
    for (int element_idx = threadIdx.x + nums_element_vec; element_idx < NUM_ROWS * VEC_PER_ROW; element_idx += blockDim.x)
    {
        smem_vec[shared_offset_Bank8x128B(element_idx, VEC_PER_ROW, 1, sizeof(uint4))] = make_uint4(0, 0, 0, 0);
    }
}

template <int Bc, int HEAD_DIM>
__device__ __forceinline__ void load_kv_from_global_to_shared(
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    int kv_stride,
    __nv_bfloat16 *smem_k,
    __nv_bfloat16 *smem_v,
    int nums_row_kv_tile)
{
    load_tile_from_global_to_shared<Bc, HEAD_DIM>(k, kv_stride, smem_k, nums_row_kv_tile);
    load_tile_from_global_to_shared<Bc, HEAD_DIM>(v, kv_stride, smem_v, nums_row_kv_tile);
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t &r0,
    uint32_t &r1,
    uint32_t &r2,
    uint32_t &r3,
    const void *smem_ptr)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr));
}

__device__ __forceinline__ void ldmatrix_x2(
    uint32_t &r0,
    uint32_t &r1,
    const void *smem_ptr)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t &r0,
    uint32_t &r1,
    const void *smem_ptr)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_addr));
}

__device__ __forceinline__ void mma_m16n8k16(
    float &d0,
    float &d1,
    float &d2,
    float &d3,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
}

__device__ __forceinline__
uint32_t pack_bf16x2(float x, float y)
{
    __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<uint32_t *>(&v);
}

__device__ __forceinline__
uint2 pack_float2(float a, float b)
{
    return make_uint2(
        __float_as_uint(a),
        __float_as_uint(b));
}

template <bool Is_last, int Br, int Bc, int HEAD_DIM, int NUM_WARPS>
__device__ void process_kv_tile(
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    float *o,
    const int kv_stride,
    const int o_stride,
    const int kv_tile_idx,
    __nv_bfloat16 *smem_q,
    __nv_bfloat16 *smem_k,
    __nv_bfloat16 *smem_v,
    float *smem_o,
    float *smem_max,
    int nums_row_q_tile,
    int nums_row_kv_tile,
    float (&regs_m)[Br / 16][2],
    float (&regs_l)[Br / 16][2])
{
    load_kv_from_global_to_shared<Bc, HEAD_DIM>(
        k + kv_tile_idx * Bc * kv_stride,
        v + kv_tile_idx * Bc * kv_stride,
        kv_stride,
        smem_k,
        smem_v,
        nums_row_kv_tile);
    __syncthreads();

    // Qtile @ Ktile^T
    constexpr int NUM_QTILE16 = Br / 16;
    constexpr int NUM_KTILE8 = (Bc / 8) / NUM_WARPS;
    uint32_t regs_q[NUM_QTILE16][16 * 16 / WARP_SIZE / 2];
    uint32_t regs_k[NUM_KTILE8][16 * 8 / WARP_SIZE / 2];
    float regs_s[NUM_QTILE16 * NUM_KTILE8][16 * 8 / WARP_SIZE] = {};

    int warp_idx = threadIdx.x / WARP_SIZE;
    int lane_idx = threadIdx.x % WARP_SIZE;

    uint4 *smem_q_vec = reinterpret_cast<uint4 *>(smem_q);
    uint4 *smem_k_vec = reinterpret_cast<uint4 *>(smem_k);
    constexpr int VEC_PER_ROW = HEAD_DIM / 8;

    for (int tile16_idx = 0; tile16_idx < HEAD_DIM / 16; tile16_idx++)
    {
        // load qtile16
        int element_q_vec_offset = tile16_idx * 2;
        #pragma unroll
        for (int qtile16_idx = 0; qtile16_idx < NUM_QTILE16; qtile16_idx++)
        {
            int col = lane_idx / 16;
            int row = lane_idx % 16;
            int element_q_vec_idx = shared_offset_Bank8x128B(
                element_q_vec_offset + qtile16_idx * 16 * VEC_PER_ROW + row * VEC_PER_ROW + col,
                VEC_PER_ROW, 1, sizeof(uint4));
            __nv_bfloat16 *ldmatrix_16B_ptr = reinterpret_cast<__nv_bfloat16 *>(smem_q_vec + element_q_vec_idx);
            ldmatrix_x4(
                regs_q[qtile16_idx][0], regs_q[qtile16_idx][1],
                regs_q[qtile16_idx][2], regs_q[qtile16_idx][3],
                ldmatrix_16B_ptr);
        }

        // load ktile8
        int element_k_vec_offset = warp_idx * (Bc / NUM_WARPS) * VEC_PER_ROW + tile16_idx * 2;
        #pragma unroll
        for (int ktile8_idx = 0; ktile8_idx < NUM_KTILE8; ktile8_idx++)
        {
            int col = (lane_idx % 16) / 8;
            int row = (lane_idx % 16) % 8;
            int element_k_vec_idx = shared_offset_Bank8x128B(
                element_k_vec_offset + ktile8_idx * 8 * VEC_PER_ROW + row * VEC_PER_ROW + col,
                VEC_PER_ROW, 1, sizeof(uint4));
            __nv_bfloat16 *ldmatrix_16B_ptr = reinterpret_cast<__nv_bfloat16 *>(smem_k_vec + element_k_vec_idx);
            ldmatrix_x2(
                regs_k[ktile8_idx][0], regs_k[ktile8_idx][1],
                ldmatrix_16B_ptr);
        }

        // mma
        #pragma unroll
        for (int qtile16_idx = 0; qtile16_idx < NUM_QTILE16; qtile16_idx++)
        {
            #pragma unroll
            for (int ktile8_idx = 0; ktile8_idx < NUM_KTILE8; ktile8_idx++)
            {
                int s_idx = qtile16_idx * NUM_KTILE8 + ktile8_idx;
                mma_m16n8k16(
                    regs_s[s_idx][0],
                    regs_s[s_idx][1],
                    regs_s[s_idx][2],
                    regs_s[s_idx][3],
                    regs_q[qtile16_idx][0],
                    regs_q[qtile16_idx][1],
                    regs_q[qtile16_idx][2],
                    regs_q[qtile16_idx][3],
                    regs_k[ktile8_idx][0],
                    regs_k[ktile8_idx][1]);
            }
        }
    }

    // Match the standard attention score: softmax((QK^T) / sqrt(HEAD_DIM)).
    float score_scale = rsqrtf((float)HEAD_DIM);
    #pragma unroll
    for (int qtile16_idx = 0; qtile16_idx < NUM_QTILE16; qtile16_idx++)
    {
        #pragma unroll
        for (int ktile8_idx = 0; ktile8_idx < NUM_KTILE8; ktile8_idx++)
        {
            int s_idx = qtile16_idx * NUM_KTILE8 + ktile8_idx;
            regs_s[s_idx][0] *= score_scale;
            regs_s[s_idx][1] *= score_scale;
            regs_s[s_idx][2] *= score_scale;
            regs_s[s_idx][3] *= score_scale;
        }
    }

    // TODO: 处理 Qtile != (Br, HEAD_DIM) or Ktile != (Bc, HEAD_DIM)
    // 处理前者：Qtile多余部分置为0，写回 global时注意越界问题
    // 处理后者：Stile多余部分置为-∞

    // TODO: 处理 mask

    // max
    #pragma unroll
    for (int qtile16_idx = 0; qtile16_idx < NUM_QTILE16; qtile16_idx++)
    {
        float max0 = -FLT_MAX;
        float max1 = -FLT_MAX;
        #pragma unroll
        for (int ktile8_idx = 0; ktile8_idx < NUM_KTILE8; ktile8_idx++)
        {
            int s_idx = qtile16_idx * NUM_KTILE8 + ktile8_idx;
            max0 = fmaxf(max0, regs_s[s_idx][0]);
            max0 = fmaxf(max0, regs_s[s_idx][1]);
            max1 = fmaxf(max1, regs_s[s_idx][2]);
            max1 = fmaxf(max1, regs_s[s_idx][3]);
        }
        max0 = fmaxf(max0, __shfl_xor_sync(0xffffffff, max0, 1));
        max0 = fmaxf(max0, __shfl_xor_sync(0xffffffff, max0, 2));
        max1 = fmaxf(max1, __shfl_xor_sync(0xffffffff, max1, 1));
        max1 = fmaxf(max1, __shfl_xor_sync(0xffffffff, max1, 2));
        // lane 0~3   持有 row 0、8 的 max
        // lane 4~7   持有 row 1、9 的 max
        // ...
        // lane 28~31 持有 row 7、15 的 max
        if (lane_idx % 4 == 0)
        {
            smem_max[warp_idx * Br + qtile16_idx * 16 + lane_idx / 4] = max0;
            smem_max[warp_idx * Br + qtile16_idx * 16 + 8 + lane_idx / 4] = max1;
        }
    }
    __syncthreads();

    // reduce max & update max
    float regs_factor[NUM_QTILE16][2];
    #pragma unroll
    for (int qtile16_idx = 0; qtile16_idx < NUM_QTILE16; qtile16_idx++)
    {
        float max0 = -FLT_MAX;
        float max1 = -FLT_MAX;
        #pragma unroll
        for (int warp_i = 0; warp_i < NUM_WARPS; warp_i++)
        {
            max0 = fmaxf(max0, smem_max[warp_i * Br + qtile16_idx * 16 + lane_idx / 4]);
            max1 = fmaxf(max1, smem_max[warp_i * Br + qtile16_idx * 16 + 8 + lane_idx / 4]);
        }
        max0 = fmaxf(regs_m[qtile16_idx][0], max0);
        regs_factor[qtile16_idx][0] = expf(regs_m[qtile16_idx][0] - max0);
        regs_l[qtile16_idx][0] *= regs_factor[qtile16_idx][0];
        regs_m[qtile16_idx][0] = max0;

        max1 = fmaxf(regs_m[qtile16_idx][1], max1);
        regs_factor[qtile16_idx][1] = expf(regs_m[qtile16_idx][1] - max1);
        regs_l[qtile16_idx][1] *= regs_factor[qtile16_idx][1];
        regs_m[qtile16_idx][1] = max1;
    }
    __syncthreads();

    // exp(x - max) & sum
    #pragma unroll
    for (int qtile16_idx = 0; qtile16_idx < NUM_QTILE16; qtile16_idx++)
    {
        float sum0 = 0;
        float sum1 = 0;
        #pragma unroll
        for (int ktile8_idx = 0; ktile8_idx < NUM_KTILE8; ktile8_idx++)
        {
            int s_idx = qtile16_idx * NUM_KTILE8 + ktile8_idx;
            regs_s[s_idx][0] = expf(regs_s[s_idx][0] - regs_m[qtile16_idx][0]);
            regs_s[s_idx][1] = expf(regs_s[s_idx][1] - regs_m[qtile16_idx][0]);
            sum0 += regs_s[s_idx][0] + regs_s[s_idx][1];

            regs_s[s_idx][2] = expf(regs_s[s_idx][2] - regs_m[qtile16_idx][1]);
            regs_s[s_idx][3] = expf(regs_s[s_idx][3] - regs_m[qtile16_idx][1]);
            sum1 += regs_s[s_idx][2] + regs_s[s_idx][3];
        }
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
        if (lane_idx % 4 == 0)
        {
            smem_max[warp_idx * Br + qtile16_idx * 16 + lane_idx / 4] = sum0;
            smem_max[warp_idx * Br + qtile16_idx * 16 + 8 + lane_idx / 4] = sum1;
        }
    }
    __syncthreads();

    // reduce sum & update sum
    #pragma unroll
    for (int qtile16_idx = 0; qtile16_idx < NUM_QTILE16; qtile16_idx++)
    {
        float sum0 = 0;
        float sum1 = 0;
        #pragma unroll
        for (int warp_i = 0; warp_i < NUM_WARPS; warp_i++)
        {
            sum0 += smem_max[warp_i * Br + qtile16_idx * 16 + lane_idx / 4];
            sum1 += smem_max[warp_i * Br + qtile16_idx * 16 + 8 + lane_idx / 4];
        }
        regs_l[qtile16_idx][0] += sum0;
        regs_l[qtile16_idx][1] += sum1;
    }

    // Ptile @ Vtile
    constexpr int NUM_DTILE8 = HEAD_DIM / 8;
    uint4 *smem_v_vec = reinterpret_cast<uint4 *>(smem_v);
    constexpr int NUM_DTILE8_PER_WARP = NUM_DTILE8 / NUM_WARPS;
    constexpr int ELEMENTS_PER_MMA = 16 * 8 / WARP_SIZE;
    uint2 *smem_o_vec = reinterpret_cast<uint2 *>(smem_o);
    constexpr int element_per_head = HEAD_DIM * sizeof(float) / sizeof(uint2);

    // Each warp owns a disjoint V column group in one round. The rounds rotate
    // the groups so every warp contributes its K-slice partial sum to O.
    #pragma unroll
    for (int vtile_round = 0; vtile_round < NUM_WARPS; vtile_round++)
    {
        int dtile8_offset = ((vtile_round + warp_idx) % NUM_WARPS) * NUM_DTILE8_PER_WARP;
        float regs_o[NUM_QTILE16][NUM_DTILE8_PER_WARP][ELEMENTS_PER_MMA] = {};

        // O-dtile = Ptile @ Vtile-dtile
        #pragma unroll
        for (int tile16_idx = 0; tile16_idx < (Bc / NUM_WARPS) / 16; tile16_idx++)
        {
            uint32_t regs_v[NUM_DTILE8_PER_WARP][2];
            // ptile 已经准备好了无需 load

            // load vtile8
            int element_v_vec_offset = warp_idx * (Bc / NUM_WARPS) * VEC_PER_ROW + tile16_idx * 16 * VEC_PER_ROW;
            #pragma unroll
            for (int local_dtile8_idx = 0; local_dtile8_idx < NUM_DTILE8_PER_WARP; local_dtile8_idx++)
            {
                int dtile8_idx = local_dtile8_idx + dtile8_offset;
                int col = (lane_idx % 16) / 16;
                int row = (lane_idx % 16) % 16;
                int element_v_vec_idx = shared_offset_Bank8x128B(
                    element_v_vec_offset + dtile8_idx + row * VEC_PER_ROW + col,
                    VEC_PER_ROW, 1, sizeof(uint4));
                __nv_bfloat16 *ldmatrix_16B_ptr = reinterpret_cast<__nv_bfloat16 *>(smem_v_vec + element_v_vec_idx);
                ldmatrix_x2_trans(
                    regs_v[local_dtile8_idx][0], regs_v[local_dtile8_idx][1],
                    ldmatrix_16B_ptr);
            }

            // mma
            #pragma unroll
            for (int qtile16_idx = 0; qtile16_idx < NUM_QTILE16; qtile16_idx++)
            {
                int p_idx = qtile16_idx * NUM_KTILE8 + 2 * tile16_idx;
                #pragma unroll
                for (int local_dtile8_idx = 0; local_dtile8_idx < NUM_DTILE8_PER_WARP; local_dtile8_idx++)
                {
                    mma_m16n8k16(
                        regs_o[qtile16_idx][local_dtile8_idx][0],
                        regs_o[qtile16_idx][local_dtile8_idx][1],
                        regs_o[qtile16_idx][local_dtile8_idx][2],
                        regs_o[qtile16_idx][local_dtile8_idx][3],
                        pack_bf16x2(regs_s[p_idx][0], regs_s[p_idx][1]),
                        pack_bf16x2(regs_s[p_idx][2], regs_s[p_idx][3]),
                        pack_bf16x2(regs_s[p_idx + 1][0], regs_s[p_idx + 1][1]),
                        pack_bf16x2(regs_s[p_idx + 1][2], regs_s[p_idx + 1][3]),
                        regs_v[local_dtile8_idx][0],
                        regs_v[local_dtile8_idx][1]);
                }
            }
        }

        // O-dtile store smem_o, smem_o[Br * HEAD_DIM]
        #pragma unroll
        for (int qtile16_idx = 0; qtile16_idx < NUM_QTILE16; qtile16_idx++)
        {
            #pragma unroll
            for (int local_dtile8_idx = 0; local_dtile8_idx < NUM_DTILE8_PER_WARP; local_dtile8_idx++)
            {
                int dtile8_idx = local_dtile8_idx + dtile8_offset;
                int t_row_idx = qtile16_idx * 16 + lane_idx / 4;
                int t_col_idx = dtile8_idx * 8 * sizeof(float) / sizeof(uint2) + lane_idx % 4;
                int element0_idx = t_row_idx * element_per_head + t_col_idx;
                uint2 r0 = smem_o_vec[shared_offset_Bank8x128B(element0_idx, element_per_head, 4, sizeof(uint2))];
                int element1_idx = (t_row_idx + 8) * element_per_head + t_col_idx;
                uint2 r1 = smem_o_vec[shared_offset_Bank8x128B(element1_idx, element_per_head, 4, sizeof(uint2))];
                if (vtile_round == 0)
                {
                    smem_o_vec[shared_offset_Bank8x128B(element0_idx, element_per_head, 4, sizeof(uint2))] = pack_float2(__uint_as_float(r0.x) * regs_factor[qtile16_idx][0] + regs_o[qtile16_idx][local_dtile8_idx][0], __uint_as_float(r0.y) * regs_factor[qtile16_idx][0] + regs_o[qtile16_idx][local_dtile8_idx][1]);
                    smem_o_vec[shared_offset_Bank8x128B(element1_idx, element_per_head, 4, sizeof(uint2))] = pack_float2(__uint_as_float(r1.x) * regs_factor[qtile16_idx][1] + regs_o[qtile16_idx][local_dtile8_idx][2], __uint_as_float(r1.y) * regs_factor[qtile16_idx][1] + regs_o[qtile16_idx][local_dtile8_idx][3]);
                }
                else
                {
                    smem_o_vec[shared_offset_Bank8x128B(element0_idx, element_per_head, 4, sizeof(uint2))] = pack_float2(__uint_as_float(r0.x) + regs_o[qtile16_idx][local_dtile8_idx][0], __uint_as_float(r0.y) + regs_o[qtile16_idx][local_dtile8_idx][1]);
                    smem_o_vec[shared_offset_Bank8x128B(element1_idx, element_per_head, 4, sizeof(uint2))] = pack_float2(__uint_as_float(r1.x) + regs_o[qtile16_idx][local_dtile8_idx][2], __uint_as_float(r1.y) + regs_o[qtile16_idx][local_dtile8_idx][3]);
                }
            }
        }
        __syncthreads();
    }

    // Is_last: normalize smem_o and write to global
    if (Is_last)
    {
        // regs_l -> shared
        float *smem_l = smem_max;
        if (warp_idx == 0)
        {
            #pragma unroll
            for (int qtile16_idx = 0; qtile16_idx < NUM_QTILE16; qtile16_idx++)
            {
                if (lane_idx % 4 == 0)
                {
                    int row0 = qtile16_idx * 16 + lane_idx / 4;
                    int row1 = row0 + 8;

                    smem_l[row0] = regs_l[qtile16_idx][0];
                    smem_l[row1] = regs_l[qtile16_idx][1];
                }
            }
        }
        __syncthreads();

        // Split Br rows among warps.
        constexpr int ROWS_PER_WARP = (Br + NUM_WARPS - 1) / NUM_WARPS;
        int row_begin = warp_idx * ROWS_PER_WARP;
        int row_end = min(row_begin + ROWS_PER_WARP, nums_row_q_tile);
        constexpr int ELEMENT_PER_HEAD = HEAD_DIM * sizeof(float) / sizeof(uint2);

        for (int row = row_begin; row < row_end; row++)
        {
            float inv_l = 1.0f / smem_l[row];
            #pragma unroll
            for (int col = lane_idx; col < ELEMENT_PER_HEAD; col += WARP_SIZE)
            {
                int element_idx = row * ELEMENT_PER_HEAD + col;
                int smem_idx = shared_offset_Bank8x128B(element_idx, ELEMENT_PER_HEAD, 4, sizeof(uint2));
                uint2 value = smem_o_vec[smem_idx];

                float o0 = __uint_as_float(value.x) * inv_l;
                float o1 = __uint_as_float(value.y) * inv_l;
                float *o_row = o + row * o_stride;
                reinterpret_cast<uint2 *>(o_row)[col] = pack_float2(o0, o1);
            }
        }
    }
}

// Packed varlen layout: [total_q_tokens, num_head, HEAD_DIM]
// K/V: [total_kv_tokens, num_kv_head, HEAD_DIM]
// GRID: (seq_idx, head_idx, br_idx)
template <int HEAD_DIM, int Br, int Bc>
__global__ void fa1(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    const int *cu_q_len,
    const int *cu_kv_len,
    int num_head,
    int num_kv_head,
    float *o)
{
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128);
    static_assert(Br == 16 || Br == 32 || Br == 64);
    static_assert(Bc == 64);
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int br_idx = blockIdx.z;
    constexpr int NUM_WARPS = 4;

    int q_start = cu_q_len[seq_idx];
    int q_end = cu_q_len[seq_idx + 1];
    int q_len = q_end - q_start;
    int q_stride = num_head * HEAD_DIM;
    int q_offset = q_start + br_idx * Br;
    if (q_offset >= q_end)
        return;
    int nums_row_q_tile = min(Br, q_end - q_offset);
    assert(q_len % Br ==0);

    int kv_start = cu_kv_len[seq_idx];
    int kv_end = cu_kv_len[seq_idx + 1];
    int kv_len = kv_end - kv_start;
    assert(kv_len > 0);
    int kv_stride = num_kv_head * HEAD_DIM;
    int kv_head_idx = head_idx / (num_head / num_kv_head);
    int nums_kv_tile = (kv_len + Bc - 1) / Bc;
    assert(kv_len % Bc ==0);

    const int o_stride = num_head * HEAD_DIM;

    const __nv_bfloat16 *q_offset_ptr =
        q + q_offset * q_stride + head_idx * HEAD_DIM;
    const __nv_bfloat16 *k_offset_ptr =
        k + kv_start * kv_stride + kv_head_idx * HEAD_DIM;
    const __nv_bfloat16 *v_offset_ptr =
        v + kv_start * kv_stride + kv_head_idx * HEAD_DIM;
    float *o_offset_ptr =
        o + q_offset * o_stride + head_idx * HEAD_DIM;

    // shared
    extern __shared__ unsigned char smem[];
    __nv_bfloat16 *smem_q = reinterpret_cast<__nv_bfloat16 *>(smem);
    __nv_bfloat16 *smem_k = smem_q + Br * HEAD_DIM;
    __nv_bfloat16 *smem_v = smem_k + Bc * HEAD_DIM;
    float *smem_o = reinterpret_cast<float *>(smem_v + Bc * HEAD_DIM);
    float *smem_max = smem_o + Br * HEAD_DIM;

    load_tile_from_global_to_shared<Br, HEAD_DIM>(q_offset_ptr, q_stride, smem_q, nums_row_q_tile);

    for (int idx = threadIdx.x; idx < Br * HEAD_DIM; idx += blockDim.x) {
        smem_o[idx] = 0.0f;
    }

    // softmax stage
    constexpr int NUM_QTILE16 = Br / 16;
    float regs_m[NUM_QTILE16][2];
    float regs_l[NUM_QTILE16][2];
    #pragma unroll
    for (int i = 0; i < NUM_QTILE16; ++i)
    {
        regs_m[i][0] = -FLT_MAX;
        regs_m[i][1] = -FLT_MAX;

        regs_l[i][0] = 0.0f;
        regs_l[i][1] = 0.0f;
    }
    __syncthreads();

    const int last_kv_tile_idx = nums_kv_tile - 1;
    for (int kv_tile_idx = 0; kv_tile_idx < last_kv_tile_idx; ++kv_tile_idx) {
        process_kv_tile<false, Br, Bc, HEAD_DIM, NUM_WARPS>(
            k_offset_ptr, v_offset_ptr, o_offset_ptr, kv_stride, o_stride,
            kv_tile_idx, smem_q, smem_k, smem_v, smem_o, smem_max, nums_row_q_tile,
            min(Bc, kv_len - kv_tile_idx * Bc), regs_m, regs_l);
    }
    process_kv_tile<true, Br, Bc, HEAD_DIM, NUM_WARPS>(
        k_offset_ptr, v_offset_ptr, o_offset_ptr, kv_stride, o_stride,
        last_kv_tile_idx, smem_q, smem_k, smem_v, smem_o, smem_max, nums_row_q_tile,
        min(Bc, kv_len - last_kv_tile_idx * Bc), regs_m, regs_l);
}
