#include<cuda_bf16.h>
#include<stdint.h>
#include<assert.h>

#define WARP_SIZE 32

__device__ __forceinline__
void cp_async_16B(
    const void* gmem_ptr,
    void* smem_ptr
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__
void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__
void cp_async_wait_group() {
    asm volatile(
        "cp.async.wait_group %0;\n"
        :
        :"n"(N)
        :
    );
}

// ldmatrix 通用 swizzle
__device__ __forceinline__ int fa2_shared_offset_Bank8x128B(
    int element_idx,
    int stride,
    int element_per_row,
    int element_size)
{
    int row = element_idx / stride;
    int col = element_idx % stride;

    int element_per_128B = 128 / element_size;
    int row_8x128B = row % 8;
    int col_8x128B = col % element_per_128B;

    int group_base = element_idx - element_idx % element_per_128B;
    return group_base + (col_8x128B + row_8x128B * element_per_row) % element_per_128B;
}

// shape: (NUM_ROWS, ELEMENT_PER_ROW)
template<int NUM_ROWS, int ELEMENT_PER_ROW, bool ASYNC>
__device__ __forceinline__
void cp_tile_from_global_to_shared(
    const __nv_bfloat16* src,
    int src_stride,
    __nv_bfloat16* smem,
    int valid_rows
) {
    const uint4* src_vec = reinterpret_cast<const uint4*>(src);
    uint4* smem_vec = reinterpret_cast<uint4*>(smem);

    constexpr int element_per_row = ELEMENT_PER_ROW / (sizeof(uint4) / sizeof(__nv_bfloat16));
    int src_vec_stride = src_stride / (sizeof(uint4) / sizeof(__nv_bfloat16));

    // load valid rows
    int num_valid_element_vec = valid_rows * element_per_row;
    for (
        int element_vec_idx = threadIdx.x;
        element_vec_idx < num_valid_element_vec;
        element_vec_idx+=blockDim.x
    ) {
        int element_vec_row = element_vec_idx / element_per_row;
        int element_vec_col = element_vec_idx % element_per_row;
        int element_vec_gmem_idx = element_vec_row * src_vec_stride + element_vec_col;

        int element_vec_smem_idx = fa2_shared_offset_Bank8x128B(element_vec_idx, element_per_row, 1, sizeof(uint4));
        if constexpr (!ASYNC) {
            smem_vec[element_vec_smem_idx] = src_vec[element_vec_gmem_idx];
        } else {
            cp_async_16B(src_vec + element_vec_gmem_idx, smem_vec + element_vec_smem_idx);
        }
    }

    // padding 0
    for (
        int element_vec_idx = threadIdx.x + num_valid_element_vec;
        element_vec_idx < NUM_ROWS * element_per_row;
        element_vec_idx+=blockDim.x
    ) {
        int element_vec_smem_idx = fa2_shared_offset_Bank8x128B(element_vec_idx, element_per_row, 1, sizeof(uint4));
        smem_vec[element_vec_smem_idx] = make_uint4(0, 0, 0, 0);
    }
}

template<int Br, int HEAD_DIM>
__device__ __forceinline__
void cp_qtile_from_global_to_shared(
    const __nv_bfloat16 *q,
    int q_stride,
    __nv_bfloat16 *smem_q,
    int num_valid_rows_q_tile
) {
    cp_tile_from_global_to_shared<Br, HEAD_DIM, false>(
        q,
        q_stride,
        smem_q,
        num_valid_rows_q_tile
    );
}

template<int Bc, int HEAD_DIM, bool ASYNC>
__device__ __forceinline__
void cp_kvtile_from_global_to_shared(
    const __nv_bfloat16 *kv,
    int kv_stride,
    __nv_bfloat16 *smem_kv,
    int num_valid_rows_kv_tile
) {
    cp_tile_from_global_to_shared<Bc, HEAD_DIM, ASYNC>(
        kv,
        kv_stride,
        smem_kv,
        num_valid_rows_kv_tile
    );
}

__device__ __forceinline__ void fa2_ldmatrix_x4(
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

__device__ __forceinline__ void fa2_ldmatrix_x2(
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

__device__ __forceinline__ void fa2_fa2_ldmatrix_x2_trans(
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

__device__ __forceinline__ void fa2_mma_m16n8k16(
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
uint32_t fa2_pack_bf16x2(
    float x,
    float y
) {
    __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<uint32_t *>(&v);
}

__device__ __forceinline__
uint2 fa2_pack_float2(
    float a,
    float b
) {
    return make_uint2(__float_as_uint(a), __float_as_uint(b));
}

// cp qtile
// cp.async ktile
// cp.async_commit ktile
template<int HEAD_DIM, int Br, int Bc, bool Is_last>
__device__ __forceinline__
void process_kv_tile(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
     float* o,
    int kv_len,
    int o_stride,
    int kv_stride,
    int* num_loadedrows_ktile,
    int* num_loadedrows_vtile,
    __nv_bfloat16* smem_q,
    __nv_bfloat16* smem_k,
    __nv_bfloat16* smem_v,
    int warp_idx,
    int lane_idx,
    float (&regs_o_acc)[HEAD_DIM / 8][4],
    float (&regs_m)[2],
    float (&regs_l)[2]
) {
    // cp.async vtile
    // cp.async_commit vtile
    const __nv_bfloat16* v_tile_ptr = v + (*num_loadedrows_vtile) * kv_stride;
    int num_rows_vtile = min(Bc, kv_len - *num_loadedrows_vtile);
    cp_kvtile_from_global_to_shared<Bc, HEAD_DIM, true>(
        v_tile_ptr,
        kv_stride,
        smem_v,
        num_rows_vtile
    );
    cp_async_commit_group();
    *num_loadedrows_vtile += num_rows_vtile;

    // cp.async_wait<1> ktile
    cp_async_wait_group<1>();
    __syncthreads();
    // stile = qtile @ ktile^T
    int VEC_PER_HEADDIM = HEAD_DIM / (sizeof(uint4) / sizeof(__nv_bfloat16));
    uint4* smem_q_vec = reinterpret_cast<uint4*>(smem_q);
    uint4* smem_k_vec = reinterpret_cast<uint4*>(smem_k);
    float regs_s[Bc / 8][4] = {};
    for (int dtile16_idx = 0; dtile16_idx < HEAD_DIM / 16; dtile16_idx++) {
        // load qtile
        int qtile_row_offset = warp_idx * 16 + lane_idx % 16;
        int qtile_col_offset = dtile16_idx * 16 + (lane_idx / 16) * 8;
        int element_vec_idx = fa2_shared_offset_Bank8x128B(
            qtile_row_offset * VEC_PER_HEADDIM + qtile_col_offset / (sizeof(uint4) / sizeof(__nv_bfloat16)),
            VEC_PER_HEADDIM,
            1,
            sizeof(uint4)
        );
        uint32_t regs_q[4];
        fa2_ldmatrix_x4(
            regs_q[0],
            regs_q[1],
            regs_q[2],
            regs_q[3],
            smem_q_vec + element_vec_idx
        );

        for (int ktile8x16_row_idx = 0; ktile8x16_row_idx < Bc / 8; ktile8x16_row_idx++) {
            // load ktile
            int ktile_row_offset = ktile8x16_row_idx * 8 + lane_idx % 8;
            int ktile_col_offset = dtile16_idx * 16 + (lane_idx / 8) * 8;
            int element_vec_idx = fa2_shared_offset_Bank8x128B(
                ktile_row_offset * VEC_PER_HEADDIM + ktile_col_offset / (sizeof(uint4) / sizeof(__nv_bfloat16)),
                VEC_PER_HEADDIM,
                1,
                sizeof(uint4)
            );
            uint32_t regs_k[2];
            fa2_ldmatrix_x2(
                regs_k[0],
                regs_k[1],
                smem_k_vec + element_vec_idx
            );

            // mma
            fa2_mma_m16n8k16(
                regs_s[ktile8x16_row_idx][0],
                regs_s[ktile8x16_row_idx][1],
                regs_s[ktile8x16_row_idx][2],
                regs_s[ktile8x16_row_idx][3],
                regs_q[0],
                regs_q[1],
                regs_q[2],
                regs_q[3],
                regs_k[0],
                regs_k[1]
            );
        }
    }

    // if !Is_last
    // cp.async ktile
    // cp.async_commit ktile
    if constexpr (!Is_last) {
        // All warps must finish reading K before its shared buffer is reused.
        __syncthreads();
        const __nv_bfloat16* k_tile_ptr = k + (*num_loadedrows_ktile) * kv_stride;
        int num_rows_ktile = min(Bc, kv_len - *num_loadedrows_ktile);
        cp_kvtile_from_global_to_shared<Bc, HEAD_DIM, true>(
            k_tile_ptr,
            kv_stride,
            smem_k,
            num_rows_ktile
        );
        cp_async_commit_group();
        *num_loadedrows_ktile += num_rows_ktile;
    }

    // TODO: mask & unaligned

    // scale & local max
    const float inv_scale = rsqrtf((float)HEAD_DIM);
    float thread_max0 = -INFINITY;
    float thread_max1 = -INFINITY;
    for (int ktile8_idx = 0; ktile8_idx < Bc / 8; ktile8_idx++) {
        regs_s[ktile8_idx][0] *= inv_scale;
        regs_s[ktile8_idx][1] *= inv_scale;
        regs_s[ktile8_idx][2] *= inv_scale;
        regs_s[ktile8_idx][3] *= inv_scale;

        thread_max0 = fmaxf(thread_max0, regs_s[ktile8_idx][0]);
        thread_max0 = fmaxf(thread_max0, regs_s[ktile8_idx][1]);
        thread_max1 = fmaxf(thread_max1, regs_s[ktile8_idx][2]);
        thread_max1 = fmaxf(thread_max1, regs_s[ktile8_idx][3]);
    }
    thread_max0 = fmaxf(thread_max0, __shfl_xor_sync(0xffffffff, thread_max0, 1));
    thread_max1 = fmaxf(thread_max1, __shfl_xor_sync(0xffffffff, thread_max1, 1));
    thread_max0 = fmaxf(thread_max0, __shfl_xor_sync(0xffffffff, thread_max0, 2));
    thread_max1 = fmaxf(thread_max1, __shfl_xor_sync(0xffffffff, thread_max1, 2));
    // global max
    thread_max0 = fmaxf(thread_max0, regs_m[0]);
    thread_max1 = fmaxf(thread_max1, regs_m[1]);
    float factor0 = expf(regs_m[0] - thread_max0);
    float factor1 = expf(regs_m[1] - thread_max1);
    regs_m[0] = thread_max0;
    regs_m[1] = thread_max1;

    // old o_acc apply factor
    for (int dtile8_idx = 0; dtile8_idx < HEAD_DIM / 8; dtile8_idx++) {
        regs_o_acc[dtile8_idx][0] *= factor0;
        regs_o_acc[dtile8_idx][1] *= factor0;
        regs_o_acc[dtile8_idx][2] *= factor1;
        regs_o_acc[dtile8_idx][3] *= factor1;
    }
    // old sum apply factor
    regs_l[0] *= factor0;
    regs_l[1] *= factor1;

    // apply max & local sum
    float thread_sum0 = 0;
    float thread_sum1 = 0;
    for (int ktile8_idx = 0; ktile8_idx < Bc / 8; ktile8_idx++) {
        regs_s[ktile8_idx][0] = expf(regs_s[ktile8_idx][0] - regs_m[0]);
        regs_s[ktile8_idx][1] = expf(regs_s[ktile8_idx][1] - regs_m[0]);
        regs_s[ktile8_idx][2] = expf(regs_s[ktile8_idx][2] - regs_m[1]);
        regs_s[ktile8_idx][3] = expf(regs_s[ktile8_idx][3] - regs_m[1]);

        thread_sum0 += regs_s[ktile8_idx][0];
        thread_sum0 += regs_s[ktile8_idx][1];
        thread_sum1 += regs_s[ktile8_idx][2];
        thread_sum1 += regs_s[ktile8_idx][3];
    }
    thread_sum0 += __shfl_xor_sync(0xffffffff, thread_sum0, 1);
    thread_sum1 += __shfl_xor_sync(0xffffffff, thread_sum1, 1);
    thread_sum0 += __shfl_xor_sync(0xffffffff, thread_sum0, 2);
    thread_sum1 += __shfl_xor_sync(0xffffffff, thread_sum1, 2);
    // global sum
    regs_l[0] += thread_sum0;
    regs_l[1] += thread_sum1;

    // if !Is_last
    if constexpr (!Is_last) {
        // cp.async_wait<1> vtile
        cp_async_wait_group<1>();
    } else {
        // cp.async_wait<0> vtile
        cp_async_wait_group<0>();
    }
    __syncthreads();

    // o_acc += ptile @ vtile
    uint4* smem_v_vec = reinterpret_cast<uint4*>(smem_v);
    for (int ptile16_idx = 0; ptile16_idx < Bc / 16; ptile16_idx++) {
        // ptile 已经就位

        for (int dtile8_idx = 0; dtile8_idx < HEAD_DIM / 8; dtile8_idx++) {
            // load vtile
            int vtile_row_offset = ptile16_idx * 16 + lane_idx % 16;
            int vtile_col_offset = dtile8_idx * 8 + (lane_idx / 16) * 8;
            int element_vec_idx = fa2_shared_offset_Bank8x128B(
                vtile_row_offset * VEC_PER_HEADDIM + vtile_col_offset / (sizeof(uint4) / sizeof(__nv_bfloat16)),
                VEC_PER_HEADDIM,
                1,
                sizeof(uint4)
            );
            uint32_t regs_v[2];
            fa2_fa2_ldmatrix_x2_trans(
                regs_v[0],
                regs_v[1],
                smem_v_vec + element_vec_idx
            );

            // mma
            fa2_mma_m16n8k16(
                regs_o_acc[dtile8_idx][0],
                regs_o_acc[dtile8_idx][1],
                regs_o_acc[dtile8_idx][2],
                regs_o_acc[dtile8_idx][3],
                fa2_pack_bf16x2(regs_s[ptile16_idx * 2][0], regs_s[ptile16_idx * 2][1]),
                fa2_pack_bf16x2(regs_s[ptile16_idx * 2][2], regs_s[ptile16_idx * 2][3]),
                fa2_pack_bf16x2(regs_s[ptile16_idx * 2 + 1][0], regs_s[ptile16_idx * 2 + 1][1]),
                fa2_pack_bf16x2(regs_s[ptile16_idx * 2 + 1][2], regs_s[ptile16_idx * 2 + 1][3]),
                regs_v[0],
                regs_v[1]
            );
        }
    }

    // Finish every warp's V reads before the next iteration overwrites V.
    __syncthreads();

    // if Is_last
    if constexpr (Is_last) {
        float inv_sum0 = 1.0f / regs_l[0];
        float inv_sum1 = 1.0f / regs_l[1];
        uint2* smem_q_vec = reinterpret_cast<uint2*>(smem_q);
        uint2* o_vec = reinterpret_cast<uint2*>(o);
        int o_vec_stride = o_stride / (sizeof(uint2) / sizeof(float));
        constexpr int UINT2_PER_SMEMQ_ROW = HEAD_DIM / (sizeof(uint2) / sizeof(__nv_bfloat16));
        // adapt smem_q size
        for (int dtile_hdinv2_idx = 0; dtile_hdinv2_idx < 2; dtile_hdinv2_idx++) {
            // o_acc swizzle to smem_q
            for (int dtile8_idx = 0; dtile8_idx < HEAD_DIM / 8 / 2; dtile8_idx++) {
                int dtile8_offset = dtile_hdinv2_idx * HEAD_DIM / 8 / 2;
                // o_acc apply sum & pack
                uint2 elem0 = fa2_pack_float2(regs_o_acc[dtile8_offset + dtile8_idx][0] * inv_sum0, regs_o_acc[dtile8_offset + dtile8_idx][1] * inv_sum0);
                uint2 elem1 = fa2_pack_float2(regs_o_acc[dtile8_offset + dtile8_idx][2] * inv_sum1, regs_o_acc[dtile8_offset + dtile8_idx][3] * inv_sum1);
                int elem0_row = warp_idx * 16 + lane_idx / 4;
                int elem1_row = warp_idx * 16 + 8 + lane_idx / 4;
                int elem_col = dtile8_idx * 8 / (sizeof(uint2) / sizeof(float)) + lane_idx % 4;
                int elem0_idx = fa2_shared_offset_Bank8x128B(
                    elem0_row * UINT2_PER_SMEMQ_ROW + elem_col,
                    UINT2_PER_SMEMQ_ROW,
                    4,
                    sizeof(uint2)
                );
                smem_q_vec[elem0_idx] = elem0;
                int elem1_idx = fa2_shared_offset_Bank8x128B(
                    elem1_row * UINT2_PER_SMEMQ_ROW + elem_col,
                    UINT2_PER_SMEMQ_ROW,
                    4,
                    sizeof(uint2)
                );
                smem_q_vec[elem1_idx] = elem1;
            }
            __syncthreads();

            // smem_q coalesce to global
            for(int el_idx = threadIdx.x; el_idx < Br * UINT2_PER_SMEMQ_ROW; el_idx+=blockDim.x) {
                int row = el_idx / UINT2_PER_SMEMQ_ROW;
                int col = dtile_hdinv2_idx * UINT2_PER_SMEMQ_ROW + el_idx % UINT2_PER_SMEMQ_ROW;
                int swizzle_idx = fa2_shared_offset_Bank8x128B(
                    el_idx,
                    UINT2_PER_SMEMQ_ROW,
                    4,
                    sizeof(uint2)
                );
                o_vec[row * o_vec_stride + col] = smem_q_vec[swizzle_idx];
            }
            __syncthreads();
        }
    }
}

// Packed varlen layout: [total_q_tokens, num_head, HEAD_DIM]
// K/V: [total_kv_tokens, num_kv_head, HEAD_DIM]
// GRID: (seq_idx, head_idx, br_idx)
template<int HEAD_DIM, int Br, int Bc, int NUM_WARPS>
__global__ void fa2(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    float* o,
    int num_head,
    int num_kv_head,
    const int* cu_q_len,
    const int* cu_kv_len
) {

    // init
    int seq_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int qtile_idx = blockIdx.x;
    int warp_idx = threadIdx.x / WARP_SIZE;
    int lane_idx = threadIdx.x % WARP_SIZE;

    int q_start = cu_q_len[seq_idx];
    int q_end = cu_q_len[seq_idx + 1];
    int q_len = q_end - q_start;
    int q_stride = num_head * HEAD_DIM;
    int q_offset = q_start + qtile_idx * Br;
    int num_rows_qtile = min(Br, q_end - q_offset);
    if (num_rows_qtile <= 0) return;

    int kv_start = cu_kv_len[seq_idx];
    int kv_end = cu_kv_len[seq_idx + 1];
    int kv_len = kv_end - kv_start;
    int kv_stride = num_kv_head * HEAD_DIM;
    int kv_head_idx = head_idx / (num_head / num_kv_head);
    int num_kvtiles = (kv_len + Bc - 1) / Bc;
    if (kv_len <= 0) return;

    // 暂不处理非整除情况
    assert(q_len % Br ==0);
    assert(kv_len % Bc ==0);

    const __nv_bfloat16* q_offset_ptr =
        q
        + q_offset * q_stride
        + head_idx * HEAD_DIM;
    const __nv_bfloat16* k_offset_ptr =
        k
        + kv_start * kv_stride
        + kv_head_idx * HEAD_DIM;
    const __nv_bfloat16* v_offset_ptr =
        v
        + kv_start * kv_stride
        + kv_head_idx * HEAD_DIM;
    float* o_offset_ptr =
        o
        + q_offset * q_stride
        + head_idx * HEAD_DIM;

    // shared:
    // Q, K, V, softmax temporary
    __shared__ __nv_bfloat16 smem_q[Br * HEAD_DIM];
    __shared__ __nv_bfloat16 smem_k[Bc * HEAD_DIM];
    __shared__ __nv_bfloat16 smem_v[Bc * HEAD_DIM];

    // 每个 warp 负责一个 qtile16
    static_assert(Br / NUM_WARPS == 16);

    // registers:
    float regs_o_acc[HEAD_DIM / 8][4] = {};
    float regs_m[2] = {-INFINITY, -INFINITY};
    float regs_l[2] = {0.0f, 0.0f};


    // prologue:
    int num_loadedrows_ktile = 0;
    int num_loadedrows_vtile = 0;

    int num_rows_ktile = min(Bc, kv_len - num_loadedrows_ktile);
    cp_kvtile_from_global_to_shared<Bc, HEAD_DIM, true>(
        k_offset_ptr,
        kv_stride,
        smem_k,
        num_rows_ktile
    );
    cp_async_commit_group();
    num_loadedrows_ktile += num_rows_ktile;

    cp_qtile_from_global_to_shared<Br, HEAD_DIM>(
        q_offset_ptr,
        q_stride,
        smem_q, num_rows_qtile
    );
    __syncthreads();

    for(int kvtile_idx = 0; kvtile_idx < num_kvtiles - 1; kvtile_idx++) {
        process_kv_tile<HEAD_DIM, Br, Bc, false>(
            k_offset_ptr,
            v_offset_ptr,
            o_offset_ptr,
            kv_len,
            q_stride,       // O 的 row stride
            kv_stride,
            &num_loadedrows_ktile,
            &num_loadedrows_vtile,
            smem_q,
            smem_k,
            smem_v,
            warp_idx,
            lane_idx,
            regs_o_acc,
            regs_m,
            regs_l
        );
    }
    process_kv_tile<HEAD_DIM, Br, Bc, true>(
        k_offset_ptr,
        v_offset_ptr,
        o_offset_ptr,
        kv_len,
        q_stride,
        kv_stride,
        &num_loadedrows_ktile,
        &num_loadedrows_vtile,
        smem_q,
        smem_k,
        smem_v,
        warp_idx,
        lane_idx,
        regs_o_acc,
        regs_m,
        regs_l
    );
}
