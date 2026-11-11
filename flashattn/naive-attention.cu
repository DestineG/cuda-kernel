#include<cuda_bf16.h>
#include <cfloat>

__global__ void kernel_attn_prefill(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    float *S,
    float *P,
    float *O,
    int N,
    int d
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= N) return;

    // S = QK^t / sqrt(d)
    float scale = rsqrtf((float)d);
    for (int i = 0; i < N; i++) {
        float vec_sum = 0;
        for (int j = 0; j < d; j++) {
            vec_sum += __bfloat162float(Q[d * row + j]) * __bfloat162float(K[d * i + j]);
        }
        S[N * row + i] = vec_sum * scale;
    }
    
    // max
    float row_max = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        row_max = fmaxf(S[N * row + i], row_max);
    }

    // exp-sum
    float row_sum = 0;
    for (int i = 0; i < N; i++) {
        row_sum += expf(S[N * row + i] - row_max);
    }

    // P = exp(x - max) / exp-sum
    for (int i = 0; i < N; i++) {
        P[N * row + i] = expf(S[N * row + i] - row_max) / row_sum;
    }

    // O = PV
    for (int i = 0; i < d; i++) {
        float vec_sum = 0;
        for (int j = 0; j < N; j++) {
            vec_sum += P[N * row + j] * __bfloat162float(V[d * j + i]);
        }
        O[d * row + i] = vec_sum;
    }
}

__global__ void kernel_attn_prefill_causal(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    float *S,
    float *P,
    float *O,
    int N,
    int d
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= N) return;
    
    // S = QK^t / sqrt(d)
    float scale = rsqrtf((float)d);
    for (int i = 0; i <= row; i++) {
        float vec_sum = 0.0f;
        for (int j = 0; j < d; j++) {
            vec_sum += __bfloat162float(Q[d * row + j]) * __bfloat162float(K[d * i + j]);
        }
        S[N * row + i] = vec_sum * scale;
    }
    for (int i = row + 1; i < N; i++) {
        S[N * row + i] = -INFINITY;
    }

    // max
    float row_max = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        row_max = fmaxf(S[N * row + i], row_max);
    }

    // exp-sum
    float row_sum = 0;
    for (int i = 0; i < N; i++) {
        row_sum += expf(S[N * row + i] - row_max);
    }

    // P = exp(x - max) / exp-sum
    for (int i = 0; i < N; i++) {
        P[N * row + i] = expf(S[N * row + i] - row_max) / row_sum;
    }

    // O = PV
    for (int i = 0; i < d; i++) {
        float vec_sum = 0;
        for (int j = 0; j < N; j++) {
            vec_sum += P[N * row + j] * __bfloat162float(V[d * j + i]);
        }
        O[d * row + i] = vec_sum;
    }
}