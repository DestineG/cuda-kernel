#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define WARP_SIZE 32

__global__ void gemm_naive(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) sum += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
    C[row * N + col] = sum;
}

static void check_cuda(cudaError_t status, const char* file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

static void check_cublas(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", file, line, (int)status);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) check_cuda((call), __FILE__, __LINE__)
#define CUBLAS_CHECK(call) check_cublas((call), __FILE__, __LINE__)

__device__ __forceinline__ int swizzle_byte16_offset(int byte16_idx, int byte16_per_row) {
    int byte16_y = byte16_idx / byte16_per_row;
    int byte16_x = byte16_idx % byte16_per_row;
    int swizzle_byte16_x = (byte16_x / 8) * 8 + (byte16_y % 8 + byte16_x) % 8;
    return byte16_y * byte16_per_row + swizzle_byte16_x;
}

__device__ __forceinline__ void cp_async_16B_from_global_to_shared(
    void* shared_destination,
    const void* global_source
) {
    uint32_t shared_address = (uint32_t)__cvta_generic_to_shared(shared_destination);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(shared_address), "l"(global_source));
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

template<int BM, int BN, int BK, int WM, int WN>
__device__ __forceinline__ void cp_async_load_AB_Ktile(
    const __nv_bfloat16* A_base, const __nv_bfloat16* B_base,
    int N, int K, __nv_bfloat16* smem_Atile, __nv_bfloat16* smem_Btile,
    int local_thread_idx
) {
    constexpr int ELEMS_PER_16B = 8;
    constexpr int THREADS = WARP_SIZE * (BM / WM) * (BN / WN);
    for (int byte16_idx = local_thread_idx; byte16_idx < BM * BK / 8; byte16_idx += THREADS) {
        int elem_idx = byte16_idx * ELEMS_PER_16B;
        int swizzle_byte16_idx = swizzle_byte16_offset(byte16_idx, BK / 8);
        cp_async_16B_from_global_to_shared(smem_Atile + swizzle_byte16_idx * ELEMS_PER_16B, A_base + elem_idx / BK * K + elem_idx % BK);
    }
    for (int byte16_idx = local_thread_idx; byte16_idx < BK * BN / 8; byte16_idx += THREADS) {
        int elem_idx = byte16_idx * ELEMS_PER_16B;
        int swizzle_byte16_idx = swizzle_byte16_offset(byte16_idx, BN / 8);
        cp_async_16B_from_global_to_shared(smem_Btile + swizzle_byte16_idx * ELEMS_PER_16B, B_base + elem_idx / BN * N + elem_idx % BN);
    }
}

__device__ __forceinline__ void load_mma_a(
    const __nv_bfloat16* address,
    uint32_t* registers
) {
    uint32_t shared_address =
        static_cast<uint32_t>(__cvta_generic_to_shared(address));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(registers[0]),
          "=r"(registers[1]),
          "=r"(registers[2]),
          "=r"(registers[3])
        : "r"(shared_address)
    );
}

__device__ __forceinline__ void load_mma_b(
    const __nv_bfloat16* address,
    uint32_t* registers
) {
    uint32_t shared_address =
        static_cast<uint32_t>(__cvta_generic_to_shared(address));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(registers[0]),
          "=r"(registers[1])
        : "r"(shared_address)
    );
}

__device__ __forceinline__ void mma(
    const uint32_t* A,
    const uint32_t* B,
    float* C
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(C[0]),
          "+f"(C[1]),
          "+f"(C[2]),
          "+f"(C[3])
        : "r"(A[0]),
          "r"(A[1]),
          "r"(A[2]),
          "r"(A[3]),
          "r"(B[0]),
          "r"(B[1])
    );
}

template<
    int BM, int BN, int BK,
    int WM, int WN,
    int MMA_M, int MMA_N, int MMA_K
>
__global__ void gemm_latest(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float* C,
    int M,
    int N,
    int K
) {
    static_assert(MMA_M == 16);
    static_assert(MMA_N == 8);
    static_assert(MMA_K == 16);

    int  block_row_base = BM * blockIdx.y;
    int  block_col_base = BN * blockIdx.x;
    int local_thread_idx = threadIdx.x;
    int warp_idx = local_thread_idx / WARP_SIZE;
    int lane_idx = local_thread_idx % WARP_SIZE;
    constexpr int NUM_WM = BM / WM;
    constexpr int NUM_WN = BN / WN;
    constexpr int THREADS = WARP_SIZE * NUM_WM * NUM_WN;
    static_assert(BM % WM == 0, "BM must be divisible by WM");
    static_assert(BN % WN == 0, "BN must be divisible by WN");
    static_assert(WM % MMA_M == 0, "WM must be divisible by MMA_M");
    static_assert(WN % MMA_N == 0, "WN must be divisible by MMA_N");
    static_assert(BK % MMA_K == 0, "BK must be divisible by MMA_K");
    static_assert(THREADS <= 1024, "CTA has too many threads");

    // shared
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    constexpr int A_STAGE_ELEMENTS = BM * BK;
    constexpr int B_STAGE_ELEMENTS = BK * BN;
    constexpr int STAGE_ELEMENTS = A_STAGE_ELEMENTS + B_STAGE_ELEMENTS;
    __nv_bfloat16* smem_A[2] = {smem, smem + STAGE_ELEMENTS};
    __nv_bfloat16* smem_B[2] = {smem + A_STAGE_ELEMENTS, smem + STAGE_ELEMENTS + A_STAGE_ELEMENTS};
    int read_stage = 0;
    int write_stage = 1;

    cp_async_load_AB_Ktile<BM, BN, BK, WM, WN>(A + block_row_base * K, B + block_col_base, N, K, smem_A[read_stage], smem_B[read_stage], local_thread_idx);
    cp_async_commit_group();

    // register
    constexpr int MMA_A_REG = MMA_M * MMA_K / WARP_SIZE / (sizeof(uint32_t) / sizeof(__nv_bfloat16));
    constexpr int MMA_B_REG = MMA_K * MMA_N / WARP_SIZE / (sizeof(uint32_t) / sizeof(__nv_bfloat16));
    constexpr int MMA_C_REG = MMA_M * MMA_N / WARP_SIZE;
    uint32_t A_r[WM / MMA_M][MMA_A_REG];
    uint32_t B_r[WN / MMA_N][MMA_B_REG];
    float C_r[(WM / MMA_M) * (WN / MMA_N)][MMA_C_REG] = {};

    cp_async_wait_group();
    __syncthreads();            // 确保数据完整加载

    for (int k_tile_idx = 0; k_tile_idx < K / BK; k_tile_idx++) {
        __nv_bfloat16* As_read = smem_A[read_stage];
        __nv_bfloat16* Bs_read = smem_B[read_stage];
        if (k_tile_idx + 1 < K / BK) {
            cp_async_load_AB_Ktile<BM, BN, BK, WM, WN>(A + block_row_base * K + (k_tile_idx + 1) * BK, B + (k_tile_idx + 1) * BK * N + block_col_base, N, K, smem_A[write_stage], smem_B[write_stage], local_thread_idx);
            cp_async_commit_group();
        }

        #pragma unroll
        for (int k_mma_idx = 0; k_mma_idx < BK / MMA_K; k_mma_idx++) {
            #pragma unroll
            for (int m_mma_idx = 0; m_mma_idx < WM / MMA_M; m_mma_idx++) {
                int warp_y = warp_idx / NUM_WN;
                int related_row = warp_y * WM + m_mma_idx * MMA_M;
                int related_col = k_mma_idx * MMA_K;
                int ldmatrix_8x8_idx = lane_idx / 8;
                int ldmatrix_8x8_inline_row = lane_idx % 8;
                int ldmatrix_8x8_c = ldmatrix_8x8_idx / 2;
                int ldmatrix_8x8_r = ldmatrix_8x8_idx % 2;
                related_row += ldmatrix_8x8_r * 8 + ldmatrix_8x8_inline_row;
                related_col += ldmatrix_8x8_c * 8;
                int byte16_idx = (related_row * BK + related_col) / 8;
                int swizzle_byte16_idx = swizzle_byte16_offset(byte16_idx, BK / 8);
                load_mma_a(As_read + swizzle_byte16_idx * 8, A_r[m_mma_idx]);
            }
            #pragma unroll
            for (int n_mma_idx = 0; n_mma_idx < WN / MMA_N; n_mma_idx++) {
                int warp_x = warp_idx % NUM_WN;
                int related_row = k_mma_idx * MMA_K;
                int related_col = warp_x * WN + n_mma_idx * MMA_N;
                int ldmatrix_8x8_idx = lane_idx / 8;
                int ldmatrix_8x8_inline_row = lane_idx % 8;
                int ldmatrix_8x8_r = ldmatrix_8x8_idx % 2;
                related_row += ldmatrix_8x8_r * 8 + ldmatrix_8x8_inline_row;
                int byte16_idx = (related_row * BN + related_col) / 8;
                int swizzle_byte16_idx = swizzle_byte16_offset(byte16_idx, BN / 8);
                load_mma_b(Bs_read + swizzle_byte16_idx * 8, B_r[n_mma_idx]);
            }
            #pragma unroll
            for (int m_mma_idx = 0; m_mma_idx < WM / MMA_M; m_mma_idx++) {
                #pragma unroll
                for (int n_mma_idx = 0; n_mma_idx < WN / MMA_N; n_mma_idx++) {
                    mma(A_r[m_mma_idx], B_r[n_mma_idx], C_r[m_mma_idx * (WN / MMA_N) + n_mma_idx]);
                }
            }
        }
        if (k_tile_idx + 1 < K / BK) {
            cp_async_wait_group();
            __syncthreads();
            int old_read_stage = read_stage;
            read_stage = write_stage;
            write_stage = old_read_stage;
        }
    }

    int warp_y = warp_idx / NUM_WN;
    int warp_x = warp_idx % NUM_WN;
    int row_in_fragment = lane_idx >> 2;
    int col_in_fragment = (lane_idx & 3) * 2;
    #pragma unroll
    for (int m_mma_idx = 0; m_mma_idx < WM / MMA_M; m_mma_idx++) {
        #pragma unroll
        for (int n_mma_idx = 0; n_mma_idx < WN / MMA_N; n_mma_idx++) {
            int fragment = m_mma_idx * (WN / MMA_N) + n_mma_idx;
            const float* values = C_r[fragment];
            int row = block_row_base + warp_y * WM + m_mma_idx * MMA_M + row_in_fragment;
            int col = block_col_base + warp_x * WN + n_mma_idx * MMA_N + col_in_fragment;
            *reinterpret_cast<float2*>(C + row * N + col) = make_float2(values[0], values[1]);
            *reinterpret_cast<float2*>(C + (row + 8) * N + col) = make_float2(values[2], values[3]);
        }
    }
}

constexpr int CUSTOM_BM = 128;
constexpr int CUSTOM_BN = 128;
constexpr int CUSTOM_BK = 64;
constexpr int CUSTOM_WM = 32;
constexpr int CUSTOM_WN = 64;
constexpr int CUSTOM_MMA_M = 16;
constexpr int CUSTOM_MMA_N = 8;
constexpr int CUSTOM_MMA_K = 16;
constexpr int CUSTOM_THREADS = WARP_SIZE * (CUSTOM_BM / CUSTOM_WM) * (CUSTOM_BN / CUSTOM_WN);
constexpr size_t CUSTOM_SHARED_BYTES = 2ull * (CUSTOM_BM * CUSTOM_BK + CUSTOM_BK * CUSTOM_BN) * sizeof(__nv_bfloat16);

static void launch_custom(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int M, int N, int K) {
    dim3 grid(N / CUSTOM_BN, M / CUSTOM_BM);
    CUDA_CHECK(cudaFuncSetAttribute(gemm_latest<CUSTOM_BM, CUSTOM_BN, CUSTOM_BK, CUSTOM_WM, CUSTOM_WN, CUSTOM_MMA_M, CUSTOM_MMA_N, CUSTOM_MMA_K>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)CUSTOM_SHARED_BYTES));
    gemm_latest<CUSTOM_BM, CUSTOM_BN, CUSTOM_BK, CUSTOM_WM, CUSTOM_WN, CUSTOM_MMA_M, CUSTOM_MMA_N, CUSTOM_MMA_K><<<grid, CUSTOM_THREADS, CUSTOM_SHARED_BYTES>>>(A, B, C, M, N, K);
}

struct TestContext {
    int M;
    int N;
    int K;
    size_t a_count;
    size_t b_count;
    size_t c_count;
    __nv_bfloat16* A;
    __nv_bfloat16* B;
    float* output;
    float* reference;
    __nv_bfloat16* d_A;
    __nv_bfloat16* d_B;
    float* d_C;
    cublasHandle_t cublas;

    TestContext(int M_value, int N_value, int K_value) : M(M_value), N(N_value), K(K_value), a_count((size_t)M_value * K_value), b_count((size_t)K_value * N_value), c_count((size_t)M_value * N_value), A((__nv_bfloat16*)malloc(a_count * sizeof(__nv_bfloat16))), B((__nv_bfloat16*)malloc(b_count * sizeof(__nv_bfloat16))), output((float*)malloc(c_count * sizeof(float))), reference((float*)malloc(c_count * sizeof(float))), d_A(nullptr), d_B(nullptr), d_C(nullptr), cublas(nullptr) {
        if (A == nullptr || B == nullptr || output == nullptr || reference == nullptr) { fprintf(stderr, "Host allocation failed\n"); exit(EXIT_FAILURE); }
        for (size_t i = 0; i < a_count; ++i) A[i] = __float2bfloat16((float)((int)((i * 17u + 3u) % 101u) - 50) / 100.0f);
        for (size_t i = 0; i < b_count; ++i) B[i] = __float2bfloat16((float)((int)((i * 29u + 7u) % 103u) - 51) / 100.0f);
        CUDA_CHECK(cudaMalloc(&d_A, a_count * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_B, b_count * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_C, c_count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, A, a_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B, b_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasCreate(&cublas));
    }

    ~TestContext() {
        if (cublas != nullptr) cublasDestroy(cublas);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(A);
        free(B);
        free(output);
        free(reference);
    }

    TestContext(const TestContext&) = delete;
    TestContext& operator=(const TestContext&) = delete;
};

static void launch_cublas(TestContext& context) {
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(context.cublas, CUBLAS_OP_N, CUBLAS_OP_N, context.N, context.M, context.K, &alpha, context.d_B, CUDA_R_16BF, context.N, context.d_A, CUDA_R_16BF, context.K, &beta, context.d_C, CUDA_R_32F, context.N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void launch_naive(TestContext& context) {
    dim3 block(16, 16);
    dim3 grid((context.N + 15) / 16, (context.M + 15) / 16);
    gemm_naive<<<grid, block>>>(context.d_A, context.d_B, context.d_C, context.M, context.N, context.K);
}

enum KernelKind { KERNEL_CUBLAS, KERNEL_NAIVE, KERNEL_CUSTOM };

static const char* kernel_name(KernelKind kind) {
    if (kind == KERNEL_CUBLAS) return "cublas";
    if (kind == KERNEL_NAIVE) return "naive";
    return "custom";
}

static void launch_kernel(TestContext& context, KernelKind kind) {
    if (kind == KERNEL_CUBLAS) launch_cublas(context);
    else if (kind == KERNEL_NAIVE) launch_naive(context);
    else launch_custom(context.d_A, context.d_B, context.d_C, context.M, context.N, context.K);
}

struct ErrorResult { double max_absolute; double mean_absolute; double relative_l2; double max_relative; };

static ErrorResult compare_results(const float* actual, const float* reference, size_t count) {
    ErrorResult result = {0.0, 0.0, 0.0, 0.0};
    double square_error = 0.0;
    double square_reference = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double difference = fabs((double)actual[i] - (double)reference[i]);
        double reference_abs = fabs((double)reference[i]);
        if (difference > result.max_absolute) result.max_absolute = difference;
        result.mean_absolute += difference;
        square_error += difference * difference;
        square_reference += (double)reference[i] * (double)reference[i];
        if (reference_abs > 1e-6) {
            double relative = difference / reference_abs;
            if (relative > result.max_relative) result.max_relative = relative;
        }
    }
    result.mean_absolute /= (double)count;
    result.relative_l2 = sqrt(square_error / square_reference);
    return result;
}

static void make_reference(TestContext& context) {
    launch_cublas(context);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(context.reference, context.d_C, context.c_count * sizeof(float), cudaMemcpyDeviceToHost));
}

static bool verify_kernel(TestContext& context, KernelKind kind) {
    launch_kernel(context, kind);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(context.output, context.d_C, context.c_count * sizeof(float), cudaMemcpyDeviceToHost));
    ErrorResult error = compare_results(context.output, context.reference, context.c_count);
    bool passed = error.max_absolute <= 1e-3 && error.relative_l2 <= 1e-5;
    printf("%-8s %-4s max_abs=% .6e mean_abs=% .6e rel_l2=% .6e max_rel=% .6e\n", kernel_name(kind), passed ? "PASS" : "FAIL", error.max_absolute, error.mean_absolute, error.relative_l2, error.max_relative);
    return passed;
}

static float benchmark_kernel(TestContext& context, KernelKind kind, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) launch_kernel(context, kind);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch_kernel(context, kind);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return milliseconds / (float)iterations;
}

static int parse_positive_int(const char* text, int* value) {
    char* end = nullptr;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed <= 0 || parsed > 2147483647L) return 0;
    *value = (int)parsed;
    return 1;
}

static void print_usage(const char* program) {
    fprintf(stderr, "Usage:\n  %s verify <all|cublas|naive|custom> M N K\n  %s benchmark <all|cublas|naive|custom> M N K [warmup] [iterations]\n  %s profile custom M N K\n", program, program, program);
}

int main(int argc, char** argv) {
    if (argc < 6 || argc > 8) { print_usage(argv[0]); return EXIT_FAILURE; }
    const char* mode = argv[1];
    const char* target = argv[2];
    int M = 0;
    int N = 0;
    int K = 0;
    if (!parse_positive_int(argv[3], &M) || !parse_positive_int(argv[4], &N) || !parse_positive_int(argv[5], &K)) { print_usage(argv[0]); return EXIT_FAILURE; }
    if (M % CUSTOM_BM != 0 || N % CUSTOM_BN != 0 || K % CUSTOM_BK != 0) { fprintf(stderr, "Custom kernel requires M%%128=0, N%%128=0, K%%64=0\n"); return EXIT_FAILURE; }
    int warmup = 10;
    int iterations = 100;
    if (argc >= 7 && !parse_positive_int(argv[6], &warmup)) { print_usage(argv[0]); return EXIT_FAILURE; }
    if (argc >= 8 && !parse_positive_int(argv[7], &iterations)) { print_usage(argv[0]); return EXIT_FAILURE; }
    bool all = strcmp(target, "all") == 0;
    KernelKind selected = KERNEL_CUSTOM;
    if (!all) {
        if (strcmp(target, "cublas") == 0) selected = KERNEL_CUBLAS;
        else if (strcmp(target, "naive") == 0) selected = KERNEL_NAIVE;
        else if (strcmp(target, "custom") == 0) selected = KERNEL_CUSTOM;
        else { print_usage(argv[0]); return EXIT_FAILURE; }
    }

    TestContext context(M, N, K);
    if (strcmp(mode, "verify") == 0) {
        if (argc != 6) { print_usage(argv[0]); return EXIT_FAILURE; }
        make_reference(context);
        bool passed = true;
        KernelKind kernels[] = {KERNEL_CUBLAS, KERNEL_NAIVE, KERNEL_CUSTOM};
        for (int i = 0; i < 3; ++i) if (all || selected == kernels[i]) passed = verify_kernel(context, kernels[i]) && passed;
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (strcmp(mode, "benchmark") == 0) {
        printf("M=%d N=%d K=%d warmup=%d iterations=%d threads=%d shared=%zu\n", M, N, K, warmup, iterations, CUSTOM_THREADS, CUSTOM_SHARED_BYTES);
        printf("%-8s %12s %12s\n", "Kernel", "Time (ms)", "TFLOPS");
        KernelKind kernels[] = {KERNEL_CUBLAS, KERNEL_NAIVE, KERNEL_CUSTOM};
        for (int i = 0; i < 3; ++i) {
            if (!all && selected != kernels[i]) continue;
            float milliseconds = benchmark_kernel(context, kernels[i], warmup, iterations);
            double tflops = 2.0 * (double)M * (double)N * (double)K / ((double)milliseconds * 1e9);
            printf("%-8s %12.4f %12.3f\n", kernel_name(kernels[i]), milliseconds, tflops);
        }
        return EXIT_SUCCESS;
    }
    if (strcmp(mode, "profile") == 0 && selected == KERNEL_CUSTOM && argc == 6) {
        launch_kernel(context, selected);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return EXIT_SUCCESS;
    }
    print_usage(argv[0]);
    return EXIT_FAILURE;
}
