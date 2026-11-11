#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

#include "naive-attention.cu"
#include "flashattn1.cu"
namespace fa2_impl {
#include "flashattn2.cu"
}

#define CUDA_CHECK(call) check_cuda((call), __FILE__, __LINE__)

static int verification_failed = 0;

static void check_cuda(cudaError_t status, const char *file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n",
                file, line, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

static float bf16_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

static void fill_input(__nv_bfloat16 *data, size_t count, unsigned int seed) {
    for (size_t i = 0; i < count; ++i) {
        int value = (int)((i * 37u + seed * 101u + 17u) % 1001u) - 500;
        data[i] = __float2bfloat16((float)value / 100.0f);
    }
}

static void attention_cpu(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    float *output,
    int n,
    int d,
    int causal
) {
    size_t score_count = (size_t)n * (size_t)n;
    float *scores = (float *)malloc(score_count * sizeof(float));
    if (scores == NULL) {
        fprintf(stderr, "CPU allocation failed for %zu scores\n", score_count);
        exit(EXIT_FAILURE);
    }

    float scale = 1.0f / sqrtf((float)d);
    for (int row = 0; row < n; ++row) {
        int limit = causal ? row + 1 : n;
        float row_max = -FLT_MAX;

        for (int col = 0; col < limit; ++col) {
            float score = 0.0f;
            for (int j = 0; j < d; ++j) {
                score += bf16_to_float(q[(size_t)row * d + j]) *
                         bf16_to_float(k[(size_t)col * d + j]);
            }
            score *= scale;
            scores[(size_t)row * n + col] = score;
            row_max = fmaxf(row_max, score);
        }

        float row_sum = 0.0f;
        for (int col = 0; col < limit; ++col) {
            row_sum += expf(scores[(size_t)row * n + col] - row_max);
        }

        for (int col = 0; col < n; ++col) {
            float probability = 0.0f;
            if (col < limit) {
                probability = expf(scores[(size_t)row * n + col] - row_max) / row_sum;
            }
            for (int j = 0; j < d; ++j) {
                output[(size_t)row * d + j] +=
                    probability * bf16_to_float(v[(size_t)col * d + j]);
            }
        }
    }

    free(scores);
}

template <int HEAD_DIM, int Br>
static void launch_fa1(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    float *o,
    int n,
    const int *d_cu_len
) {
    constexpr int Bc = 64;
    constexpr int threads = 128;
    constexpr size_t shared_bytes =
        ((size_t)Br * HEAD_DIM + (size_t)Bc * HEAD_DIM * 2) * sizeof(__nv_bfloat16) +
        (size_t)Br * HEAD_DIM * sizeof(float) +
        (size_t)4 * Br * sizeof(float);
    static bool configured = false;
    if (!configured) {
        CUDA_CHECK(cudaFuncSetAttribute(
            fa1<HEAD_DIM, Br, Bc>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shared_bytes));
        configured = true;
    }
    dim3 grid(1, 1, (n + Br - 1) / Br);
    fa1<HEAD_DIM, Br, Bc><<<grid, threads, shared_bytes>>>(
        q, k, v, d_cu_len, d_cu_len, 1, 1, o);
}

template <int HEAD_DIM, int Br>
static void launch_fa2(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    float *o,
    int n,
    const int *d_cu_len
) {
    constexpr int Bc = 64;
    constexpr int num_warps = Br / 16;
    constexpr int threads = num_warps * 32;
    dim3 grid((n + Br - 1) / Br, 1, 1);
    fa2_impl::fa2<HEAD_DIM, Br, Bc, num_warps><<<grid, threads>>>(
        q, k, v, o, 1, 1, d_cu_len, d_cu_len);
}

static void launch_attention(
    int kernel,
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    float *s,
    float *p,
    float *o,
    int n,
    int d,
    int br,
    const int *d_cu_len
) {
    if (kernel == 0) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        kernel_attn_prefill<<<blocks, threads>>>(q, k, v, s, p, o, n, d);
    } else if (kernel == 1) {
        if (d == 64 && br == 16) launch_fa1<64, 16>(q, k, v, o, n, d_cu_len);
        else if (d == 64 && br == 32) launch_fa1<64, 32>(q, k, v, o, n, d_cu_len);
        else if (d == 64 && br == 64) launch_fa1<64, 64>(q, k, v, o, n, d_cu_len);
        else if (d == 128 && br == 16) launch_fa1<128, 16>(q, k, v, o, n, d_cu_len);
        else if (d == 128 && br == 32) launch_fa1<128, 32>(q, k, v, o, n, d_cu_len);
        else if (d == 128 && br == 64) launch_fa1<128, 64>(q, k, v, o, n, d_cu_len);
        else {
            fprintf(stderr, "Unsupported FA1 shape d=%d Br=%d\n", d, br);
            exit(EXIT_FAILURE);
        }
    } else {
        if (d == 64 && br == 16) launch_fa2<64, 16>(q, k, v, o, n, d_cu_len);
        else if (d == 64 && br == 32) launch_fa2<64, 32>(q, k, v, o, n, d_cu_len);
        else if (d == 64 && br == 64) launch_fa2<64, 64>(q, k, v, o, n, d_cu_len);
        else if (d == 128 && br == 16) launch_fa2<128, 16>(q, k, v, o, n, d_cu_len);
        else if (d == 128 && br == 32) launch_fa2<128, 32>(q, k, v, o, n, d_cu_len);
        else if (d == 128 && br == 64) launch_fa2<128, 64>(q, k, v, o, n, d_cu_len);
        else {
            fprintf(stderr, "Unsupported FA2 shape d=%d Br=%d\n", d, br);
            exit(EXIT_FAILURE);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

static void compare_output(
    const float *actual,
    const float *expected,
    size_t count,
    const char *name
) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    double sum_abs = 0.0;
    for (size_t i = 0; i < count; ++i) {
        float abs_error = fabsf(actual[i] - expected[i]);
        float denominator = fmaxf(fabsf(expected[i]), 1.0e-6f);
        float rel_error = abs_error / denominator;
        max_abs = fmaxf(max_abs, abs_error);
        max_rel = fmaxf(max_rel, rel_error);
        sum_abs += (double)abs_error;
    }

    float mean_abs = (float)(sum_abs / (double)count);
    // Relative error is not a useful gate for output values close to zero.
    // Use absolute and mean error for the BF16/Tensor Core comparison, while
    // still reporting the maximum relative error for diagnosis.
    int passed = max_abs <= 5.0e-2f && mean_abs <= 5.0e-3f;
    printf("verify %-7s max_abs=%.6e mean_abs=%.6e max_rel=%.6e %s\n",
           name, max_abs, mean_abs, max_rel, passed ? "PASS" : "FAIL");
    if (!passed) verification_failed = 1;
}

static void run_verify(
    int kernel,
    const __nv_bfloat16 *h_q,
    const __nv_bfloat16 *h_k,
    const __nv_bfloat16 *h_v,
    __nv_bfloat16 *d_q,
    __nv_bfloat16 *d_k,
    __nv_bfloat16 *d_v,
    float *d_s,
    float *d_p,
    float *d_o,
    const int *d_cu_len,
    int n,
    int d,
    int br
) {
    size_t output_count = (size_t)n * (size_t)d;
    float *reference = (float *)calloc(output_count, sizeof(float));
    float *actual = (float *)malloc(output_count * sizeof(float));
    if (reference == NULL || actual == NULL) {
        fprintf(stderr, "CPU allocation failed for verification\n");
        exit(EXIT_FAILURE);
    }

    attention_cpu(h_q, h_k, h_v, reference, n, d, 0);
    launch_attention(kernel, d_q, d_k, d_v, d_s, d_p, d_o, n, d, br, d_cu_len);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual, d_o, output_count * sizeof(float), cudaMemcpyDeviceToHost));
    compare_output(actual, reference, output_count,
                   kernel == 0 ? "naive" : (kernel == 1 ? "fa1" : "fa2"));

    free(reference);
    free(actual);
}

static float run_benchmark(
    int kernel,
    const __nv_bfloat16 *d_q,
    const __nv_bfloat16 *d_k,
    const __nv_bfloat16 *d_v,
    float *d_s,
    float *d_p,
    float *d_o,
    const int *d_cu_len,
    int n,
    int d,
    int br,
    int warmup,
    int iterations
) {
    for (int i = 0; i < warmup; ++i) {
        launch_attention(kernel, d_q, d_k, d_v, d_s, d_p, d_o, n, d, br, d_cu_len);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        launch_attention(kernel, d_q, d_k, d_v, d_s, d_p, d_o, n, d, br, d_cu_len);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / (float)iterations;
}

static void run_profile(
    int kernel,
    const __nv_bfloat16 *d_q,
    const __nv_bfloat16 *d_k,
    const __nv_bfloat16 *d_v,
    float *d_s,
    float *d_p,
    float *d_o,
    const int *d_cu_len,
    int n,
    int d,
    int br
) {
    // Nsight Compute replays this launch for its metric passes. Do not wrap it
    // in CUDA events: that elapsed time would measure profiler replay overhead.
    launch_attention(kernel, d_q, d_k, d_v, d_s, d_p, d_o, n, d, br, d_cu_len);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static int parse_positive(const char *text, const char *name) {
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || value <= 0 || value > 2147483647L) {
        fprintf(stderr, "Invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

static int parse_nonnegative(const char *text, const char *name) {
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || value < 0 || value > 2147483647L) {
        fprintf(stderr, "Invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

int main(int argc, char **argv) {
    if (argc >= 6 && strcmp(argv[1], "profile_batch") == 0) {
        const char *kernel = argv[2];
        const char *warmup = argv[3];
        const char *iterations = argv[4];
        int case_args = argc - 5;
        if (case_args < 3 || case_args % 3 != 0) {
            fprintf(stderr, "Usage: %s profile_batch <all|naive|fa1|fa2> warmup iterations N HEAD_DIM Br [... ]\n", argv[0]);
            return EXIT_FAILURE;
        }
        for (int i = 5; i < argc; i += 3) {
            char *const child_argv[] = {
                argv[0], (char *)"profile", (char *)kernel,
                argv[i], argv[i + 1], argv[i + 2],
                (char *)warmup, (char *)iterations, NULL
            };
            pid_t child = fork();
            if (child == 0) {
                execv(argv[0], child_argv);
                perror("execv profile case");
                _exit(EXIT_FAILURE);
            }
            if (child < 0) {
                perror("fork profile case");
                return EXIT_FAILURE;
            }
            int status = 0;
            if (waitpid(child, &status, 0) < 0 ||
                !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                fprintf(stderr, "profile case failed: N=%s d=%s Br=%s\n",
                        argv[i], argv[i + 1], argv[i + 2]);
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }
    if (argc < 6 || argc > 8) {
        fprintf(stderr, "Usage: %s <verify|benchmark|profile> <all|naive|fa1|fa2> N HEAD_DIM Br [warmup] [iterations]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *mode = argv[1];
    const char *kernel = argv[2];
    int n = parse_positive(argv[3], "N");
    int d = parse_positive(argv[4], "HEAD_DIM");
    int br = parse_positive(argv[5], "Br");
    int warmup = argc >= 7 ? parse_nonnegative(argv[6], "warmup") : 10;
    int iterations = argc >= 8 ? parse_positive(argv[7], "iterations") : 100;
    int run_naive = strcmp(kernel, "all") == 0 || strcmp(kernel, "naive") == 0;
    int run_fa1 = strcmp(kernel, "all") == 0 || strcmp(kernel, "fa1") == 0;
    int run_fa2 = strcmp(kernel, "all") == 0 || strcmp(kernel, "fa2") == 0;

    if ((!run_naive && !run_fa1 && !run_fa2) ||
        (strcmp(mode, "verify") != 0 && strcmp(mode, "benchmark") != 0 && strcmp(mode, "profile") != 0)) {
        fprintf(stderr, "Usage: %s <verify|benchmark|profile> <all|naive|fa1|fa2> N HEAD_DIM Br [warmup] [iterations]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (run_fa1 && d != 64 && d != 128) {
        fprintf(stderr, "FA1 supports HEAD_DIM=64 or 128, got %d\n", d);
        return EXIT_FAILURE;
    }
    if (run_fa1 && br != 16 && br != 32 && br != 64) {
        fprintf(stderr, "FA1 supports Br=16, 32, or 64, got %d\n", br);
        return EXIT_FAILURE;
    }
    if (run_fa1 && n % 64 != 0) {
        fprintf(stderr, "FA1 benchmark requires N to be a multiple of 64, got N=%d\n", n);
        return EXIT_FAILURE;
    }
    if (run_fa1 && n % br != 0) {
        fprintf(stderr, "FA1 benchmark requires N to be a multiple of Br, got N=%d Br=%d\n", n, br);
        return EXIT_FAILURE;
    }
    if (run_fa2 && d != 64 && d != 128) {
        fprintf(stderr, "FA2 supports HEAD_DIM=64 or 128, got %d\n", d);
        return EXIT_FAILURE;
    }
    if (run_fa2 && (br != 16 && br != 32 && br != 64)) {
        fprintf(stderr, "FA2 supports Br=16, 32, or 64, got %d\n", br);
        return EXIT_FAILURE;
    }
    if (run_fa2 && (n % 64 != 0 || n % br != 0)) {
        fprintf(stderr, "FA2 benchmark requires N to be a multiple of 64 and Br, got N=%d Br=%d\n", n, br);
        return EXIT_FAILURE;
    }
    if (strcmp(mode, "verify") == 0 && argc >= 7) {
        fprintf(stderr, "verify does not take warmup/iterations\n");
        return EXIT_FAILURE;
    }

    size_t qkv_count = (size_t)n * (size_t)d;
    size_t matrix_count = (size_t)n * (size_t)n;
    __nv_bfloat16 *h_q = (__nv_bfloat16 *)malloc(qkv_count * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_k = (__nv_bfloat16 *)malloc(qkv_count * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_v = (__nv_bfloat16 *)malloc(qkv_count * sizeof(__nv_bfloat16));
    if (h_q == NULL || h_k == NULL || h_v == NULL) {
        fprintf(stderr, "Host allocation failed for N=%d d=%d\n", n, d);
        return EXIT_FAILURE;
    }
    fill_input(h_q, qkv_count, 1);
    fill_input(h_k, qkv_count, 2);
    fill_input(h_v, qkv_count, 3);

    __nv_bfloat16 *d_q;
    __nv_bfloat16 *d_k;
    __nv_bfloat16 *d_v;
    float *d_s;
    float *d_p;
    float *d_o;
    int *d_cu_len;
    CUDA_CHECK(cudaMalloc(&d_q, qkv_count * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, qkv_count * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, qkv_count * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_s, matrix_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p, matrix_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o, qkv_count * sizeof(float)));
    int h_cu_len[2] = {0, n};
    CUDA_CHECK(cudaMalloc(&d_cu_len, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cu_len, h_cu_len, 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, qkv_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k, qkv_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, qkv_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    if (strcmp(mode, "verify") == 0) {
        if (run_naive) run_verify(0, h_q, h_k, h_v, d_q, d_k, d_v, d_s, d_p, d_o, d_cu_len, n, d, br);
        if (run_fa1) run_verify(1, h_q, h_k, h_v, d_q, d_k, d_v, d_s, d_p, d_o, d_cu_len, n, d, br);
        if (run_fa2) run_verify(2, h_q, h_k, h_v, d_q, d_k, d_v, d_s, d_p, d_o, d_cu_len, n, d, br);
    } else if (strcmp(mode, "benchmark") == 0) {
        // The CPU reference and naive path are O(N^2) in storage and work.
        // Keep correctness-before-benchmark for normal sizes, but allow long
        // sequence performance runs without spending minutes on the reference.
        if (n <= 4096) {
            printf("correctness before benchmark\n");
            if (run_naive) run_verify(0, h_q, h_k, h_v, d_q, d_k, d_v, d_s, d_p, d_o, d_cu_len, n, d, br);
            if (run_fa1) run_verify(1, h_q, h_k, h_v, d_q, d_k, d_v, d_s, d_p, d_o, d_cu_len, n, d, br);
            if (run_fa2) run_verify(2, h_q, h_k, h_v, d_q, d_k, d_v, d_s, d_p, d_o, d_cu_len, n, d, br);
        } else {
            printf("correctness skipped for long sequence N=%d (CPU/naive O(N^2)); benchmark only\n", n);
        }
        if (run_naive) {
            float ms = run_benchmark(0, d_q, d_k, d_v, d_s, d_p, d_o, d_cu_len, n, d, br, warmup, iterations);
            printf("benchmark naive  N=%d d=%d Br=%d time_ms=%.6f\n", n, d, br, ms);
        }
        if (run_fa1) {
            float ms = run_benchmark(1, d_q, d_k, d_v, d_s, d_p, d_o, d_cu_len, n, d, br, warmup, iterations);
            printf("benchmark fa1    N=%d d=%d Br=%d Bc=64 time_ms=%.6f\n", n, d, br, ms);
        }
        if (run_fa2) {
            float ms = run_benchmark(2, d_q, d_k, d_v, d_s, d_p, d_o, d_cu_len, n, d, br, warmup, iterations);
            printf("benchmark fa2    N=%d d=%d Br=%d Bc=64 time_ms=%.6f\n", n, d, br, ms);
        }
    } else {
        if (run_naive) run_profile(0, d_q, d_k, d_v, d_s, d_p, d_o, d_cu_len, n, d, br);
        if (run_fa1) run_profile(1, d_q, d_k, d_v, d_s, d_p, d_o, d_cu_len, n, d, br);
        if (run_fa2) run_profile(2, d_q, d_k, d_v, d_s, d_p, d_o, d_cu_len, n, d, br);
    }

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_cu_len));
    free(h_q);
    free(h_k);
    free(h_v);
    return verification_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
