#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define WARP_SIZE 32

void check_cuda(cudaError_t status, const char* file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

void check_cudnn(cudnnStatus_t status, const char* file, int line) {
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "cuDNN error at %s:%d: %s\n", file, line, cudnnGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) check_cuda((call), __FILE__, __LINE__)
#define CUDNN_CHECK(call) check_cudnn((call), __FILE__, __LINE__)

cudnnBackendDescriptor_t create_tensor_descriptor(int64_t uid, const int64_t* dimensions, const int64_t* strides, int64_t dimension_count, bool by_value) {
    cudnnBackendDescriptor_t descriptor = nullptr;
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    int64_t alignment = sizeof(float);
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &descriptor));
    CUDNN_CHECK(cudnnBackendSetAttribute(descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &data_type));
    CUDNN_CHECK(cudnnBackendSetAttribute(descriptor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, dimension_count, dimensions));
    CUDNN_CHECK(cudnnBackendSetAttribute(descriptor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, dimension_count, strides));
    CUDNN_CHECK(cudnnBackendSetAttribute(descriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid));
    CUDNN_CHECK(cudnnBackendSetAttribute(descriptor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    if (by_value) CUDNN_CHECK(cudnnBackendSetAttribute(descriptor, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1, &by_value));
    CUDNN_CHECK(cudnnBackendFinalize(descriptor));
    return descriptor;
}

void destroy_backend_descriptor(cudnnBackendDescriptor_t descriptor) {
    if (descriptor != nullptr) cudnnBackendDestroyDescriptor(descriptor);
}

class CudnnRmsNorm {
public:
    CudnnRmsNorm(int N, int M, const float* input, float* output, const float* gamma, float epsilon) : handle_(nullptr), x_desc_(nullptr), y_desc_(nullptr), scale_desc_(nullptr), bias_desc_(nullptr), epsilon_desc_(nullptr), inv_variance_desc_(nullptr), operation_(nullptr), operation_graph_(nullptr), engine_config_(nullptr), execution_plan_(nullptr), variant_pack_(nullptr), workspace_(nullptr), d_bias_(nullptr), d_inv_variance_(nullptr), workspace_size_(0), epsilon_(epsilon) {
        CUDNN_CHECK(cudnnCreate(&handle_));

        int64_t tensor_dimensions[4] = {N, M, 1, 1};
        int64_t tensor_strides[4] = {M, 1, 1, 1};
        int64_t parameter_dimensions[4] = {1, M, 1, 1};
        int64_t parameter_strides[4] = {M, 1, 1, 1};
        int64_t scalar_dimensions[4] = {1, 1, 1, 1};
        int64_t scalar_strides[4] = {1, 1, 1, 1};
        int64_t statistic_dimensions[4] = {N, 1, 1, 1};
        int64_t statistic_strides[4] = {1, 1, 1, 1};
        x_desc_ = create_tensor_descriptor(0, tensor_dimensions, tensor_strides, 4, false);
        y_desc_ = create_tensor_descriptor(1, tensor_dimensions, tensor_strides, 4, false);
        scale_desc_ = create_tensor_descriptor(2, parameter_dimensions, parameter_strides, 4, false);
        bias_desc_ = create_tensor_descriptor(3, parameter_dimensions, parameter_strides, 4, false);
        epsilon_desc_ = create_tensor_descriptor(4, scalar_dimensions, scalar_strides, 4, true);
        inv_variance_desc_ = create_tensor_descriptor(5, statistic_dimensions, statistic_strides, 4, false);

        cudnnBackendNormMode_t mode = CUDNN_RMS_NORM;
        cudnnBackendNormFwdPhase_t phase = CUDNN_NORM_FWD_TRAINING;
        CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR, &operation_));
        CUDNN_CHECK(cudnnBackendSetAttribute(operation_, CUDNN_ATTR_OPERATION_NORM_FWD_MODE, CUDNN_TYPE_NORM_MODE, 1, &mode));
        CUDNN_CHECK(cudnnBackendSetAttribute(operation_, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE, CUDNN_TYPE_NORM_FWD_PHASE, 1, &phase));
        CUDNN_CHECK(cudnnBackendSetAttribute(operation_, CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x_desc_));
        CUDNN_CHECK(cudnnBackendSetAttribute(operation_, CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y_desc_));
        CUDNN_CHECK(cudnnBackendSetAttribute(operation_, CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &scale_desc_));
        CUDNN_CHECK(cudnnBackendSetAttribute(operation_, CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &bias_desc_));
        CUDNN_CHECK(cudnnBackendSetAttribute(operation_, CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &epsilon_desc_));
        CUDNN_CHECK(cudnnBackendSetAttribute(operation_, CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &inv_variance_desc_));
        CUDNN_CHECK(cudnnBackendFinalize(operation_));

        CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &operation_graph_));
        CUDNN_CHECK(cudnnBackendSetAttribute(operation_graph_, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle_));
        CUDNN_CHECK(cudnnBackendSetAttribute(operation_graph_, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &operation_));
        CUDNN_CHECK(cudnnBackendFinalize(operation_graph_));

        create_execution_plan();
        CUDA_CHECK(cudaMalloc(&d_bias_, (size_t)M * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_bias_, 0, (size_t)M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_inv_variance_, (size_t)N * sizeof(float)));
        create_variant_pack(input, output, gamma);
    }

    ~CudnnRmsNorm() {
        destroy_backend_descriptor(variant_pack_);
        destroy_backend_descriptor(execution_plan_);
        destroy_backend_descriptor(engine_config_);
        destroy_backend_descriptor(operation_graph_);
        destroy_backend_descriptor(operation_);
        destroy_backend_descriptor(inv_variance_desc_);
        destroy_backend_descriptor(epsilon_desc_);
        destroy_backend_descriptor(bias_desc_);
        destroy_backend_descriptor(scale_desc_);
        destroy_backend_descriptor(y_desc_);
        destroy_backend_descriptor(x_desc_);
        if (d_inv_variance_ != nullptr) cudaFree(d_inv_variance_);
        if (d_bias_ != nullptr) cudaFree(d_bias_);
        if (workspace_ != nullptr) cudaFree(workspace_);
        if (handle_ != nullptr) cudnnDestroy(handle_);
    }

    void launch() const { CUDNN_CHECK(cudnnBackendExecute(handle_, execution_plan_, variant_pack_)); }

private:
    void create_execution_plan() {
        const int maximum_config_count = 16;
        cudnnBackendDescriptor_t heuristic = nullptr;
        cudnnBackendDescriptor_t configs[maximum_config_count];
        for (int i = 0; i < maximum_config_count; ++i) {
            configs[i] = nullptr;
            CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &configs[i]));
        }
        cudnnBackendHeurMode_t heuristic_mode = CUDNN_HEUR_MODE_INSTANT;
        CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heuristic));
        CUDNN_CHECK(cudnnBackendSetAttribute(heuristic, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &operation_graph_));
        CUDNN_CHECK(cudnnBackendSetAttribute(heuristic, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &heuristic_mode));
        CUDNN_CHECK(cudnnBackendFinalize(heuristic));
        int64_t config_count = 0;
        CUDNN_CHECK(cudnnBackendGetAttribute(heuristic, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, maximum_config_count, &config_count, configs));

        for (int i = 0; i < config_count; ++i) {
            cudnnBackendDescriptor_t candidate = nullptr;
            CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &candidate));
            cudnnStatus_t status = cudnnBackendSetAttribute(candidate, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &configs[i]);
            if (status == CUDNN_STATUS_SUCCESS) status = cudnnBackendFinalize(candidate);
            if (status == CUDNN_STATUS_SUCCESS) {
                execution_plan_ = candidate;
                engine_config_ = configs[i];
                configs[i] = nullptr;
                break;
            }
            destroy_backend_descriptor(candidate);
        }

        for (int i = 0; i < maximum_config_count; ++i) destroy_backend_descriptor(configs[i]);
        destroy_backend_descriptor(heuristic);
        if (execution_plan_ == nullptr) {
            fprintf(stderr, "cuDNN did not return a usable RMSNorm execution plan\n");
            exit(EXIT_FAILURE);
        }

        int64_t element_count = 0;
        CUDNN_CHECK(cudnnBackendGetAttribute(execution_plan_, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &element_count, &workspace_size_));
        if (workspace_size_ > 0) CUDA_CHECK(cudaMalloc(&workspace_, (size_t)workspace_size_));
    }

    void create_variant_pack(const float* input, float* output, const float* gamma) {
        int64_t uids[6] = {0, 1, 2, 3, 4, 5};
        void* data_pointers[6] = {(void*)input, (void*)output, (void*)gamma, d_bias_, (void*)&epsilon_, d_inv_variance_};
        CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &variant_pack_));
        CUDNN_CHECK(cudnnBackendSetAttribute(variant_pack_, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 6, uids));
        CUDNN_CHECK(cudnnBackendSetAttribute(variant_pack_, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 6, data_pointers));
        CUDNN_CHECK(cudnnBackendSetAttribute(variant_pack_, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace_));
        CUDNN_CHECK(cudnnBackendFinalize(variant_pack_));
    }

    cudnnHandle_t handle_;
    cudnnBackendDescriptor_t x_desc_;
    cudnnBackendDescriptor_t y_desc_;
    cudnnBackendDescriptor_t scale_desc_;
    cudnnBackendDescriptor_t bias_desc_;
    cudnnBackendDescriptor_t epsilon_desc_;
    cudnnBackendDescriptor_t inv_variance_desc_;
    cudnnBackendDescriptor_t operation_;
    cudnnBackendDescriptor_t operation_graph_;
    cudnnBackendDescriptor_t engine_config_;
    cudnnBackendDescriptor_t execution_plan_;
    cudnnBackendDescriptor_t variant_pack_;
    void* workspace_;
    float* d_bias_;
    float* d_inv_variance_;
    int64_t workspace_size_;
    float epsilon_;
};

__global__ void rmsnorm_shared(const float *input, float *output, int N, int M, const float* gamma, float epsilon) {
    int row_idx = blockIdx.x;
    if (row_idx >= N) return;

    int local_thread_idx = threadIdx.x;
    int warp_idx = local_thread_idx / WARP_SIZE;
    int lane_idx = local_thread_idx % WARP_SIZE;

    extern __shared__ float shared_dyn[];
    float *shared_row = shared_dyn;
    float *shared_warp = shared_dyn + M;

    const float *input_row = input + row_idx * M;
    float *output_row = output + row_idx * M;

    // global -> shared & sum_squares
    float thread_sum_squares = 0;
    for (int i = local_thread_idx; i < M; i+=blockDim.x) {
        float val = input_row[i];
        thread_sum_squares += val * val;
        shared_row[i] = val;
    }
    // warp_sum_squares 内规约
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_sum_squares += __shfl_down_sync(0xffffffff, thread_sum_squares, offset);
    }
    if (lane_idx == 0) shared_warp[warp_idx] = thread_sum_squares;
    __syncthreads();
    // warp_sum_squares 间规约
    if (warp_idx == 0) {
        thread_sum_squares = 0;
        if (lane_idx < blockDim.x / WARP_SIZE) thread_sum_squares = shared_warp[lane_idx];
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            thread_sum_squares += __shfl_down_sync(0xffffffff, thread_sum_squares, offset);
        }
        if (lane_idx == 0) shared_warp[0] = rsqrtf(thread_sum_squares / static_cast<float>(M) + epsilon);
    }
    __syncthreads();
    float inv_std = shared_warp[0];

    // 写回
    for (int i = local_thread_idx; i < M; i += blockDim.x) {
        float centered = shared_row[i];
        output_row[i] = centered * inv_std * gamma[i];
    }
}

template <int BSIZE>
void rmsnorm_shared_launcher(const float *input, float *output, int N, int M, const float* gamma, float epsilon) {
    if (M > 4096) return;
    dim3 grid_size(N);
    dim3 block_size(BSIZE);
    int num_warps = BSIZE / WARP_SIZE;
    size_t shared_bytes = (static_cast<size_t>(M) + num_warps) * sizeof(float);
    rmsnorm_shared<<<grid_size, block_size, shared_bytes>>>(input, output, N, M, gamma, epsilon);
}

template<int BLOCK_SIZE, int BUCKET_SIZE>
__global__ void rmsnorm_register(const float *input, float *output, int N, int M, const float* gamma, float epsilon) {
    int row_idx = blockIdx.x;
    if (row_idx >= N) return;

    int local_thread_idx = threadIdx.x;
    int warp_idx = local_thread_idx / WARP_SIZE;
    int lane_idx = local_thread_idx % WARP_SIZE;

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    constexpr int REG_SIZE = (BUCKET_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    __shared__ float shared_warp[NUM_WARPS];
    float reg_input[REG_SIZE];

    const float *input_row = input + row_idx * M;
    float *output_row = output + row_idx * M;

    // global -> register & sum_squares
    float thread_sum_squares = 0;
    #pragma unroll
    for (int reg_idx = 0; reg_idx < REG_SIZE; reg_idx++) {
        int col_idx = BLOCK_SIZE  * reg_idx + local_thread_idx;
        if (col_idx < M) {
            float value = input_row[col_idx];
            reg_input[reg_idx] = value;
            thread_sum_squares += value * value;
        }
    }
    // warp_sum_squares 内规约
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_sum_squares += __shfl_down_sync(0xffffffff, thread_sum_squares, offset);
    }
    if (lane_idx == 0) shared_warp[warp_idx] = thread_sum_squares;
    __syncthreads();
    // warp_sum_squares 间规约
    if (warp_idx == 0) {
        thread_sum_squares = 0;
        if (lane_idx < NUM_WARPS) thread_sum_squares = shared_warp[lane_idx];
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            thread_sum_squares += __shfl_down_sync(0xffffffff, thread_sum_squares, offset);
        }
        if (lane_idx == 0) shared_warp[0] = rsqrtf(thread_sum_squares / static_cast<float>(M) + epsilon);
    }
    __syncthreads();
    float inv_std = shared_warp[0];

    // 写回
    #pragma unroll
    for (int reg_idx = 0; reg_idx < REG_SIZE; reg_idx++) {
        int col_idx = BLOCK_SIZE  * reg_idx + local_thread_idx;
        if (col_idx < M) {
            output_row[col_idx] = reg_input[reg_idx] * inv_std * gamma[col_idx];
        }
    }
}

template <int BSIZE>
void rmsnorm_register_launcher(const float *input, float *output, int N, int M, const float* gamma, float epsilon) {
    if (M > 4096) return;
    dim3 grid_size(N);
    dim3 block_size(BSIZE);
    if (M <= 128) {
        rmsnorm_register<BSIZE, 128><<<grid_size, block_size>>>(input, output, N, M, gamma, epsilon);
    }
    else if (M <= 256) {
        rmsnorm_register<BSIZE, 256><<<grid_size, block_size>>>(input, output, N, M, gamma, epsilon);
    }
    else if (M <= 512) {
        rmsnorm_register<BSIZE, 512><<<grid_size, block_size>>>(input, output, N, M, gamma, epsilon);
    }
    else if (M <= 1024) {
        rmsnorm_register<BSIZE, 1024><<<grid_size, block_size>>>(input, output, N, M, gamma, epsilon);
    }
    else if (M <= 2048) {
        rmsnorm_register<BSIZE, 2048><<<grid_size, block_size>>>(input, output, N, M, gamma, epsilon);
    }
    else if (M <= 4096) {
        rmsnorm_register<BSIZE, 4096><<<grid_size, block_size>>>(input, output, N, M, gamma, epsilon);
    }
    else {
        return;
    }
}

struct TestContext {
    int N;
    int M;
    size_t element_count;
    size_t tensor_bytes;
    size_t parameter_bytes;
    float epsilon;
    float* input;
    float* output;
    float* gamma;
    float* d_input;
    float* d_output;
    float* d_gamma;

    TestContext(int N_value, int M_value) : N(N_value), M(M_value), element_count((size_t)N_value * (size_t)M_value), tensor_bytes(element_count * sizeof(float)), parameter_bytes((size_t)M_value * sizeof(float)), epsilon(1e-5f), input((float*)malloc(tensor_bytes)), output((float*)malloc(tensor_bytes)), gamma((float*)malloc(parameter_bytes)), d_input(nullptr), d_output(nullptr), d_gamma(nullptr) {
        if (input == nullptr || output == nullptr || gamma == nullptr) {
            fprintf(stderr, "Host memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < element_count; ++i) input[i] = (float)((int)((i * 37u + 17u) % 1001u) - 500) / 100.0f;
        for (int i = 0; i < M; ++i) gamma[i] = 0.5f + (float)((i * 13) % 101) / 100.0f;
        CUDA_CHECK(cudaMalloc(&d_input, tensor_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, tensor_bytes));
        CUDA_CHECK(cudaMalloc(&d_gamma, parameter_bytes));
        CUDA_CHECK(cudaMemcpy(d_input, input, tensor_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma, gamma, parameter_bytes, cudaMemcpyHostToDevice));
    }

    ~TestContext() {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_gamma);
        free(input);
        free(output);
        free(gamma);
    }

    TestContext(const TestContext&) = delete;
    TestContext& operator=(const TestContext&) = delete;
};

static double max_double(double a, double b) { return a > b ? a : b; }
static double abs_double(double value) { return value < 0.0 ? -value : value; }

double* rmsnorm_cpu_reference(const TestContext& context) {
    double* reference = (double*)malloc(context.element_count * sizeof(double));
    if (reference == nullptr) {
        fprintf(stderr, "CPU reference allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int row = 0; row < context.N; ++row) {
        size_t row_offset = (size_t)row * (size_t)context.M;
        double mean_square = 0.0;
        for (int col = 0; col < context.M; ++col) {
            double value = (double)context.input[row_offset + col];
            mean_square += value * value;
        }
        mean_square /= (double)context.M;
        double inv_rms = 1.0 / sqrt(mean_square + (double)context.epsilon);
        for (int col = 0; col < context.M; ++col) reference[row_offset + col] = (double)context.input[row_offset + col] * inv_rms * (double)context.gamma[col];
    }
    return reference;
}

struct ValidationResult {
    bool passed;
    double max_abs_error;
    double max_rel_error;
    size_t mismatch_count;
};

template <typename Launcher>
ValidationResult validate_kernel(TestContext& context, const double* reference, Launcher launcher) {
    const double absolute_tolerance = 2e-5;
    const double relative_tolerance = 2e-4;
    for (size_t i = 0; i < context.element_count; ++i) context.output[i] = NAN;
    CUDA_CHECK(cudaMemcpy(context.d_output, context.output, context.tensor_bytes, cudaMemcpyHostToDevice));
    launcher();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(context.output, context.d_output, context.tensor_bytes, cudaMemcpyDeviceToHost));
    ValidationResult result = {true, 0.0, 0.0, 0};
    for (size_t i = 0; i < context.element_count; ++i) {
        double actual = (double)context.output[i];
        double expected = reference[i];
        double abs_error = abs_double(actual - expected);
        double expected_abs = abs_double(expected);
        double tolerance = absolute_tolerance + relative_tolerance * expected_abs;
        result.max_abs_error = max_double(result.max_abs_error, abs_error);
        if (expected_abs > 1e-12) result.max_rel_error = max_double(result.max_rel_error, abs_error / expected_abs);
        if (!isfinite(actual) || abs_error > tolerance) ++result.mismatch_count;
    }
    result.passed = result.mismatch_count == 0;
    return result;
}

struct BenchmarkResult { float median_us; float min_us; float p90_us; };

template <typename Launcher>
BenchmarkResult benchmark_kernel(Launcher launcher, int warmup_iterations, int measured_iterations, int samples) {
    for (int i = 0; i < warmup_iterations; ++i) launcher();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float* sample_us = (float*)malloc((size_t)samples * sizeof(float));
    if (sample_us == nullptr) {
        fprintf(stderr, "Benchmark sample allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int sample = 0; sample < samples; ++sample) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int iteration = 0; iteration < measured_iterations; ++iteration) launcher();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        sample_us[sample] = elapsed_ms * 1000.0f / (float)measured_iterations;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    for (int i = 1; i < samples; ++i) {
        float value = sample_us[i];
        int j = i - 1;
        while (j >= 0 && sample_us[j] > value) { sample_us[j + 1] = sample_us[j]; --j; }
        sample_us[j + 1] = value;
    }
    int middle = samples / 2;
    float median = samples % 2 == 0 ? (sample_us[middle - 1] + sample_us[middle]) / 2.0f : sample_us[middle];
    int p90_index = (9 * samples + 9) / 10 - 1;
    BenchmarkResult result = {median, sample_us[0], sample_us[p90_index]};
    free(sample_us);
    return result;
}

enum RunMode { RUN_VERIFY, RUN_BENCHMARK, RUN_PROFILE };

struct Options {
    RunMode mode;
    const char* target;
    int N;
    int M;
    int warmup_iterations;
    int measured_iterations;
    int samples;
};

bool parse_positive_int(const char* text, int* value) {
    errno = 0;
    char* end = nullptr;
    long parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX) return false;
    *value = (int)parsed;
    return true;
}

void print_usage(const char* program) {
    fprintf(stderr, "Usage:\n  %s <verify|benchmark|profile> <all|kernel> <N> <M> [warmup] [iterations] [samples]\n  %s <N> <M>  # profile all (legacy)\n\nKernels:\n  cudnn, shared_128, shared_256, shared_512, register_128, register_256, register_512\n", program, program);
}

bool parse_options(int argc, char* argv[], Options* options) {
    options->mode = RUN_PROFILE;
    options->target = "all";
    options->warmup_iterations = 10;
    options->measured_iterations = 100;
    options->samples = 10;
    if (argc == 3) return parse_positive_int(argv[1], &options->N) && parse_positive_int(argv[2], &options->M);
    if (argc < 5 || argc > 8) return false;
    if (strcmp(argv[1], "verify") == 0) options->mode = RUN_VERIFY;
    else if (strcmp(argv[1], "benchmark") == 0) options->mode = RUN_BENCHMARK;
    else if (strcmp(argv[1], "profile") == 0) options->mode = RUN_PROFILE;
    else return false;
    options->target = argv[2];
    if (!parse_positive_int(argv[3], &options->N) || !parse_positive_int(argv[4], &options->M)) return false;
    if (options->mode != RUN_BENCHMARK && argc != 5) return false;
    if ((argc >= 6 && !parse_positive_int(argv[5], &options->warmup_iterations)) || (argc >= 7 && !parse_positive_int(argv[6], &options->measured_iterations)) || (argc >= 8 && !parse_positive_int(argv[7], &options->samples))) return false;
    return true;
}

template <typename Launcher>
void run_candidate(const char* name, bool supported, const Options& options, TestContext& context, const double* reference, bool* matched, bool* succeeded, Launcher launcher) {
    if (strcmp(options.target, "all") != 0 && strcmp(options.target, name) != 0) return;
    *matched = true;
    if (!supported) {
        printf("%-20sSKIP (unsupported)\n", name);
        if (strcmp(options.target, "all") != 0) *succeeded = false;
        return;
    }
    if (options.mode == RUN_VERIFY) {
        ValidationResult result = validate_kernel(context, reference, launcher);
        printf("%-20s%-8smax_abs=% .3e  max_rel=% .3e  mismatches=%zu\n", name, result.passed ? "PASS" : "FAIL", result.max_abs_error, result.max_rel_error, result.mismatch_count);
        *succeeded = *succeeded && result.passed;
        return;
    }
    if (options.mode == RUN_BENCHMARK) {
        BenchmarkResult result = benchmark_kernel(launcher, options.warmup_iterations, options.measured_iterations, options.samples);
        printf("%-20s%14.3f%14.3f%14.3f\n", name, result.median_us, result.min_us, result.p90_us);
        return;
    }
    launcher();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char* argv[]) {
    Options options;
    if (!parse_options(argc, argv, &options)) { print_usage(argv[0]); return EXIT_FAILURE; }
    TestContext context(options.N, options.M);
    CudnnRmsNorm cudnn(options.N, options.M, context.d_input, context.d_output, context.d_gamma, context.epsilon);
    double* reference = nullptr;
    if (options.mode == RUN_VERIFY) {
        reference = rmsnorm_cpu_reference(context);
        printf("%-20s%-8sErrors\n", "Kernel", "Status");
    } else if (options.mode == RUN_BENCHMARK) {
        printf("warmup=%d, iterations=%d, samples=%d\n", options.warmup_iterations, options.measured_iterations, options.samples);
        printf("%-20s%14s%14s%14s\n", "Kernel", "Median (us)", "Min (us)", "P90 (us)");
    }
    bool matched = false;
    bool succeeded = true;
    bool custom_supported = options.M <= 4096;
    const double* reference_ptr = options.mode == RUN_VERIFY ? reference : nullptr;
    run_candidate("cudnn", true, options, context, reference_ptr, &matched, &succeeded, [&] { cudnn.launch(); });
    run_candidate("shared_128", custom_supported, options, context, reference_ptr, &matched, &succeeded, [&] { rmsnorm_shared_launcher<128>(context.d_input, context.d_output, context.N, context.M, context.d_gamma, context.epsilon); });
    run_candidate("shared_256", custom_supported, options, context, reference_ptr, &matched, &succeeded, [&] { rmsnorm_shared_launcher<256>(context.d_input, context.d_output, context.N, context.M, context.d_gamma, context.epsilon); });
    run_candidate("shared_512", custom_supported, options, context, reference_ptr, &matched, &succeeded, [&] { rmsnorm_shared_launcher<512>(context.d_input, context.d_output, context.N, context.M, context.d_gamma, context.epsilon); });
    run_candidate("register_128", custom_supported, options, context, reference_ptr, &matched, &succeeded, [&] { rmsnorm_register_launcher<128>(context.d_input, context.d_output, context.N, context.M, context.d_gamma, context.epsilon); });
    run_candidate("register_256", custom_supported, options, context, reference_ptr, &matched, &succeeded, [&] { rmsnorm_register_launcher<256>(context.d_input, context.d_output, context.N, context.M, context.d_gamma, context.epsilon); });
    run_candidate("register_512", custom_supported, options, context, reference_ptr, &matched, &succeeded, [&] { rmsnorm_register_launcher<512>(context.d_input, context.d_output, context.N, context.M, context.d_gamma, context.epsilon); });
    if (!matched) {
        fprintf(stderr, "Unknown kernel: %s\n", options.target);
        print_usage(argv[0]);
        free(reference);
        return EXIT_FAILURE;
    }
    free(reference);
    return succeeded ? EXIT_SUCCESS : EXIT_FAILURE;
}
