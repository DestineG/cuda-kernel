#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
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

float* create_input(int batch, int num_classes) {
    size_t num_elements =
        static_cast<size_t>(batch) * static_cast<size_t>(num_classes);

    float* input = (float*)malloc(num_elements * sizeof(float));
    for (size_t i = 0; i < num_elements; ++i) {
        input[i] = (float)((int)((i * 37u + 17u) % 1001u) - 500) / 100.0f;
    }

    return input;
}

class CudnnSoftmax {
public:
    CudnnSoftmax(int batch, int num_classes) {
        CUDNN_CHECK(cudnnCreate(&handle_));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensor_desc_));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, num_classes, 1, 1));
    }

    ~CudnnSoftmax() {
        cudnnDestroyTensorDescriptor(tensor_desc_);
        cudnnDestroy(handle_);
    }

    void launch(const float* input, float* output) const {
        float alpha = 1.0f;
        float beta = 0.0f;
        CUDNN_CHECK(cudnnSoftmaxForward(handle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, tensor_desc_, input, &beta, tensor_desc_, output));
    }

private:
    cudnnHandle_t handle_;
    cudnnTensorDescriptor_t tensor_desc_;
};

// 一个线程处理一个样本的 softmax 计算
__global__ void softmax_naive(const float* input, float* output, int batch, int num_classes) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch) {
        return;
    }

    int offset = row * num_classes;

    float row_max = -FLT_MAX;
    for (int i = 0; i < num_classes; ++i) {
        row_max = fmaxf(input[offset + i], row_max);
    }

    float row_sum = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        row_sum += expf(input[offset + i] - row_max);
    }

    for (int i = 0; i < num_classes; ++i) {
        output[offset + i] =
            expf(input[offset + i] - row_max) / row_sum;
    }
}

template <int BSIZE>
void softmax_naive_launcher(
    const float* input,
    float* output,
    int batch,
    int num_classes
) {
    int blocks = (batch + BSIZE - 1) / BSIZE;
    softmax_naive<<<blocks, BSIZE>>>(input, output, batch, num_classes);
}

// 一个线程处理一个样本的 softmax 计算
__global__ void softmax_vectorized(const float* input, float* output, int batch, int num_classes) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch) {
        return;
    }

    int offset = row * num_classes;
    const float4 *input4 = reinterpret_cast<const float4*>(input + offset);
    float4* output4 = reinterpret_cast<float4*>(output + offset);
    int vec_num = num_classes / 4;

    float row_max = -FLT_MAX;
    for (int i = 0; i < vec_num; i++) {
        float4 val = input4[i];
        row_max = fmaxf(row_max, val.x);
        row_max = fmaxf(row_max, val.y);
        row_max = fmaxf(row_max, val.z);
        row_max = fmaxf(row_max, val.w);
    }

    float row_sum = 0;
    for (int i = 0; i < vec_num; i++) {
        float4 val = input4[i];
        row_sum += expf(val.x - row_max);
        row_sum += expf(val.y - row_max);
        row_sum += expf(val.z - row_max);
        row_sum += expf(val.w - row_max);
    }

    for (int i = 0; i < vec_num; i++) {
        float4 val = input4[i];
        float4 out;
        out.x = expf(val.x - row_max) / row_sum;
        out.y = expf(val.y - row_max) / row_sum;
        out.z = expf(val.z - row_max) / row_sum;
        out.w = expf(val.w - row_max) / row_sum;
        output4[i] = out;
    }
}

template <int BSIZE>
void softmax_vectorized_launcher(
    const float* input,
    float* output,
    int batch,
    int num_classes
) {
    if (num_classes % 4 != 0) {
        return;
    }

    unsigned long long input_addr =
        reinterpret_cast<unsigned long long>(input);

    unsigned long long output_addr =
        reinterpret_cast<unsigned long long>(output);

    if (input_addr % alignof(float4) != 0) {
        return;
    }

    if (output_addr % alignof(float4) != 0) {
        return;
    }

    int blocks = (batch + BSIZE - 1) / BSIZE;

    softmax_vectorized<<<blocks, BSIZE>>>(
        input,
        output,
        batch,
        num_classes
    );
}

// 一个warp处理一个样本的 softmax 计算
__global__ void softmax_warp(const float *input, float *output, int batch, int num_classes) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * blockDim.x / WARP_SIZE + warp_id;

    if (row >= batch) return;
    int row_offset = row * num_classes;

    float row_tile_max = -FLT_MAX;
    for (int i = lane_id; i < num_classes; i+=WARP_SIZE) {
        row_tile_max = fmaxf(row_tile_max, input[row_offset + i]);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        row_tile_max = fmaxf(row_tile_max, __shfl_xor_sync(0xffffffff, row_tile_max, offset));
    }

    float row_tile_sum = 0;
    for (int i = lane_id; i < num_classes; i+=WARP_SIZE) {
        row_tile_sum += expf(input[row_offset + i] - row_tile_max);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        row_tile_sum += __shfl_xor_sync(0xffffffff, row_tile_sum, offset);
    }

    for (int i = lane_id; i < num_classes; i+=WARP_SIZE) {
        output[row_offset + i] = expf(input[row_offset + i] - row_tile_max) / row_tile_sum;
    }
}

template <int BSIZE>
void softmax_warp_launcher(
    const float* input,
    float* output,
    int batch,
    int num_classes
) {
    int warps_per_block = BSIZE / WARP_SIZE;
    int blocks = (batch + warps_per_block - 1) / warps_per_block;
    softmax_warp<<<blocks, BSIZE>>>(input, output, batch, num_classes);
}

__global__ void softmax_warp_vectorized(const float *input, float *output, int batch, int num_classes) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * blockDim.x / WARP_SIZE + warp_id;

    if (row >= batch) return;
    int row_offset = row * num_classes;
    const float4 *input4 = reinterpret_cast<const float4*>(input + row_offset);
    float4 *output4 = reinterpret_cast<float4*>(output + row_offset);
    int vec_num = num_classes / 4;

    float row_tile_max = -FLT_MAX;
    for (int i = lane_id; i < vec_num; i+=WARP_SIZE) {
        float4 val = input4[i];
        row_tile_max = fmaxf(row_tile_max, val.x);
        row_tile_max = fmaxf(row_tile_max, val.y);
        row_tile_max = fmaxf(row_tile_max, val.z);
        row_tile_max = fmaxf(row_tile_max, val.w);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        row_tile_max = fmaxf(row_tile_max, __shfl_xor_sync(0xffffffff, row_tile_max, offset));
    }

    float row_tile_sum = 0;
    for (int i = lane_id; i < vec_num; i+=WARP_SIZE) {
        float4 val = input4[i];
        row_tile_sum += expf(val.x - row_tile_max);
        row_tile_sum += expf(val.y - row_tile_max);
        row_tile_sum += expf(val.z - row_tile_max);
        row_tile_sum += expf(val.w - row_tile_max);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        row_tile_sum += __shfl_xor_sync(0xffffffff, row_tile_sum, offset);
    }

    for (int i = lane_id; i < vec_num; i += WARP_SIZE) {
        float4 val = input4[i];
        float4 out;
        out.x = expf(val.x - row_tile_max) / row_tile_sum;
        out.y = expf(val.y - row_tile_max) / row_tile_sum;
        out.z = expf(val.z - row_tile_max) / row_tile_sum;
        out.w = expf(val.w - row_tile_max) / row_tile_sum;
        output4[i] = out;
    }
}

template <int BSIZE>
void softmax_warp_vectorized_launcher(
    const float *input,
    float *output,
    int batch,
    int num_classes
) {
    if (num_classes % 4 != 0) {
        return;
    }

    unsigned long long input_addr = reinterpret_cast<unsigned long long>(input);
    unsigned long long output_addr = reinterpret_cast<unsigned long long>(output);

    if (input_addr % alignof(float4) != 0) {
        return;
    }

    if (output_addr % alignof(float4) != 0) {
        return;
    }

    int warps_per_block = BSIZE / WARP_SIZE;
    int blocks = (batch + warps_per_block - 1) / warps_per_block;

    softmax_warp_vectorized<<<blocks, BSIZE>>>(input, output, batch, num_classes);
}

struct MAX_SUM {
    float max;
    float sum;
};

__device__ __forceinline__ MAX_SUM merge_max_sum(MAX_SUM ms1, MAX_SUM ms2) {
    MAX_SUM ret;

    bool first_is_max = ms1.max >= ms2.max;
    ret.max = fmaxf(ms1.max, ms2.max);
    float scale = expf(-fabsf(ms1.max - ms2.max));
    ret.sum = first_is_max? ms1.sum + ms2.sum * scale : ms1.sum * scale + ms2.sum;

    return ret;
}

__global__ void softmax_warp_vectorized_online(const float *input, float *output, int batch, int num_classes) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * blockDim.x / WARP_SIZE + warp_id;

    if (row >= batch) return;
    int row_offset = row * num_classes;
    const float4 *input4 = reinterpret_cast<const float4*>(input + row_offset);
    float4 *output4 = reinterpret_cast<float4*>(output + row_offset);
    int vec_num = num_classes / 4;

    MAX_SUM row_tile_max_sum = {-FLT_MAX, 0.0f};
    for (int i = lane_id; i < vec_num; i += WARP_SIZE) {
        float4 val = input4[i];

        row_tile_max_sum = merge_max_sum(row_tile_max_sum, MAX_SUM{val.x, 1.0f});
        row_tile_max_sum = merge_max_sum(row_tile_max_sum, MAX_SUM{val.y, 1.0f});
        row_tile_max_sum = merge_max_sum(row_tile_max_sum, MAX_SUM{val.z, 1.0f});
        row_tile_max_sum = merge_max_sum(row_tile_max_sum, MAX_SUM{val.w, 1.0f});
    }
    // for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    //     MAX_SUM other;
    //     other.max = __shfl_xor_sync(0xffffffff, row_tile_max_sum.max, offset);
    //     other.sum = __shfl_xor_sync(0xffffffff, row_tile_max_sum.sum, offset);
    //     row_tile_max_sum = merge_max_sum(row_tile_max_sum, other);
    // }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        MAX_SUM other;
        other.max = __shfl_down_sync(0xffffffff, row_tile_max_sum.max, offset);
        other.sum = __shfl_down_sync(0xffffffff, row_tile_max_sum.sum, offset);
        row_tile_max_sum = merge_max_sum(row_tile_max_sum, other);
    }
    row_tile_max_sum.max = __shfl_sync(0xffffffff, row_tile_max_sum.max, 0);
    row_tile_max_sum.sum = __shfl_sync(0xffffffff, row_tile_max_sum.sum, 0);


    for (int i = lane_id; i < vec_num; i += WARP_SIZE) {
        float4 val = input4[i];
        float4 out;
        out.x = expf(val.x - row_tile_max_sum.max) / row_tile_max_sum.sum;
        out.y = expf(val.y - row_tile_max_sum.max) / row_tile_max_sum.sum;
        out.z = expf(val.z - row_tile_max_sum.max) / row_tile_max_sum.sum;
        out.w = expf(val.w - row_tile_max_sum.max) / row_tile_max_sum.sum;
        output4[i] = out;
    }
}

template <int BSIZE>
void softmax_warp_vectorized_online_launcher(
    const float *input,
    float *output,
    int batch,
    int num_classes
) {
    if (num_classes % 4 != 0) {
        return;
    }

    unsigned long long input_addr = reinterpret_cast<unsigned long long>(input);
    unsigned long long output_addr = reinterpret_cast<unsigned long long>(output);

    if (input_addr % alignof(float4) != 0) {
        return;
    }

    if (output_addr % alignof(float4) != 0) {
        return;
    }

    int warps_per_block = BSIZE / WARP_SIZE;
    int blocks = (batch + warps_per_block - 1) / warps_per_block;

    softmax_warp_vectorized_online<<<blocks, BSIZE>>>(input, output, batch, num_classes);
}

__global__ void softmax_block_vectorized_shared(const float* input, float* output, int batch, int num_classes) {
    int row_idx = blockIdx.x;
    if (row_idx >= batch) return;

    int local_thread_idx = threadIdx.x;
    int warp_idx = local_thread_idx / WARP_SIZE;
    int lane_idx = local_thread_idx % WARP_SIZE;
    int num_warps_per_block = blockDim.x / WARP_SIZE;
    
    int row_offset = row_idx * num_classes;
    int num_vec4 = num_classes / 4;

    extern __shared__ float4 shared_vec4[];
    float4 *shared_row = shared_vec4;
    float *shared_warp = reinterpret_cast<float*>(shared_vec4 + num_vec4);

    const float4 *input4 = reinterpret_cast<const float4*>(input + row_offset);
    float4 *output4 = reinterpret_cast<float4*>(output + row_offset);

    // 数据加载：global -> shared
    for (int i = local_thread_idx; i < num_vec4; i+=blockDim.x) {
        shared_row[i] = input4[i];
    }
    // 保证数据完整加载
    __syncthreads();

    // 计算 max
    float thread_max = -FLT_MAX;
    for (int i = local_thread_idx; i < num_vec4; i+=blockDim.x) {
        float4 val4 = shared_row[i];
        thread_max = fmaxf(thread_max, val4.x);
        thread_max = fmaxf(thread_max, val4.y);
        thread_max = fmaxf(thread_max, val4.z);
        thread_max = fmaxf(thread_max, val4.w);
    }
    // warp内规约
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    // 将warp_max写入shared
    if (lane_idx == 0) shared_warp[warp_idx] = thread_max;
    // 保证warp_max完整写入
    __syncthreads();
    // warp间规约
    if (warp_idx == 0) {
        thread_max = -FLT_MAX;
        if (lane_idx < num_warps_per_block) {
            thread_max = shared_warp[lane_idx];
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
        }
        if (lane_idx == 0) shared_warp[0] = thread_max;
    }
    // 保证max写入shared后再进行读取
    __syncthreads();
    // 读取max
    thread_max = shared_warp[0];
    // 保证所有线程读取完成，再复用shared_warp
    __syncthreads();

    // 计算 sum
    float thread_sum = 0;
    for (int i = local_thread_idx; i < num_vec4; i+=blockDim.x) {
        float4 val4 = shared_row[i];
        val4.x = expf(val4.x - thread_max);
        val4.y = expf(val4.y - thread_max);
        val4.z = expf(val4.z - thread_max);
        val4.w = expf(val4.w - thread_max);
        thread_sum += val4.x + val4.y + val4.z + val4.w;
        shared_row[i] = val4;
    }
    // warp内规约
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    // 将warp_sum写入shared
    if (lane_idx == 0) shared_warp[warp_idx] = thread_sum;
    // 保证warp_sum完整写入
    __syncthreads();
    // warp间规约
    if (warp_idx == 0) {
        thread_sum = 0;
        if (lane_idx < num_warps_per_block) {
            thread_sum = shared_warp[lane_idx];
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        if (lane_idx == 0) shared_warp[0] = thread_sum;
    }
    // 保证sum写入shared后再进行读取
    __syncthreads();
    // 读取sum
    thread_sum = shared_warp[0];

    // 写回
    for (int i = local_thread_idx; i < num_vec4; i+=blockDim.x) {
        float4 val4 = shared_row[i];
        val4.x = val4.x / thread_sum;
        val4.y = val4.y / thread_sum;
        val4.z = val4.z / thread_sum;
        val4.w = val4.w / thread_sum;
        output4[i] = val4;
    }
}

template <int BSIZE>
void softmax_block_vectorized_shared_launcher(
    const float *input,
    float *output,
    int batch,
    int num_classes
) {
    if (num_classes % 4 != 0 || num_classes > 4096) {
        return;
    }

    unsigned long long input_addr = reinterpret_cast<unsigned long long>(input);
    unsigned long long output_addr = reinterpret_cast<unsigned long long>(output);

    if (input_addr % alignof(float4) != 0) {
        return;
    }

    if (output_addr % alignof(float4) != 0) {
        return;
    }

    int num_vec4 = num_classes / 4;
    int warps_per_block = BSIZE / WARP_SIZE;
    int shared_bytes = num_vec4 * sizeof(float4) + warps_per_block * sizeof(float);

    softmax_block_vectorized_shared<<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
}

template <int MAX_NUM_CLASSES, int BSIZE>
__global__ void softmax_block_vectorized_register(const float* input, float* output, int batch, int num_classes) {
    int row_idx = blockIdx.x;
    if (row_idx >= batch) return;

    int local_thread_idx = threadIdx.x;
    int warp_idx = local_thread_idx / WARP_SIZE;
    int lane_idx = local_thread_idx % WARP_SIZE;
    int num_warps_per_block = blockDim.x / WARP_SIZE;
    
    int row_offset = row_idx * num_classes;
    const float4 *input4 = reinterpret_cast<const float4*>(input + row_offset);
    float4 *output4 = reinterpret_cast<float4*>(output + row_offset);

    constexpr int MAX_NUM_VEC4 = MAX_NUM_CLASSES / 4;
    constexpr int MAX_VECS_PER_THREAD = (MAX_NUM_VEC4 + BSIZE - 1) / BSIZE;
    float4 thread_input4[MAX_VECS_PER_THREAD];
    int num_vec4 = num_classes / 4;
    extern __shared__ float shared_warp[];

    // 数据加载：global -> register
    #pragma unroll
    for (int reg_idx = 0; reg_idx < MAX_VECS_PER_THREAD; reg_idx++) {
        int vec_idx = local_thread_idx + reg_idx * BSIZE;
        if (vec_idx < num_vec4) thread_input4[reg_idx] = input4[vec_idx];
    }

    // 计算 max
    // 线程内部计算max
    float thread_max = -FLT_MAX;
    #pragma unroll
    for (int reg_idx = 0; reg_idx < MAX_VECS_PER_THREAD; reg_idx++) {
        int vec_idx = local_thread_idx + reg_idx * BSIZE;
        if (vec_idx < num_vec4) {
            float4 val4 = thread_input4[reg_idx];
            thread_max = fmaxf(thread_max, val4.x);
            thread_max = fmaxf(thread_max, val4.y);
            thread_max = fmaxf(thread_max, val4.z);
            thread_max = fmaxf(thread_max, val4.w);
        }
    }
    // warp内部计算max
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    // 将warp_max写入shared
    if (lane_idx == 0) shared_warp[warp_idx] = thread_max;
    // 保证warp_max完整写入
    __syncthreads();
    // warp间规约
    if (warp_idx == 0) {
        thread_max = -FLT_MAX;
        if (lane_idx < num_warps_per_block) {
            thread_max = shared_warp[lane_idx];
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
        }
        if (lane_idx == 0) shared_warp[0] = thread_max;
    }
    // 保证max写入shared后再进行读取
    __syncthreads();
    // 读取max
    thread_max = shared_warp[0];
    // 保证所有线程读取完成，再复用shared_warp
    __syncthreads();

    // 计算 sum
    // 线程内部计算sum
    float thread_sum = 0;
    #pragma unroll
    for (int reg_idx = 0; reg_idx < MAX_VECS_PER_THREAD; reg_idx++) {
        int vec_idx = local_thread_idx + reg_idx * BSIZE;
        if (vec_idx < num_vec4) {
            float4 val4 = thread_input4[reg_idx];
            val4.x = expf(val4.x - thread_max);
            val4.y = expf(val4.y - thread_max);
            val4.z = expf(val4.z - thread_max);
            val4.w = expf(val4.w - thread_max);
            thread_sum += val4.x + val4.y + val4.z + val4.w;
            thread_input4[reg_idx] = val4;
        }
    }
    // warp内部计算sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    // 将warp_sum写入shared
    if (lane_idx == 0) shared_warp[warp_idx] = thread_sum;
    // 保证warp_sum完整写入
    __syncthreads();
    // warp间规约
    if (warp_idx == 0) {
        thread_sum = 0;
        if (lane_idx < num_warps_per_block) {
            thread_sum = shared_warp[lane_idx];
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        if (lane_idx == 0) shared_warp[0] = thread_sum;
    }
    // 保证sum写入shared后再进行读取
    __syncthreads();
    // 读取sum
    thread_sum = shared_warp[0];

    // 写回
    #pragma unroll
    for (int reg_idx = 0; reg_idx < MAX_VECS_PER_THREAD; reg_idx++) {
        int vec_idx = local_thread_idx + reg_idx * BSIZE;
        if (vec_idx < num_vec4) {
            float4 val4 = thread_input4[reg_idx];
            val4.x = val4.x / thread_sum;
            val4.y = val4.y / thread_sum;
            val4.z = val4.z / thread_sum;
            val4.w = val4.w / thread_sum;
            output4[vec_idx] = val4;
        }
    }
}

template <int BSIZE>
void softmax_block_vectorized_register_launcher(
    const float *input,
    float *output,
    int batch,
    int num_classes
) {
    if (num_classes % 4 != 0 || num_classes > 4096) {
        return;
    }

    unsigned long long input_addr = reinterpret_cast<unsigned long long>(input);
    unsigned long long output_addr = reinterpret_cast<unsigned long long>(output);

    if (input_addr % alignof(float4) != 0) {
        return;
    }

    if (output_addr % alignof(float4) != 0) {
        return;
    }

    int warps_per_block = BSIZE / WARP_SIZE;
    int shared_bytes = warps_per_block * sizeof(float);

    if (num_classes <= 128) {
        softmax_block_vectorized_register<128, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 256) {
        softmax_block_vectorized_register<256, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 512) {
        softmax_block_vectorized_register<512, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 1024) {
        softmax_block_vectorized_register<1024, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 2048) {
        softmax_block_vectorized_register<2048, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 4096) {
        softmax_block_vectorized_register<4096, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else {
        return;
    }
}

template <int MAX_NUM_CLASSES, int BSIZE>
__global__ void softmax_block_vectorized_register_latest(const float* input, float* output, int batch, int num_classes) {
    int row_idx = blockIdx.x;
    if (row_idx >= batch) return;

    int local_thread_idx = threadIdx.x;
    int warp_idx = local_thread_idx / WARP_SIZE;
    int lane_idx = local_thread_idx % WARP_SIZE;
    int num_warps_per_block = blockDim.x / WARP_SIZE;
    
    int row_offset = row_idx * num_classes;
    const float4 *input4 = reinterpret_cast<const float4*>(input + row_offset);
    float4 *output4 = reinterpret_cast<float4*>(output + row_offset);

    constexpr int MAX_NUM_VEC4 = MAX_NUM_CLASSES / 4;
    constexpr int MAX_VECS_PER_THREAD = (MAX_NUM_VEC4 + BSIZE - 1) / BSIZE;
    float4 thread_input4[MAX_VECS_PER_THREAD];
    int num_vec4 = num_classes / 4;
    extern __shared__ float shared_warp[];

    // 数据加载：global -> register
    // 线程内部计算max
    float thread_max = -FLT_MAX;
    #pragma unroll
    for (int reg_idx = 0; reg_idx < MAX_VECS_PER_THREAD; ++reg_idx) {
        int vec_idx = local_thread_idx + reg_idx * BSIZE;

        if (vec_idx < num_vec4) {
            float4 val4 = input4[vec_idx];
            thread_input4[reg_idx] = val4;

            float max_xy = fmaxf(val4.x, val4.y);
            float max_zw = fmaxf(val4.z, val4.w);
            thread_max = fmaxf(thread_max, fmaxf(max_xy, max_zw));
        }
    }
    // warp内部计算max
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    // 将warp_max写入shared
    if (lane_idx == 0) shared_warp[warp_idx] = thread_max;
    // 保证warp_max完整写入
    __syncthreads();
    // warp间规约
    if (warp_idx == 0) {
        thread_max = -FLT_MAX;
        if (lane_idx < num_warps_per_block) {
            thread_max = shared_warp[lane_idx];
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
        }
        if (lane_idx == 0) shared_warp[0] = thread_max;
    }
    // 保证max写入shared后再进行读取
    __syncthreads();
    // 读取max
    thread_max = shared_warp[0];
    // 保证所有线程读取完成，再复用shared_warp
    __syncthreads();

    // 计算 sum
    // 线程内部计算sum
    float thread_sum = 0;
    #pragma unroll
    for (int reg_idx = 0; reg_idx < MAX_VECS_PER_THREAD; reg_idx++) {
        int vec_idx = local_thread_idx + reg_idx * BSIZE;
        if (vec_idx < num_vec4) {
            float4 val4 = thread_input4[reg_idx];
            val4.x = __expf(val4.x - thread_max);
            val4.y = __expf(val4.y - thread_max);
            val4.z = __expf(val4.z - thread_max);
            val4.w = __expf(val4.w - thread_max);
            thread_sum += val4.x + val4.y + val4.z + val4.w;
            thread_input4[reg_idx] = val4;
        }
    }
    // warp内部计算sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    // 将warp_sum写入shared
    if (lane_idx == 0) shared_warp[warp_idx] = thread_sum;
    // 保证warp_sum完整写入
    __syncthreads();
    // warp间规约
    if (warp_idx == 0) {
        thread_sum = 0;
        if (lane_idx < num_warps_per_block) {
            thread_sum = shared_warp[lane_idx];
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        if (lane_idx == 0) shared_warp[0] = 1.0f / thread_sum;
    }
    // 保证sum写入shared后再进行读取
    __syncthreads();
    // 读取sum
    thread_sum = shared_warp[0];

    // 写回
    float inv_sum = shared_warp[0];
    #pragma unroll
    for (int reg_idx = 0; reg_idx < MAX_VECS_PER_THREAD; reg_idx++) {
        int vec_idx = local_thread_idx + reg_idx * BSIZE;
        if (vec_idx < num_vec4) {
            float4 val4 = thread_input4[reg_idx];
            val4.x *= inv_sum;
            val4.y *= inv_sum;
            val4.z *= inv_sum;
            val4.w *= inv_sum;
            output4[vec_idx] = val4;
        }
    }
}

template <int BSIZE>
void softmax_block_vectorized_register_latest_launcher(
    const float *input,
    float *output,
    int batch,
    int num_classes
) {
    if (num_classes % 4 != 0 || num_classes > 4096) {
        return;
    }

    unsigned long long input_addr = reinterpret_cast<unsigned long long>(input);
    unsigned long long output_addr = reinterpret_cast<unsigned long long>(output);

    if (input_addr % alignof(float4) != 0) {
        return;
    }

    if (output_addr % alignof(float4) != 0) {
        return;
    }

    int warps_per_block = BSIZE / WARP_SIZE;
    int shared_bytes = warps_per_block * sizeof(float);

    if (num_classes <= 128) {
        softmax_block_vectorized_register_latest<128, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 256) {
        softmax_block_vectorized_register_latest<256, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 512) {
        softmax_block_vectorized_register_latest<512, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 1024) {
        softmax_block_vectorized_register_latest<1024, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 2048) {
        softmax_block_vectorized_register_latest<2048, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 4096) {
        softmax_block_vectorized_register_latest<4096, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else {
        return;
    }
}

struct TestContext {
    int batch;
    int num_classes;
    size_t num_elements;
    size_t bytes;
    float* input;
    float* output;
    float* d_input;
    float* d_output;

    TestContext(int batch_value, int num_classes_value) : batch(batch_value), num_classes(num_classes_value), num_elements((size_t)batch_value * (size_t)num_classes_value), bytes(num_elements * sizeof(float)), input(create_input(batch_value, num_classes_value)), output((float*)malloc(bytes)), d_input(nullptr), d_output(nullptr) {
        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));
        CUDA_CHECK(cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice));
    }

    ~TestContext() {
        cudaFree(d_input);
        cudaFree(d_output);
        free(input);
        free(output);
    }

    TestContext(const TestContext&) = delete;
    TestContext& operator=(const TestContext&) = delete;
};

static double max_double(double a, double b) { return a > b ? a : b; }
static double abs_double(double value) { return value < 0.0 ? -value : value; }

double* softmax_cpu_reference(const TestContext& context) {
    double* reference = (double*)malloc(context.num_elements * sizeof(double));
    for (int row = 0; row < context.batch; ++row) {
        size_t row_offset = (size_t)row * (size_t)context.num_classes;
        double row_max = -DBL_MAX;
        for (int col = 0; col < context.num_classes; ++col) row_max = max_double(row_max, (double)context.input[row_offset + col]);
        double row_sum = 0.0;
        for (int col = 0; col < context.num_classes; ++col) {
            double value = exp((double)context.input[row_offset + col] - row_max);
            reference[row_offset + col] = value;
            row_sum += value;
        }
        for (int col = 0; col < context.num_classes; ++col) reference[row_offset + col] /= row_sum;
    }
    return reference;
}

struct ValidationResult {
    bool passed;
    double max_abs_error;
    double max_rel_error;
    double max_row_sum_error;
    size_t mismatch_count;
};

template <typename Launcher>
ValidationResult validate_kernel(TestContext& context, const double* reference, Launcher launcher) {
    const double absolute_tolerance = 2e-6;
    const double relative_tolerance = 1e-4;
    const double row_sum_tolerance = 1e-4;
    for (size_t i = 0; i < context.num_elements; ++i) context.output[i] = NAN;
    CUDA_CHECK(cudaMemcpy(context.d_output, context.output, context.bytes, cudaMemcpyHostToDevice));
    launcher();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(context.output, context.d_output, context.bytes, cudaMemcpyDeviceToHost));
    ValidationResult result = {true, 0.0, 0.0, 0.0, 0};
    for (int row = 0; row < context.batch; ++row) {
        size_t row_offset = (size_t)row * (size_t)context.num_classes;
        double row_sum = 0.0;
        bool row_is_finite = true;
        for (int col = 0; col < context.num_classes; ++col) {
            size_t index = row_offset + col;
            double actual = (double)context.output[index];
            double expected = reference[index];
            double abs_error = abs_double(actual - expected);
            double tolerance = absolute_tolerance + relative_tolerance * abs_double(expected);
            bool finite = isfinite(actual);
            result.max_abs_error = max_double(result.max_abs_error, abs_error);
            if (abs_double(expected) > 1e-12) result.max_rel_error = max_double(result.max_rel_error, abs_error / abs_double(expected));
            if (!finite || actual < 0.0 || abs_error > tolerance) ++result.mismatch_count;
            row_is_finite = row_is_finite && finite;
            row_sum += actual;
        }
        double row_sum_error = row_is_finite ? abs_double(row_sum - 1.0) : DBL_MAX;
        result.max_row_sum_error = max_double(result.max_row_sum_error, row_sum_error);
    }
    result.passed = result.mismatch_count == 0 && result.max_row_sum_error <= row_sum_tolerance;
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
    float median = (samples % 2 == 0) ? (sample_us[middle - 1] + sample_us[middle]) / 2.0f : sample_us[middle];
    int p90_index = (9 * samples + 9) / 10 - 1;
    BenchmarkResult result = {median, sample_us[0], sample_us[p90_index]};
    free(sample_us);
    return result;
}

enum RunMode { RUN_VERIFY, RUN_BENCHMARK, RUN_PROFILE };

struct Options {
    RunMode mode;
    const char* target;
    int batch;
    int num_classes;
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
    fprintf(stderr, "Usage:\n  %s <verify|benchmark|profile> <all|kernel> <batch> <num_classes> [warmup] [iterations] [samples]\n  %s <batch> <num_classes>  # profile all (legacy)\n\nKernels:\n  cudnn, naive, vectorized, warp, warp_vectorized,\n  warp_vectorized_online, block_shared_128, block_shared_256,\n  block_register_128, block_register_256,\n  block_register_latest_128, block_register_latest_256\n", program, program);
}

bool parse_options(int argc, char* argv[], Options* options) {
    options->mode = RUN_PROFILE;
    options->target = "all";
    options->warmup_iterations = 10;
    options->measured_iterations = 100;
    options->samples = 10;
    if (argc == 3) return parse_positive_int(argv[1], &options->batch) && parse_positive_int(argv[2], &options->num_classes);
    if (argc < 5 || argc > 8) return false;
    if (strcmp(argv[1], "verify") == 0) options->mode = RUN_VERIFY;
    else if (strcmp(argv[1], "benchmark") == 0) options->mode = RUN_BENCHMARK;
    else if (strcmp(argv[1], "profile") == 0) options->mode = RUN_PROFILE;
    else return false;
    options->target = argv[2];
    if (!parse_positive_int(argv[3], &options->batch) || !parse_positive_int(argv[4], &options->num_classes)) return false;
    if (options->mode != RUN_BENCHMARK && argc != 5) return false;
    if ((argc >= 6 && !parse_positive_int(argv[5], &options->warmup_iterations)) || (argc >= 7 && !parse_positive_int(argv[6], &options->measured_iterations)) || (argc >= 8 && !parse_positive_int(argv[7], &options->samples)) ) return false;
    return true;
}

template <typename Launcher>
void run_candidate(const char* name, bool supported, const Options& options, TestContext& context, const double* reference, bool* matched, bool* succeeded, Launcher launcher) {
    if (strcmp(options.target, "all") != 0 && strcmp(options.target, name) != 0) return;
    *matched = true;
    if (!supported) {
        printf("%-40sSKIP (unsupported)\n", name);
        if (strcmp(options.target, "all") != 0) *succeeded = false;
        return;
    }
    if (options.mode == RUN_VERIFY) {
        ValidationResult result = validate_kernel(context, reference, launcher);
        printf("%-40s%-8smax_abs=% .3e  max_rel=% .3e  row_sum=% .3e  mismatches=%zu\n", name, result.passed ? "PASS" : "FAIL", result.max_abs_error, result.max_rel_error, result.max_row_sum_error, result.mismatch_count);
        *succeeded = *succeeded && result.passed;
        return;
    }
    if (options.mode == RUN_BENCHMARK) {
        BenchmarkResult result = benchmark_kernel(launcher, options.warmup_iterations, options.measured_iterations, options.samples);
        printf("%-40s%14.3f%14.3f%14.3f\n", name, result.median_us, result.min_us, result.p90_us);
        return;
    }
    launcher();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char* argv[]) {
    Options options;
    if (!parse_options(argc, argv, &options)) { print_usage(argv[0]); return EXIT_FAILURE; }
    TestContext context(options.batch, options.num_classes);
    CudnnSoftmax cudnn(options.batch, options.num_classes);
    double* reference = nullptr;
    if (options.mode == RUN_VERIFY) {
        reference = softmax_cpu_reference(context);
        printf("%-40s%-8sErrors\n", "Kernel", "Status");
    } else if (options.mode == RUN_BENCHMARK) {
        printf("warmup=%d, iterations=%d, samples=%d\n", options.warmup_iterations, options.measured_iterations, options.samples);
        printf("%-40s%14s%14s%14s\n", "Kernel", "Median (us)", "Min (us)", "P90 (us)");
    }
    bool matched = false;
    bool succeeded = true;
    bool vectorized_supported = options.num_classes % 4 == 0;
    bool block_supported = vectorized_supported && options.num_classes <= 4096;
    const double* reference_ptr = options.mode == RUN_VERIFY ? reference : nullptr;

    run_candidate("cudnn", true, options, context, reference_ptr, &matched, &succeeded, [&] { cudnn.launch(context.d_input, context.d_output); });
    run_candidate("naive", true, options, context, reference_ptr, &matched, &succeeded, [&] { softmax_naive_launcher<256>(context.d_input, context.d_output, context.batch, context.num_classes); });
    run_candidate("vectorized", vectorized_supported, options, context, reference_ptr, &matched, &succeeded, [&] { softmax_vectorized_launcher<256>(context.d_input, context.d_output, context.batch, context.num_classes); });
    run_candidate("warp", true, options, context, reference_ptr, &matched, &succeeded, [&] { softmax_warp_launcher<256>(context.d_input, context.d_output, context.batch, context.num_classes); });
    run_candidate("warp_vectorized", vectorized_supported, options, context, reference_ptr, &matched, &succeeded, [&] { softmax_warp_vectorized_launcher<256>(context.d_input, context.d_output, context.batch, context.num_classes); });
    run_candidate("warp_vectorized_online", vectorized_supported, options, context, reference_ptr, &matched, &succeeded, [&] { softmax_warp_vectorized_online_launcher<256>(context.d_input, context.d_output, context.batch, context.num_classes); });
    run_candidate("block_shared_128", block_supported, options, context, reference_ptr, &matched, &succeeded, [&] { softmax_block_vectorized_shared_launcher<128>(context.d_input, context.d_output, context.batch, context.num_classes); });
    run_candidate("block_shared_256", block_supported, options, context, reference_ptr, &matched, &succeeded, [&] { softmax_block_vectorized_shared_launcher<256>(context.d_input, context.d_output, context.batch, context.num_classes); });
    run_candidate("block_register_128", block_supported, options, context, reference_ptr, &matched, &succeeded, [&] { softmax_block_vectorized_register_launcher<128>(context.d_input, context.d_output, context.batch, context.num_classes); });
    run_candidate("block_register_256", block_supported, options, context, reference_ptr, &matched, &succeeded, [&] { softmax_block_vectorized_register_launcher<256>(context.d_input, context.d_output, context.batch, context.num_classes); });
    run_candidate("block_register_latest_128", block_supported, options, context, reference_ptr, &matched, &succeeded, [&] { softmax_block_vectorized_register_latest_launcher<128>(context.d_input, context.d_output, context.batch, context.num_classes); });
    run_candidate("block_register_latest_256", block_supported, options, context, reference_ptr, &matched, &succeeded, [&] { softmax_block_vectorized_register_latest_launcher<256>(context.d_input, context.d_output, context.batch, context.num_classes); });

    if (!matched) {
        fprintf(stderr, "Unknown kernel: %s\n", options.target);
        print_usage(argv[0]);
        free(reference);
        return EXIT_FAILURE;
    }

    free(reference);
    return succeeded ? EXIT_SUCCESS : EXIT_FAILURE;
}
