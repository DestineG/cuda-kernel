#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../../flashattn/naive-attention.cu"
namespace fa1_impl {
#include "../../../flashattn/flashattn1.cu"
}
namespace fa2_impl {
#include "../../../flashattn/flashattn2.cu"
}

namespace {

__global__ void naive_packed(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    float *o,
    float *scores,
    float *probs,
    int bsz,
    int heads,
    int n,
    int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    if (row >= n || head >= heads || batch >= bsz) return;
    size_t q_base = ((size_t)batch * n * heads + row * heads) * d + head * d;
    size_t kv_base = (size_t)batch * n * heads * d;
    size_t matrix_base = ((size_t)batch * heads + head) * n * n;
    float scale = rsqrtf((float)d);
    for (int col = 0; col < n; ++col) {
        float sum = 0.0f;
        for (int j = 0; j < d; ++j) {
            sum += __bfloat162float(q[q_base + j]) *
                   __bfloat162float(k[kv_base + ((size_t)col * heads + head) * d + j]);
        }
        scores[matrix_base + (size_t)row * n + col] = sum * scale;
    }
    float row_max = -FLT_MAX;
    for (int col = 0; col < n; ++col)
        row_max = fmaxf(row_max, scores[matrix_base + (size_t)row * n + col]);
    float row_sum = 0.0f;
    for (int col = 0; col < n; ++col) {
        float p = expf(scores[matrix_base + (size_t)row * n + col] - row_max);
        probs[matrix_base + (size_t)row * n + col] = p;
        row_sum += p;
    }
    for (int col = 0; col < n; ++col)
        probs[matrix_base + (size_t)row * n + col] /= row_sum;
    for (int j = 0; j < d; ++j) {
        float sum = 0.0f;
        for (int col = 0; col < n; ++col)
            sum += probs[matrix_base + (size_t)row * n + col] *
                   __bfloat162float(v[kv_base + ((size_t)col * heads + head) * d + j]);
        o[q_base + j] = sum;
    }
}

template <int D, int Br>
void launch_fa1(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    float *o,
    int n,
    int *cu_len,
    int bsz,
    int heads,
    cudaStream_t stream) {
    constexpr int Bc = 64;
    constexpr int threads = 128;
    constexpr size_t smem =
        ((size_t)Br * D + (size_t)Bc * D * 2) * sizeof(__nv_bfloat16) +
        (size_t)Br * D * sizeof(float) + (size_t)4 * Br * sizeof(float);
    static bool configured = false;
    if (!configured) {
        TORCH_CHECK(cudaFuncSetAttribute(
            fa1_impl::fa1<D, Br, Bc>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem) == cudaSuccess,
            "failed to configure FA1 shared memory");
        configured = true;
    }
    dim3 grid(bsz, heads, (n + Br - 1) / Br);
    fa1_impl::fa1<D, Br, Bc><<<grid, threads, smem, stream>>>(
        q, k, v, cu_len, cu_len, heads, heads, o);
}

template <int D, int Br>
void launch_fa2(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    float *o,
    int n,
    int *cu_len,
    int bsz,
    int heads,
    cudaStream_t stream) {
    constexpr int Bc = 64;
    constexpr int warps = Br / 16;
    dim3 grid((n + Br - 1) / Br, heads, bsz);
    fa2_impl::fa2<D, Br, Bc, warps><<<grid, warps * 32, 0, stream>>>(
        q, k, v, o, heads, heads, cu_len, cu_len);
}

void check_shape(const torch::Tensor &q, const torch::Tensor &k,
                 const torch::Tensor &v, int64_t br) {
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, v must have shape [B, N, H, D] internally");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "q, k, v must have the same shape");
    TORCH_CHECK(q.size(3) == 64 || q.size(3) == 128,
                "HEAD_DIM must be 64 or 128");
    TORCH_CHECK(br == 16 || br == 32 || br == 64,
                "Br must be 16, 32, or 64");
    TORCH_CHECK(q.size(1) % 64 == 0 && q.size(1) % br == 0,
                "N must be divisible by 64 and Br");
}

} // namespace

torch::Tensor forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int64_t kernel,
    int64_t br) {
    TORCH_CHECK(kernel >= 0 && kernel <= 2,
                "kernel must be 0 (naive), 1 (fa1), or 2 (fa2)");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q, k, v must be on the same CUDA device");
    const c10::cuda::CUDAGuard device_guard(q.device());
    q = q.permute({0, 2, 1, 3}).contiguous();
    k = k.permute({0, 2, 1, 3}).contiguous();
    v = v.permute({0, 2, 1, 3}).contiguous();
    check_shape(q, k, v, br);
    const int bsz = (int)q.size(0);
    const int n = (int)q.size(1);
    const int heads = (int)q.size(2);
    const int d = (int)q.size(3);
    auto output = torch::zeros({bsz, n, heads, d},
                               q.options().dtype(torch::kFloat32));
    auto lengths = torch::arange(0, (bsz + 1) * n, n,
                                 torch::TensorOptions().dtype(torch::kInt32)
                                     .device(q.device()));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto *q_ptr = reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr<at::BFloat16>());
    const auto *k_ptr = reinterpret_cast<const __nv_bfloat16 *>(k.data_ptr<at::BFloat16>());
    const auto *v_ptr = reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr<at::BFloat16>());
    auto *o_ptr = output.data_ptr<float>();
    auto *len_ptr = lengths.data_ptr<int>();

    if (kernel == 0) {
        auto scores = torch::empty({bsz, heads, n, n}, output.options());
        auto probs = torch::empty({bsz, heads, n, n}, output.options());
        naive_packed<<<dim3((n + 255) / 256, heads, bsz), 256, 0, stream>>>(
            q_ptr, k_ptr, v_ptr, o_ptr, scores.data_ptr<float>(),
            probs.data_ptr<float>(), bsz, heads, n, d);
    } else if (kernel == 1) {
        if (d == 64 && br == 16) launch_fa1<64, 16>(q_ptr, k_ptr, v_ptr, o_ptr, n, len_ptr, bsz, heads, stream);
        else if (d == 64 && br == 32) launch_fa1<64, 32>(q_ptr, k_ptr, v_ptr, o_ptr, n, len_ptr, bsz, heads, stream);
        else if (d == 64 && br == 64) launch_fa1<64, 64>(q_ptr, k_ptr, v_ptr, o_ptr, n, len_ptr, bsz, heads, stream);
        else if (d == 128 && br == 16) launch_fa1<128, 16>(q_ptr, k_ptr, v_ptr, o_ptr, n, len_ptr, bsz, heads, stream);
        else if (d == 128 && br == 32) launch_fa1<128, 32>(q_ptr, k_ptr, v_ptr, o_ptr, n, len_ptr, bsz, heads, stream);
        else launch_fa1<128, 64>(q_ptr, k_ptr, v_ptr, o_ptr, n, len_ptr, bsz, heads, stream);
    } else if (kernel == 2) {
        if (d == 64 && br == 16) launch_fa2<64, 16>(q_ptr, k_ptr, v_ptr, o_ptr, n, len_ptr, bsz, heads, stream);
        else if (d == 64 && br == 32) launch_fa2<64, 32>(q_ptr, k_ptr, v_ptr, o_ptr, n, len_ptr, bsz, heads, stream);
        else if (d == 64 && br == 64) launch_fa2<64, 64>(q_ptr, k_ptr, v_ptr, o_ptr, n, len_ptr, bsz, heads, stream);
        else if (d == 128 && br == 16) launch_fa2<128, 16>(q_ptr, k_ptr, v_ptr, o_ptr, n, len_ptr, bsz, heads, stream);
        else if (d == 128 && br == 32) launch_fa2<128, 32>(q_ptr, k_ptr, v_ptr, o_ptr, n, len_ptr, bsz, heads, stream);
        else launch_fa2<128, 64>(q_ptr, k_ptr, v_ptr, o_ptr, n, len_ptr, bsz, heads, stream);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "attention kernel launch failed");
    return output.permute({0, 2, 1, 3});
}
