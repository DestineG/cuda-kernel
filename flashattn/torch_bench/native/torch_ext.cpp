#include <torch/extension.h>

torch::Tensor forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int64_t kernel,
    int64_t br);

torch::Tensor forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int64_t kernel,
    int64_t br) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda() && v.is_cuda(), "k and v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16,
                "q must have bfloat16 dtype");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16 &&
                    v.scalar_type() == torch::kBFloat16,
                "k and v must have bfloat16 dtype");
    return forward_cuda(q, k, v, kernel, br);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "FlashAttention forward (CUDA)");
}
