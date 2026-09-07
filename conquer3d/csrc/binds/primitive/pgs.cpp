#include <torch/extension.h>
#include "../../primitive/pgs.h"
#include "../../check.h"

#include <cuda_fp16.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;

/**
 * @brief Tensor-level entry point for the periodic Gaussian tangency radius query.
 * @details Sits between pybind11 and the host dispatcher: it applies the `CHECK_INPUT`
 * contract -- CUDA device, contiguous layout, expected dtype -- then unwraps
 * `data_ptr` and calls the kernel launcher. Validating here keeps the launch path
 * free of checks and gives Python callers a clear error instead of a device fault.
 * @return The operator's results as PyTorch tensors.
 */
std::tuple<torch::Tensor, torch::Tensor> solve_pgs_cluster_tangency_radius_wrapper(
    const torch::Tensor &means,
    const torch::Tensor &normals,
    const torch::Tensor &covis,
    const uint32_t k
)
{
    CHECK_INPUT(means);
    CHECK_INPUT(normals);
    CHECK_INPUT(covis);

    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(normals.scalar_type() == torch::kFloat32, "normals must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");

    const uint32_t num_gaussians = means.size(0);
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(normals.size(0) == num_gaussians && normals.size(1) == 3, "normals must have shape (N, 3)");
    TORCH_CHECK(covis.size(0) == num_gaussians && covis.size(1) == 6, "covis must have shape (N, 6)");

    uint32_t search_k = k + 1;

    TORCH_CHECK(search_k > 1, "k must be greater than 0");
    TORCH_CHECK(search_k <= 32, "Requested k exceeds the hardware register limit (MAX_K = 32).");
    TORCH_CHECK(search_k <= num_gaussians, "Requested k is larger than the number of points in the tree.");

    auto options = means.options();
    torch::Tensor isos = torch::empty({num_gaussians}, options.dtype(torch::kFloat32));
    torch::Tensor invalid_mask = torch::empty({num_gaussians}, options.dtype(torch::kBool));

    pgs::solve_pgs_cluster_tangency_radius(
        num_gaussians,
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float3 *>(normals.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        search_k,
        reinterpret_cast<float *>(isos.data_ptr<float>()),
        reinterpret_cast<bool *>(invalid_mask.data_ptr<bool>()));

    return std::make_tuple(isos, invalid_mask);
}

void bind_primitive_pgs(py::module_ &m) {
    m.def("solve_pgs_cluster_tangency_radius_func", &solve_pgs_cluster_tangency_radius_wrapper,
          py::arg("means"), py::arg("normals"), py::arg("covis"), py::arg("k") = 16,
          R"pbdoc(
          Computes pairwise tangency contact radii for Planar Gaussians from k-NN clusters (CUDA).

          Args:
              means (torch.Tensor): (N, 3) float32 coordinates on CUDA.
              normals (torch.Tensor): (N, 3) float32 principal planar normal axes on CUDA.
              covis (torch.Tensor): (N, 6) float32 inverse covariance matrices on CUDA.
              k (int, optional): Number of nearest neighbors. Defaults to 16.

          Returns:
              Tuple[torch.Tensor, torch.Tensor]:
                  - isos (torch.Tensor): (N,) float32 optimal tangency radii.
                  - invalid_mask (torch.Tensor): (N,) bool mask indicating failure or degenerate pairs.

          Example:
              >>> import torch
              >>> from conquer3d._C import solve_pgs_cluster_tangency_radius_func
              >>> isos, invalid_mask = solve_pgs_cluster_tangency_radius_func(means, normals, covis, k=16)
          )pbdoc");
}