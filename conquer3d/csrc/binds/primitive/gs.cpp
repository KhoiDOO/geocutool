#include <torch/extension.h>
#include "../../primitive/gs.h"
#include "../../check.h"

#include <cuda_fp16.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;

/**
 * @brief Tensor-level entry point for the Gaussian inverse covariance query.
 * @details Sits between pybind11 and the host dispatcher: it applies the `CHECK_INPUT`
 * contract -- CUDA device, contiguous layout, expected dtype -- then unwraps
 * `data_ptr` and calls the kernel launcher. Validating here keeps the launch path
 * free of checks and gives Python callers a clear error instead of a device fault.
 * @return The operator's results as PyTorch tensors.
 */
torch::Tensor compute_gs_covi_wrapper(
    const torch::Tensor &means,
    const torch::Tensor &rotations,
    const torch::Tensor &scales,
    const bool rotnorm,
    const float tol,
    const uint32_t level)
{
    CHECK_INPUT(means);
    CHECK_INPUT(rotations);
    CHECK_INPUT(scales);

    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(rotations.scalar_type() == torch::kFloat32, "rotations must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");

    const uint32_t num_gaussians = means.size(0);
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(rotations.size(1) == 4, "rotations must have shape (N, 4)");
    TORCH_CHECK(scales.size(1) == 3, "scales must have shape (N, 3)");

    auto options = means.options();
    torch::Tensor covi = torch::empty({num_gaussians, 6}, options);

    gs::compute_gs_covi(
        num_gaussians,
        reinterpret_cast<const float4 *>(rotations.data_ptr<float>()),
        reinterpret_cast<const float3 *>(scales.data_ptr<float>()),
        rotnorm,
        tol,
        level,
        reinterpret_cast<float *>(covi.data_ptr<float>()));
    
    return covi;
}

/**
 * @brief Tensor-level entry point for the Gaussian AABB query.
 * @details Sits between pybind11 and the host dispatcher: it applies the `CHECK_INPUT`
 * contract -- CUDA device, contiguous layout, expected dtype -- then unwraps
 * `data_ptr` and calls the kernel launcher. Validating here keeps the launch path
 * free of checks and gives Python callers a clear error instead of a device fault.
 * @return The operator's results as PyTorch tensors.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_gs_aabb_wrapper(
    const torch::Tensor &means,
    const torch::Tensor &scales,
    const torch::Tensor &covis,
    const std::optional<torch::Tensor> &isos,
    const float iso,
    const float tol,
    const uint32_t level)
{
    CHECK_INPUT(means);
    CHECK_INPUT(scales);
    CHECK_INPUT(covis);

    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");

    const uint32_t num_gaussians = means.size(0);
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(scales.size(1) == 3, "scales must have shape (N, 3)");

    float *isos_ptr = nullptr;
    if (isos.has_value())
    {
        CHECK_INPUT(isos.value());
        TORCH_CHECK(isos.value().scalar_type() == torch::kFloat32, "isos must be float32");
        TORCH_CHECK(isos.value().size(0) == num_gaussians, "isos must have shape (N,)");
        isos_ptr = isos.value().data_ptr<float>();
    }

    auto options = means.options();
    torch::Tensor aabb_min = torch::empty({num_gaussians, 3}, options);
    torch::Tensor aabb_max = torch::empty({num_gaussians, 3}, options);
    torch::Tensor contact_points = torch::empty({num_gaussians, 9}, options);

    gs_aabb::compute_gs_aabb(
        num_gaussians,
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float3 *>(scales.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        isos_ptr,
        iso,
        tol,
        level,
        reinterpret_cast<float3 *>(aabb_min.data_ptr<float>()),
        reinterpret_cast<float3 *>(aabb_max.data_ptr<float>()),
        reinterpret_cast<float3 *>(contact_points.data_ptr<float>()));
    
    return std::make_tuple(aabb_min, aabb_max, contact_points);
}

/**
 * @brief Tensor-level entry point for the Gaussian neighbour Mahalanobis radius query.
 * @details Sits between pybind11 and the host dispatcher: it applies the `CHECK_INPUT`
 * contract -- CUDA device, contiguous layout, expected dtype -- then unwraps
 * `data_ptr` and calls the kernel launcher. Validating here keeps the launch path
 * free of checks and gives Python callers a clear error instead of a device fault.
 * @return The operator's results as PyTorch tensors.
 */
torch::Tensor solve_gs_neighbor_mahalanobis_radius_wrapper(
    const torch::Tensor &means,
    const torch::Tensor &covis,
    const int k)
{
    CHECK_INPUT(means);
    CHECK_INPUT(covis);

    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");

    const uint32_t num_gaussians = means.size(0);
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(covis.size(0) == num_gaussians && covis.size(1) == 6, "covis must have shape (N, 6)");

    uint32_t search_k = k + 1;

    TORCH_CHECK(search_k > 1, "k must be greater than 0");
    TORCH_CHECK(search_k <= 32, "Requested k exceeds the hardware register limit (MAX_K = 32).");
    TORCH_CHECK(search_k <= num_gaussians, "Requested k is larger than the number of points in the tree.");

    auto options = means.options();
    torch::Tensor isos = torch::empty({num_gaussians}, options.dtype(torch::kFloat32));

    gs::solve_gs_neighbor_mahalanobis_radius(
        num_gaussians,
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        search_k,
        reinterpret_cast<float *>(isos.data_ptr<float>()));

    return isos;
}

void bind_primitive_gs(py::module_ &m) {
    m.def("compute_gs_covi_func", &compute_gs_covi_wrapper,
          py::arg("means"), py::arg("rotations"), py::arg("scales"),
          py::arg("rotnorm"), py::arg("tol"), py::arg("level"),
          R"pbdoc(
          Computes upper-triangular inverse covariance matrix entries for 3D Gaussians (CUDA).

          Args:
              means (torch.Tensor): (N, 3) float32 coordinates on CUDA.
              rotations (torch.Tensor): (N, 4) float32 quaternion orientations (w, x, y, z) on CUDA.
              scales (torch.Tensor): (N, 3) float32 scaling components on CUDA.
              rotnorm (bool): Whether to normalize quaternions.
              tol (float): Numerical tolerance epsilon.
              level (int): Optimization unroll level.

          Returns:
              torch.Tensor: (N, 6) float32 inverse covariance entries (xx, xy, xz, yy, yz, zz).

          Example:
              >>> import torch
              >>> from conquer3d._C import compute_gs_covi_func
              >>> covis = compute_gs_covi_func(means, rotations, scales, True, 1e-6, 0)
          )pbdoc");
    m.def("solve_gs_neighbor_mahalanobis_radius_func", &solve_gs_neighbor_mahalanobis_radius_wrapper,
          py::arg("means"), py::arg("covis"), py::arg("k"),
          R"pbdoc(
          Computes optimal Mahalanobis isosurface radii for 3D Gaussians from k-NN neighbors (CUDA).

          Args:
              means (torch.Tensor): (N, 3) float32 coordinates on CUDA.
              covis (torch.Tensor): (N, 6) float32 inverse covariances on CUDA.
              k (int): Number of nearest neighbors.

          Returns:
              torch.Tensor: (N,) float32 optimal Mahalanobis radii $r_i$.

          Example:
              >>> radii = solve_gs_neighbor_mahalanobis_radius_func(means, covis, 16)
          )pbdoc");
    m.def("compute_gs_aabb_func", &compute_gs_aabb_wrapper,
          py::arg("means"), py::arg("scales"), py::arg("covis"),
          py::arg("isos"), py::arg("iso"), py::arg("tol"), py::arg("level"),
          R"pbdoc(
          Computes tight Axis-Aligned Bounding Boxes (AABBs) for 3D Gaussians (CUDA).

          Args:
              means (torch.Tensor): (N, 3) float32 Gaussian centroids on CUDA.
              scales (torch.Tensor): (N, 3) float32 scaling components on CUDA.
              covis (torch.Tensor): (N, 6) float32 inverse covariance entries on CUDA.
              isos (torch.Tensor, optional): (N,) float32 per-Gaussian isosurface thresholds.
              iso (float): Global fallback isosurface threshold.
              tol (float): Numerical tolerance.
              level (int): Optimization unroll level.

          Returns:
              Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (aabb_mins, aabb_maxs, contact_points)

          Example:
              >>> mins, maxs, contacts = compute_gs_aabb_func(means, scales, covis, isos, 3.0, 1e-6, 0)
          )pbdoc");
}