#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../ops/chamfer.h"
#include "../../check.h"

namespace py = pybind11;

/**
 * @brief Tensor-level entry point for the one-sided Chamfer distance query.
 * @details Sits between pybind11 and the host dispatcher: it applies the `CHECK_INPUT`
 * contract -- CUDA device, contiguous layout, expected dtype -- then unwraps
 * `data_ptr` and calls the kernel launcher. Validating here keeps the launch path
 * free of checks and gives Python callers a clear error instead of a device fault.
 * @return The operator's results as PyTorch tensors.
 */
std::tuple<torch::Tensor, torch::Tensor> one_sided_chamfer_distance_wrapper(
    torch::Tensor query_points,
    torch::Tensor reference_points
) {
    CHECK_INPUT(query_points);
    CHECK_INPUT(reference_points);

    uint32_t num_query_points = query_points.size(0);
    uint32_t num_reference_points = reference_points.size(0);

    if (num_query_points == 0) {
        return std::make_tuple(
            torch::empty({0}, query_points.options()),
            torch::empty({0}, query_points.options().dtype(torch::kInt64))
        );
    }

    if (num_reference_points == 0) {
        return std::make_tuple(
            torch::full({(int64_t)num_query_points}, std::numeric_limits<float>::infinity(), query_points.options()),
            torch::full({(int64_t)num_query_points}, -1, query_points.options().dtype(torch::kInt64))
        );
    }

    const float3* __restrict__ p_query_points = (float3*)query_points.data_ptr<float>();
    const float3* __restrict__ p_reference_points = (float3*)reference_points.data_ptr<float>();

    torch::Tensor distances = torch::empty({(int64_t)num_query_points}, query_points.options());
    torch::Tensor indices = torch::empty({(int64_t)num_query_points}, query_points.options().dtype(torch::kInt64));

    float* __restrict__ p_distances = distances.data_ptr<float>();
    int64_t* __restrict__ p_indices = indices.data_ptr<int64_t>();

    one_sided_chamfer_distance(
        num_query_points,
        p_query_points,
        num_reference_points,
        p_reference_points,
        p_distances,
        p_indices
    );

    return std::make_tuple(distances, indices);
}

/**
 * @brief Registers the one-sided Chamfer distance operator on the extension module.
 * @details Called once from `pybind.cpp` with the root module, so every symbol
 * defined here lands directly on `conquer3d._C`.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_chamfer(py::module_& m) {
    m.def("one_sided_chamfer_distance", &one_sided_chamfer_distance_wrapper,
          py::arg("query_points"), py::arg("reference_points"),
          R"pbdoc(
          Computes one-sided nearest-neighbor squared Euclidean distances from query to reference point clouds on CUDA.

          Args:
              query_points (torch.Tensor): (N, 3) float32 coordinates on CUDA.
              reference_points (torch.Tensor): (M, 3) float32 coordinates on CUDA.

          Returns:
              Tuple[torch.Tensor, torch.Tensor]:
                  - distances (torch.Tensor): (N,) float32 squared Euclidean distance to closest reference point.
                  - indices (torch.Tensor): (N,) int64 index of the closest reference point for each query point.

          Example:
              >>> import torch
              >>> from conquer3d._C import one_sided_chamfer_distance
              >>> dists, inds = one_sided_chamfer_distance(query_pts, ref_pts)
          )pbdoc");
}
