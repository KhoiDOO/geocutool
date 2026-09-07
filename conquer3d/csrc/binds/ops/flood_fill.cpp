#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../ops/flood_fill.h"
#include "../../check.h"

namespace py = pybind11;

/**
 * @brief Tensor-level entry point for the volumetric flood fill query.
 * @details Sits between pybind11 and the host dispatcher: it applies the `CHECK_INPUT`
 * contract -- CUDA device, contiguous layout, expected dtype -- then unwraps
 * `data_ptr` and calls the kernel launcher. Validating here keeps the launch path
 * free of checks and gives Python callers a clear error instead of a device fault.
 * @return The operator's results as PyTorch tensors.
 */
torch::Tensor compute_flood_fill_wrapper(
    torch::Tensor vertices,
    torch::Tensor triangles,
    torch::Tensor aabb_mins,
    torch::Tensor aabb_maxs,
    torch::Tensor bvh_children,
    torch::Tensor object_ids,
    std::vector<float> grid_min,
    std::vector<float> grid_max,
    std::vector<int64_t> grid_res,
    int connectivity
) {
    CHECK_INPUT(vertices);
    CHECK_INPUT(triangles);
    CHECK_INPUT(aabb_mins);
    CHECK_INPUT(aabb_maxs);
    CHECK_INPUT(bvh_children);
    CHECK_INPUT(object_ids);
    TORCH_CHECK(grid_min.size() == 3, "grid_min must have 3 elements.");
    TORCH_CHECK(grid_max.size() == 3, "grid_max must have 3 elements.");
    TORCH_CHECK(grid_res.size() == 3, "grid_res must have 3 elements.");
    return ops::compute_flood_fill(vertices, triangles, aabb_mins, aabb_maxs, bvh_children, object_ids, grid_min, grid_max, grid_res, connectivity);
}

/**
 * @brief Registers the volumetric flood fill operator on the extension module.
 * @details Called once from `pybind.cpp` with the root module, so every symbol
 * defined here lands directly on `conquer3d._C`.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_flood_fill(py::module_& m) {
    m.def("compute_flood_fill", &compute_flood_fill_wrapper,
          py::arg("vertices"), py::arg("triangles"), py::arg("aabb_mins"), py::arg("aabb_maxs"),
          py::arg("bvh_children"), py::arg("object_ids"), py::arg("grid_min"), py::arg("grid_max"),
          py::arg("grid_res"), py::arg("connectivity") = 6,
          R"pbdoc(
          Computes a 3D volumetric flood-fill binary occupancy mask on GPU.

          Args:
              vertices (torch.Tensor): (V, 3) float32 mesh vertices on CUDA.
              triangles (torch.Tensor): (F, 3) int32 triangle indices on CUDA.
              aabb_mins (torch.Tensor): (2F-1, 3) float32 BVH node min corners.
              aabb_maxs (torch.Tensor): (2F-1, 3) float32 BVH node max corners.
              bvh_children (torch.Tensor): (2F-1, 2) int32 BVH child node indices.
              object_ids (torch.Tensor): (F,) int32 leaf-to-triangle map.
              grid_min (List[float]): Lower grid extents [x_min, y_min, z_min].
              grid_max (List[float]): Upper grid extents [x_max, y_max, z_max].
              grid_res (List[int]): Grid resolution [rx, ry, rz].
              connectivity (int, optional): Connectivity neighborhood (6, 18, 26). Defaults to 6.

          Returns:
              torch.Tensor: (rx, ry, rz) int8 tensor (0: exterior, 1: surface, -1: interior).

          Example:
              >>> import torch
              >>> from conquer3d._C import compute_flood_fill
              >>> mask = compute_flood_fill(verts, tris, a_mins, a_maxs, children, obj_ids, [-1,-1,-1], [1,1,1], [64,64,64], 6)
          )pbdoc");
}
