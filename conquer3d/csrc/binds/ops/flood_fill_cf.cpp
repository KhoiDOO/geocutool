#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../ops/flood_fill_cf.h"
#include "../../check.h"

namespace py = pybind11;

/**
 * @brief Tensor-level entry point for the coarse-fine volumetric flood fill query.
 * @details Sits between pybind11 and the host dispatcher: it applies the `CHECK_INPUT`
 * contract -- CUDA device, contiguous layout, expected dtype -- then unwraps
 * `data_ptr` and calls the kernel launcher. Validating here keeps the launch path
 * free of checks and gives Python callers a clear error instead of a device fault.
 * @return The operator's results as PyTorch tensors.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<int64_t>, std::vector<int64_t>> compute_flood_fill_cf_wrapper(
    torch::Tensor vertices,
    torch::Tensor triangles,
    torch::Tensor aabb_mins,
    torch::Tensor aabb_maxs,
    torch::Tensor bvh_children,
    torch::Tensor object_ids,
    std::vector<float> grid_min,
    std::vector<float> grid_max,
    std::vector<int64_t> grid_res,
    std::vector<int64_t> block_size,
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
    
    auto res = ops::compute_flood_fill_cf(
        vertices, triangles, aabb_mins, aabb_maxs, bvh_children, object_ids,
        grid_min, grid_max, grid_res, block_size, connectivity
    );

    return std::make_tuple(
        res.coarse_mask,
        res.boundary_block_coords,
        res.boundary_block_lookup,
        res.fine_boundary_masks,
        res.block_size,
        res.coarse_res
    );
}

/**
 * @brief Registers the coarse-fine volumetric flood fill operator on the extension module.
 * @details Called once from `pybind.cpp` with the root module, so every symbol
 * defined here lands directly on `conquer3d._C`.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_flood_fill_cf(py::module_& m) {
    m.def("compute_flood_fill_cf", &compute_flood_fill_cf_wrapper,
          py::arg("vertices"), py::arg("triangles"), py::arg("aabb_mins"), py::arg("aabb_maxs"),
          py::arg("bvh_children"), py::arg("object_ids"), py::arg("grid_min"), py::arg("grid_max"),
          py::arg("grid_res"), py::arg("block_size") = std::vector<int64_t>{}, py::arg("connectivity") = 6,
          R"pbdoc(
          Computes a 2-level Coarse-to-Fine (CF) Volumetric Flood Fill on GPU (< 10 MB VRAM at 1024^3).

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
              block_size (List[int], optional): Macro-block size [bx, by, bz]. If omitted, dynamically computed.
              connectivity (int, optional): Connectivity neighborhood. Defaults to 6.

          Returns:
              Tuple[Tensor, Tensor, Tensor, Tensor, List[int], List[int]]:
                  - coarse_mask (Tensor): (Cx, Cy, Cz) int8 tensor.
                  - boundary_block_coords (Tensor): (N_boundary, 3) int32 coordinates.
                  - boundary_block_lookup (Tensor): (Cx, Cy, Cz) int32 lookup tensor.
                  - fine_boundary_masks (Tensor): (N_boundary, Bx, By, Bz) int8 fine masks.
                  - block_size (List[int]): [Bx, By, Bz].
                  - coarse_res (List[int]): [Cx, Cy, Cz].
          )pbdoc");
}
