#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../ops/dc.h"
#include "../../check.h"

namespace py = pybind11;

/**
 * @brief Tensor-level entry point for the Dual Contouring extraction.
 * @details Sits between pybind11 and the host dispatcher: it applies the `CHECK_INPUT`
 * contract -- CUDA device, contiguous layout, expected dtype -- then unwraps
 * `data_ptr` and calls the kernel launcher. Validating here keeps the launch path
 * free of checks and gives Python callers a clear error instead of a device fault.
 * @return The operator's results as PyTorch tensors.
 */
std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> dual_contouring_wrapper(
    torch::Tensor grid_vertices,
    torch::Tensor voxels,
    torch::Tensor sdf,
    std::optional<torch::Tensor> grid_normals,
    std::optional<torch::Tensor> colors,
    std::optional<torch::Tensor> voxel_vertices,
    float iso,
    bool quad_split
) {
    CHECK_INPUT(grid_vertices);
    CHECK_INPUT(voxels);
    CHECK_INPUT(sdf);

    if (grid_normals.has_value() && grid_normals.value().defined()) {
        CHECK_INPUT(grid_normals.value());
    }
    if (colors.has_value() && colors.value().defined()) {
        CHECK_INPUT(colors.value());
    }
    if (voxel_vertices.has_value() && voxel_vertices.value().defined()) {
        CHECK_INPUT(voxel_vertices.value());
    }

    return conquer3d::ops::dual_contouring(
        grid_vertices,
        voxels,
        sdf,
        grid_normals,
        colors,
        voxel_vertices,
        iso,
        quad_split
    );
}

/**
 * @brief Tensor-level entry point for the Dual Contouring backward pass.
 * @details Validates that every incoming gradient is CUDA-resident and contiguous,
 * unwraps the raw device pointers, and dispatches to the analytical adjoint kernel.
 * @return Gradients with respect to the differentiable inputs.
 * @warning Requires the same inputs the forward pass received; the adjoint recomputes
 * topology rather than storing it.
 */
std::tuple<torch::Tensor, std::optional<torch::Tensor>> dual_contouring_backward_wrapper(
    torch::Tensor grad_verts,
    std::optional<torch::Tensor> grad_colors,
    torch::Tensor grid_vertices,
    torch::Tensor voxels,
    torch::Tensor sdf,
    std::optional<torch::Tensor> grid_normals,
    std::optional<torch::Tensor> colors,
    float iso
) {
    CHECK_INPUT(grad_verts);
    CHECK_INPUT(grid_vertices);
    CHECK_INPUT(voxels);
    CHECK_INPUT(sdf);

    if (grad_colors.has_value() && grad_colors.value().defined()) {
        CHECK_INPUT(grad_colors.value());
    }
    if (grid_normals.has_value() && grid_normals.value().defined()) {
        CHECK_INPUT(grid_normals.value());
    }
    if (colors.has_value() && colors.value().defined()) {
        CHECK_INPUT(colors.value());
    }

    return conquer3d::ops::dual_contouring_backward(
        grad_verts,
        grad_colors,
        grid_vertices,
        voxels,
        sdf,
        grid_normals,
        colors,
        iso
    );
}

/**
 * @brief Registers the Dual Contouring operator and its backward pass on the extension module.
 * @details Called once from `pybind.cpp` with the root module, so every symbol defined
 * here lands directly on `conquer3d._C`.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_dc(py::module &m) {
    m.def("dual_contouring", &dual_contouring_wrapper,
          py::arg("grid_vertices"), py::arg("voxels"), py::arg("sdf"),
          py::arg("grid_normals") = py::none(), py::arg("colors") = py::none(),
          py::arg("voxel_vertices") = py::none(),
          py::arg("iso") = 0.0f, py::arg("quad_split") = true,
          R"pbdoc(
          Extracts a sharp-feature preserving surface mesh using Dual Contouring with GPU QEF solver (Ju et al. 2002) or precomputed voxel vertices.

          Args:
              grid_vertices (torch.Tensor): (N, 3) float32 corner coordinates on CUDA.
              voxels (torch.Tensor): (M, 8) int32 corner indices per voxel cell.
              sdf (torch.Tensor): (N,) float32 scalar SDF values on CUDA.
              grid_normals (torch.Tensor, optional): (N, 3) float32 explicit vertex normals for sharp CAD features. Defaults to None.
              colors (torch.Tensor, optional): (N, C) float32 vertex features/colors on CUDA. Defaults to None.
              voxel_vertices (torch.Tensor, optional): (M, 3) float32 precomputed inside-voxel vertex coordinates on CUDA. Defaults to None.
              iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.
              quad_split (bool, optional): If True, splits quads into Delaunay triangles; if False, returns quads. Defaults to True.

          Returns:
              Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                  - vertices (torch.Tensor): (V, 3) float32 extracted surface vertices.
                  - faces (torch.Tensor): (F, 3) int32 triangles if quad_split=True, or (Q, 4) quads if quad_split=False.
                  - [colors] (torch.Tensor, optional): (V, C) float32 interpolated colors.

          Example:
              >>> import torch
              >>> from conquer3d._C import dual_contouring
              >>> verts, faces, _ = dual_contouring(grid_verts, voxels, sdf, iso=0.0)
          )pbdoc");
    m.def("dual_contouring_backward", &dual_contouring_backward_wrapper,
          py::arg("grad_verts"), py::arg("grad_colors"), py::arg("grid_vertices"),
          py::arg("voxels"), py::arg("sdf"), py::arg("grid_normals") = py::none(),
          py::arg("colors") = py::none(), py::arg("iso") = 0.0f,
          R"pbdoc(
          Analytical backward gradient propagation for Dual Contouring w.r.t. SDF and colors.

          Args:
              grad_verts (torch.Tensor): Upstream gradient w.r.t. extracted vertices (V, 3).
              grad_colors (torch.Tensor, optional): Upstream gradient w.r.t. colors (V, C).
              grid_vertices (torch.Tensor): (N, 3) float32 corner coordinates on CUDA.
              voxels (torch.Tensor): (M, 8) int32 corner indices.
              sdf (torch.Tensor): (N,) float32 scalar SDF values on CUDA.
              grid_normals (torch.Tensor, optional): (N, 3) float32 normals on CUDA.
              colors (torch.Tensor, optional): (N, C) float32 vertex colors on CUDA.
              iso (float, optional): Isosurface threshold. Defaults to 0.0.

          Returns:
              Tuple[torch.Tensor, Optional[torch.Tensor]]: (grad_sdf, grad_colors)

          Example:
              >>> grad_sdf, grad_colors = dual_contouring_backward(g_verts, g_colors, grid_verts, voxels, sdf, iso=0.0)
          )pbdoc");
}
