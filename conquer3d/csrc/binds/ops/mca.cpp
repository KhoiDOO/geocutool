#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../ops/mca.h"
#include "../../check.h"

namespace py = pybind11;

/**
 * @brief Tensor-level entry point for the Marching Cubes Asymptotic extraction.
 * @details Sits between pybind11 and the host dispatcher: it applies the `CHECK_INPUT`
 * contract -- CUDA device, contiguous layout, expected dtype -- then unwraps
 * `data_ptr` and calls the kernel launcher. Validating here keeps the launch path
 * free of checks and gives Python callers a clear error instead of a device fault.
 * @return The operator's results as PyTorch tensors.
 */
std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> marching_cubes_asymptotic_wrapper(
    torch::Tensor grid_vertices,
    torch::Tensor voxels,
    torch::Tensor sdf,
    std::optional<torch::Tensor> colors,
    float iso
) {
    CHECK_INPUT(grid_vertices);
    CHECK_INPUT(voxels);
    CHECK_INPUT(sdf);
    if (colors.has_value() && colors.value().defined()) {
        CHECK_INPUT(colors.value());
    }

    return mca::marching_cubes_asymptotic(
        grid_vertices,
        voxels,
        sdf,
        colors,
        iso
    );
}

/**
 * @brief Tensor-level entry point for the Marching Cubes Asymptotic backward pass.
 * @details Validates that every incoming gradient is CUDA-resident and contiguous,
 * unwraps the raw device pointers, and dispatches to the analytical adjoint kernel.
 * @return Gradients with respect to the differentiable inputs.
 * @warning Requires the same inputs the forward pass received; the adjoint recomputes
 * topology rather than storing it.
 */
std::tuple<torch::Tensor, std::optional<torch::Tensor>> marching_cubes_asymptotic_backward_wrapper(
    torch::Tensor grad_vertices,
    std::optional<torch::Tensor> grad_colors,
    torch::Tensor grid_vertices,
    torch::Tensor unique_edges,
    torch::Tensor sdf,
    std::optional<torch::Tensor> colors,
    float iso
) {
    CHECK_INPUT(grad_vertices);
    CHECK_INPUT(grid_vertices);
    CHECK_INPUT(unique_edges);
    CHECK_INPUT(sdf);

    if (grad_colors.has_value() && grad_colors.value().defined()) {
        CHECK_INPUT(grad_colors.value());
    }
    if (colors.has_value() && colors.value().defined()) {
        CHECK_INPUT(colors.value());
    }

    return mca::marching_cubes_asymptotic_backward(
        grad_vertices,
        grad_colors,
        grid_vertices,
        unique_edges,
        sdf,
        colors,
        iso
    );
}

/**
 * @brief Registers the Marching Cubes Asymptotic operator and its backward pass on the extension module.
 * @details Called once from `pybind.cpp` with the root module, so every symbol defined
 * here lands directly on `conquer3d._C`.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_mca(py::module &m) {
    m.def("marching_cubes_asymptotic", &marching_cubes_asymptotic_wrapper,
          py::arg("grid_vertices"), py::arg("voxels"), py::arg("sdf"),
          py::arg("colors") = py::none(), py::arg("iso") = 0.0f,
          R"pbdoc(
          Extracts a watertight 2-manifold surface using Marching Cubes with Asymptotic Deciders (Nielson & Hamann 1991).

          Args:
              grid_vertices (torch.Tensor): (N, 3) float32 coordinates on CUDA.
              voxels (torch.Tensor): (M, 8) int32 corner indices per voxel.
              sdf (torch.Tensor): (N,) float32 scalar SDF values on CUDA.
              colors (torch.Tensor, optional): (N, C) float32 vertex colors on CUDA. Defaults to None.
              iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.

          Returns:
              Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                  - vertices (torch.Tensor): (V, 3) float32 extracted surface coordinates.
                  - triangles (torch.Tensor): (F, 3) int32 triangle face indices.
                  - [colors] (torch.Tensor, optional): (V, C) float32 interpolated features.

          Example:
              >>> import torch
              >>> from conquer3d._C import marching_cubes_asymptotic
              >>> verts, tris, colors = marching_cubes_asymptotic(grid_verts, voxels, sdf, colors, iso=0.0)
          )pbdoc");
    m.def("marching_cubes_asymptotic_backward", &marching_cubes_asymptotic_backward_wrapper,
          py::arg("grad_vertices"), py::arg("grad_colors"), py::arg("grid_vertices"),
          py::arg("unique_edges"), py::arg("sdf"), py::arg("colors"), py::arg("iso") = 0.0f,
          R"pbdoc(
          Analytical backward gradient propagation for Marching Cubes with Asymptotic Deciders on GPU.

          Args:
              grad_vertices (torch.Tensor): (V, 3) float32 upstream vertex gradients.
              grad_colors (torch.Tensor, optional): (V, C) float32 upstream color gradients.
              grid_vertices (torch.Tensor): (N, 3) float32 grid coordinates on CUDA.
              unique_edges (torch.Tensor): (V, 2) int64 unique edge indices.
              sdf (torch.Tensor): (N,) float32 scalar field values on CUDA.
              colors (torch.Tensor, optional): (N, C) float32 colors on CUDA.
              iso (float, optional): Isosurface threshold. Defaults to 0.0.

          Returns:
              Tuple[torch.Tensor, Optional[torch.Tensor]]: (grad_sdf, grad_colors)

          Example:
              >>> grad_sdf, grad_colors = marching_cubes_asymptotic_backward(g_verts, g_colors, grid_verts, u_edges, sdf, colors, 0.0)
          )pbdoc");
}
