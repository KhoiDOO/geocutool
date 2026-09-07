#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../ops/mt.h"
#include "../../check.h"

namespace py = pybind11;

/**
 * @brief Tensor-level entry point for the Marching Tetrahedra extraction.
 * @details Sits between pybind11 and the host dispatcher: it applies the `CHECK_INPUT`
 * contract -- CUDA device, contiguous layout, expected dtype -- then unwraps
 * `data_ptr` and calls the kernel launcher. Validating here keeps the launch path
 * free of checks and gives Python callers a clear error instead of a device fault.
 * @return The operator's results as PyTorch tensors.
 */
std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>> marching_tetrahedra_wrapper(
    torch::Tensor grid_vertices,
    torch::Tensor tets,
    torch::Tensor vert_values,
    std::optional<torch::Tensor> grid_normals,
    std::optional<torch::Tensor> grid_colors,
    float iso,
    bool return_unique_edges
) {
    CHECK_INPUT(grid_vertices);
    CHECK_INPUT(tets);
    CHECK_INPUT(vert_values);

    uint32_t num_tets = tets.size(0);

    const float3* __restrict__ p_grid_vertices = (float3*)grid_vertices.data_ptr<float>();
    const uint32_t* __restrict__ p_tets = (uint32_t*)tets.data_ptr<int32_t>();
    const float* __restrict__ p_vert_values = vert_values.data_ptr<float>();

    const float3* __restrict__ p_grid_normals = nullptr;
    if (grid_normals.has_value()) {
        CHECK_INPUT(grid_normals.value());
        p_grid_normals = (float3*)grid_normals.value().data_ptr<float>();
    }

    const float3* __restrict__ p_grid_colors = nullptr;
    if (grid_colors.has_value()) {
        CHECK_INPUT(grid_colors.value());
        p_grid_colors = (float3*)grid_colors.value().data_ptr<float>();
    }

    return mt::marching_tetrahedra(
        num_tets,
        p_grid_vertices,
        p_tets,
        p_vert_values,
        p_grid_normals,
        p_grid_colors,
        iso,
        grid_vertices.options(),
        tets.options(),
        return_unique_edges
    );
}

/**
 * @brief Tensor-level entry point for the Marching Tetrahedra backward pass.
 * @details Validates that every incoming gradient is CUDA-resident and contiguous,
 * unwraps the raw device pointers, and dispatches to the analytical adjoint kernel.
 * @return Gradients with respect to the differentiable inputs.
 * @warning Requires the same inputs the forward pass received; the adjoint recomputes
 * topology rather than storing it.
 */
void marching_tetrahedra_backward_wrapper(
    torch::Tensor unique_edges,
    torch::Tensor grid_vertices,
    std::optional<torch::Tensor> grid_colors,
    torch::Tensor values,
    torch::Tensor adj_verts,
    std::optional<torch::Tensor> adj_colors,
    torch::Tensor adj_values,
    std::optional<torch::Tensor> adj_grid_colors,
    float iso
) {
    CHECK_INPUT(unique_edges);
    CHECK_INPUT(grid_vertices);
    CHECK_INPUT(values);
    CHECK_INPUT(adj_verts);
    CHECK_INPUT(adj_values);

    uint32_t n_verts = unique_edges.size(0);

    const Edge* p_unique_edges = (const Edge*)unique_edges.data_ptr<int32_t>();
    const float3* p_grid_vertices = (const float3*)grid_vertices.data_ptr<float>();
    const float* p_values = values.data_ptr<float>();
    const float3* p_adj_verts = (const float3*)adj_verts.data_ptr<float>();
    float* p_adj_values = adj_values.data_ptr<float>();

    const float3* p_grid_colors = nullptr;
    if (grid_colors.has_value()) {
        CHECK_INPUT(grid_colors.value());
        p_grid_colors = (const float3*)grid_colors.value().data_ptr<float>();
    }

    const float3* p_adj_colors = nullptr;
    if (adj_colors.has_value()) {
        CHECK_INPUT(adj_colors.value());
        p_adj_colors = (const float3*)adj_colors.value().data_ptr<float>();
    }

    float3* p_adj_grid_colors = nullptr;
    if (adj_grid_colors.has_value()) {
        CHECK_INPUT(adj_grid_colors.value());
        p_adj_grid_colors = (float3*)adj_grid_colors.value().data_ptr<float>();
    }

    mt::backward(
        n_verts,
        p_unique_edges,
        p_grid_vertices,
        p_grid_colors,
        p_values,
        p_adj_verts,
        p_adj_colors,
        p_adj_values,
        p_adj_grid_colors,
        iso
    );
}

/**
 * @brief Registers the Marching Tetrahedra operator and its backward pass on the extension module.
 * @details Called once from `pybind.cpp` with the root module, so every symbol
 * defined here lands directly on `conquer3d._C`.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_mt(py::module_& m) {
    m.def("marching_tetrahedra", &marching_tetrahedra_wrapper,
          py::arg("grid_vertices"), py::arg("tets"), py::arg("vert_values"),
          py::arg("grid_normals") = std::nullopt, py::arg("grid_colors") = std::nullopt,
          py::arg("iso") = 0.0f, py::arg("return_unique_edges") = false,
          R"pbdoc(
          GPU-accelerated Marching Tetrahedra algorithm for unstructured tetrahedral meshes (Treece et al. 1999).

          Args:
              grid_vertices (torch.Tensor): (N, 3) float32 coordinates on CUDA.
              tets (torch.Tensor): (T, 4) int32 tetrahedron corner indices on CUDA.
              vert_values (torch.Tensor): (N,) float32 scalar SDF values on CUDA.
              grid_normals (torch.Tensor, optional): (N, 3) float32 vertex normals on CUDA. Defaults to None.
              grid_colors (torch.Tensor, optional): (N, 3) float32 vertex colors on CUDA. Defaults to None.
              iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.
              return_unique_edges (bool, optional): Track unique edges for autograd. Defaults to False.

          Returns:
              Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
                  - vertices (torch.Tensor): (V_out, 3) float32 extracted surface coordinates.
                  - triangles (torch.Tensor): (F, 3) int32 triangle face indices.
                  - [normals] (torch.Tensor, optional): (V_out, 3) float32 interpolated normals.
                  - [colors] (torch.Tensor, optional): (V_out, 3) float32 interpolated colors.
                  - [unique_edges] (torch.Tensor, optional): (V_out, 2) int64 unique tet edge indices.

          Example:
              >>> import torch
              >>> from conquer3d._C import marching_tetrahedra
              >>> verts, tris, _, _, _ = marching_tetrahedra(grid_verts, tets, sdf, iso=0.0)
          )pbdoc");
    m.def("marching_tetrahedra_backward", &marching_tetrahedra_backward_wrapper,
          py::arg("unique_edges"), py::arg("grid_vertices"), py::arg("grid_colors"), py::arg("values"),
          py::arg("adj_verts"), py::arg("adj_colors"), py::arg("adj_values"), py::arg("adj_grid_colors"), py::arg("iso") = 0.0f,
          R"pbdoc(
          Analytical backward gradient propagation for Marching Tetrahedra on GPU.

          Args:
              unique_edges (torch.Tensor): (V_out, 2) int64 unique edge indices.
              grid_vertices (torch.Tensor): (N, 3) float32 coordinates on CUDA.
              grid_colors (torch.Tensor, optional): (N, 3) float32 colors on CUDA.
              values (torch.Tensor): (N,) float32 scalar values on CUDA.
              adj_verts (torch.Tensor): (V_out, 3) float32 upstream vertex gradients.
              adj_colors (torch.Tensor, optional): (V_out, 3) float32 upstream color gradients.
              adj_values (torch.Tensor): (N,) float32 scalar field gradient accumulator on CUDA.
              adj_grid_colors (torch.Tensor, optional): (N, 3) float32 color gradient accumulator on CUDA.
              iso (float, optional): Isosurface threshold. Defaults to 0.0.

          Example:
              >>> marching_tetrahedra_backward(u_edges, g_verts, g_colors, vals, adj_verts, adj_colors, adj_vals, adj_g_colors, 0.0)
          )pbdoc");
}
