#ifndef CONQUER3D_OPS_DMC_H
#define CONQUER3D_OPS_DMC_H

#include <torch/extension.h>
#include <tuple>
#include <c10/util/Optional.h>

namespace conquer3d {
namespace ops {

/**
 * @brief Forward pass of Differentiable Dual Marching Cubes (DMC).
 * 
 * @param grid_vertices (N, 3) 3D coordinate tensor of unique grid vertices.
 * @param voxels (M, 8) integer voxel index grid.
 * @param sdf (N,) 1D scalar SDF tensor on grid vertices.
 * @param colors Optional (N, C) vertex color/feature tensor.
 * @param iso Isosurface threshold value (default: 0.0).
 * @param quad_split If true, splits quads into Delaunay triangles; if false, emits quads.
 * @param project_iters Number of Newton-Raphson level-set projection iterations.
 * @return std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> 
 *         (vertices, faces/quads, colors)
 */
std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> dual_marching_cubes(
    const at::Tensor &grid_vertices,
    const at::Tensor &voxels,
    const at::Tensor &sdf,
    const c10::optional<at::Tensor> &colors = c10::nullopt,
    const c10::optional<at::Tensor> &voxel_vertices = c10::nullopt,
    float iso = 0.0f,
    bool quad_split = true,
    int project_iters = 5,
    const c10::optional<at::Tensor> &edge_points = c10::nullopt,
    const c10::optional<at::Tensor> &edge_normals = c10::nullopt
);

/**
 * @brief Analytical backward pass of Differentiable Dual Marching Cubes (DMC).
 */
std::tuple<at::Tensor, c10::optional<at::Tensor>> dual_marching_cubes_backward(
    const at::Tensor &grad_vertices,
    const c10::optional<at::Tensor> &grad_colors,
    const at::Tensor &grid_vertices,
    const at::Tensor &voxels,
    const at::Tensor &sdf,
    const c10::optional<at::Tensor> &colors,
    float iso = 0.0f,
    int project_iters = 5
);

} // namespace ops
} // namespace conquer3d

#endif // CONQUER3D_OPS_DMC_H
