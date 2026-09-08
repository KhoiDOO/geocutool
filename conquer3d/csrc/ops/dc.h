#ifndef DC_H
#define DC_H

#include <torch/extension.h>
#include <tuple>

namespace conquer3d {
namespace ops {

/**
 * @brief Forward Dual Contouring with GPU QEF solver.
 *
 * @param grid_vertices (N, 3) float32 corner coordinates.
 * @param voxels (M, 8) int32 corner indices in CCW order.
 * @param sdf (N,) float32 scalar field.
 * @param grid_normals (N, 3) float32 optional explicit vertex normals.
 * @param colors (N, C) float32 optional vertex feature colors.
 * @param iso Isolevel threshold (default: 0.0).
 * @param quad_split If true, splits quads into optimal triangles; if false, outputs quads.
 * @return std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
 *         - Extracted vertices: (V, 3) float32
 *         - Extracted faces: (F, 3) int32 triangles or (Q, 4) int32 quads
 *         - Extracted colors: (V, C) float32 if colors provided, else nullopt
 */
std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> dual_contouring(
    const at::Tensor &grid_vertices,
    const at::Tensor &voxels,
    const at::Tensor &sdf,
    const c10::optional<at::Tensor> &grid_normals = c10::nullopt,
    const c10::optional<at::Tensor> &colors = c10::nullopt,
    const c10::optional<at::Tensor> &voxel_vertices = c10::nullopt,
    float iso = 0.0f,
    bool quad_split = true,
    const c10::optional<at::Tensor> &edge_points = c10::nullopt,
    const c10::optional<at::Tensor> &edge_normals = c10::nullopt
);

/**
 * @brief Analytical Backward Dual Contouring w.r.t. scalar SDF and colors.
 */
std::tuple<at::Tensor, c10::optional<at::Tensor>> dual_contouring_backward(
    const at::Tensor &grad_verts,
    const c10::optional<at::Tensor> &grad_colors,
    const at::Tensor &grid_vertices,
    const at::Tensor &voxels,
    const at::Tensor &sdf,
    const c10::optional<at::Tensor> &grid_normals,
    const c10::optional<at::Tensor> &colors,
    float iso = 0.0f
);

} // namespace ops
} // namespace conquer3d

#endif // DC_H
