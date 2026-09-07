/**
 * @file flood_fill_cf.h
 * @brief High-performance GPU Coarse-to-Fine (CF) Hierarchical Volumetric Flood-Fill.
 * 
 * Provides a 2-level hierarchical spatial flood-fill pipeline consuming < 10 MB of VRAM
 * at 1024^3 resolution with exact topological inside/outside sign determination.
 */

#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

namespace ops {

    /**
     * @brief Result container for Coarse-to-Fine Volumetric Flood Fill.
     */
    struct CFFloodFillResult {
        torch::Tensor coarse_mask; ///< (Cx, Cy, Cz) int8 labels: 2 exterior, -1 interior, 1 surface-crossing.
        torch::Tensor boundary_block_coords; ///< (N, 3) int32 coarse coordinates of the surface-crossing blocks.
        torch::Tensor boundary_block_lookup; ///< (Cx, Cy, Cz) int32 index into the boundary arrays, -1 when not a boundary block.
        torch::Tensor fine_boundary_masks; ///< (N, Bx, By, Bz) int8 per-voxel labels inside each surface-crossing block.
        std::vector<int64_t> block_size; ///< Fine voxels per coarse block, [Bx, By, Bz].
        std::vector<int64_t> coarse_res; ///< Coarse grid resolution, [Cx, Cy, Cz].
        std::vector<float> grid_min; ///< World coordinate of the grid's lower corner.
        std::vector<float> grid_max; ///< World coordinate of the grid's upper corner.
        std::vector<int64_t> grid_res; ///< Fine grid resolution.
    };

    /**
     * @brief Computes 2-Level Coarse-to-Fine Volumetric Flood Fill on GPU.
     *
     * @param[in] vertices      (V, 3) float32 mesh vertex tensor.
     * @param[in] triangles     (F, 3) int32 triangle index tensor.
     * @param[in] aabb_mins     (2F-1, 3) BVH lower box coordinates.
     * @param[in] aabb_maxs     (2F-1, 3) BVH upper box coordinates.
     * @param[in] bvh_children  (2F-1, 2) BVH child node indices.
     * @param[in] object_ids    (F,) leaf-to-triangle map.
     * @param[in] grid_min      3D lower coordinate bounds [x_min, y_min, z_min].
     * @param[in] grid_max      3D upper coordinate bounds [x_max, y_max, z_max].
     * @param[in] grid_res      3D fine grid resolution [rx, ry, rz].
     * @param[in] block_size    Optional macro-block size [bx, by, bz]. If empty, computed dynamically.
     * @param[in] connectivity  Voxel neighbor connectivity (6, 18, 26).
     * @return CFFloodFillResult struct holding coarse mask and fine boundary masks.
     */
    CFFloodFillResult compute_flood_fill_cf(
        const torch::Tensor& vertices,
        const torch::Tensor& triangles,
        const torch::Tensor& aabb_mins,
        const torch::Tensor& aabb_maxs,
        const torch::Tensor& bvh_children,
        const torch::Tensor& object_ids,
        std::vector<float> grid_min,
        std::vector<float> grid_max,
        std::vector<int64_t> grid_res,
        std::vector<int64_t> block_size = {},
        int connectivity = 6
    );

} // namespace ops
