/**
 * @file mesh_bvh.h
 * @brief High-performance Triangle Mesh BVH with multi-mode SDF & Fast Winding Number queries.
 */

#ifndef MESH_BVH_H
#define MESH_BVH_H

#include "bvh.h"
#include <torch/extension.h>
#include <tuple>
#include <optional>
#include <vector>

/**
 * @brief Hierarchical dipole node representation for Fast Winding Number (FWN) evaluation.
 */
struct WindingData {
    float3 n;          ///< Area-weighted unnormalized normal vector.
    float3 area_p;     ///< Area-weighted centroid position.
    float area;        ///< Total surface area of subtree.
    float3 average_p;  ///< Average centroid (area_p / area).
    float max_p_dist2; ///< Maximum squared distance from centroid to primitive boundary.
};

/**
 * @brief Triangle Mesh BVH subclass supporting exact Ray-Mesh, Mesh-Mesh, and SDF sign evaluation.
 */
class MeshBVH : public BVH
{
public:
    torch::Tensor winding_data; ///< Size: [2N - 1] * sizeof(WindingData) on device.
    bool has_winding_data = false; ///< Whether build_winding_data() has been called.

    using BVH::BVH;
    using BVH::query;
    using BVH::query_self;
    using BVH::query_ray;

    /**
     * @brief Builds hierarchical dipole winding data for Fast Winding Number queries.
     * 
     * @param[in] vertices  (V, 3) float32 mesh vertex tensor on CUDA.
     * @param[in] triangles (F, 3) int32 triangle index tensor on CUDA.
     */
    void build_winding_data(
        const torch::Tensor &vertices,
        const torch::Tensor &triangles);

    /**
     * @brief Discovers all colliding triangle index pairs in the mesh.
     * 
     * @param[in] vertices  (V, 3) float32 mesh vertex tensor.
     * @param[in] triangles (F, 3) int32 triangle index tensor.
     * @return (N, 2) int64 tensor of intersecting triangle index pairs.
     */
    torch::Tensor get_self_intersection(
        const torch::Tensor &vertices,
        const torch::Tensor &triangles);

    /**
     * @brief Checks if the triangle mesh contains any self-intersecting faces.
     * 
     * @param[in] vertices  (V, 3) float32 mesh vertex tensor.
     * @param[in] triangles (F, 3) int32 triangle index tensor.
     * @return True if self-intersections exist, False otherwise.
     */
    bool is_self_intersection(
        const torch::Tensor &vertices,
        const torch::Tensor &triangles);

    /**
     * @brief Performs accelerated ray-triangle intersection queries (Möller-Trumbore).
     * 
     * @param[in] ray_origins     (R, 3) float32 ray origins.
     * @param[in] ray_dirs        (R, 3) float32 ray unit directions.
     * @param[in] vertices        (V, 3) float32 mesh vertices.
     * @param[in] triangles       (F, 3) int32 triangle indices.
     * @param[in] return_distance If true, computes and returns exact ray hit distances.
     * 
     * @return Tuple of (ray_ids, triangle_ids, intersect_points, distances).
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_ray_intersection(
        const torch::Tensor &ray_origins,
        const torch::Tensor &ray_dirs,
        const torch::Tensor &vertices,
        const torch::Tensor &triangles,
        bool return_distance = false);

    /**
     * @brief Queries closest triangle projections and Signed Distance Fields (SDF) for points.
     * 
     * @param[in] query_points      (P, 3) float32 query coordinates on CUDA.
     * @param[in] vertices          (V, 3) float32 mesh vertices.
     * @param[in] triangles         (F, 3) int32 triangle indices.
     * @param[in] return_sdf        If true, computes signed distance.
     * @param[in] return_prj_pts    If true, computes closest projected points on surface.
     * @param[in] sign_mode         Sign evaluation mode:
     *                              - 0: Ray parity casting.
     *                              - 1: Fast Winding Number (FWN).
     *                              - 2: Angle-weighted pseudonormals.
     *                              - 3: Volumetric 3D flood fill mask (dense).
     *                              - 4: Hybrid Winding Number + Pseudonormals.
     *                              - 5: Coarse-to-Fine (CF) Hierarchical Volumetric Flood Fill.
     * @param[in] triangle_normals  Optional (F, 3) triangle face normals.
     * @param[in] vertex_normals    Optional (V, 3) vertex pseudonormals.
     * @param[in] edge_normals      Optional (3*F, 3) edge pseudonormals.
     * @param[in] flood_fill_mask   Optional 3D grid flood fill mask.
     * @param[in] flood_grid_min    Optional flood grid min bounds.
     * @param[in] flood_grid_max    Optional flood grid max bounds.
     * @param[in] flood_grid_res    Optional flood grid resolution.
     * @param[in] cf_coarse_mask    Optional (Cx, Cy, Cz) int8 coarse mask for sign_mode=5.
     * @param[in] cf_boundary_lookup Optional (Cx, Cy, Cz) int32 boundary lookup table for sign_mode=5.
     * @param[in] cf_fine_masks    Optional (N_boundary, Bx, By, Bz) int8 fine masks for sign_mode=5.
     * @param[in] cf_block_size     Optional macro-block size [Bx, By, Bz].
     * @param[in] cf_coarse_res     Optional coarse grid resolution [Cx, Cy, Cz].
     * 
     * @return Tuple of (query_ids, closest_triangle_ids, projected_points, signed_distances).
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> query_point(
        const torch::Tensor &query_points,
        const torch::Tensor &vertices,
        const torch::Tensor &triangles,
        bool return_sdf = false,
        bool return_prj_pts = true,
        int sign_mode = 0,
        std::optional<torch::Tensor> triangle_normals = std::nullopt,
        std::optional<torch::Tensor> vertex_normals = std::nullopt,
        std::optional<torch::Tensor> edge_normals = std::nullopt,
        std::optional<torch::Tensor> flood_fill_mask = std::nullopt,
        std::optional<std::vector<float>> flood_grid_min = std::nullopt,
        std::optional<std::vector<float>> flood_grid_max = std::nullopt,
        std::optional<std::vector<int64_t>> flood_grid_res = std::nullopt,
        std::optional<torch::Tensor> cf_coarse_mask = std::nullopt,
        std::optional<torch::Tensor> cf_boundary_lookup = std::nullopt,
        std::optional<torch::Tensor> cf_fine_masks = std::nullopt,
        std::optional<std::vector<int64_t>> cf_block_size = std::nullopt,
        std::optional<std::vector<int64_t>> cf_coarse_res = std::nullopt);

    /**
     * @brief Performs triangle-box intersection tests against voxel cells.
     */
    torch::Tensor query_voxel(
        const torch::Tensor &query_mins,
        const torch::Tensor &query_maxs,
        const torch::Tensor &vertices,
        const torch::Tensor &triangles);

    /**
     * @brief Extracts linear IDs of active surface-intersecting voxels in a 3D grid.
     */
    torch::Tensor get_active_voxel_ids_from_grid(
        std::vector<float> grid_min,
        std::vector<float> grid_max,
        std::vector<int64_t> res,
        const torch::Tensor &vertices,
        const torch::Tensor &triangles);
};

namespace mesh_bvh
{
    __host__ void filter_self_intersections(
        const int num_pairs,
        const int64_t *query_ids,
        const int64_t *object_ids,
        const float3 *vertices,
        const int3 *triangles,
        int64_t *out_query_ids,
        int64_t *out_object_ids,
        int64_t *valid_counter);

    __host__ void filter_ray_triangle_intersections(
        const int num_pairs,
        const int64_t *query_ids,
        const int64_t *object_ids,
        const float3 *ray_origins,
        const float3 *ray_dirs,
        const float3 *vertices,
        const int3 *triangles,
        int64_t *out_query_ids,
        int64_t *out_object_ids,
        float3 *out_intersect_pts,
        float *out_distances,
        bool return_distance,
        int64_t *valid_counter);
    
    __host__ void query_point_mesh_bvh(
        const int num_queries,
        const int num_objects,
        const float3 *query_points,
        const float3 *vertices,
        const int3 *triangles,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        const WindingData *winding_data,
        const float3 *pseudonormal_vertices,
        const float3 *pseudonormal_edges,
        const float3 *pseudonormal_faces,
        int64_t *out_query_ids,
        int64_t *out_object_ids,
        float3 *out_projected_pts,
        float *out_distances,
        bool return_sdf,
        bool return_prj_pts,
        int sign_mode,
        const int8_t *flood_mask = nullptr,
        float3 flood_min = make_float3(0.0f, 0.0f, 0.0f),
        float3 flood_spacing = make_float3(1.0f, 1.0f, 1.0f),
        int3 flood_dims = make_int3(0, 0, 0),
        const int8_t *cf_coarse_mask = nullptr,
        const int32_t *cf_boundary_lookup = nullptr,
        const int8_t *cf_fine_masks = nullptr,
        int3 cf_block_size = make_int3(8, 8, 8),
        int3 cf_coarse_dims = make_int3(0, 0, 0));
    
    __host__ void bottom_up_winding_data(
        const int num_objects,
        const int *object_ids,
        const float3 *vertices,
        const int3 *triangles,
        const int *bvh_parents,
        const int2 *bvh_children,
        WindingData *winding_data);

    __host__ void query_voxel_mesh_bvh(
        const int num_queries,
        const int num_objects,
        const float3 *query_mins,
        const float3 *query_maxs,
        const float3 *vertices,
        const int3 *triangles,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        bool *out_intersect);

    __host__ void count_active_voxels_mesh_bvh(
        const int3 res,
        const float3 grid_min,
        const float3 voxel_size,
        const int num_objects,
        const float3 *vertices,
        const int3 *triangles,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        int64_t *active_counter);

    __host__ void collect_active_voxels_mesh_bvh(
        const int3 res,
        const float3 grid_min,
        const float3 voxel_size,
        const int num_objects,
        const float3 *vertices,
        const int3 *triangles,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        int64_t *active_counter,
        int64_t *out_active_ids);
}

#endif // MESH_BVH_H
