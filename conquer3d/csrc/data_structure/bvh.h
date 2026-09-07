/**
 * @file bvh.h
 * @brief High-performance GPU Linear Bounding Volume Hierarchy (LBVH - Karras 2012).
 */

#ifndef BVH_H
#define BVH_H

#include "../maths/maths.h"
#include "../constants.h"
#include "../primitive/aabb.h"
#include "../primitive/ray.h"

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <tuple>

/**
 * @brief High-level Linear BVH class for GPU-accelerated spatial queries.
 */
class BVH
{
public:
    uint32_t num_objects; ///< Number of primitives the hierarchy was built over.
    uint32_t num_nodes; ///< Always $2N - 1$ nodes for $N$ leaf primitives.

    torch::Tensor aabb_mins;    ///< Shape: [2N - 1, 3] minimum bounds.
    torch::Tensor aabb_maxs;    ///< Shape: [2N - 1, 3] maximum bounds.
    torch::Tensor bvh_children; ///< Shape: [2N - 1, 2] child node indices (left, right).
    torch::Tensor bvh_parents;  ///< Shape: [2N - 1] parent node indices.
    torch::Tensor object_ids;   ///< Shape: [N] mapping from sorted leaf index to original primitive ID.

    /**
     * @brief Constructs a Linear BVH from axis-aligned bounding boxes (AABBs).
     * 
     * @param[in] in_aabb_mins (N, 3) float32 tensor of lower box coordinates.
     * @param[in] in_aabb_maxs (N, 3) float32 tensor of upper box coordinates.
     */
    BVH(const torch::Tensor &in_aabb_mins, const torch::Tensor &in_aabb_maxs);

    /**
     * @brief Performs parallel AABB-AABB overlap queries against the BVH.
     * 
     * @param[in] query_aabb_mins (Q, 3) float32 lower bounds of query boxes.
     * @param[in] query_aabb_maxs (Q, 3) float32 upper bounds of query boxes.
     * @return Tuple of (query_ids, object_ids) for all overlapping pairs.
     */
    std::tuple<torch::Tensor, torch::Tensor> query(
        const torch::Tensor &query_aabb_mins,
        const torch::Tensor &query_aabb_maxs);

    /**
     * @brief Performs self-intersection collision detection among all leaves in the BVH.
     * 
     * @return Tuple of (query_ids, object_ids) for all colliding leaf pairs.
     */
    std::tuple<torch::Tensor, torch::Tensor> query_self();

    /**
     * @brief Performs parallel Ray-AABB intersection queries against the BVH.
     * 
     * @param[in] ray_origins   (R, 3) float32 ray origin coordinates.
     * @param[in] ray_dirs      (R, 3) float32 normalized ray directions.
     * @param[in] max_capacity  Maximum capacity for the output pair buffer.
     * @return Tuple of (ray_ids, object_ids) for all ray-box intersections.
     */
    std::tuple<torch::Tensor, torch::Tensor> query_ray(
        const torch::Tensor &ray_origins,
        const torch::Tensor &ray_dirs,
        int64_t max_capacity = BVH_MAX_CAPACITY);

    /**
     * @brief Finds the closest bounding box for each query point.
     * 
     * @param[in] query_points (P, 3) float32 point coordinates.
     * @return Tuple of (query_ids, object_ids, distances).
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> query_point(
        const torch::Tensor &query_points);
};

namespace bvh
{
    /**
     * @brief Constructs the Karras (2012) Radix Linear BVH hierarchy on GPU.
     * 
     * @param[in]  num_objects   Number of leaf objects ($N$).
     * @param[in]  num_nodes     Total internal and leaf nodes ($2N - 1$).
     * @param[in]  in_aabb_mins  Unsorted input lower bounding box coordinates.
     * @param[in]  in_aabb_maxs  Unsorted input upper bounding box coordinates.
     * @param[out] bvh_aabb_mins Output BVH lower bounds array of size $2N - 1$.
     * @param[out] bvh_aabb_maxs Output BVH upper bounds array of size $2N - 1$.
     * @param[out] bvh_children  Output BVH child indices array of size $2N - 1$.
     * @param[out] bvh_parents   Output BVH parent indices array of size $2N - 1$.
     * @param[out] object_ids    Output leaf-to-original primitive ID map of size $N$.
     */
    __host__ void build(
        const uint32_t num_objects,
        const uint32_t num_nodes,
        const float3 *__restrict__ in_aabb_mins,
        const float3 *__restrict__ in_aabb_maxs,
        float3 *__restrict__ bvh_aabb_mins,
        float3 *__restrict__ bvh_aabb_maxs,
        int2 *__restrict__ bvh_children,
        int *__restrict__ bvh_parents,
        int *__restrict__ object_ids
    );

    /**
     * @brief Dispatches parallel AABB-AABB query kernel on GPU.
     */
    __host__ void query(
        const uint32_t num_queries,
        const uint32_t num_objects,
        const float3 *__restrict__ query_mins,
        const float3 *__restrict__ query_maxs,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        int64_t *__restrict__ out_query_ids,
        int64_t *__restrict__ out_object_ids,
        int64_t *__restrict__ hit_counter,
        const int64_t max_capacity);

    /**
     * @brief Dispatches parallel BVH self-collision query kernel on GPU.
     */
    __host__ void query_self(
        const uint32_t num_objects,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        int64_t *__restrict__ out_query_ids,
        int64_t *__restrict__ out_object_ids,
        int64_t *__restrict__ hit_counter,
        const int64_t max_capacity);

    /**
     * @brief Dispatches parallel Ray-AABB intersection query kernel on GPU.
     */
    __host__ void query_ray(
        const uint32_t num_queries,
        const uint32_t num_objects,
        const float3 *__restrict__ ray_origins,
        const float3 *__restrict__ ray_dirs,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        int64_t *__restrict__ out_query_ids,
        int64_t *__restrict__ out_object_ids,
        int64_t *__restrict__ hit_counter,
        const int64_t max_capacity);

    /**
     * @brief Dispatches parallel Point-AABB proximity query kernel on GPU.
     */
    __host__ void query_point(
        const uint32_t num_queries,
        const uint32_t num_objects,
        const float3 *__restrict__ query_points,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        int64_t *__restrict__ out_query_ids,
        int64_t *__restrict__ out_object_ids,
        float *__restrict__ out_distances);
}

#endif // BVH_H