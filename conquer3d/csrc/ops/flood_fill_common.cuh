/**
 * @file flood_fill_common.cuh
 * @brief Helpers shared by the single-level and coarse-fine volumetric flood fills.
 *
 * @details Both fills need the same two primitives: a byte-wide compare-and-swap, because
 * CUDA offers no 8-bit `atomicCAS` and storing occupancy as bytes is what keeps a large
 * grid resident, and a segment-versus-mesh test, because deciding whether two neighbouring
 * voxels are connected means asking whether the segment between them crosses the surface.
 * They were duplicated verbatim in both translation units; keeping one copy here means a
 * change to either lands in both fills at once.
 */

#ifndef FLOOD_FILL_COMMON_CUH
#define FLOOD_FILL_COMMON_CUH

#include "../constants.h"
#include "../maths/maths.h"
#include "../primitive/ray.h"
#include "../primitive/triangle.h"
#include "../data_structure/bvh_traverse.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

namespace ops {

/**
     * @brief Compare-and-swap on a single byte, emulated with 32-bit atomics.
     * @details CUDA provides no 8-bit `atomicCAS`, so this reads the containing aligned word,
     * splices the byte, and retries until the word-wide CAS succeeds. Storing occupancy labels
     * as bytes rather than words cuts the mask's footprint fourfold, which is what keeps a
     * $1024^3$ grid resident.
     * @param[in,out] address Byte to update; may be unaligned.
     * @param[in] compare Expected current value.
     * @param[in] val Replacement value.
     * @return The value found at @p address before the attempt.
     * @warning Contention is per 32-bit word, not per byte: four threads updating adjacent
     * bytes serialise against each other even though their targets are disjoint.
     */
    __device__ __forceinline__ int8_t atomicCAS_int8(int8_t* address, int8_t compare, int8_t val) {
        int32_t* address_as_int = (int32_t*)((uintptr_t)address & ~3);
        int shift = (((uintptr_t)address & 3) * 8);
        int32_t old = *address_as_int;
        int32_t assumed;
        do {
            assumed = old;
            int8_t current_val = (int8_t)((assumed >> shift) & 0xff);
            if (current_val != compare) {
                break;
            }
            int32_t new_val = (assumed & ~(0xff << shift)) | ((int32_t)(uint8_t)val << shift);
            old = atomicCAS(address_as_int, assumed, new_val);
        } while (assumed != old);
        return (int8_t)((old >> shift) & 0xff);
    }

/**
     * @brief Tests whether a segment intersects any mesh triangle.
     * @details Traverses the BVH with the segment's own bounding box and applies the exact
     * Moller-Trumbore test to surviving leaves, returning at the first hit. Testing the whole
     * segment rather than its endpoints is what stops the flood fill tunnelling through a wall
     * thinner than the voxel spacing.
     * @param[in] p0 Segment start.
     * @param[in] p1 Segment end.
     * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
     * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
     * @param[in] bvh_children Device array of BVH child index pairs.
     * @param[in] object_ids Device array mapping leaves to triangle indices.
     * @param[in] vertices Device array of mesh vertex coordinates.
     * @param[in] triangles Device array of triangle vertex indices.
     * @return True if the segment meets any triangle.
     * @note Degenerate segments shorter than 1e-8 return false.
     * @warning Uses a per-thread stack of `BVH_STACK_SIZE` entries in local memory.
     */
    __device__ __forceinline__ bool test_segment_intersect_mesh(
        const float3& p0, const float3& p1,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects)
    {
        float3 dir = p1 - p0;
        float len = maths::norm(dir);
        if (len < 1e-8f) return false;
        float3 norm_dir = dir / len;
        
        Ray ray(p0, norm_dir, 0.0f, len);
        
        bool hit = false;
        bvh::traverse(num_objects, bvh_children,
            [&] (int node_idx) {
                float t_hit_aabb;
                float3 box_min = bvh_aabb_mins[node_idx] - make_float3(1e-4f, 1e-4f, 1e-4f);
                float3 box_max = bvh_aabb_maxs[node_idx] + make_float3(1e-4f, 1e-4f, 1e-4f);
                return ray.is_intersect_aabb(box_min, box_max, t_hit_aabb);
            },
            [&] (int leaf_idx) {
                int3 tri = triangles[object_ids[leaf_idx]];
                float3 v0 = vertices[tri.x];
                float3 v1 = vertices[tri.y];
                float3 v2 = vertices[tri.z];

                float3 edge1 = v1 - v0;
                float3 edge2 = v2 - v0;
                float3 h = maths::cross(ray.direction, edge2);
                float a = maths::dot(edge1, h);
                if (a > -1e-8f && a < 1e-8f) return true;

                float f = 1.0f / a;
                float3 s = ray.origin - v0;
                float u = f * maths::dot(s, h);
                if (u < -1e-4f || u > 1.0f + 1e-4f) return true;

                float3 q = maths::cross(s, edge1);
                float v = f * maths::dot(ray.direction, q);
                if (v < -1e-4f || u + v > 1.0f + 1e-4f) return true;

                float t = f * maths::dot(edge2, q);
                if (t >= -1e-4f && t <= len + 1e-4f) { hit = true; return false; }
                return true;
            });
        return hit;
    }

}

#endif // FLOOD_FILL_COMMON_CUH
