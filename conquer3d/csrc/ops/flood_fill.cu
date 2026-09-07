/**
 * @file flood_fill.cu
 * @brief CUDA kernel implementations for BVH segment intersection and 3D volumetric flood-fill occupancy.
 */

#include "flood_fill.h"
#include "../constants.h"
#include "../primitive/ray.h"
#include "../primitive/triangle.h"
#include "../maths/maths.h"
#include <cuda.h>
#include <cuda_runtime.h>

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
    __device__ int8_t atomicCAS_int8(int8_t* address, int8_t compare, int8_t val) {
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
    __device__ __forceinline__ bool test_segment_intersect_bvh(
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
        
        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0;
        
        while (stack_ptr >= 0)
        {
            int node_idx = stack[stack_ptr--];
            
            float t_hit_aabb;
            float3 box_min = bvh_aabb_mins[node_idx] - make_float3(1e-4f, 1e-4f, 1e-4f);
            float3 box_max = bvh_aabb_maxs[node_idx] + make_float3(1e-4f, 1e-4f, 1e-4f);
            if (!ray.is_intersect_aabb(box_min, box_max, t_hit_aabb)) {
                continue;
            }
            
            if (node_idx >= num_objects - 1)
            {
                int tri_id = object_ids[node_idx - (num_objects - 1)];
                int3 tri = triangles[tri_id];
                float3 v0 = vertices[tri.x];
                float3 v1 = vertices[tri.y];
                float3 v2 = vertices[tri.z];
                
                float3 edge1 = v1 - v0;
                float3 edge2 = v2 - v0;
                float3 h = maths::cross(ray.direction, edge2);
                float a = maths::dot(edge1, h);

                if (a > -1e-8f && a < 1e-8f) continue; 

                float f = 1.0f / a;
                float3 s = ray.origin - v0;
                float u = f * maths::dot(s, h);

                if (u < -1e-4f || u > 1.0f + 1e-4f) continue;

                float3 q = maths::cross(s, edge1);
                float v = f * maths::dot(ray.direction, q);

                if (v < -1e-4f || u + v > 1.0f + 1e-4f) continue;

                float t = f * maths::dot(edge2, q);
                if (t >= -1e-4f && t <= len + 1e-4f)
                {
                    return true;
                }
            }
            else
            {
                if (stack_ptr + 2 < BVH_STACK_SIZE)
                {
                    int2 children = bvh_children[node_idx];
                    if (children.y != -1) stack[++stack_ptr] = children.y;
                    if (children.x != -1) stack[++stack_ptr] = children.x;
                }
            }
        }
        return false;
    }

/**
 * @brief Seeds the flood fill from the grid's outer boundary.
 * @details Every vertex on the domain's six faces is definitionally outside the geometry,
 * so the fill starts there and works inwards. One thread per grid vertex; boundary
 * vertices still marked unknown ($-2$) are relabelled as exterior ($2$) and appended to
 * the initial frontier through an atomic bump of the shared counter.
 *
 * @param[in,out] mask Device array of $R_X R_Y R_Z$ occupancy labels, updated in place.
 * @param[out] frontier Device array receiving the seeded vertex indices.
 * @param[in,out] frontier_size Device counter, atomically incremented per seed.
 * @param[in] RX Grid resolution along $x$.
 * @param[in] RY Grid resolution along $y$.
 * @param[in] RZ Grid resolution along $z$.
 * @note Launched with one thread per vertex over a 1D grid; indices are computed in
 * `int64_t` because a $1024^3$ grid overflows 32-bit addressing.
 * @warning Assumes the grid bounds enclose the geometry with at least one vertex of
 * padding. Without it, surface vertices sit on the boundary, get seeded as exterior, and
 * the fill leaks into the interior.
 */
__global__ void init_perimeter_kernel(
        int8_t* __restrict__ mask,
        int* __restrict__ frontier,
        int* __restrict__ frontier_size,
        int RX, int RY, int RZ)
    {
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t num_vertices = (int64_t)RX * RY * RZ;
        if (idx >= num_vertices) return;

        int vi = idx / (RY * RZ);
        int rem = idx % (RY * RZ);
        int vj = rem / RZ;
        int vk = rem % RZ;

        if (vi == 0 || vi == RX - 1 || vj == 0 || vj == RY - 1 || vk == 0 || vk == RZ - 1)
        {
            if (mask[idx] == -2)
            {
                mask[idx] = 2; // Water (Open Sea)
                int pos = atomicAdd(frontier_size, 1);
                frontier[pos] = idx;
            }
        }
    }

/**
 * @brief Expands the flood-fill frontier by one layer, stopping at the surface.
 * @details One thread per frontier vertex. Each examines its 6- or 26-connected
 * neighbours and, for every unvisited one, tests whether the connecting segment
 * intersects the mesh by traversing the BVH. Unobstructed neighbours inherit the exterior
 * label and join the next frontier; blocked ones are left for the interior pass. Testing
 * the segment rather than the endpoint is what keeps the fill from tunnelling through
 * thin walls between adjacent samples.
 *
 * The host relaunches this kernel until the frontier empties, so the number of launches
 * is the exterior region's graph diameter rather than a fixed count.
 *
 * @param[in,out] mask Device array of occupancy labels, updated in place.
 * @param[in] current_frontier Device array of vertex indices to expand.
 * @param[in] frontier_size Number of entries in @p current_frontier.
 * @param[out] next_frontier Device array receiving the following layer.
 * @param[in,out] next_frontier_size Device counter, atomically incremented per insertion.
 * @param[in] RX Grid resolution along $x$.
 * @param[in] RY Grid resolution along $y$.
 * @param[in] RZ Grid resolution along $z$.
 * @param[in] connectivity Neighbourhood size, 6 or 26.
 * @param[in] grid_min World coordinate of the grid origin.
 * @param[in] grid_spacing Per-axis distance between adjacent vertices.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping BVH leaves to triangle indices.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of mesh triangle vertex indices.
 * @param[in] num_objects Number of triangles in the mesh.
 * @note Launched with `NTHREADS` threads per block over a 1D grid sized to the frontier.
 * @warning Load is highly irregular: a thread whose neighbours are all visited exits at
 * once, while one near the surface performs up to 26 BVH traversals. Warp divergence here
 * is intrinsic to the algorithm.
 * @warning Several threads may reach the same neighbour in one step. The label write is
 * idempotent, but the frontier counter must stay atomic or entries would be lost.
 */
__global__ void flood_fill_step_kernel(
        int8_t* __restrict__ mask,
        const int* __restrict__ current_frontier,
        int frontier_size,
        int* __restrict__ next_frontier,
        int* __restrict__ next_frontier_size,
        int RX, int RY, int RZ,
        int connectivity,
        float3 grid_min,
        float3 grid_spacing,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= frontier_size) return;

        int vertex_idx = current_frontier[idx];
        int vi = vertex_idx / (RY * RZ);
        int rem = vertex_idx % (RY * RZ);
        int vj = rem / RZ;
        int vk = rem % RZ;
        
        float3 pA = make_float3(
            grid_min.x + vi * grid_spacing.x,
            grid_min.y + vj * grid_spacing.y,
            grid_min.z + vk * grid_spacing.z
        );

        for (int di = -1; di <= 1; di++)
        {
            for (int dj = -1; dj <= 1; dj++)
            {
                for (int dk = -1; dk <= 1; dk++)
                {
                    if (di == 0 && dj == 0 && dk == 0) continue;

                    int dist = abs(di) + abs(dj) + abs(dk);
                    if (connectivity == 6 && dist > 1) continue;
                    if (connectivity == 18 && dist > 2) continue;

                    int ni = vi + di;
                    int nj = vj + dj;
                    int nk = vk + dk;

                    if (ni < 0 || ni >= RX || nj < 0 || nj >= RY || nk < 0 || nk >= RZ) continue;

                    int n_idx = ni * (RY * RZ) + nj * RZ + nk;
                    if (mask[n_idx] == -2)
                    {
                        float3 pB = make_float3(
                            grid_min.x + ni * grid_spacing.x,
                            grid_min.y + nj * grid_spacing.y,
                            grid_min.z + nk * grid_spacing.z
                        );
                        
                        if (!test_segment_intersect_bvh(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects))
                        {
                            int8_t old_val = atomicCAS_int8(&mask[n_idx], -2, 2);
                            if (old_val == -2)
                            {
                                int pos = atomicAdd(next_frontier_size, 1);
                                next_frontier[pos] = n_idx;
                            }
                        }
                    }
                }
            }
        }
    }

    torch::Tensor compute_flood_fill(
        const torch::Tensor& vertices,
        const torch::Tensor& triangles,
        const torch::Tensor& aabb_mins,
        const torch::Tensor& aabb_maxs,
        const torch::Tensor& bvh_children,
        const torch::Tensor& object_ids,
        std::vector<float> grid_min,
        std::vector<float> grid_max,
        std::vector<int64_t> grid_res,
        int connectivity
    ) {
        int64_t rx = grid_res[0];
        int64_t ry = grid_res[1];
        int64_t rz = grid_res[2];
        int64_t num_vertices = rx * ry * rz;
        auto options = torch::TensorOptions().device(vertices.device()).dtype(torch::kInt32);

        auto mask = torch::full({num_vertices}, -2, options.dtype(torch::kInt8));

        auto current_frontier = torch::empty({num_vertices}, options);
        auto next_frontier = torch::empty({num_vertices}, options);
        auto frontier_size = torch::zeros({1}, options);
        auto next_frontier_size = torch::zeros({1}, options);

        int threads = NTHREADS;
        int blocks = (num_vertices + threads - 1) / threads;

        init_perimeter_kernel<<<blocks, threads>>>(
            mask.data_ptr<int8_t>(),
            current_frontier.data_ptr<int>(),
            frontier_size.data_ptr<int>(),
            static_cast<int>(rx),
            static_cast<int>(ry),
            static_cast<int>(rz)
        );

        float3 f_min = make_float3(grid_min[0], grid_min[1], grid_min[2]);
        float3 f_spacing = make_float3(
            (rx > 1) ? (grid_max[0] - grid_min[0]) / (rx - 1) : 1.0f,
            (ry > 1) ? (grid_max[1] - grid_min[1]) / (ry - 1) : 1.0f,
            (rz > 1) ? (grid_max[2] - grid_min[2]) / (rz - 1) : 1.0f
        );

        int curr_size = frontier_size.item<int>();

        while (curr_size > 0)
        {
            next_frontier_size.zero_();
            int step_blocks = (curr_size + threads - 1) / threads;

            flood_fill_step_kernel<<<step_blocks, threads>>>(
                mask.data_ptr<int8_t>(),
                current_frontier.data_ptr<int>(),
                curr_size,
                next_frontier.data_ptr<int>(),
                next_frontier_size.data_ptr<int>(),
                static_cast<int>(rx),
                static_cast<int>(ry),
                static_cast<int>(rz),
                connectivity,
                f_min,
                f_spacing,
                (const float3*)aabb_mins.data_ptr<float>(),
                (const float3*)aabb_maxs.data_ptr<float>(),
                (const int2*)bvh_children.data_ptr<int>(),
                object_ids.data_ptr<int>(),
                (const float3*)vertices.data_ptr<float>(),
                (const int3*)triangles.data_ptr<int>(),
                static_cast<int>(object_ids.size(0))
            );

            curr_size = next_frontier_size.item<int>();
            std::swap(current_frontier, next_frontier);
        }

        return mask.view({rx, ry, rz});
    }

} // namespace ops
