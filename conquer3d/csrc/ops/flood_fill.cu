/**
 * @file flood_fill.cu
 * @brief CUDA kernel implementations for BVH segment intersection and 3D volumetric flood-fill occupancy.
 */

#include "flood_fill.h"
#include "flood_fill_common.cuh"
#include "../data_structure/bvh_traverse.cuh"
#include "../constants.h"
#include "../primitive/ray.h"
#include "../primitive/triangle.h"
#include "../maths/maths.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace ops {

    
    
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
                        
                        if (!test_segment_intersect_mesh(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects))
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
