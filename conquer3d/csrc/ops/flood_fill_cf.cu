/**
 * @file flood_fill_cf.cu
 * @brief CUDA kernel implementations for 2-Level Coarse-to-Fine (CF) Volumetric Flood Fill.
 */

#include "flood_fill_cf.h"
#include "../constants.h"
#include "../primitive/ray.h"
#include "../primitive/triangle.h"
#include "../maths/maths.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

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
    __device__ static int8_t atomicCAS_int8_cf(int8_t* address, int8_t compare, int8_t val) {
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
    __device__ __forceinline__ bool test_segment_intersect_bvh_cf(
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
     * @brief Tests whether an axis-aligned box overlaps any mesh triangle.
     * @details Traverses the BVH and applies the Akenine-Moller separating-axis test to
     * surviving leaves, stopping at the first hit. Used to classify whole coarse blocks in one
     * test, so empty regions can be resolved without descending to fine voxels.
     * @param[in] box_min Box lower bound.
     * @param[in] box_max Box upper bound.
     * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
     * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
     * @param[in] bvh_children Device array of BVH child index pairs.
     * @param[in] object_ids Device array mapping leaves to triangle indices.
     * @param[in] vertices Device array of mesh vertex coordinates.
     * @param[in] triangles Device array of triangle vertex indices.
     * @return True if the box meets any triangle.
     * @warning Uses a per-thread stack of `BVH_STACK_SIZE` entries in local memory.
     */
    __device__ __forceinline__ bool test_box_overlap_bvh_cf(
        const float3& box_min, const float3& box_max,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects)
    {
        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0;

        while (stack_ptr >= 0)
        {
            int node_idx = stack[stack_ptr--];
            float3 node_min = bvh_aabb_mins[node_idx];
            float3 node_max = bvh_aabb_maxs[node_idx];

            // AABB overlap test
            if (box_max.x < node_min.x || box_min.x > node_max.x ||
                box_max.y < node_min.y || box_min.y > node_max.y ||
                box_max.z < node_min.z || box_min.z > node_max.z)
            {
                continue;
            }

            if (node_idx >= num_objects - 1)
            {
                int tri_id = object_ids[node_idx - (num_objects - 1)];
                int3 tri = triangles[tri_id];
                Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
                if (T.is_voxel_intersect(box_min, box_max)) {
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

    // -------------------------------------------------------------
    // Initialization Kernels
    // -------------------------------------------------------------

/**
 * @brief Classifies each coarse block as empty, surface-crossing, or unknown.
 * @details Stage 1 of the coarse-fine flood fill. The domain is tiled into blocks of
 * $B_X \times B_Y \times B_Z$ fine voxels and each block's bounds are tested against the
 * mesh BVH. Blocks the surface misses can be filled wholesale at coarse resolution; only
 * blocks it crosses are refined to individual voxels later. That two-level split is what
 * makes the fill scale to $1024^3$, where a flat frontier would need thousands of launches.
 * @param[out] coarse_mask Device array of one label per coarse block.
 * @param[in] CX Coarse grid resolution along $x$.
 * @param[in] CY Coarse grid resolution along $y$.
 * @param[in] CZ Coarse grid resolution along $z$.
 * @param[in] BX Fine voxels per coarse block along $x$.
 * @param[in] BY Fine voxels per coarse block along $y$.
 * @param[in] BZ Fine voxels per coarse block along $z$.
 * @param[in] grid_min World coordinate of the grid origin.
 * @param[in] fine_spacing Per-axis fine voxel size.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to triangle indices.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] num_objects Number of triangles.
 * @note Launched with one thread per coarse block, `int64_t` indexed.
 * @warning Traversal uses a per-thread stack of `BVH_STACK_SIZE` entries in local memory;
 * a pathologically unbalanced hierarchy can overflow it.
 */
__global__ void init_coarse_grid_kernel(
        int8_t* __restrict__ coarse_mask,
        int CX, int CY, int CZ,
        int BX, int BY, int BZ,
        float3 grid_min,
        float3 fine_spacing,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects)
    {
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t total_coarse = (int64_t)CX * CY * CZ;
        if (idx >= total_coarse) return;

        int ci = idx / (CY * CZ);
        int rem = idx % (CY * CZ);
        int cj = rem / CZ;
        int ck = rem % CZ;

        float3 box_min = make_float3(
            grid_min.x + ci * BX * fine_spacing.x,
            grid_min.y + cj * BY * fine_spacing.y,
            grid_min.z + ck * BZ * fine_spacing.z
        );
        float3 box_max = make_float3(
            grid_min.x + (ci + 1) * BX * fine_spacing.x,
            grid_min.y + (cj + 1) * BY * fine_spacing.y,
            grid_min.z + (ck + 1) * BZ * fine_spacing.z
        );

        bool intersects = test_box_overlap_bvh_cf(
            box_min, box_max,
            bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids,
            vertices, triangles, num_objects
        );

        coarse_mask[idx] = intersects ? (int8_t)1 : (int8_t)-2;
    }

/**
 * @brief Seeds the coarse fill from blocks on the domain boundary.
 * @details Stage 2. One thread per coarse block; those on the outer faces are definitionally
 * outside the geometry and are relabelled as exterior, giving the propagation its starting
 * set.
 * @param[in,out] coarse_mask Device array of coarse block labels.
 * @param[in] CX Coarse grid resolution along $x$.
 * @param[in] CY Coarse grid resolution along $y$.
 * @param[in] CZ Coarse grid resolution along $z$.
 * @note Launched with one thread per coarse block.
 * @warning Assumes the grid bounds enclose the geometry with padding. Without it, surface
 * blocks touch the boundary and the fill leaks inside.
 */
__global__ void init_perimeter_seeds_kernel(
        int8_t* __restrict__ coarse_mask,
        int CX, int CY, int CZ)
    {
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t total_coarse = (int64_t)CX * CY * CZ;
        if (idx >= total_coarse) return;

        int ci = idx / (CY * CZ);
        int rem = idx % (CY * CZ);
        int cj = rem / CZ;
        int ck = rem % CZ;

        if (ci == 0 || ci == CX - 1 || cj == 0 || cj == CY - 1 || ck == 0 || ck == CZ - 1) {
            if (coarse_mask[idx] == -2) {
                coarse_mask[idx] = 2; // Exterior Water
            }
        }
    }

/**
 * @brief Seeds fine voxels on the outer faces of boundary-adjacent blocks.
 * @details Stage 3. Surface-crossing blocks on the domain boundary need their fine voxels
 * seeded individually, since only part of such a block is exterior.
 * @param[in] boundary_coords Device array of coarse coordinates for surface blocks.
 * @param[in,out] fine_masks Device array of per-block fine voxel labels.
 * @param[in] num_boundary_blocks Number of surface-crossing blocks.
 * @param[in] CX Coarse grid resolution along $x$.
 * @param[in] CY Coarse grid resolution along $y$.
 * @param[in] CZ Coarse grid resolution along $z$.
 * @param[in] BX Fine voxels per coarse block along $x$.
 * @param[in] BY Fine voxels per coarse block along $y$.
 * @param[in] BZ Fine voxels per coarse block along $z$.
 * @note Launched with one CUDA block per boundary coarse block, threads cooperating over
 * that block's fine voxels through shared memory.
 */
__global__ void init_boundary_perimeter_faces_kernel(
        const int3* __restrict__ boundary_coords,
        int8_t* __restrict__ fine_masks,
        int num_boundary_blocks,
        int CX, int CY, int CZ,
        int BX, int BY, int BZ)
    {
        int block_id = blockIdx.x;
        if (block_id >= num_boundary_blocks) return;

        int3 c_coord = boundary_coords[block_id];
        int ci = c_coord.x, cj = c_coord.y, ck = c_coord.z;
        int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;

        int local_idx = fi * (BY * BZ) + fj * BZ + fk;
        int8_t* my_fine_mask = fine_masks + (int64_t)block_id * (BX * BY * BZ);

        bool on_outer_perimeter = false;
        if (ci == 0 && fi == 0) on_outer_perimeter = true;
        if (ci == CX - 1 && fi == BX - 1) on_outer_perimeter = true;
        if (cj == 0 && fj == 0) on_outer_perimeter = true;
        if (cj == CY - 1 && fj == BY - 1) on_outer_perimeter = true;
        if (ck == 0 && fk == 0) on_outer_perimeter = true;
        if (ck == CZ - 1 && fk == BZ - 1) on_outer_perimeter = true;

        if (on_outer_perimeter) {
            my_fine_mask[local_idx] = 2;
        }
    }

    // -------------------------------------------------------------
    // Unified Graph Wavefront Propagation Kernels
    // -------------------------------------------------------------

    // Step A: Coarse-to-Coarse Expansion
/**
 * @brief Propagates the exterior label between neighbouring coarse blocks.
 * @details Stage 4, the cheap bulk of the fill. One thread per unknown block: if any
 * face-adjacent neighbour is exterior and the connecting region is unobstructed, the block
 * becomes exterior too. Empty space is resolved a whole block at a time here, so the
 * expensive per-voxel work is confined to the surface.
 * @param[in,out] coarse_mask Device array of coarse block labels.
 * @param[in] CX Coarse grid resolution along $x$.
 * @param[in] CY Coarse grid resolution along $y$.
 * @param[in] CZ Coarse grid resolution along $z$.
 * @param[in] BX Fine voxels per coarse block along $x$.
 * @param[in] BY Fine voxels per coarse block along $y$.
 * @param[in] BZ Fine voxels per coarse block along $z$.
 * @param[in] grid_min World coordinate of the grid origin.
 * @param[in] fine_spacing Per-axis fine voxel size.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to triangle indices.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] num_objects Number of triangles.
 * @param[out] changed_flag Device flag set when any label changes.
 * @note Launched with one thread per coarse block, relaunched by the host until quiescent.
 * @note Sets @p changed_flag when any label changes, which is how the host knows whether
 * another sweep is required; the fill has converged once a full round leaves it clear.
 */
__global__ void coarse_to_coarse_kernel(
        int8_t* __restrict__ coarse_mask,
        int CX, int CY, int CZ,
        int BX, int BY, int BZ,
        float3 grid_min,
        float3 fine_spacing,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects,
        int* __restrict__ changed_flag)
    {
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t total_coarse = (int64_t)CX * CY * CZ;
        if (idx >= total_coarse) return;

        if (coarse_mask[idx] != -2) return;

        int ci = idx / (CY * CZ);
        int rem = idx % (CY * CZ);
        int cj = rem / CZ;
        int ck = rem % CZ;

        float3 pA = make_float3(
            grid_min.x + (ci + 0.5f) * BX * fine_spacing.x,
            grid_min.y + (cj + 0.5f) * BY * fine_spacing.y,
            grid_min.z + (ck + 0.5f) * BZ * fine_spacing.z
        );

        const int di[6] = {-1, 1, 0, 0, 0, 0};
        const int dj[6] = {0, 0, -1, 1, 0, 0};
        const int dk[6] = {0, 0, 0, 0, -1, 1};

        for (int k = 0; k < 6; ++k) {
            int ni = ci + di[k];
            int nj = cj + dj[k];
            int nk = ck + dk[k];
            if (ni >= 0 && ni < CX && nj >= 0 && nj < CY && nk >= 0 && nk < CZ) {
                int n_idx = ni * (CY * CZ) + nj * CZ + nk;
                if (coarse_mask[n_idx] == 2) {
                    float3 pB = make_float3(
                        grid_min.x + (ni + 0.5f) * BX * fine_spacing.x,
                        grid_min.y + (nj + 0.5f) * BY * fine_spacing.y,
                        grid_min.z + (nk + 0.5f) * BZ * fine_spacing.z
                    );
                    if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) {
                        coarse_mask[idx] = 2;
                        atomicExch(changed_flag, 1);
                        return;
                    }
                }
            }
        }
    }

    // Step B: Coarse-to-Fine Face Seeding
/**
 * @brief Transfers the exterior label from coarse blocks into fine voxels.
 * @details Stage 5. Where a resolved exterior block adjoins a surface-crossing one, its
 * label is pushed across the shared face onto that block's boundary fine voxels, handing
 * the fill down a level.
 * @param[in] boundary_coords Device array of coarse coordinates for surface blocks.
 * @param[in] coarse_mask Device array of coarse block labels.
 * @param[in,out] fine_masks Device array of per-block fine voxel labels.
 * @param[in] num_boundary_blocks Number of surface-crossing blocks.
 * @param[in] CX Coarse grid resolution along $x$.
 * @param[in] CY Coarse grid resolution along $y$.
 * @param[in] CZ Coarse grid resolution along $z$.
 * @param[in] BX Fine voxels per coarse block along $x$.
 * @param[in] BY Fine voxels per coarse block along $y$.
 * @param[in] BZ Fine voxels per coarse block along $z$.
 * @param[in] grid_min World coordinate of the grid origin.
 * @param[in] fine_spacing Per-axis fine voxel size.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to triangle indices.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] num_objects Number of triangles.
 * @param[out] changed_flag Device flag set when any label changes.
 * @note Launched with one CUDA block per boundary coarse block, threads cooperating over
 * that block's fine voxels through shared memory.
 * @note Sets @p changed_flag when any label changes, which is how the host knows whether
 * another sweep is required; the fill has converged once a full round leaves it clear.
 */
__global__ void coarse_to_fine_kernel(
        const int3* __restrict__ boundary_coords,
        const int8_t* __restrict__ coarse_mask,
        int8_t* __restrict__ fine_masks,
        int num_boundary_blocks,
        int CX, int CY, int CZ,
        int BX, int BY, int BZ,
        float3 grid_min,
        float3 fine_spacing,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects,
        int* __restrict__ changed_flag)
    {
        int block_id = blockIdx.x;
        if (block_id >= num_boundary_blocks) return;

        int3 c_coord = boundary_coords[block_id];
        int ci = c_coord.x, cj = c_coord.y, ck = c_coord.z;
        int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;

        int local_idx = fi * (BY * BZ) + fj * BZ + fk;
        int8_t* my_fine_mask = fine_masks + (int64_t)block_id * (BX * BY * BZ);

        if (my_fine_mask[local_idx] != -2) return;

        float3 pA = make_float3(
            grid_min.x + (ci * BX + fi) * fine_spacing.x,
            grid_min.y + (cj * BY + fj) * fine_spacing.y,
            grid_min.z + (ck * BZ + fk) * fine_spacing.z
        );

        bool seed = false;
        if (fi == 0 && ci > 0 && coarse_mask[(ci - 1) * (CY * CZ) + cj * CZ + ck] == 2) {
            float3 pB = make_float3(grid_min.x + ((ci - 1) * BX + (BX - 1)) * fine_spacing.x, pA.y, pA.z);
            if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
        }
        if (!seed && fi == BX - 1 && ci < CX - 1 && coarse_mask[(ci + 1) * (CY * CZ) + cj * CZ + ck] == 2) {
            float3 pB = make_float3(grid_min.x + ((ci + 1) * BX + 0) * fine_spacing.x, pA.y, pA.z);
            if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
        }
        if (!seed && fj == 0 && cj > 0 && coarse_mask[ci * (CY * CZ) + (cj - 1) * CZ + ck] == 2) {
            float3 pB = make_float3(pA.x, grid_min.y + ((cj - 1) * BY + (BY - 1)) * fine_spacing.y, pA.z);
            if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
        }
        if (!seed && fj == BY - 1 && cj < CY - 1 && coarse_mask[ci * (CY * CZ) + (cj + 1) * CZ + ck] == 2) {
            float3 pB = make_float3(pA.x, grid_min.y + ((cj + 1) * BY + 0) * fine_spacing.y, pA.z);
            if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
        }
        if (!seed && fk == 0 && ck > 0 && coarse_mask[ci * (CY * CZ) + cj * CZ + (ck - 1)] == 2) {
            float3 pB = make_float3(pA.x, pA.y, grid_min.z + ((ck - 1) * BZ + (BZ - 1)) * fine_spacing.z);
            if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
        }
        if (!seed && fk == BZ - 1 && ck < CZ - 1 && coarse_mask[ci * (CY * CZ) + cj * CZ + (ck + 1)] == 2) {
            float3 pB = make_float3(pA.x, pA.y, grid_min.z + ((ck + 1) * BZ + 0) * fine_spacing.z);
            if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
        }

        if (seed) {
            my_fine_mask[local_idx] = 2;
            atomicExch(changed_flag, 1);
        }
    }

    // Step C: Fine Intra-Block Shared Memory BFS
/**
 * @brief Runs a breadth-first fill within each surface-crossing block.
 * @details Stage 6, where the actual surface resolution happens. One CUDA block per coarse
 * block iterates its fine voxels in shared memory until no further label changes, testing
 * each candidate step against the mesh so the fill cannot pass through a wall. Keeping the
 * whole sweep in shared memory means the many iterations needed near the surface never
 * touch global memory.
 * @param[in] boundary_coords Device array of coarse coordinates for surface blocks.
 * @param[in,out] fine_masks Device array of per-block fine voxel labels.
 * @param[in] num_boundary_blocks Number of surface-crossing blocks.
 * @param[in] BX Fine voxels per coarse block along $x$.
 * @param[in] BY Fine voxels per coarse block along $y$.
 * @param[in] BZ Fine voxels per coarse block along $z$.
 * @param[in] grid_min World coordinate of the grid origin.
 * @param[in] fine_spacing Per-axis fine voxel size.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to triangle indices.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] num_objects Number of triangles.
 * @param[out] changed_flag Device flag set when any label changes.
 * @note Launched with one CUDA block per boundary coarse block, threads cooperating over
 * that block's fine voxels through shared memory.
 * @warning Shared memory must accommodate one block's fine voxel labels, which bounds how
 * large $B_X \times B_Y \times B_Z$ may be.
 * @note Sets @p changed_flag when any label changes, which is how the host knows whether
 * another sweep is required; the fill has converged once a full round leaves it clear.
 */
__global__ void fine_intra_block_bfs_kernel(
        const int3* __restrict__ boundary_coords,
        int8_t* __restrict__ fine_masks,
        int num_boundary_blocks,
        int BX, int BY, int BZ,
        float3 grid_min,
        float3 fine_spacing,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects,
        int* __restrict__ changed_flag)
    {
        int block_id = blockIdx.x;
        if (block_id >= num_boundary_blocks) return;

        int3 c_coord = boundary_coords[block_id];
        int ci = c_coord.x, cj = c_coord.y, ck = c_coord.z;
        int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;

        int local_idx = fi * (BY * BZ) + fj * BZ + fk;
        int8_t* my_fine_mask = fine_masks + (int64_t)block_id * (BX * BY * BZ);

        __shared__ int8_t s_mask[8][8][8];
        s_mask[fi][fj][fk] = my_fine_mask[local_idx];
        __syncthreads();

        float3 pA = make_float3(
            grid_min.x + (ci * BX + fi) * fine_spacing.x,
            grid_min.y + (cj * BY + fj) * fine_spacing.y,
            grid_min.z + (ck * BZ + fk) * fine_spacing.z
        );

        const int di[6] = {-1, 1, 0, 0, 0, 0};
        const int dj[6] = {0, 0, -1, 1, 0, 0};
        const int dk[6] = {0, 0, 0, 0, -1, 1};

        bool any_local_change = false;

        for (int iter = 0; iter < (BX + BY + BZ); ++iter)
        {
            if (s_mask[fi][fj][fk] == 2)
            {
                for (int k = 0; k < 6; ++k)
                {
                    int nfi = fi + di[k];
                    int nfj = fj + dj[k];
                    int nfk = fk + dk[k];

                    if (nfi >= 0 && nfi < BX && nfj >= 0 && nfj < BY && nfk >= 0 && nfk < BZ)
                    {
                        if (s_mask[nfi][nfj][nfk] == -2)
                        {
                            float3 pB = make_float3(
                                grid_min.x + (ci * BX + nfi) * fine_spacing.x,
                                grid_min.y + (cj * BY + nfj) * fine_spacing.y,
                                grid_min.z + (ck * BZ + nfk) * fine_spacing.z
                            );

                            if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects))
                            {
                                s_mask[nfi][nfj][nfk] = 2;
                                any_local_change = true;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        my_fine_mask[local_idx] = s_mask[fi][fj][fk];
        if (any_local_change) {
            atomicExch(changed_flag, 1);
        }
    }

    // Step D: Fine-to-Fine Inter-Block Boundary Exchange
/**
 * @brief Propagates fine voxel labels across adjoining surface blocks.
 * @details Stage 7. Intra-block sweeps cannot see past their own boundary, so this kernel
 * carries labels across shared faces between neighbouring surface-crossing blocks,
 * consulting @p boundary_lookup to find the neighbour's storage.
 * @param[in] boundary_coords Device array of coarse coordinates for surface blocks.
 * @param[in] boundary_lookup Device array mapping coarse index to boundary block slot.
 * @param[in,out] fine_masks Device array of per-block fine voxel labels.
 * @param[in] num_boundary_blocks Number of surface-crossing blocks.
 * @param[in] CX Coarse grid resolution along $x$.
 * @param[in] CY Coarse grid resolution along $y$.
 * @param[in] CZ Coarse grid resolution along $z$.
 * @param[in] BX Fine voxels per coarse block along $x$.
 * @param[in] BY Fine voxels per coarse block along $y$.
 * @param[in] BZ Fine voxels per coarse block along $z$.
 * @param[in] grid_min World coordinate of the grid origin.
 * @param[in] fine_spacing Per-axis fine voxel size.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to triangle indices.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] num_objects Number of triangles.
 * @param[out] changed_flag Device flag set when any label changes.
 * @note Launched with one CUDA block per boundary coarse block, threads cooperating over
 * that block's fine voxels through shared memory.
 * @note Sets @p changed_flag when any label changes, which is how the host knows whether
 * another sweep is required; the fill has converged once a full round leaves it clear.
 */
__global__ void fine_to_fine_inter_block_kernel(
        const int3* __restrict__ boundary_coords,
        const int32_t* __restrict__ boundary_lookup,
        int8_t* __restrict__ fine_masks,
        int num_boundary_blocks,
        int CX, int CY, int CZ,
        int BX, int BY, int BZ,
        float3 grid_min,
        float3 fine_spacing,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects,
        int* __restrict__ changed_flag)
    {
        int block_id = blockIdx.x;
        if (block_id >= num_boundary_blocks) return;

        int3 c_coord = boundary_coords[block_id];
        int ci = c_coord.x, cj = c_coord.y, ck = c_coord.z;
        int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;

        int local_idx = fi * (BY * BZ) + fj * BZ + fk;
        int8_t* my_fine_mask = fine_masks + (int64_t)block_id * (BX * BY * BZ);

        if (my_fine_mask[local_idx] != -2) return;

        float3 pA = make_float3(
            grid_min.x + (ci * BX + fi) * fine_spacing.x,
            grid_min.y + (cj * BY + fj) * fine_spacing.y,
            grid_min.z + (ck * BZ + fk) * fine_spacing.z
        );

        const int di[6] = {-1, 1, 0, 0, 0, 0};
        const int dj[6] = {0, 0, -1, 1, 0, 0};
        const int dk[6] = {0, 0, 0, 0, -1, 1};

        for (int k = 0; k < 6; ++k) {
            int nfi = fi + di[k];
            int nfj = fj + dj[k];
            int nfk = fk + dk[k];

            int nci = ci, ncj = cj, nck = ck;
            if (nfi < 0) { nci -= 1; nfi = BX - 1; }
            else if (nfi >= BX) { nci += 1; nfi = 0; }
            if (nfj < 0) { ncj -= 1; nfj = BY - 1; }
            else if (nfj >= BY) { ncj += 1; nfj = 0; }
            if (nfk < 0) { nck -= 1; nfk = BZ - 1; }
            else if (nfk >= BZ) { nck += 1; nfk = 0; }

            if (nci != ci || ncj != cj || nck != ck) {
                if (nci >= 0 && nci < CX && ncj >= 0 && ncj < CY && nck >= 0 && nck < CZ) {
                    int neighbor_block_id = boundary_lookup[nci * (CY * CZ) + ncj * CZ + nck];
                    if (neighbor_block_id >= 0) {
                        const int8_t* neighbor_fine = fine_masks + (int64_t)neighbor_block_id * (BX * BY * BZ);
                        int neighbor_local = nfi * (BY * BZ) + nfj * BZ + nfk;
                        if (neighbor_fine[neighbor_local] == 2) {
                            float3 pB = make_float3(
                                grid_min.x + (nci * BX + nfi) * fine_spacing.x,
                                grid_min.y + (ncj * BY + nfj) * fine_spacing.y,
                                grid_min.z + (nck * BZ + nfk) * fine_spacing.z
                            );
                            if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) {
                                if (atomicCAS_int8_cf(&my_fine_mask[local_idx], -2, 2) == -2) {
                                    atomicExch(changed_flag, 1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Step E: Fine-to-Coarse Seeding
/**
 * @brief Promotes fine voxel labels back up to the coarse grid.
 * @details Stage 8, closing the loop. A surface block that has reached its outer face at
 * fine resolution can now mark the adjacent coarse block exterior, letting the cheap
 * coarse propagation resume through the passage the fine sweep just opened. Alternating
 * between the two levels is what lets the fill navigate narrow channels without paying
 * fine-resolution cost everywhere.
 * @param[in] boundary_coords Device array of coarse coordinates for surface blocks.
 * @param[in] fine_masks Device array of per-block fine voxel labels.
 * @param[in,out] coarse_mask Device array of coarse block labels.
 * @param[in] num_boundary_blocks Number of surface-crossing blocks.
 * @param[in] CX Coarse grid resolution along $x$.
 * @param[in] CY Coarse grid resolution along $y$.
 * @param[in] CZ Coarse grid resolution along $z$.
 * @param[in] BX Fine voxels per coarse block along $x$.
 * @param[in] BY Fine voxels per coarse block along $y$.
 * @param[in] BZ Fine voxels per coarse block along $z$.
 * @param[in] grid_min World coordinate of the grid origin.
 * @param[in] fine_spacing Per-axis fine voxel size.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to triangle indices.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] num_objects Number of triangles.
 * @param[out] changed_flag Device flag set when any label changes.
 * @note Launched with one CUDA block per boundary coarse block, threads cooperating over
 * that block's fine voxels through shared memory.
 * @note Sets @p changed_flag when any label changes, which is how the host knows whether
 * another sweep is required; the fill has converged once a full round leaves it clear.
 */
__global__ void fine_to_coarse_kernel(
        const int3* __restrict__ boundary_coords,
        const int8_t* __restrict__ fine_masks,
        int8_t* __restrict__ coarse_mask,
        int num_boundary_blocks,
        int CX, int CY, int CZ,
        int BX, int BY, int BZ,
        float3 grid_min,
        float3 fine_spacing,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects,
        int* __restrict__ changed_flag)
    {
        int block_id = blockIdx.x;
        if (block_id >= num_boundary_blocks) return;

        int3 c_coord = boundary_coords[block_id];
        int ci = c_coord.x, cj = c_coord.y, ck = c_coord.z;
        int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;

        int local_idx = fi * (BY * BZ) + fj * BZ + fk;
        const int8_t* my_fine_mask = fine_masks + (int64_t)block_id * (BX * BY * BZ);

        if (my_fine_mask[local_idx] != 2) return;

        float3 pA = make_float3(
            grid_min.x + (ci * BX + fi) * fine_spacing.x,
            grid_min.y + (cj * BY + fj) * fine_spacing.y,
            grid_min.z + (ck * BZ + fk) * fine_spacing.z
        );

        if (fi == 0 && ci > 0 && coarse_mask[(ci - 1) * (CY * CZ) + cj * CZ + ck] == -2) {
            float3 pB = make_float3(grid_min.x + ((ci - 1) * BX + (BX - 1)) * fine_spacing.x, pA.y, pA.z);
            if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) {
                coarse_mask[(ci - 1) * (CY * CZ) + cj * CZ + ck] = 2;
                atomicExch(changed_flag, 1);
            }
        }
        if (fi == BX - 1 && ci < CX - 1 && coarse_mask[(ci + 1) * (CY * CZ) + cj * CZ + ck] == -2) {
            float3 pB = make_float3(grid_min.x + ((ci + 1) * BX + 0) * fine_spacing.x, pA.y, pA.z);
            if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) {
                coarse_mask[(ci + 1) * (CY * CZ) + cj * CZ + ck] = 2;
                atomicExch(changed_flag, 1);
            }
        }
        if (fj == 0 && cj > 0 && coarse_mask[ci * (CY * CZ) + (cj - 1) * CZ + ck] == -2) {
            float3 pB = make_float3(pA.x, grid_min.y + ((cj - 1) * BY + (BY - 1)) * fine_spacing.y, pA.z);
            if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) {
                coarse_mask[ci * (CY * CZ) + (cj - 1) * CZ + ck] = 2;
                atomicExch(changed_flag, 1);
            }
        }
        if (fj == BY - 1 && cj < CY - 1 && coarse_mask[ci * (CY * CZ) + (cj + 1) * CZ + ck] == -2) {
            float3 pB = make_float3(pA.x, grid_min.y + ((cj + 1) * BY + 0) * fine_spacing.y, pA.z);
            if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) {
                coarse_mask[ci * (CY * CZ) + (cj + 1) * CZ + ck] = 2;
                atomicExch(changed_flag, 1);
            }
        }
        if (fk == 0 && ck > 0 && coarse_mask[ci * (CY * CZ) + cj * CZ + (ck - 1)] == -2) {
            float3 pB = make_float3(pA.x, pA.y, grid_min.z + ((ck - 1) * BZ + (BZ - 1)) * fine_spacing.z);
            if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) {
                coarse_mask[ci * (CY * CZ) + cj * CZ + (ck - 1)] = 2;
                atomicExch(changed_flag, 1);
            }
        }
        if (fk == BZ - 1 && ck < CZ - 1 && coarse_mask[ci * (CY * CZ) + cj * CZ + (ck + 1)] == -2) {
            float3 pB = make_float3(pA.x, pA.y, grid_min.z + ((ck + 1) * BZ + 0) * fine_spacing.z);
            if (!test_segment_intersect_bvh_cf(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) {
                coarse_mask[ci * (CY * CZ) + cj * CZ + (ck + 1)] = 2;
                atomicExch(changed_flag, 1);
            }
        }
    }

/**
 * @brief Labels every block the fill never reached as interior.
 * @details Final stage. The fill only ever propagates exterior labels, so any block still
 * unknown once it has converged is enclosed by the surface -- interior by elimination,
 * without a separate inside test. One thread per coarse block.
 * @param[in,out] coarse_mask Device array of coarse block labels.
 * @param[in] total_coarse Number of coarse blocks.
 * @note Launched with one thread per coarse block, `int64_t` indexed.
 * @warning Correct only after the fill has fully converged. Running it early freezes
 * still-unresolved blocks as interior.
 */
__global__ void finalize_coarse_interior_kernel(int8_t* coarse_mask, int64_t total_coarse) {
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_coarse) return;
        if (coarse_mask[idx] == -2) {
            coarse_mask[idx] = -1; // Unreached dry coarse block -> Solid Interior
        }
    }

    // -------------------------------------------------------------
    // Host Pipeline
    // -------------------------------------------------------------

    /**
     * @brief Picks a coarse block size that divides the grid evenly.
     * @details Searches for the divisor closest to the requested block size, so the coarse
     * grid tiles the domain exactly and no partial blocks need special handling.
     * @return The chosen block dimension.
     */
    static int choose_best_divisor(int res, int max_b = 8) {
        int best = 1;
        for (int b = max_b; b >= 1; --b) {
            if (res % b == 0) {
                return b;
            }
        }
        return best;
    }

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
        std::vector<int64_t> block_size,
        int connectivity
    ) {
        int64_t rx = grid_res[0], ry = grid_res[1], rz = grid_res[2];
        
        int64_t bx = (block_size.size() >= 3 && block_size[0] > 0) ? block_size[0] : choose_best_divisor(static_cast<int>(rx), 8);
        int64_t by = (block_size.size() >= 3 && block_size[1] > 0) ? block_size[1] : choose_best_divisor(static_cast<int>(ry), 8);
        int64_t bz = (block_size.size() >= 3 && block_size[2] > 0) ? block_size[2] : choose_best_divisor(static_cast<int>(rz), 8);

        int64_t cx = rx / bx;
        int64_t cy = ry / by;
        int64_t cz = rz / bz;

        int64_t total_coarse = cx * cy * cz;
        auto dev = vertices.device();
        auto opt_i8 = torch::TensorOptions().device(dev).dtype(torch::kInt8);
        auto opt_i32 = torch::TensorOptions().device(dev).dtype(torch::kInt32);

        auto coarse_mask = torch::full({cx, cy, cz}, -2, opt_i8);

        float3 f_min = make_float3(grid_min[0], grid_min[1], grid_min[2]);
        float3 f_spacing = make_float3(
            (rx > 1) ? (grid_max[0] - grid_min[0]) / (rx - 1) : 1.0f,
            (ry > 1) ? (grid_max[1] - grid_min[1]) / (ry - 1) : 1.0f,
            (rz > 1) ? (grid_max[2] - grid_min[2]) / (rz - 1) : 1.0f
        );

        int threads = NTHREADS;
        int blocks = (total_coarse + threads - 1) / threads;

        // Phase 1: Coarse Classification
        init_coarse_grid_kernel<<<blocks, threads>>>(
            coarse_mask.data_ptr<int8_t>(),
            cx, cy, cz,
            bx, by, bz,
            f_min, f_spacing,
            (const float3*)aabb_mins.data_ptr<float>(),
            (const float3*)aabb_maxs.data_ptr<float>(),
            (const int2*)bvh_children.data_ptr<int>(),
            object_ids.data_ptr<int>(),
            (const float3*)vertices.data_ptr<float>(),
            (const int3*)triangles.data_ptr<int>(),
            static_cast<int>(object_ids.size(0))
        );

        // Phase 2: Index Boundary Blocks
        auto boundary_mask = (coarse_mask == 1);
        auto boundary_indices_1d = torch::nonzero(boundary_mask.flatten()).squeeze(1);
        int64_t num_boundary_blocks = boundary_indices_1d.size(0);

        auto boundary_coords = torch::empty({num_boundary_blocks, 3}, opt_i32);
        auto boundary_block_lookup = torch::full({cx, cy, cz}, -1, opt_i32);
        auto fine_boundary_masks = torch::full({num_boundary_blocks, bx, by, bz}, -2, opt_i8);

        if (num_boundary_blocks > 0) {
            auto b_1d = boundary_indices_1d.to(torch::kInt64);
            auto ci = b_1d.div(cy * cz, "trunc");
            auto rem = b_1d.remainder(cy * cz);
            auto cj = rem.div(cz, "trunc");
            auto ck = rem.remainder(cz);

            boundary_coords = torch::stack({ci, cj, ck}, 1).to(torch::kInt32).contiguous();
            boundary_block_lookup.flatten().index_put_({b_1d}, torch::arange(num_boundary_blocks, opt_i32));
        }

        // Phase 3: Perimeter Seeding
        init_perimeter_seeds_kernel<<<blocks, threads>>>(
            coarse_mask.data_ptr<int8_t>(),
            cx, cy, cz
        );

        if (num_boundary_blocks > 0) {
            dim3 fine_block(bx, by, bz);
            init_boundary_perimeter_faces_kernel<<<static_cast<int>(num_boundary_blocks), fine_block>>>(
                (const int3*)boundary_coords.data_ptr<int>(),
                fine_boundary_masks.data_ptr<int8_t>(),
                static_cast<int>(num_boundary_blocks),
                cx, cy, cz,
                bx, by, bz
            );
        }

        // Phase 4: Unified Multi-Scale Wavefront Propagation Loop
        auto changed_flag = torch::zeros({1}, opt_i32);
        dim3 fine_block(bx, by, bz);

        for (int iter = 0; iter < 100; ++iter)
        {
            changed_flag.zero_();

            // 4a. Coarse-to-Coarse Expansion
            coarse_to_coarse_kernel<<<blocks, threads>>>(
                coarse_mask.data_ptr<int8_t>(),
                cx, cy, cz,
                bx, by, bz,
                f_min, f_spacing,
                (const float3*)aabb_mins.data_ptr<float>(),
                (const float3*)aabb_maxs.data_ptr<float>(),
                (const int2*)bvh_children.data_ptr<int>(),
                object_ids.data_ptr<int>(),
                (const float3*)vertices.data_ptr<float>(),
                (const int3*)triangles.data_ptr<int>(),
                static_cast<int>(object_ids.size(0)),
                changed_flag.data_ptr<int>()
            );

            if (num_boundary_blocks > 0)
            {
                // 4b. Coarse-to-Fine Seeding
                coarse_to_fine_kernel<<<static_cast<int>(num_boundary_blocks), fine_block>>>(
                    (const int3*)boundary_coords.data_ptr<int>(),
                    coarse_mask.data_ptr<int8_t>(),
                    fine_boundary_masks.data_ptr<int8_t>(),
                    static_cast<int>(num_boundary_blocks),
                    cx, cy, cz,
                    bx, by, bz,
                    f_min, f_spacing,
                    (const float3*)aabb_mins.data_ptr<float>(),
                    (const float3*)aabb_maxs.data_ptr<float>(),
                    (const int2*)bvh_children.data_ptr<int>(),
                    object_ids.data_ptr<int>(),
                    (const float3*)vertices.data_ptr<float>(),
                    (const int3*)triangles.data_ptr<int>(),
                    static_cast<int>(object_ids.size(0)),
                    changed_flag.data_ptr<int>()
                );

                // 4c. Fine Intra-Block Shared Memory BFS
                fine_intra_block_bfs_kernel<<<static_cast<int>(num_boundary_blocks), fine_block>>>(
                    (const int3*)boundary_coords.data_ptr<int>(),
                    fine_boundary_masks.data_ptr<int8_t>(),
                    static_cast<int>(num_boundary_blocks),
                    bx, by, bz,
                    f_min, f_spacing,
                    (const float3*)aabb_mins.data_ptr<float>(),
                    (const float3*)aabb_maxs.data_ptr<float>(),
                    (const int2*)bvh_children.data_ptr<int>(),
                    object_ids.data_ptr<int>(),
                    (const float3*)vertices.data_ptr<float>(),
                    (const int3*)triangles.data_ptr<int>(),
                    static_cast<int>(object_ids.size(0)),
                    changed_flag.data_ptr<int>()
                );

                // 4d. Fine-to-Fine Inter-Block Boundary Exchange
                fine_to_fine_inter_block_kernel<<<static_cast<int>(num_boundary_blocks), fine_block>>>(
                    (const int3*)boundary_coords.data_ptr<int>(),
                    boundary_block_lookup.data_ptr<int32_t>(),
                    fine_boundary_masks.data_ptr<int8_t>(),
                    static_cast<int>(num_boundary_blocks),
                    cx, cy, cz,
                    bx, by, bz,
                    f_min, f_spacing,
                    (const float3*)aabb_mins.data_ptr<float>(),
                    (const float3*)aabb_maxs.data_ptr<float>(),
                    (const int2*)bvh_children.data_ptr<int>(),
                    object_ids.data_ptr<int>(),
                    (const float3*)vertices.data_ptr<float>(),
                    (const int3*)triangles.data_ptr<int>(),
                    static_cast<int>(object_ids.size(0)),
                    changed_flag.data_ptr<int>()
                );

                // 4e. Fine-to-Coarse Seeding
                fine_to_coarse_kernel<<<static_cast<int>(num_boundary_blocks), fine_block>>>(
                    (const int3*)boundary_coords.data_ptr<int>(),
                    fine_boundary_masks.data_ptr<int8_t>(),
                    coarse_mask.data_ptr<int8_t>(),
                    static_cast<int>(num_boundary_blocks),
                    cx, cy, cz,
                    bx, by, bz,
                    f_min, f_spacing,
                    (const float3*)aabb_mins.data_ptr<float>(),
                    (const float3*)aabb_maxs.data_ptr<float>(),
                    (const int2*)bvh_children.data_ptr<int>(),
                    object_ids.data_ptr<int>(),
                    (const float3*)vertices.data_ptr<float>(),
                    (const int3*)triangles.data_ptr<int>(),
                    static_cast<int>(object_ids.size(0)),
                    changed_flag.data_ptr<int>()
                );
            }

            if (changed_flag.item<int>() == 0) {
                break;
            }
        }

        // Phase 5: Finalize Coarse Interior
        finalize_coarse_interior_kernel<<<blocks, threads>>>(coarse_mask.data_ptr<int8_t>(), total_coarse);

        CFFloodFillResult result;
        result.coarse_mask = coarse_mask;
        result.boundary_block_coords = boundary_coords;
        result.boundary_block_lookup = boundary_block_lookup;
        result.fine_boundary_masks = fine_boundary_masks;
        result.block_size = {bx, by, bz};
        result.coarse_res = {cx, cy, cz};
        result.grid_min = grid_min;
        result.grid_max = grid_max;
        result.grid_res = grid_res;

        return result;
    }

} // namespace ops
