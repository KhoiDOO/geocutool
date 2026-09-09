/**
 * @file pgs_aabb.cu
 * @brief CUDA kernel implementations for analytical Axis-Aligned Bounding Box (AABB) bounds of Periodic Gaussians.
 */

#include "../data_structure/bvh_traverse.cuh"
#include "pgs.h"
#include "pgs_math.cuh"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <math_constants.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

namespace pgs_aabb
{
    // Fast Broad-Phase Collision (No 12-edge loops!)
    /**
     * @brief Tests whether a periodic Gaussian's support meets a voxel.
     * @details Begins with an AABB rejection, then accounts for the splat's orientation: a
     * periodic Gaussian's support is a slab about its normal, not an ellipsoid, so a plain
     * ellipsoid test would accept regions the splat does not actually occupy.
     * @param[in] mean Gaussian centre.
     * @param[in] normal Orientation vector.
     * @param[in] gs_ab_min Gaussian AABB lower bound.
     * @param[in] gs_ab_max Gaussian AABB upper bound.
     * @param[in] vx_ab_min Voxel lower bound.
     * @param[in] vx_ab_max Voxel upper bound.
     * @return True if the support meets the voxel.
     */
    __device__ __forceinline__ bool test_pgs_intersect_voxel(
        const float3 &mean,
        const float3 &normal,
        const float3 &gs_ab_min,
        const float3 &gs_ab_max,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max)
    {
        // 1. Standard AABB Overlap Check
        if (!aabb::test_aabb_overlap(gs_ab_min, gs_ab_max, vx_ab_min, vx_ab_max))
            return false;

        bool has_pos = false;
        bool has_neg = false;

        // 2. The 8-Corner Sign Check (Lightning Fast Dot Products)
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float3 corner = make_float3(
                (i & 1) ? vx_ab_max.x : vx_ab_min.x,
                (i & 2) ? vx_ab_max.y : vx_ab_min.y,
                (i & 4) ? vx_ab_max.z : vx_ab_min.z
            );
            
            float dist = maths::dot(corner - mean, normal);

            if (dist > 0.0f) has_pos = true;
            if (dist <= 0.0f) has_neg = true; 

            // If the plane cuts through the voxel, the corners will have mixed signs!
            if (has_pos && has_neg) return true; 
        }

        return false; // Voxel is entirely above or below the flat plane
    }

    /**
     * @brief Computes the centroid of a periodic Gaussian's occupancy within a voxel.
     * @details Samples the voxel and averages the positions whose density clears the isovalue,
     * giving a representative point for the intersection. Returns false when no sample
     * qualifies, meaning the overlap is empty despite the bounds suggesting otherwise.
     * @param[in] mean Gaussian centre.
     * @param[in] normal Orientation vector.
     * @param[in] covi Six upper-triangular entries of the inverse covariance.
     * @param[in] iso Isovalue defining the surface.
     * @param[in] vx_ab_min Voxel lower bound.
     * @param[in] vx_ab_max Voxel upper bound.
     * @param[in] return_centroids Whether the centroid is written.
     * @param[out] out_centroid Centroid of the occupied region.
     * @return True if any sample qualified.
     * @note Sampling is discrete, so a support far thinner than the voxel may be missed.
     */
    __device__ __forceinline__ bool compute_pgs_voxel_centroid(
        const float3 &mean,
        const float3 &normal,
        const float *covi,
        const float iso,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max,
        const bool return_centroids,
        float3 &out_centroid)
    {
        float vsize = vx_ab_max.x - vx_ab_min.x; 
        float3 p = vx_ab_min; 

        float3 sum_points = make_float3(0.0f, 0.0f, 0.0f);
        int total_hits = 0;

        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float t_hit;

            // --- X-Axis Edges ---
            float3 start_x = make_float3(p.x, p.y + (i / 2 ? vsize : 0.0f), p.z + (i % 2 ? vsize : 0.0f));
            float3 end_x   = make_float3(p.x + vsize, start_x.y, start_x.z);
            if (pgs::test_pgs_segment(mean, normal, covi, iso, start_x, end_x, t_hit)) {
                if (return_centroids) {
                    sum_points.x += start_x.x + (t_hit * vsize);
                    sum_points.y += start_x.y;
                    sum_points.z += start_x.z;
                }
                total_hits++;
            }

            // --- Y-Axis Edges ---
            float3 start_y = make_float3(p.x + (i / 2 ? vsize : 0.0f), p.y, p.z + (i % 2 ? vsize : 0.0f));
            float3 end_y   = make_float3(start_y.x, p.y + vsize, start_y.z);
            if (pgs::test_pgs_segment(mean, normal, covi, iso, start_y, end_y, t_hit)) {
                if (return_centroids) {
                    sum_points.x += start_y.x;
                    sum_points.y += start_y.y + (t_hit * vsize);
                    sum_points.z += start_y.z;
                }
                total_hits++;
            }

            // --- Z-Axis Edges ---
            float3 start_z = make_float3(p.x + (i / 2 ? vsize : 0.0f), p.y + (i % 2 ? vsize : 0.0f), p.z);
            float3 end_z   = make_float3(start_z.x, start_z.y, p.z + vsize);
            if (pgs::test_pgs_segment(mean, normal, covi, iso, start_z, end_z, t_hit)) {
                if (return_centroids) {
                    sum_points.x += start_z.x;
                    sum_points.y += start_z.y;
                    sum_points.z += start_z.z + (t_hit * vsize);
                }
                total_hits++;
            }
        }

        if (total_hits > 0) {
            // Average the accumulated surface points
            if (return_centroids) {
                out_centroid = sum_points * (1.0f / total_hits);
            }
            return true; 
        }
        
        return false; 
    }

    template <bool multiple_isos>
/**
 * @brief Reports every (voxel, periodic Gaussian) pair whose supports overlap.
 * @details The periodic counterpart of the 3DGS voxel query. One thread per voxel; leaves
 * surviving the AABB test are checked against the oriented, normal-aware support of a
 * periodic Gaussian rather than a plain ellipsoid.
 * @param[in] num_voxels Number of voxels $V$.
 * @param[in] num_gaussians Number of Gaussians $N$.
 * @param[in] vx_aabb_mins Device array of $V$ voxel lower bounds.
 * @param[in] vx_aabb_maxs Device array of $V$ voxel upper bounds.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to Gaussian indices.
 * @param[in] means Device array of $N$ Gaussian centres.
 * @param[in] normals Device array of $N$ orientation vectors.
 * @param[in] covis Device array of $6N$ inverse covariance entries.
 * @param[in] gs_aabb_mins Device array of $N$ Gaussian AABB lower bounds.
 * @param[in] gs_aabb_maxs Device array of $N$ Gaussian AABB upper bounds.
 * @param[in] isos Device array of $N$ per-Gaussian isovalues.
 * @param[in] iso Uniform isovalue fallback.
 * @param[in] return_centroids Whether to emit per-pair centroids.
 * @param[in] return_centroid_densities Whether to emit per-pair densities.
 * @param[out] hit_mask Device array of $V$ flags marking voxels with any hit.
 * @param[out] out_voxel_ids Device array receiving the voxel index of each pair.
 * @param[out] out_gaus_ids Device array receiving the Gaussian index of each pair.
 * @param[out] centroids Device array of per-pair centroids, when requested.
 * @param[out] densities Device array of per-pair densities, when requested.
 * @param[in,out] global_counter Device counter, atomically incremented per pair.
 * @param[in] max_capacity Capacity of the output arrays.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Traversal uses a per-thread stack of `BVH_STACK_SIZE` entries in local memory;
 * a pathologically unbalanced hierarchy can overflow it.
 * @warning Output slots are claimed atomically, so pair ordering varies between runs.
 * Emission stops at @p max_capacity; compare the final counter against it to detect
 * truncation.
 */
__global__ void query_pgs_voxel_pair_intersection_bvh_kernel(
        const uint32_t num_voxels,
        const uint32_t num_gaussians,
        const float3 *__restrict__ vx_aabb_mins,
        const float3 *__restrict__ vx_aabb_maxs,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ covis,
        const float3 *__restrict__ gs_aabb_mins,
        const float3 *__restrict__ gs_aabb_maxs,
        const float *__restrict__ isos,
        const float iso,
        const bool return_centroids,
        const bool return_centroid_densities,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_voxel_ids,
        int64_t *__restrict__ out_gaus_ids,
        float3 *__restrict__ centroids,
        float *__restrict__ densities,
        int64_t *__restrict__ global_counter,
        const int64_t max_capacity)
    {
        uint32_t v_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (v_idx >= num_voxels) return;

        float3 vx_ab_min = vx_aabb_mins[v_idx];
        float3 vx_ab_max = vx_aabb_maxs[v_idx];
        bool any_hit = false;

        // --- BVH LOCAL STACK ---
        bvh::traverse(num_gaussians, bvh_children,
            [&] (int node_idx) {
                return aabb::test_aabb_overlap(vx_ab_min, vx_ab_max, bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx]);
            },
            [&] (int leaf_idx) {
                        uint32_t g_idx = object_ids[leaf_idx];
    
                        float3 mean = means[g_idx];
                        float3 normal = normals[g_idx];
                        const float *covi = covis + (g_idx * 6);
                        float3 gs_ab_min = gs_aabb_mins[g_idx];
                        float3 gs_ab_max = gs_aabb_maxs[g_idx];
    
                        bool broad_hit = test_pgs_intersect_voxel(
                            mean, normal, gs_ab_min, gs_ab_max, vx_ab_min, vx_ab_max);
    
                        if (broad_hit)
                        {
                            float __iso = multiple_isos ? isos[g_idx] : iso;
                            float3 centroid;
                            
                            bool narrow_hit = compute_pgs_voxel_centroid(
                                mean, normal, covi, __iso, vx_ab_min, vx_ab_max, return_centroids, centroid);
    
                            if (narrow_hit)
                            {
                                any_hit = true;
                                uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int *)global_counter, 1ULL);
    
                                if (write_idx < max_capacity)
                                {
                                    out_voxel_ids[write_idx] = v_idx;
                                    out_gaus_ids[write_idx] = g_idx;
    
                                    if (return_centroids)
                                    {
                                        centroids[write_idx] = centroid;
    
                                        if (return_centroid_densities)
                                        {
                                            float density;
                                            gs::compute_density(centroid, mean, covi, 1.0f, density);
                                            densities[write_idx] = density;
                                        }
                                    }
                                }
                            }
                        }
                return true;
            });
        hit_mask[v_idx] = any_hit;
    }

    void query_pgs_voxel_pair_intersection_bvh(
        const uint32_t num_voxels,
        const uint32_t num_gaussians,
        const float3 *__restrict__ vx_aabb_mins,
        const float3 *__restrict__ vx_aabb_maxs,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ covis,
        const float3 *__restrict__ gs_aabb_mins,
        const float3 *__restrict__ gs_aabb_maxs,
        const float *__restrict__ isos,
        const float iso,
        const bool return_centroids,
        const bool return_centroid_densities,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_voxel_ids,
        int64_t *__restrict__ out_gaus_ids,
        float3 *__restrict__ centroids,
        float *__restrict__ densities,
        int64_t *__restrict__ global_counter,
        const int64_t max_capacity)
    {
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_voxels + threads - 1) / threads;

        if (isos != nullptr) {
            query_pgs_voxel_pair_intersection_bvh_kernel<true><<<blocks, threads>>>(
                num_voxels, 
                num_gaussians, 
                vx_aabb_mins, 
                vx_aabb_maxs, 
                bvh_aabb_mins, 
                bvh_aabb_maxs, 
                bvh_children, 
                object_ids, 
                means, 
                normals, 
                covis, 
                gs_aabb_mins, 
                gs_aabb_maxs, 
                isos, 
                iso, 
                return_centroids, 
                return_centroid_densities, 
                hit_mask, 
                out_voxel_ids, 
                out_gaus_ids, 
                centroids, 
                densities, 
                global_counter, 
                max_capacity);
        } else {
            query_pgs_voxel_pair_intersection_bvh_kernel<false><<<blocks, threads>>>(
                num_voxels, 
                num_gaussians, 
                vx_aabb_mins, 
                vx_aabb_maxs, 
                bvh_aabb_mins, 
                bvh_aabb_maxs, 
                bvh_children, 
                object_ids, 
                means, 
                normals, 
                covis, 
                gs_aabb_mins, 
                gs_aabb_maxs, 
                isos, 
                iso, 
                return_centroids, 
                return_centroid_densities, 
                hit_mask, 
                out_voxel_ids, 
                out_gaus_ids, 
                centroids, 
                densities, 
                global_counter, 
                max_capacity);
        }
    }

    template <bool multiple_isos>
/**
 * @brief Finds the first periodic Gaussian each edge intersects.
 * @details One thread per edge, writing a single result per edge at a fixed slot, so no
 * atomics are required. The exact test accounts for the Gaussian's orientation, which a
 * plain ellipsoid test would ignore.
 * @param[in] num_edges Number of edges $E$.
 * @param[in] num_gaussians Number of Gaussians $N$.
 * @param[in] edge_starts Device array of $E$ segment start points.
 * @param[in] edge_ends Device array of $E$ segment end points.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to Gaussian indices.
 * @param[in] means Device array of $N$ Gaussian centres.
 * @param[in] normals Device array of $N$ orientation vectors.
 * @param[in] opacities Device array of $N$ opacities.
 * @param[in] covis Device array of $6N$ inverse covariance entries.
 * @param[in] isos Device array of $N$ per-Gaussian isovalues.
 * @param[in] iso Uniform isovalue fallback.
 * @param[out] hit_mask Device array of $E$ flags marking intersected edges.
 * @param[out] out_gaus_ids Device array of $E$ Gaussian indices, $-1$ where none.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Traversal uses a per-thread stack of `BVH_STACK_SIZE` entries in local memory;
 * a pathologically unbalanced hierarchy can overflow it.
 */
__global__ void query_pgs_edge_intersection_bvh_kernel(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ opacities,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_gaus_ids)
    {
        uint32_t e_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (e_idx >= num_edges) return;

        float3 edge_start = edge_starts[e_idx];
        float3 edge_end = edge_ends[e_idx];

        // Ray AABB for BVH traversal
        float3 e_ab_min = make_float3(fminf(edge_start.x, edge_end.x), fminf(edge_start.y, edge_end.y), fminf(edge_start.z, edge_end.z));
        float3 e_ab_max = make_float3(fmaxf(edge_start.x, edge_end.x), fmaxf(edge_start.y, edge_end.y), fmaxf(edge_start.z, edge_end.z));

        bool any_hit = false;
        float max_density = -1.0f;
        int best_gs = -1;

        bvh::traverse(num_gaussians, bvh_children,
            [&] (int node_idx) {
                return aabb::test_aabb_overlap(e_ab_min, e_ab_max, bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx]);
            },
            [&] (int leaf_idx) {
                        uint32_t g_idx = object_ids[leaf_idx];
    
                        float3 mean = means[g_idx];
                        float3 normal = normals[g_idx];
                        const float *covi = covis + (g_idx * 6);
                        float __iso = multiple_isos ? isos[g_idx] : iso;
    
                        float t_hit;
    
                        bool hit = pgs::test_pgs_segment(
                            mean, normal, covi, __iso, edge_start, edge_end, t_hit);
    
                        if (hit)
                        {
                            any_hit = true;
    
                            float3 p_hit = make_float3(
                                edge_start.x + t_hit * (edge_end.x - edge_start.x),
                                edge_start.y + t_hit * (edge_end.y - edge_start.y),
                                edge_start.z + t_hit * (edge_end.z - edge_start.z)
                            );
    
                            float density;
                            gs::compute_density(p_hit, mean, covi, opacities[g_idx], density);
    
                            if (density > max_density)
                            {
                                max_density = density;
                                best_gs = g_idx;
                            }
                        }
                return true;
            });

        out_gaus_ids[e_idx] = best_gs;
        hit_mask[e_idx] = any_hit;
    }

    void query_pgs_edge_intersection_bvh(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ opacities,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_gaus_ids)
    {
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_edges + threads - 1) / threads;

        if (isos != nullptr) {
            query_pgs_edge_intersection_bvh_kernel<true><<<blocks, threads>>>(
                num_edges, 
                num_gaussians, 
                edge_starts, 
                edge_ends, 
                bvh_aabb_mins, 
                bvh_aabb_maxs, 
                bvh_children, 
                object_ids, 
                means, 
                normals, 
                opacities, 
                covis, 
                isos, 
                iso, 
                hit_mask, 
                out_gaus_ids);
        } else {
            query_pgs_edge_intersection_bvh_kernel<false><<<blocks, threads>>>(
                num_edges, 
                num_gaussians, 
                edge_starts, 
                edge_ends, 
                bvh_aabb_mins, 
                bvh_aabb_maxs, 
                bvh_children, 
                object_ids, 
                means, 
                normals, 
                opacities, 
                covis, 
                isos, 
                iso, 
                hit_mask, 
                out_gaus_ids);
        }
    }
}