#include "gs_math.cuh"
#include "aabb.h"

#include <math_constants.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

namespace pgs
{
    __device__ __forceinline__ bool test_pgs_segment(
        const float3 &mean,
        const float3 &normal,
        const float *covi,
        const float iso,
        const float3 &segment_start,
        const float3 &segment_end,
        float &t_hit)
    {
        float3 dir = make_float3(
            segment_end.x - segment_start.x,
            segment_end.y - segment_start.y,
            segment_end.z - segment_start.z);

        float denom = dir.x * normal.x + dir.y * normal.y + dir.z * normal.z;

        if (fabsf(denom) < 1e-6f) return false; 

        float num = (mean.x - segment_start.x) * normal.x +
                    (mean.y - segment_start.y) * normal.y +
                    (mean.z - segment_start.z) * normal.z;

        float t = num / denom;

        if (t >= 0.0f && t <= 1.0f) 
        {
            float3 p_hit = make_float3(
                segment_start.x + t * dir.x,
                segment_start.y + t * dir.y,
                segment_start.z + t * dir.z);

            float dist_sq;
            gs::compute_mahalanobis_distance(p_hit, mean, covi, dist_sq);

            if (dist_sq <= iso) 
            {
                t_hit = t;
                return true; 
            }
        }

        return false;
    }
}

namespace pgs_aabb
{
    // Fast Broad-Phase Collision (No 12-edge loops!)
    __device__ __forceinline__ bool test_pgs_intersect_voxel(
        const float3 &mean,
        const float3 &normal,
        const float3 &gs_ab_min,
        const float3 &gs_ab_max,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max)
    {
        // 1. Standard AABB Overlap Check
        if (!gs_aabb::test_gs_aabb_overlap_voxel(gs_ab_min, gs_ab_max, vx_ab_min, vx_ab_max))
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
            
            float dist = (corner.x - mean.x) * normal.x + 
                         (corner.y - mean.y) * normal.y + 
                         (corner.z - mean.z) * normal.z;

            if (dist > 0.0f) has_pos = true;
            if (dist <= 0.0f) has_neg = true; 

            // If the plane cuts through the voxel, the corners will have mixed signs!
            if (has_pos && has_neg) return true; 
        }

        return false; // Voxel is entirely above or below the flat plane
    }

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
                out_centroid = make_float3(
                    sum_points.x / total_hits,
                    sum_points.y / total_hits,
                    sum_points.z / total_hits
                );
            }
            return true; 
        }
        
        return false; 
    }

    __global__ void query_pgs_voxel_pair_intersection_brute_force_kernel(
        const uint32_t num_voxels,
        const uint32_t num_gaussians,
        const float3 *__restrict__ vx_aabb_mins,
        const float3 *__restrict__ vx_aabb_maxs,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ covis,
        const float3 *__restrict__ gs_aabb_mins,
        const float3 *__restrict__ gs_aabb_maxs,
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

        for (uint32_t g_idx = 0; g_idx < num_gaussians; g_idx++)
        {
            float3 mean = means[g_idx];
            float3 normal = normals[g_idx];
            const float *covi = covis + (g_idx * 6);
            float3 gs_ab_min = gs_aabb_mins[g_idx];
            float3 gs_ab_max = gs_aabb_maxs[g_idx];

            // 1. BROAD PHASE (8-Corner + AABB Check)
            bool broad_hit = test_pgs_intersect_voxel(
                mean, normal, gs_ab_min, gs_ab_max, vx_ab_min, vx_ab_max);

            if (broad_hit)
            {
                float3 centroid;
                
                // 2. NARROW PHASE & CENTROID CALCULATION (12-Edge Check + Iso)
                bool narrow_hit = compute_pgs_voxel_centroid(
                    mean, normal, covi, iso, vx_ab_min, vx_ab_max, return_centroids, centroid);

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
        }

        hit_mask[v_idx] = any_hit;
    }

    void query_pgs_voxel_pair_intersection_brute_force(
        const uint32_t num_voxels,
        const uint32_t num_gaussians,
        const float3 *__restrict__ vx_aabb_mins,
        const float3 *__restrict__ vx_aabb_maxs,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ covis,
        const float3 *__restrict__ gs_aabb_mins,
        const float3 *__restrict__ gs_aabb_maxs,
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
        uint32_t threads = 256;
        uint32_t blocks = (num_voxels + threads - 1) / threads;

        query_pgs_voxel_pair_intersection_brute_force_kernel<<<blocks, threads>>>(
            num_voxels, 
            num_gaussians, 
            vx_aabb_mins, 
            vx_aabb_maxs,
            means, 
            normals, 
            covis, 
            gs_aabb_mins, 
            gs_aabb_maxs,
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

    __global__ void query_pgs_edge_intersection_brute_force_kernel(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ opacities,
        const float *__restrict__ covis,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_gaus_ids)
    {
        uint32_t e_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (e_idx >= num_edges)
            return;

        float3 edge_start = edge_starts[e_idx];
        float3 edge_end = edge_ends[e_idx];

        bool any_hit = false;

        float max_density = -1.0f;
        int best_gs = -1;

        for (uint32_t g_idx = 0; g_idx < num_gaussians; g_idx++)
        {
            float3 mean = means[g_idx];
            float3 normal = normals[g_idx];
            const float *covi = covis + (g_idx * 6);

            float t_hit;

            bool hit = pgs::test_pgs_segment(
                mean,
                normal,
                covi,
                iso,
                edge_start,
                edge_end,
                t_hit);

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
        }

        out_gaus_ids[e_idx] = best_gs;
        hit_mask[e_idx] = any_hit;
    }

    void query_pgs_edge_intersection_brute_force(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ opacities,
        const float *__restrict__ covis,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_gaus_ids
    )
    {
        uint32_t threads = 256;
        uint32_t blocks = (num_edges + threads - 1) / threads;

        query_pgs_edge_intersection_brute_force_kernel<<<blocks, threads>>>(
            num_edges,
            num_gaussians,
            edge_starts,
            edge_ends,
            means,
            normals,
            opacities,
            covis,
            iso,
            hit_mask,
            out_gaus_ids);
    }
}