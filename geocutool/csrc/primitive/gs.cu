#include "gs.h"

#include <math_constants.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

namespace gs_aabb
{
    __device__ __forceinline__ void compute_single_aabb(
        const float3 &mean,
        const float4 &rot,
        const float3 &scale,
        const float &iso,
        const float &tol,
        const uint32_t level,
        const bool rotnorm,
        float3 &out_min,
        float3 &out_max,
        float3 *contact_points,
        float *covi)
    {
        float two_level = (float)(0x1 << level);
        float voxelSize = 2.0f / two_level;
        float min_scale = tol * voxelSize;
        float3 modified_scale = scale;

        modified_scale.x = fmaxf(scale.x, min_scale);
        modified_scale.y = fmaxf(scale.y, min_scale);
        modified_scale.z = fmaxf(scale.z, min_scale);

        double detS = ((double)modified_scale.x) * ((double)modified_scale.y) * ((double)modified_scale.z);

        float3x3 S_inv = make_float3x3(
            1.0f / modified_scale.x, 0.0f, 0.0f,
            0.0f, 1.0f / modified_scale.y, 0.0f,
            0.0f, 0.0f, 1.0f / modified_scale.z); // $S^{-1}$

        float4 q = rot;
        float r = q.x;
        float x = q.y;
        float y = q.z;
        float z = q.w;

        if (rotnorm)
        {
            float inv_norm = rsqrtf(r * r + x * x + y * y + z * z);
            r *= inv_norm;
            x *= inv_norm;
            y *= inv_norm;
            z *= inv_norm;
        }

        float3x3 R_T = make_float3x3(
            1.f - 2.f * (y * y + z * z), 2.f * (x * y + r * z), 2.f * (x * z - r * y),
            2.f * (x * y - r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + r * x),
            2.f * (x * z + r * y), 2.f * (y * z - r * x), 1.f - 2.f * (x * x + y * y)); // $R^T$

        // M = S^{-1} R^T
        float3x3 M = S_inv * R_T;

        // M^T M
        // = (S^{-1} R^T)^T (S^{-1} R^T)
        // = R S^{-T} S^{-1} R^T
        // = R S^{-2} R^T
        // = \Sigma
        float3x3 Covi = transpose(M) * M;

        covi[0] = Covi.m[0][0];
        covi[1] = Covi.m[0][1];
        covi[2] = Covi.m[0][2];
        covi[3] = Covi.m[1][1];
        covi[4] = Covi.m[1][2];
        covi[5] = Covi.m[2][2];

        double c0 = covi[0];
        double c1 = covi[1];
        double c2 = covi[2];
        double c3 = covi[3];
        double c4 = covi[4];
        double c5 = covi[5];

        double h0 = c3 * c5 - c4 * c4;
        double h1 = c2 * c4 - c1 * c5;
        double h2 = c1 * c4 - c2 * c3;
        double h3 = c0 * c5 - c2 * c2;
        double h4 = c1 * c2 - c0 * c4;
        double h5 = c0 * c3 - c1 * c1;

        double w[3];
        w[0] = detS * sqrt(iso / h0);
        w[1] = detS * sqrt(iso / h3);
        w[2] = detS * sqrt(iso / h5);

        double3 Q[3];
        Q[0] = make_double3(h0, h1, h2);
        Q[1] = make_double3(h1, h3, h4);
        Q[2] = make_double3(h2, h4, h5);

        float3 P[6];
        for (int i = 0; i < 3; i++)
        {
            P[2 * i] = make_float3((float)(w[i] * Q[i].x), (float)(w[i] * Q[i].y), (float)(w[i] * Q[i].z));
            P[2 * i + 1] = -1.0f * P[2 * i];
        }

        contact_points[0] = P[0];
        contact_points[1] = P[2];
        contact_points[2] = P[4];

        float3 Pmin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 Pmax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (int i = 0; i < 6; i++)
        {
            Pmin.x = fminf(Pmin.x, P[i].x);
            Pmin.y = fminf(Pmin.y, P[i].y);
            Pmin.z = fminf(Pmin.z, P[i].z);
            Pmax.x = fmaxf(Pmax.x, P[i].x);
            Pmax.y = fmaxf(Pmax.y, P[i].y);
            Pmax.z = fmaxf(Pmax.z, P[i].z);
        }

        out_min = mean + Pmin;
        out_max = mean + Pmax;
    }

    __global__ void compute_aabb_kernel(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float4 *__restrict__ rotations,
        const float3 *__restrict__ scales,
        const float iso,
        const float tol,
        const uint32_t level,
        const bool rotnorm,
        float3 *__restrict__ aabb_min,
        float3 *__restrict__ aabb_max,
        float3 *__restrict__ contact_points,
        float *__restrict__ covi)
    {
        uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

        if (tidx >= num_gaussians)
            return;

        compute_single_aabb(
            means[tidx],
            rotations[tidx],
            scales[tidx],
            iso,
            tol,
            level,
            rotnorm,
            aabb_min[tidx],
            aabb_max[tidx],
            contact_points + (tidx * 3),
            covi + (tidx * 6));
    }

    void compute_aabb(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float4 *__restrict__ rotations,
        const float3 *__restrict__ scales,
        const float iso,
        const float tol,
        const uint32_t level,
        const bool rotnorm,
        float3 *__restrict__ aabb_min,
        float3 *__restrict__ aabb_max,
        float3 *__restrict__ contact_points,
        float *__restrict__ covi)
    {
        uint32_t threads = 256;
        uint32_t blocks = (num_gaussians + threads - 1) / threads;

        compute_aabb_kernel<<<blocks, threads>>>(
            num_gaussians,
            means,
            rotations,
            scales,
            iso,
            tol,
            level,
            rotnorm,
            aabb_min,
            aabb_max,
            contact_points,
            covi);
    }

    __device__ __forceinline__ bool aabb_inside_voxel(
        const float3 &gs_ab_min,
        const float3 &gs_ab_max,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max)
    {
        return (
            vx_ab_min.x <= gs_ab_min.x &&
            vx_ab_min.y <= gs_ab_min.y &&
            vx_ab_min.z <= gs_ab_min.z &&
            (vx_ab_max.x) >= gs_ab_max.x &&
            (vx_ab_max.y) >= gs_ab_max.y &&
            (vx_ab_max.z) >= gs_ab_max.z);
    }

    __device__ __forceinline__ bool aabb_overlap_voxel(
        const float3 &gs_ab_min,
        const float3 &gs_ab_max,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max)
    {

        if (gs_ab_max.x < vx_ab_min.x)
            return false;
        if (gs_ab_max.y < vx_ab_min.y)
            return false;
        if (gs_ab_max.z < vx_ab_min.z)
            return false;
        if (gs_ab_min.x > (vx_ab_max.x))
            return false;
        if (gs_ab_min.y > (vx_ab_max.y))
            return false;
        if (gs_ab_min.z > (vx_ab_max.z))
            return false;

        return true;
    }

    __device__ __forceinline__ bool edge_test(
        const float c0,
        const float c1,
        const float c2,
        const float c3,
        const float c4,
        const float c5,
        const float cp6,
        const float s,
        const float t,
        const float l,
        const float u)
    {
        double a = c0;
        double b = 2 * (c1 * s + c2 * t);
        double c = (c3 * s * s + 2 * c4 * s * t + c5 * t * t) - cp6;
        double dcrm = max(b * b - 4 * a * c, 0.0);
        double b2a = -0.5 * b / a;
        double r0 = b2a;
        double r1 = b2a;

        if (dcrm > 0)
        {
            double rdcrm = 0.5 * sqrt(dcrm) / a;
            r0 = b2a - rdcrm;
            r1 = b2a + rdcrm;

            return !(u <= r0 || r1 <= l);
        }

        return false;
    }

    __device__ __forceinline__ bool gs_intersect_voxel_edge(
        const float3 &mean,
        const float *covi,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max,
        const float iso)
    {

        float3 p = vx_ab_min - mean;             // move to local space of Gaussian
        float vsize = vx_ab_max.x - vx_ab_min.x; // assume cubic voxel

        // if edge crossings, true true
        for (int i = 0; i < 4; i++)
        {
            if (edge_test(covi[0], covi[1], covi[2], covi[3], covi[4], covi[5], iso,
                          p.y + (i / 2 ? vsize : 0.0), p.z + (i % 2 ? vsize : 0.0), p.x, p.x + vsize))
                return true;

            if (edge_test(covi[3], covi[1], covi[4], covi[0], covi[2], covi[5], iso,
                          p.x + (i / 2 ? vsize : 0.0), p.z + (i % 2 ? vsize : 0.0), p.y, p.y + vsize))
                return true;

            if (edge_test(covi[5], covi[2], covi[4], covi[0], covi[1], covi[3], iso,
                          p.x + (i / 2 ? vsize : 0.0), p.y + (i % 2 ? vsize : 0.0), p.z, p.z + vsize))
                return true;
        }

        return false;
    }

    __device__ __forceinline__ bool facetest(
        const float *p,
        float *q,
        const float vsize,
        const int i,
        const int j,
        const int k,
        const float vs)
    {
        float t = (q[i] - (p[i] + vs)) / (2.0 * q[i]);

        if (0.0f <= t && t <= 1.0f)
        {
            float h[3];
            for (int l = 0; l < 3; l++)
                h[l] = (1.0 - 2.0 * t) * q[l];

            return p[j] <= h[j] && h[j] <= (p[j] + vsize) && p[k] <= h[k] && h[k] <= (p[k] + vsize);
        }

        return false;
    }

    __device__ __forceinline__ bool gs_intersect_voxel_face(
        const float3 &mean,
        float3 cp0,
        float3 cp1,
        float3 cp2,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max)
    {
        float3 p = vx_ab_min - mean;
        float vsize = vx_ab_max.x - vx_ab_min.x; // assume cubic voxel
        bool b[6];

        // need to index float3 components, so cast to float[]
        b[0] = facetest((float *)(&p), (float *)(&cp0), vsize, 0, 1, 2, 0.0);
        b[1] = facetest((float *)(&p), (float *)(&cp0), vsize, 0, 1, 2, vsize);

        b[2] = facetest((float *)(&p), (float *)(&cp1), vsize, 1, 0, 2, 0.0);
        b[3] = facetest((float *)(&p), (float *)(&cp1), vsize, 1, 0, 2, vsize);

        b[4] = facetest((float *)(&p), (float *)(&cp2), vsize, 2, 0, 1, 0.0);
        b[5] = facetest((float *)(&p), (float *)(&cp2), vsize, 2, 0, 1, vsize);

        return (b[0] || b[1] || b[2] || b[3] || b[4] || b[5]);
    }

    __device__ __forceinline__ bool gs_intersect_voxel(
        const uint64_t gaus_idx,
        const float3& mean, 
        const float* covi, 
        const float3& gs_ab_min,
        const float3& gs_ab_max,
        const float3& cp0,
        const float3& cp1,
        const float3& cp2,
        const float3& vx_ab_min, 
        const float3& vx_ab_max,
        const float iso) {

        if (aabb_inside_voxel(gs_ab_min, gs_ab_max, vx_ab_min, vx_ab_max)) 
            return true;

        if (!aabb_overlap_voxel(gs_ab_min, gs_ab_max, vx_ab_min, vx_ab_max))
            return false;

        if (gs_intersect_voxel_face(mean, cp0, cp1, cp2, vx_ab_min, vx_ab_max))
            return true;

        return gs_intersect_voxel_edge(mean, covi, vx_ab_min, vx_ab_max, iso);
    }

    __device__ __forceinline__ void compute_aabb_centroid(
        const float3 &gs_ab_min,
        const float3 &gs_ab_max,
        float3 &out_centroid)
    {
        out_centroid = make_float3(
            (gs_ab_min.x + gs_ab_max.x) * 0.5f,
            (gs_ab_min.y + gs_ab_max.y) * 0.5f,
            (gs_ab_min.z + gs_ab_max.z) * 0.5f
        );
    }

    __device__ __forceinline__ void compute_density(
        const float3& point,
        const float3& mean,
        const float* covi,
        const float opacity,
        float& out_density
    ) {
        float3 d = make_float3(point.x - mean.x, point.y - mean.y, point.z - mean.z);
        
        float power = -0.5f * (
            d.x * (d.x * covi[0] + d.y * covi[1] + d.z * covi[2]) +
            d.y * (d.x * covi[1] + d.y * covi[3] + d.z * covi[4]) +
            d.z * (d.x * covi[2] + d.y * covi[4] + d.z * covi[5])
        );

        if (power > 0.0f || power < -15.0f) {
            out_density = 0.0f;
        } else {
            out_density = opacity * expf(power);
        }
    }

    __device__ __forceinline__ void compute_overlap_metrics(
        const float3& gs_ab_min, 
        const float3& gs_ab_max,
        const float3& vx_ab_min, 
        const float3& vx_ab_max,
        const float3& mean,
        const float* covi,
        const float opacity,
        const bool return_centroids,
        float3& out_centroid,
        float& out_density,
        float& out_aspect_ratio,
        float& out_penetration
    ) {
        // 1. Calculate Overlap Box boundaries
        float3 overlap_min = make_float3(
            fmaxf(gs_ab_min.x, vx_ab_min.x), 
            fmaxf(gs_ab_min.y, vx_ab_min.y), 
            fmaxf(gs_ab_min.z, vx_ab_min.z)
        );

        float3 overlap_max = make_float3(
            fminf(gs_ab_max.x, vx_ab_max.x), 
            fminf(gs_ab_max.y, vx_ab_max.y), 
            fminf(gs_ab_max.z, vx_ab_max.z)
        );

        if (return_centroids) {
            compute_aabb_centroid(overlap_min, overlap_max, out_centroid);
            compute_density(out_centroid, mean, covi, opacity, out_density);
        }

        // 3. Get the physical dimensions of the overlap box
        float3 dims = make_float3(
            fmaxf(0.0f, overlap_max.x - overlap_min.x),
            fmaxf(0.0f, overlap_max.y - overlap_min.y),
            fmaxf(0.0f, overlap_max.z - overlap_min.z)
        );

        // 4. Find the smallest and largest dimensions
        float min_dim = fminf(dims.x, fminf(dims.y, dims.z));
        float max_dim = fmaxf(dims.x, fmaxf(dims.y, dims.z));
        
        // Assume cubic voxels: size is max - min on any axis
        float voxel_size = vx_ab_max.x - vx_ab_min.x; 

        // 5. Calculate Metrics (with safeguards against divide-by-zero)
        out_aspect_ratio = (max_dim > 1e-6f) ? (min_dim / max_dim) : 0.0f;
        out_penetration  = (voxel_size > 1e-6f) ? (min_dim / voxel_size) : 0.0f;
    }

    __global__ void query_gs_voxel_intersection_brute_force_kernel(
        const uint32_t num_voxels,
        const uint32_t num_gaussians,
        const float3* __restrict__ vx_aabb_mins,
        const float3* __restrict__ vx_aabb_maxs,
        const float3* __restrict__ means,
        const float* __restrict__ covis,
        const float* __restrict__ opacities,
        const float3* __restrict__ gs_aabb_mins,
        const float3* __restrict__ gs_aabb_maxs,
        const float3* __restrict__ contact_points,
        const float iso,
        const float ar_threshold,
        const float p_threshold,
        const bool return_centroids,
        bool* __restrict__ hit_mask,
        int64_t* __restrict__ out_voxel_ids,
        int64_t* __restrict__ out_gaus_ids,
        float3* __restrict__ centroids,
        float* __restrict__ densities,
        int64_t* __restrict__ global_counter,
        const int64_t max_capacity
    ) {
        uint32_t v_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (v_idx >= num_voxels) return;

        float3 vx_ab_min = vx_aabb_mins[v_idx];
        float3 vx_ab_max = vx_aabb_maxs[v_idx];

        bool any_hit = false;

        for (uint32_t g_idx = 0; g_idx < num_gaussians; g_idx++) {
            float3 mean = means[g_idx];
            const float* covi = covis + (g_idx * 6);
            float opacity = opacities[g_idx];
            float3 gs_ab_min = gs_aabb_mins[g_idx];
            float3 gs_ab_max = gs_aabb_maxs[g_idx];
            float3 cp0 = contact_points[g_idx * 3 + 0];
            float3 cp1 = contact_points[g_idx * 3 + 1];
            float3 cp2 = contact_points[g_idx * 3 + 2];

            bool hit = gs_intersect_voxel(
                g_idx, 
                mean, 
                covi, 
                gs_ab_min, 
                gs_ab_max, 
                cp0, 
                cp1, 
                cp2, 
                vx_ab_min, 
                vx_ab_max, 
                iso
            );

            if (hit) {
                float3 centroid;
                float density;
                float aspect_ratio;
                float penetration;

                // Calculate the metrics of the intersection
                compute_overlap_metrics(
                    gs_ab_min, 
                    gs_ab_max, 
                    vx_ab_min, 
                    vx_ab_max, 
                    mean,
                    covi,
                    opacity,
                    return_centroids,
                    centroid, 
                    density,
                    aspect_ratio, 
                    penetration
                );

                if (aspect_ratio >= ar_threshold && penetration >= p_threshold) {
                    any_hit = true;
                    uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int*)global_counter, 1ULL);
                    
                    if (write_idx < max_capacity) {
                        out_voxel_ids[write_idx] = v_idx;
                        out_gaus_ids[write_idx] = g_idx;

                        if (return_centroids) {
                            centroids[write_idx] = centroid;
                            densities[write_idx] = density;
                        }
                    }
                }
            }
        }

        hit_mask[v_idx] = any_hit;
    }

    void query_gs_voxel_intersection_brute_force(
        const uint32_t num_voxels,
        const uint32_t num_gaussians,
        const float3* __restrict__ vx_aabb_mins,
        const float3* __restrict__ vx_aabb_maxs,
        const float3* __restrict__ means,
        const float* __restrict__ covis,
        const float* __restrict__ opacities,
        const float3* __restrict__ gs_aabb_mins,
        const float3* __restrict__ gs_aabb_maxs,
        const float3* __restrict__ contact_points,
        const float iso,
        const float ar_threshold,
        const float p_threshold,
        const bool return_centroids,
        bool* __restrict__ hit_mask,
        int64_t* __restrict__ out_voxel_ids,
        int64_t* __restrict__ out_gaus_ids,
        float3* __restrict__ centroids,
        float* __restrict__ densities,
        int64_t* __restrict__ global_counter,
        const int64_t max_capacity
    ) {
        uint32_t threads = 256;
        uint32_t blocks = (num_voxels + threads - 1) / threads;

        query_gs_voxel_intersection_brute_force_kernel<<<blocks, threads>>>(
            num_voxels,
            num_gaussians,
            vx_aabb_mins,
            vx_aabb_maxs,
            means,
            covis,
            opacities,
            gs_aabb_mins,
            gs_aabb_maxs,
            contact_points,
            iso,
            ar_threshold,
            p_threshold,
            return_centroids,
            hit_mask,
            out_voxel_ids,
            out_gaus_ids,
            centroids,
            densities,
            global_counter,
            max_capacity
        );
    }
}