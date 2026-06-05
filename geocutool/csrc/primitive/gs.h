#ifndef GS_H
#define GS_H

#include "base.h"
#include "../check.h"

#include <cuda_runtime.h>
#include <cstdint>


namespace gs_aabb
{
    __host__ void compute_aabb(
        const uint32_t num_gaussians,
        const float3* __restrict__ means,
        const float4* __restrict__ rotations,
        const float3* __restrict__ scales,
        const float iso,
        const float tol,
        const uint32_t level,
        const bool rotnorm,
        float3* __restrict__ aabb_min,
        float3* __restrict__ aabb_max,
        float3* __restrict__ contact_points,
        float* __restrict__ covi
    );

    __host__ void query_gs_voxel_intersection_brute_force(
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
    );
}

#endif // GS_H