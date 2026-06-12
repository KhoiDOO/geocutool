#pragma once
#include "../base.h"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace gs
{

    __device__ __forceinline__ void compute_mahalanobis_distance(
        const float3 &point,
        const float3 &mean,
        const float *covi,
        float &out_distance)
    {
        float3 d = make_float3(point.x - mean.x, point.y - mean.y, point.z - mean.z);

        out_distance = 
            d.x * (d.x * covi[0] + d.y * covi[1] + d.z * covi[2]) + 
            d.y * (d.x * covi[1] + d.y * covi[3] + d.z * covi[4]) +
            d.z * (d.x * covi[2] + d.y * covi[4] + d.z * covi[5]);
    }

    __device__ __forceinline__ void compute_density(
        const float3 &point,
        const float3 &mean,
        const float *covi,
        const float opacity,
        float &out_density)
    {
        float mahal_dist;
        compute_mahalanobis_distance(point, mean, covi, mahal_dist);

        float power = -0.5f * mahal_dist;

        if (power > 0.0f || power < -15.0f)
        {
            out_density = 0.0f;
        }
        else
        {
            out_density = opacity * expf(power);
        }
    }

    __device__ __forceinline__ void compute_density_local(
        const float3 &point,
        const float *covi,
        const float opacity,
        float &out_density)
    {
        float mahal_dist;
        compute_mahalanobis_distance(point, make_float3(0.0f, 0.0f, 0.0f), covi, mahal_dist);
        float power = -0.5f * mahal_dist;

        if (power > 0.0f || power < -15.0f)
        {
            out_density = 0.0f;
        }
        else
        {
            out_density = opacity * expf(power);
        }
    }

    __device__ __forceinline__ void compute_inverse_scale(
        const float3 &scale, 
        float3x3 &out_inv_scale
    ) {
        out_inv_scale = make_float3x3(
            1.0f / scale.x, 0.0f, 0.0f,
            0.0f, 1.0f / scale.y, 0.0f,
            0.0f, 0.0f, 1.0f / scale.z);
    }

    __device__ __forceinline__ void compute_rotation(
        const float4 &rot,
        float3x3 &out_rotation,
        const bool rotnorm,
        const bool transpose
    ) {
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

        if (transpose)
        {
            out_rotation = make_float3x3(
                1.f - 2.f * (y * y + z * z), 2.f * (x * y + r * z), 2.f * (x * z - r * y),
                2.f * (x * y - r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + r * x),
                2.f * (x * z + r * y), 2.f * (y * z - r * x), 1.f - 2.f * (x * x + y * y));
        } else {
            out_rotation = make_float3x3(
                1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
                2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
        }
    }

    __device__ __forceinline__ void compute_cov_inverse(
        const float3x3 &inv_scale,
        const float3x3 &rotation_transpose,
        float *covi)
    {
        // M = S^{-1} R^T
        float3x3 M = inv_scale * rotation_transpose;

        // M^T M
        // = (S^{-1} R^T)^T (S^{-1} R^T)
        // = R S^{-T} S^{-1} R^T
        // = R S^{-2} R^T
        // = \Sigma^{-1}
        float3x3 out_cov_inv = transpose(M) * M;

        covi[0] = out_cov_inv.m[0][0];
        covi[1] = out_cov_inv.m[0][1];
        covi[2] = out_cov_inv.m[0][2];
        covi[3] = out_cov_inv.m[1][1];
        covi[4] = out_cov_inv.m[1][2];
        covi[5] = out_cov_inv.m[2][2];
    }

    __device__ __forceinline__ bool test_gs_segment(
        const float c0, const float c1, const float c2,
        const float c3, const float c4, const float c5,
        const float iso,
        const float3 &segment_start,
        const float3 &segment_end,
        const bool return_t,
        float &t_entry, float &t_exit
    )
    {
        // d = P1 - P0
        float3 d = make_float3(
            segment_end.x - segment_start.x,
            segment_end.y - segment_start.y,
            segment_end.z - segment_start.z);

        // Sigma^-1 * d
        float3 v_d = make_float3(
            c0 * d.x + c1 * d.y + c2 * d.z,
            c1 * d.x + c3 * d.y + c4 * d.z,
            c2 * d.x + c4 * d.y + c5 * d.z);

        // Sigma^-1 * P0
        float3 v_p0 = make_float3(
            c0 * segment_start.x + c1 * segment_start.y + c2 * segment_start.z,
            c1 * segment_start.x + c3 * segment_start.y + c4 * segment_start.z,
            c2 * segment_start.x + c4 * segment_start.y + c5 * segment_start.z);

        // A = d^T * (Sigma^-1 * d)
        float a = d.x * v_d.x + d.y * v_d.y + d.z * v_d.z;

        // B = 2 * P0^T * (Sigma^-1 * d)
        float b = 2.0f * (segment_start.x * v_d.x + segment_start.y * v_d.y + segment_start.z * v_d.z);

        // C = P0^T * (Sigma^-1 * P0) - iso
        float c = (segment_start.x * v_p0.x + segment_start.y * v_p0.y + segment_start.z * v_p0.z) - iso;

        // B^2 - 4AC
        float dcrm = fmaxf(b * b - 4.0f * a * c, 0.0f);

        // 6. Check for Intersection
        if (dcrm > 0.0f)
        {
            // Calculate the two roots (t_entry and t_exit)
            float rdcrm = sqrtf(dcrm) / (2.0f * a);
            float b2a = -b / (2.0f * a);

            t_entry = b2a - rdcrm;
            t_exit = b2a + rdcrm;

            // If it exits before t=0.0, it's behind the start point.
            // If it enters after t=1.0, it's past the end point.
            if (return_t)
            {
                t_entry = fmaxf(t_entry, 0.0f);
                t_exit = fminf(t_exit, 1.0f);
            }

            return !(1.0f <= t_entry || t_exit <= 0.0f);
        }
        
        if (return_t)
        {
            t_entry = -1.0f;
            t_exit = -1.0f;
        }

        return false;
    }
}

namespace gs_aabb
{
    __device__ __forceinline__ bool test_gs_aabb_overlap_voxel(
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
}