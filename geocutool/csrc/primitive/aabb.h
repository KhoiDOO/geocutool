#ifndef AABB_H
#define AABB_H

#include "../base.h"

#include <cuda_runtime.h>

namespace aabb 
{
    __device__ __forceinline__ void compute_aabb_centroid(
        const float3 &aabb_min,
        const float3 &aabb_max,
        float3 &out_centroid)
    {
        out_centroid = make_float3(
            (aabb_min.x + aabb_max.x) * 0.5f,
            (aabb_min.y + aabb_max.y) * 0.5f,
            (aabb_min.z + aabb_max.z) * 0.5f);
    }
}

#endif // AABB_H