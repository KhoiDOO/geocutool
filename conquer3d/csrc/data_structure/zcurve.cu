/**
 * @file zcurve.cu
 * @brief CUDA kernel implementations for 3D Morton space-filling Z-curve code computation.
 */

#include "zcurve.h"
#include "../constants.h"
#include <cuda_runtime.h>

namespace zcurve
{

/**
 * @brief Computes a 3D Morton code for every point.
 * @details One thread per point. Interleaving the bits of the quantised $x$, $y$ and $z$
 * coordinates maps 3D position onto a one-dimensional space-filling curve, so points that
 * are close in space receive close codes. Sorting by that code is what gives every
 * hierarchy in the library its cache-coherent memory layout.
 * @param[in] points Device array of `num_points` coordinates, expected pre-normalised to
 *     the unit cube.
 * @param[in] num_points Number of points.
 * @param[out] codes Device array of `num_points` Morton codes, widened to 64-bit for the
 *     radix sort that follows.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note The code is computed at 10 bits per axis, so points closer than $2^{-10}$ of the
 * grid extent collide. Collisions are harmless -- they only leave the relative order of
 * near-coincident points unspecified.
 */
__global__ void compute_zcurve_kernel(const float3 *__restrict__ points, uint32_t num_points, int64_t *__restrict__ codes)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points)
            return;

        float3 p = points[idx];
        unsigned int code = morton3D(p.x, p.y, p.z);
        codes[idx] = (int64_t)code;
    }

    void compute_zcurve(const float *points, uint32_t num_points, int64_t *codes)
    {
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_points + threads - 1) / threads;

        compute_zcurve_kernel<<<blocks, threads>>>((const float3 *)points, num_points, codes);
    }

}
