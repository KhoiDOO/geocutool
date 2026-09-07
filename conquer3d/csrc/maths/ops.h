#ifndef OPS_H
#define OPS_H

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
/**
 * @file ops.h
 * @brief Host fallbacks for CUDA math intrinsics.
 *
 * @details `rsqrt` and `rsqrtf` are device intrinsics that do not exist when a translation
 * unit is compiled by the host compiler alone. These definitions appear only outside
 * `__CUDACC__`, so headers shared between host and device code compile either way without
 * `#ifdef` guards at every call site.
 */

#include <math_constants.h>

#ifndef __CUDACC__
/**
 * @brief Reciprocal square root, double precision (host fallback).
 * @param[in] a Value whose reciprocal square root is taken; must be positive.
 * @return The value $1 / \sqrt{a}$.
 * @note Defined only when not compiling with nvcc, which supplies the intrinsic.
 */
static inline __host__ __device__ double rsqrt(double a) {
    return 1. / sqrt(a);
}

/**
 * @brief Reciprocal square root, single precision (host fallback).
 * @param[in] a Value whose reciprocal square root is taken; must be positive.
 * @return The value $1 / \sqrt{a}$.
 * @note Defined only when not compiling with nvcc. The device intrinsic is an
 * approximation, so host and device results may differ in the last bits.
 */
static inline __host__ __device__ float rsqrtf(float a) {
    return 1. / sqrtf(a);
}
#endif

#endif // OPS_H