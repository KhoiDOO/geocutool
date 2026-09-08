#ifndef OPS_H
#define OPS_H

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
/**
 * @file ops.h
 * @brief Host fallbacks for CUDA math intrinsics, and scalar interpolation helpers.
 *
 * @details `rsqrt` and `rsqrtf` are device intrinsics that do not exist when a translation
 * unit is compiled by the host compiler alone. These definitions appear only outside
 * `__CUDACC__`, so headers shared between host and device code compile either way without
 * `#ifdef` guards at every call site.
 *
 * The scalar helpers below are the counterparts of the `float3` routines in f3x1.h and
 * overload on the same names, so `maths::clamp` and `maths::lerp` read identically whether
 * the operand is a scalar or a vector. They live here rather than in f3x1.h because this
 * header is included first and must not depend on the vector operators.
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

namespace maths
{
    /**
     * @brief Confines a scalar to a range.
     * @details Scalar overload of the `float3` clamp in f3x1.h. Spelling the two branches
     * as one call keeps the intent legible where the bound is itself an expression.
     * @param[in] v Value to clamp.
     * @param[in] min_val Lower bound.
     * @param[in] max_val Upper bound.
     * @return @p v confined to $[\text{min\_val}, \text{max\_val}]$.
     * @note Ordered `fmaxf(lo, fminf(hi, v))` to match the form these call sites already
     * used. The order is irrelevant for finite input but decides which bound a NaN
     * collapses to, so keeping it preserves the previous behaviour exactly.
     */
    static inline __host__ __device__ float clamp(float v, float min_val, float max_val) {
        return fmaxf(min_val, fminf(max_val, v));
    }

    /**
     * @brief Confines a scalar to the unit interval.
     * @details The overwhelmingly common case of clamp(), used wherever an interpolation
     * parameter or a normalised cell coordinate must not escape $[0, 1]$ through rounding.
     * @param[in] v Value to clamp.
     * @return @p v confined to $[0, 1]$.
     */
    static inline __host__ __device__ float saturate(float v) {
        return fmaxf(0.0f, fminf(1.0f, v));
    }

    /**
     * @brief Linear interpolation between two scalars.
     * @param[in] a Value returned at $t = 0$.
     * @param[in] b Value returned at $t = 1$.
     * @param[in] t Interpolation parameter; not clamped.
     * @return $a + t\,(b - a)$.
     */
    static inline __host__ __device__ float lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }
}

#endif // OPS_H