#ifndef F3x1_H
#define F3x1_H

/**
 * @file f3x1.h
 * @brief Arithmetic operators and vector routines for CUDA's `float3` type.
 *
 * @details CUDA ships `float3` as a plain struct with no operators, so every geometric
 * kernel in the library would otherwise spell out component arithmetic by hand. These
 * `__host__ __device__` inlines supply the missing algebra once, compile to the same
 * instructions as the expanded form, and keep kernel code readable.
 *
 * All functions are branch-free and safe to call from divergent warp contexts.
 */

#include "ops.h"

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>

/**
 * @brief Component-wise sum of two vectors.
 * @param[in] a First operand.
 * @param[in] b Second operand.
 * @return The vector $\mathbf{a} + \mathbf{b}$.
 */
static inline __host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/**
 * @brief Adds a vector in place.
 * @param[in,out] a Vector accumulated into.
 * @param[in] b Vector added to @p a.
 */
static inline __host__ __device__ void operator+=(float3 &a, float3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

/**
 * @brief Component-wise difference of two vectors.
 * @param[in] a Minuend.
 * @param[in] b Subtrahend.
 * @return The vector $\mathbf{a} - \mathbf{b}$.
 */
static inline __host__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/**
 * @brief Subtracts a vector in place.
 * @param[in,out] a Vector decremented.
 * @param[in] b Vector subtracted from @p a.
 */
static inline __host__ __device__ void operator-=(float3 &a, float3 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

/**
 * @brief Negates a vector.
 * @param[in] a Vector to negate.
 * @return The vector $-\mathbf{a}$.
 */
static inline __host__ __device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

/**
 * @brief Scales a vector by a scalar.
 * @param[in] a Vector operand.
 * @param[in] b Scalar multiplier.
 * @return The vector $b\,\mathbf{a}$.
 */
static inline __host__ __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

/**
 * @brief Scales a vector by a scalar, with the scalar on the left.
 * @param[in] b Scalar multiplier.
 * @param[in] a Vector operand.
 * @return The vector $b\,\mathbf{a}$.
 */
static inline __host__ __device__ float3 operator*(float b, float3 a) {
    return make_float3(b * a.x, b * a.y, b * a.z);
}

/**
 * @brief Scales a vector in place.
 * @param[in,out] a Vector scaled.
 * @param[in] b Scalar multiplier.
 */
static inline __host__ __device__ void operator*=(float3 &a, float b) {
    a.x *= b; a.y *= b; a.z *= b;
}

/**
 * @brief Divides a vector by a scalar.
 * @details Computes the reciprocal once and multiplies, trading exact rounding for a
 * single division instead of three.
 * @param[in] a Vector operand.
 * @param[in] b Scalar divisor.
 * @return The vector $\mathbf{a} / b$.
 * @warning No zero check; a zero divisor yields infinities or NaNs.
 */
static inline __host__ __device__ float3 operator/(float3 a, const float b) {
    float inv = 1.0f / b;
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

/**
 * @brief Divides a vector by a scalar in place.
 * @param[in,out] a Vector divided.
 * @param[in] b Scalar divisor.
 * @warning No zero check; see operator/(float3, const float).
 */
static inline __host__ __device__ void operator/=(float3 &a, float b) {
    float inv = 1.0f / b;
    a.x *= inv; a.y *= inv; a.z *= inv;
}

#ifdef __CUDACC__
/**
 * @brief Atomically accumulates a vector into device memory.
 * @details Issues three independent scalar `atomicAdd` calls. The components are
 * therefore atomic individually but not as a unit -- a concurrent reader may observe a
 * partially updated vector. That is sufficient for gradient accumulation, where only
 * the final total matters.
 * @param[in,out] address Destination vector in global or shared memory.
 * @param[in] val Vector added to the destination.
 * @return The component-wise values held before the update.
 * @note Device-only; requires `__CUDACC__`.
 */
static inline __device__ float3 atomicAdd(float3* address, float3 val) {
    float3 old;
    old.x = ::atomicAdd(&address->x, val.x);
    old.y = ::atomicAdd(&address->y, val.y);
    old.z = ::atomicAdd(&address->z, val.z);
    return old;
}
#endif

namespace maths
{
    /**
     * @brief Euclidean inner product.
     * @param[in] a First operand.
     * @param[in] b Second operand.
     * @return The scalar $\mathbf{a} \cdot \mathbf{b}$.
     */
    static inline __host__ __device__ float dot(float3 a, float3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    /**
     * @brief Squared length of a vector.
     * @details Equivalent to `dot(a, a)`. Preferred over norm() when only comparing
     * magnitudes, since it avoids a square root.
     * @param[in] a Vector operand.
     * @return The scalar $\|\mathbf{a}\|^2$.
     */
    static inline __host__ __device__ float dot2(float3 a) {
        return dot(a, a);
    }

    /**
     * @brief Euclidean length of a vector.
     * @param[in] a Vector operand.
     * @return The scalar $\|\mathbf{a}\|$.
     */
    static inline __host__ __device__ float norm(float3 a) {
        return sqrtf(dot2(a));
    }

    /**
     * @brief Cross product of two vectors.
     * @details The result is orthogonal to both operands, with magnitude equal to the area
     * of the parallelogram they span -- the basis for triangle normals and areas throughout
     * the library.
     * @param[in] a First operand.
     * @param[in] b Second operand.
     * @return The vector $\mathbf{a} \times \mathbf{b}$.
     */
    static inline __host__ __device__ float3 cross(float3 a, float3 b) {
        return make_float3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    /**
     * @brief Scales a vector to unit length.
     * @details Uses the hardware reciprocal square root rather than dividing by norm(),
     * which is faster and accurate enough for shading and geometric predicates.
     * @param[in] v Vector to normalise.
     * @return The unit vector $\mathbf{v} / \|\mathbf{v}\|$.
     * @warning Undefined for the zero vector, which yields NaNs. Guard degenerate inputs
     * with dot2() before calling.
     */
    static inline __host__ __device__ float3 normalize(float3 v) {
        float invLen = rsqrtf(dot2(v));
        return v * invLen;
    }

    /**
     * @brief Exact component-wise equality test.
     * @param[in] a First operand.
     * @param[in] b Second operand.
     * @return True when all three components match bit-for-bit.
     * @warning Exact floating-point comparison; use a tolerance for computed values.
     */
    static inline __host__ __device__ bool equals(float3 a, float3 b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }

    /**
     * @brief Component-wise absolute value.
     * @param[in] a Vector operand.
     * @return A vector holding $|a_x|, |a_y|, |a_z|$.
     */
    static inline __host__ __device__ float3 abs(float3 a) {
        return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
    }

    /**
     * @brief Component-wise minimum of two vectors.
     * @param[in] a First operand.
     * @param[in] b Second operand.
     * @return A vector of per-component minima, as used to grow AABBs.
     */
    static inline __host__ __device__ float3 min(float3 a, float3 b) {
        return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
    }

    /**
     * @brief Component-wise maximum of two vectors.
     * @param[in] a First operand.
     * @param[in] b Second operand.
     * @return A vector of per-component maxima, as used to grow AABBs.
     */
    static inline __host__ __device__ float3 max(float3 a, float3 b) {
        return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
    }

    /**
     * @brief Clamps each component into a per-axis range.
     * @param[in] v Vector to clamp.
     * @param[in] min_val Per-component lower bounds.
     * @param[in] max_val Per-component upper bounds.
     * @return @p v with every component confined to its range.
     */
    static inline __host__ __device__ float3 clamp(float3 v, float3 min_val, float3 max_val) {
        return make_float3(
            fminf(fmaxf(v.x, min_val.x), max_val.x),
            fminf(fmaxf(v.y, min_val.y), max_val.y),
            fminf(fmaxf(v.z, min_val.z), max_val.z)
        );
    }
}

/**
 * @brief Component-wise (Hadamard) product of two vectors.
 * @details Distinct from dot() and cross(): this multiplies matching components, as
 * needed for per-axis scaling.
 * @param[in] a First operand.
 * @param[in] b Second operand.
 * @return A vector holding $a_x b_x, a_y b_y, a_z b_z$.
 */
static inline __host__ __device__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

#endif // F3x1_H