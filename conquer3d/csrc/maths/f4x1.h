#ifndef F4x1_H
#define F4x1_H

#include "ops.h"

/**
 * @file f4x1.h
 * @brief Arithmetic operators and vector routines for CUDA's `float4` type.
 *
 * @details The `float4` counterpart to f3x1.h. Four-component vectors carry homogeneous
 * coordinates for matrix transforms and quaternion rotations, and their 16-byte width lets
 * the hardware move one per instruction, so `float4` is also the preferred layout for
 * coalesced global-memory traffic.
 */

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
static inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

/**
 * @brief Adds a vector in place.
 * @param[in,out] a Vector accumulated into.
 * @param[in] b Vector added to @p a.
 */
static inline __host__ __device__ void operator+=(float4 &a, float4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

/**
 * @brief Component-wise difference of two vectors.
 * @param[in] a Minuend.
 * @param[in] b Subtrahend.
 * @return The vector $\mathbf{a} - \mathbf{b}$.
 */
static inline __host__ __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

/**
 * @brief Subtracts a vector in place.
 * @param[in,out] a Vector decremented.
 * @param[in] b Vector subtracted from @p a.
 */
static inline __host__ __device__ void operator-=(float4 &a, float4 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

/**
 * @brief Scales a vector by a scalar.
 * @param[in] a Vector operand.
 * @param[in] b Scalar multiplier.
 * @return The vector $b\,\mathbf{a}$.
 */
static inline __host__ __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

/**
 * @brief Scales a vector by a scalar, with the scalar on the left.
 * @param[in] b Scalar multiplier.
 * @param[in] a Vector operand.
 * @return The vector $b\,\mathbf{a}$.
 */
static inline __host__ __device__ float4 operator*(float b, float4 a) {
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

/**
 * @brief Scales a vector in place.
 * @param[in,out] a Vector scaled.
 * @param[in] b Scalar multiplier.
 */
static inline __host__ __device__ void operator*=(float4 &a, float b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

/**
 * @brief Divides a vector by a scalar.
 * @details Multiplies by the precomputed reciprocal, one division instead of four.
 * @param[in] a Vector operand.
 * @param[in] b Scalar divisor.
 * @return The vector $\mathbf{a} / b$.
 * @warning No zero check; a zero divisor yields infinities or NaNs.
 */
static inline __host__ __device__ float4 operator/(float4 a, const float b) {
    float inv = 1.0f / b;
    return make_float4(a.x * inv, a.y * inv, a.z * inv, a.w * inv);
}

/**
 * @brief Divides a vector by a scalar in place.
 * @param[in,out] a Vector divided.
 * @param[in] b Scalar divisor.
 * @warning No zero check.
 */
static inline __host__ __device__ void operator/=(float4 &a, float b) {
    float inv = 1.0f / b;
    a.x *= inv; a.y *= inv; a.z *= inv; a.w *= inv;
}

namespace maths
{
    /**
     * @brief Euclidean inner product over all four components.
     * @param[in] a First operand.
     * @param[in] b Second operand.
     * @return The scalar $\mathbf{a} \cdot \mathbf{b}$.
     * @note Includes the $w$ component; for a homogeneous point this is rarely the
     * geometrically meaningful quantity.
     */
    static inline __host__ __device__ float dot(float4 a, float4 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    /**
     * @brief Squared length of a vector.
     * @param[in] a Vector operand.
     * @return The scalar $\|\mathbf{a}\|^2$, avoiding the square root of norm().
     */
    static inline __host__ __device__ float dot2(float4 a) {
        return dot(a, a);
    }

    /**
     * @brief Euclidean length of a vector.
     * @param[in] a Vector operand.
     * @return The scalar $\|\mathbf{a}\|$.
     */
    static inline __host__ __device__ float norm(float4 a) {
        return sqrtf(dot2(a));
    }

    /**
     * @brief Scales a vector to unit length.
     * @details Uses the hardware reciprocal square root. Also the operation that
     * renormalises a quaternion after accumulated drift.
     * @param[in] v Vector to normalise.
     * @return The unit vector $\mathbf{v} / \|\mathbf{v}\|$.
     * @warning Undefined for the zero vector.
     */
    static inline __host__ __device__ float4 normalize(float4 v) {
        float invLen = rsqrtf(dot2(v));
        return v * invLen;
    }

    /**
     * @brief Exact component-wise equality test.
     * @param[in] a First operand.
     * @param[in] b Second operand.
     * @return True when all four components match bit-for-bit.
     * @warning Exact floating-point comparison.
     */
    static inline __host__ __device__ bool equals(float4 a, float4 b) {
        return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
    }
}

#endif // F4x1_H