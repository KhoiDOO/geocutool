/**
 * @file f4x4.h
 * @brief 4x4 single-precision homogeneous transformation matrix struct and GPU device operations.
 */

#ifndef F4x4_H
#define F4x4_H

#include "f3x4.h"

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>

/**
 * @file f4x4.h
 * @brief The `float4x4` homogeneous transform type and its products.
 *
 * @details Full $4 \times 4$ transforms, as used for camera projection and view matrices
 * where the bottom row is not constant and the compact `float3x4` of f3x4.h will not do.
 * Products follow the row-vector convention: translation lives in row 3, and points are
 * multiplied on the left.
 */

/**
 * @brief 4x4 float matrix.
 */
typedef struct
{
    float m[4][4]; ///< Row-major elements, `m[row][col]`.
} float4x4;

/**
 * @brief Constructs a matrix from its sixteen elements in row-major order.
 * @return The assembled matrix.
 */
static __inline__ __host__ __device__ float4x4 make_float4x4(
    float a00, float a01, float a02, float a03,
    float a10, float a11, float a12, float a13,
    float a20, float a21, float a22, float a23,
    float a30, float a31, float a32, float a33) {
    float4x4 a;
    a.m[0][0] = a00; a.m[0][1] = a01; a.m[0][2] = a02; a.m[0][3] = a03;
    a.m[1][0] = a10; a.m[1][1] = a11; a.m[1][2] = a12; a.m[1][3] = a13;
    a.m[2][0] = a20; a.m[2][1] = a21; a.m[2][2] = a22; a.m[2][3] = a23;
    a.m[3][0] = a30; a.m[3][1] = a31; a.m[3][2] = a32; a.m[3][3] = a33;
    return a;
}

// [4, 4] x [4, 4] = [4, 4]
/**
 * @brief Matrix-matrix product.
 * @details Composes two transforms. Under the row-vector convention the left operand is
 * applied first, so `model * view * projection` reads in execution order.
 * @param[in] a Left operand.
 * @param[in] b Right operand.
 * @return The product $\mathbf{a}\mathbf{b}$.
 */
static __inline__ __host__ __device__ float4x4  operator*(const float4x4& a, const float4x4& b)
{
    float4x4 c;

    c.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0] + a.m[0][2] * b.m[2][0] + a.m[0][3] * b.m[3][0];
    c.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1] + a.m[0][2] * b.m[2][1] + a.m[0][3] * b.m[3][1];
    c.m[0][2] = a.m[0][0] * b.m[0][2] + a.m[0][1] * b.m[1][2] + a.m[0][2] * b.m[2][2] + a.m[0][3] * b.m[3][2];
    c.m[0][3] = a.m[0][0] * b.m[0][3] + a.m[0][1] * b.m[1][3] + a.m[0][2] * b.m[2][3] + a.m[0][3] * b.m[3][3];

    c.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0] + a.m[1][2] * b.m[2][0] + a.m[1][3] * b.m[3][0];
    c.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1] + a.m[1][2] * b.m[2][1] + a.m[1][3] * b.m[3][1];
    c.m[1][2] = a.m[1][0] * b.m[0][2] + a.m[1][1] * b.m[1][2] + a.m[1][2] * b.m[2][2] + a.m[1][3] * b.m[3][2];
    c.m[1][3] = a.m[1][0] * b.m[0][3] + a.m[1][1] * b.m[1][3] + a.m[1][2] * b.m[2][3] + a.m[1][3] * b.m[3][3];

    c.m[2][0] = a.m[2][0] * b.m[0][0] + a.m[2][1] * b.m[1][0] + a.m[2][2] * b.m[2][0] + a.m[2][3] * b.m[3][0];
    c.m[2][1] = a.m[2][0] * b.m[0][1] + a.m[2][1] * b.m[1][1] + a.m[2][2] * b.m[2][1] + a.m[2][3] * b.m[3][1];
    c.m[2][2] = a.m[2][0] * b.m[0][2] + a.m[2][1] * b.m[1][2] + a.m[2][2] * b.m[2][2] + a.m[2][3] * b.m[3][2];
    c.m[2][3] = a.m[2][0] * b.m[0][3] + a.m[2][1] * b.m[1][3] + a.m[2][2] * b.m[2][3] + a.m[2][3] * b.m[3][3];

    c.m[3][0] = a.m[3][0] * b.m[0][0] + a.m[3][1] * b.m[1][0] + a.m[3][2] * b.m[2][0] + a.m[3][3] * b.m[3][0];
    c.m[3][1] = a.m[3][0] * b.m[0][1] + a.m[3][1] * b.m[1][1] + a.m[3][2] * b.m[2][1] + a.m[3][3] * b.m[3][1];
    c.m[3][2] = a.m[3][0] * b.m[0][2] + a.m[3][1] * b.m[1][2] + a.m[3][2] * b.m[2][2] + a.m[3][3] * b.m[3][2];
    c.m[3][3] = a.m[3][0] * b.m[0][3] + a.m[3][1] * b.m[1][3] + a.m[3][2] * b.m[2][3] + a.m[3][3] * b.m[3][3];

    return c;
}

// [4, 4] x [4, 1] = [4, 1]
/**
 * @brief Row vector times matrix.
 * @param[in] a Homogeneous row vector.
 * @param[in] m Transform matrix.
 * @return The vector $\mathbf{a}\mathbf{m}$, with $w$ left un-divided.
 */
static __inline__ __host__ __device__ float4 operator* (const float4& a, const float4x4& m)
{
    return make_float4(
        a.x * m.m[0][0] + a.y * m.m[1][0] + a.z * m.m[2][0] + a.w * m.m[3][0],
        a.x * m.m[0][1] + a.y * m.m[1][1] + a.z * m.m[2][1] + a.w * m.m[3][1],
        a.x * m.m[0][2] + a.y * m.m[1][2] + a.z * m.m[2][2] + a.w * m.m[3][2],
        a.x * m.m[0][3] + a.y * m.m[1][3] + a.z * m.m[2][3] + a.w * m.m[3][3]
    );
}

// [4, 4] x [4, 1] = [4, 1]
/**
 * @brief Matrix times homogeneous column vector.
 * @param[in] m Transform matrix.
 * @param[in] a Homogeneous column vector.
 * @return The vector $\mathbf{m}\mathbf{a}$.
 */
static __inline__ __host__ __device__ float4 operator* (const float4x4& m, const float4& a)
{
    return make_float4(
        a.x * m.m[0][0] + a.y * m.m[0][1] + a.z * m.m[0][2] + a.w * m.m[0][3],
        a.x * m.m[1][0] + a.y * m.m[1][1] + a.z * m.m[1][2] + a.w * m.m[1][3],
        a.x * m.m[2][0] + a.y * m.m[2][1] + a.z * m.m[2][2] + a.w * m.m[2][3],
        a.x * m.m[3][0] + a.y * m.m[3][1] + a.z * m.m[3][2] + a.w * m.m[3][3]
    );
}

// [3, 4] x [4, 1] = [3, 1]
/**
 * @brief Transforms a 3D point, taking $w = 1$ implicitly.
 * @details Applies the linear part and adds the translation row, then returns the first
 * three components without a perspective divide. Correct for affine transforms; for a
 * projection matrix, use the `float4` overload and divide by $w$ yourself.
 * @param[in] a Point to transform.
 * @param[in] m Transform matrix.
 * @return The transformed point.
 * @warning Translation is always applied, so this transforms positions, not directions.
 */
static __inline__ __host__ __device__ float3 operator*(const float3& a, const float4x4& m) {
    return make_float3(
        a.x * m.m[0][0] + a.y * m.m[1][0] + a.z * m.m[2][0] + m.m[3][0],
        a.x * m.m[0][1] + a.y * m.m[1][1] + a.z * m.m[2][1] + m.m[3][1],
        a.x * m.m[0][2] + a.y * m.m[1][2] + a.z * m.m[2][2] + m.m[3][2]
    );
}

// [3, 4] x [4, 4] = [3, 4]
/**
 * @brief Composes a compact affine transform with a full homogeneous one.
 * @details Computes a $3 \times 4$ by $4 \times 4$ product, keeping the compact form so a
 * chain of transforms never materialises the constant bottom row.
 * @param[in] a Left operand, a compact affine transform.
 * @param[in] b Right operand, a full transform.
 * @return The composed $3 \times 4$ transform.
 */
static __inline__ __host__ __device__ float3x4 operator* (const float3x4& a, const float4x4& b)
{
    float3x4 c;

    c.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0] + a.m[0][2] * b.m[2][0] + a.m[0][3] * b.m[3][0];
    c.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1] + a.m[0][2] * b.m[2][1] + a.m[0][3] * b.m[3][1];
    c.m[0][2] = a.m[0][0] * b.m[0][2] + a.m[0][1] * b.m[1][2] + a.m[0][2] * b.m[2][2] + a.m[0][3] * b.m[3][2];
    c.m[0][3] = a.m[0][0] * b.m[0][3] + a.m[0][1] * b.m[1][3] + a.m[0][2] * b.m[2][3] + a.m[0][3] * b.m[3][3];

    c.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0] + a.m[1][2] * b.m[2][0] + a.m[1][3] * b.m[3][0];
    c.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1] + a.m[1][2] * b.m[2][1] + a.m[1][3] * b.m[3][1];
    c.m[1][2] = a.m[1][0] * b.m[0][2] + a.m[1][1] * b.m[1][2] + a.m[1][2] * b.m[2][2] + a.m[1][3] * b.m[3][2];
    c.m[1][3] = a.m[1][0] * b.m[0][3] + a.m[1][1] * b.m[1][3] + a.m[1][2] * b.m[2][3] + a.m[1][3] * b.m[3][3];

    c.m[2][0] = a.m[2][0] * b.m[0][0] + a.m[2][1] * b.m[1][0] + a.m[2][2] * b.m[2][0] + a.m[2][3] * b.m[3][0];
    c.m[2][1] = a.m[2][0] * b.m[0][1] + a.m[2][1] * b.m[1][1] + a.m[2][2] * b.m[2][1] + a.m[2][3] * b.m[3][1];
    c.m[2][2] = a.m[2][0] * b.m[0][2] + a.m[2][1] * b.m[1][2] + a.m[2][2] * b.m[2][2] + a.m[2][3] * b.m[3][2];
    c.m[2][3] = a.m[2][0] * b.m[0][3] + a.m[2][1] * b.m[1][3] + a.m[2][2] * b.m[2][3] + a.m[2][3] * b.m[3][3];

    return c;
}

namespace maths
{
    /**
     * @brief Transpose of a $4 \times 4$ matrix.
     * @param[in] a Matrix to transpose.
     * @return The matrix $\mathbf{a}^\top$, as needed to switch between row- and
     * column-vector conventions.
     */
    static __inline__ __host__ __device__ float4x4 transpose(const float4x4& a) {
        float4x4 b;
        b.m[0][0] = a.m[0][0]; b.m[0][1] = a.m[1][0]; b.m[0][2] = a.m[2][0]; b.m[0][3] = a.m[3][0];
        b.m[1][0] = a.m[0][1]; b.m[1][1] = a.m[1][1]; b.m[1][2] = a.m[2][1]; b.m[1][3] = a.m[3][1];
        b.m[2][0] = a.m[0][2]; b.m[2][1] = a.m[1][2]; b.m[2][2] = a.m[2][2]; b.m[2][3] = a.m[3][2];
        b.m[3][0] = a.m[0][3]; b.m[3][1] = a.m[1][3]; b.m[3][2] = a.m[2][3]; b.m[3][3] = a.m[3][3];
        return b;
    }

    /**
     * @brief Copies one matrix into another.
     * @param[out] a Destination matrix.
     * @param[in] b Source matrix.
     */
    static inline __host__ __device__ void copy(float4x4 &a, float4x4 b) {
        a.m[0][0] = b.m[0][0]; a.m[0][1] = b.m[0][1]; a.m[0][2] = b.m[0][2]; a.m[0][3] = b.m[0][3];
        a.m[1][0] = b.m[1][0]; a.m[1][1] = b.m[1][1]; a.m[1][2] = b.m[1][2]; a.m[1][3] = b.m[1][3];
        a.m[2][0] = b.m[2][0]; a.m[2][1] = b.m[2][1]; a.m[2][2] = b.m[2][2]; a.m[2][3] = b.m[2][3];
        a.m[3][0] = b.m[3][0]; a.m[3][1] = b.m[3][1]; a.m[3][2] = b.m[3][2]; a.m[3][3] = b.m[3][3]; 
    }
}

#endif // F4x4_H