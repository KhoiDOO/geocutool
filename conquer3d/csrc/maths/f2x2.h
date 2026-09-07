#ifndef F2X2_H
#define F2X2_H

#include "ops.h"

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>

/**
 * @file f2x2.h
 * @brief The `float2x2` matrix type, its products, and closed-form inversion.
 *
 * @details Two-dimensional linear algebra, used for planar projections and for the
 * $2 \times 2$ subproblems that appear inside larger solves. Every routine is closed form --
 * no loops, no pivoting -- so it costs a fixed number of instructions regardless of input.
 */

/**
 * @brief 2x2 float matrix.
 */
typedef struct
{
    float m[2][2]; ///< Row-major elements, `m[row][col]`.
} float2x2;

/**
 * @brief Constructs a matrix from its four elements in row-major order.
 * @return The assembled matrix.
 */
static __inline__ __host__ __device__ float2x2 make_float2x2(
    float a00, float a01,
    float a10, float a11) {
    float2x2 a;
    a.m[0][0] = a00; a.m[0][1] = a01;
    a.m[1][0] = a10; a.m[1][1] = a11;
    return a;
}

// [2, 2] x [2, 2] = [2, 2]
/**
 * @brief Matrix-matrix product.
 * @param[in] a Left operand.
 * @param[in] b Right operand.
 * @return The product $\mathbf{a}\mathbf{b}$.
 */
static __inline__ __host__ __device__ float2x2 operator* (const float2x2& a, const float2x2& b)
{
    float2x2 c;

    c.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0];
    c.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1];
    c.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0];
    c.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1];

    return c;
}

// [2, 1]^T x [2, 2] = [2, 1]^T
/**
 * @brief Row-vector times matrix.
 * @param[in] a Row vector.
 * @param[in] m Matrix operand.
 * @return The vector $\mathbf{a}\mathbf{m}$.
 */
static __inline__ __host__ __device__ float2 operator*(const float2& a, const float2x2& m) {
    return make_float2(
        a.x * m.m[0][0] + a.y * m.m[1][0],
        a.x * m.m[0][1] + a.y * m.m[1][1]
    );
}

// [2, 2] x [2, 1] = [2, 1]
/**
 * @brief Matrix times column vector.
 * @param[in] m Matrix operand.
 * @param[in] a Column vector.
 * @return The vector $\mathbf{m}\mathbf{a}$.
 */
static __inline__ __host__ __device__ float2 operator*(const float2x2& m, const float2& a) {
    return make_float2(
        m.m[0][0] * a.x + m.m[0][1] * a.y,
        m.m[1][0] * a.x + m.m[1][1] * a.y
    );
}

namespace maths
{
    /**
     * @brief Transpose of a $2 \times 2$ matrix.
     * @param[in] a Matrix to transpose.
     * @return The matrix $\mathbf{a}^\top$.
     */
    static __inline__ __host__ __device__ float2x2 transpose(const float2x2& a) {
        float2x2 b;
        b.m[0][0] = a.m[0][0]; b.m[0][1] = a.m[1][0];
        b.m[1][0] = a.m[0][1]; b.m[1][1] = a.m[1][1];
        return b;
    }

    /**
     * @brief Determinant of a $2 \times 2$ matrix.
     * @param[in] a Matrix operand.
     * @return The scalar $\det(\mathbf{a})$; zero indicates a singular matrix.
     */
    static __inline__ __host__ __device__ float det(const float2x2& a) {
        return a.m[0][0] * a.m[1][1] - a.m[0][1] * a.m[1][0];
    }

    /**
     * @brief Copies one matrix into another.
     * @param[out] a Destination matrix.
     * @param[in] b Source matrix.
     */
    static inline __host__ __device__ void copy(float2x2 &a, float2x2 b) {
        a.m[0][0] = b.m[0][0]; a.m[0][1] = b.m[0][1];
        a.m[1][0] = b.m[1][0]; a.m[1][1] = b.m[1][1];
    }

    /**
     * @brief Inverts a matrix via the adjugate, if it is non-singular.
     * @param[in] m Matrix to invert.
     * @param[out] out_inv Receives $\mathbf{m}^{-1}$; untouched when the matrix is singular.
     * @return True on success, false when the determinant is too close to zero.
     * @note Always check the return value -- ignoring it leaves @p out_inv uninitialised.
     */
    static __host__ __device__ __forceinline__ bool invert(const float2x2& m, float2x2& out_inv) 
    {
        float det_val = m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0];

        if (fabsf(det_val) < 1e-8f) return false;

        float inv_det = 1.0f / det_val;

        out_inv.m[0][0] =  m.m[1][1] * inv_det;
        out_inv.m[0][1] = -m.m[0][1] * inv_det;
        out_inv.m[1][0] = -m.m[1][0] * inv_det;
        out_inv.m[1][1] =  m.m[0][0] * inv_det;

        return true;
    }
}

#endif // F2X2_H