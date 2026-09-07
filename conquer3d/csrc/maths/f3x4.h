#ifndef F3x4_H
#define F3x4_H

/**
 * @file f3x4.h
 * @brief The `float3x4` affine transform type and its products.
 *
 * @details A $3 \times 4$ matrix stores a rotation and a translation without the constant
 * bottom row of a full $4 \times 4$, which is the compact form used for camera extrinsics
 * and instance transforms. Dropping the row saves four floats per transform and a row of
 * multiply-adds per point.
 */

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>

/**
 * @brief 3x4 affine transform matrix.
 *
 * @details Stores a rotation and translation without the constant bottom row of a full
 * 4x4, the compact form used for camera extrinsics and instance transforms.
 */
typedef struct
{
    float m[3][4]; ///< Row-major elements, `m[row][col]`.
} float3x4;

/**
 * @brief Applies an affine transform to a homogeneous point.
 * @details Computes $\mathbf{m}\,\mathbf{a}$ under the column-vector convention. Pass
 * $w = 1$ to transform a position and $w = 0$ to transform a direction, which skips the
 * translation column.
 * @param[in] m Affine transform matrix.
 * @param[in] a Homogeneous input vector.
 * @return The transformed 3D vector.
 */
static __inline__ __host__ __device__ float3 mul3x4(float3x4 m, float4 a) {
    return make_float3(
        a.x * m.m[0][0] + a.y * m.m[0][1] + a.z * m.m[0][2] + a.w * m.m[0][3],
        a.x * m.m[1][0] + a.y * m.m[1][1] + a.z * m.m[1][2] + a.w * m.m[1][3],
        a.x * m.m[2][0] + a.y * m.m[2][1] + a.z * m.m[2][2] + a.w * m.m[2][3]
    );
}

/**
 * @brief Operator spelling of mul3x4().
 * @param[in] m Affine transform matrix.
 * @param[in] a Homogeneous input vector.
 * @return The transformed 3D vector.
 */
static __inline__ __host__ __device__ float3 operator* (const float3x4 m, const float4 a)
{
    return mul3x4(m, a);
}

#endif // F3x4_H