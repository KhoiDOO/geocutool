#ifndef QEF_H
#define QEF_H

#include "f3x3.h"
#include "f3x1.h"
#include "ops.h"
#include <cuda_runtime.h>
#include <math.h>

namespace maths {

/**
 * @brief Computes the eigensystem of a real symmetric 3x3 matrix via Jacobi rotations.
 * 
 * @param A Input symmetric 3x3 matrix.
 * @param eigenvalues Output eigenvalues (diagonal).
 * @param eigenvectors Output orthonormal eigenvectors (columns of V).
 * @param max_sweeps Maximum Jacobi sweeps (default: 6).
 */
__host__ __device__ __forceinline__ void symmetric_eigen_3x3(
    const float3x3 &A,
    float3 &eigenvalues,
    float3x3 &eigenvectors,
    int max_sweeps = 6
) {
    float3x3 D = A;
    eigenvectors = make_float3x3(
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    );

    #pragma unroll
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        float off_diag_sum = fabsf(D.m[0][1]) + fabsf(D.m[0][2]) + fabsf(D.m[1][2]);
        if (off_diag_sum < 1e-7f) break;

        // Pair (0, 1)
        if (fabsf(D.m[0][1]) > 1e-7f) {
            float theta = (D.m[1][1] - D.m[0][0]) / (2.0f * D.m[0][1]);
            float t = (theta >= 0.0f) ? (1.0f / (theta + sqrtf(1.0f + theta * theta)))
                                      : (-1.0f / (-theta + sqrtf(1.0f + theta * theta)));
            float c = 1.0f / sqrtf(1.0f + t * t);
            float s = t * c;
            float tau = s / (1.0f + c);

            float h = t * D.m[0][1];
            D.m[0][0] -= h;
            D.m[1][1] += h;
            D.m[0][1] = 0.0f;
            D.m[1][0] = 0.0f;

            float g0 = D.m[0][2];
            float g1 = D.m[1][2];
            D.m[0][2] = g0 - s * (g1 + g0 * tau);
            D.m[2][0] = D.m[0][2];
            D.m[1][2] = g1 + s * (g0 - g1 * tau);
            D.m[2][1] = D.m[1][2];

            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                float v0 = eigenvectors.m[k][0];
                float v1 = eigenvectors.m[k][1];
                eigenvectors.m[k][0] = v0 - s * (v1 + v0 * tau);
                eigenvectors.m[k][1] = v1 + s * (v0 - v1 * tau);
            }
        }

        // Pair (0, 2)
        if (fabsf(D.m[0][2]) > 1e-7f) {
            float theta = (D.m[2][2] - D.m[0][0]) / (2.0f * D.m[0][2]);
            float t = (theta >= 0.0f) ? (1.0f / (theta + sqrtf(1.0f + theta * theta)))
                                      : (-1.0f / (-theta + sqrtf(1.0f + theta * theta)));
            float c = 1.0f / sqrtf(1.0f + t * t);
            float s = t * c;
            float tau = s / (1.0f + c);

            float h = t * D.m[0][2];
            D.m[0][0] -= h;
            D.m[2][2] += h;
            D.m[0][2] = 0.0f;
            D.m[2][0] = 0.0f;

            float g0 = D.m[0][1];
            float g2 = D.m[1][2];
            D.m[0][1] = g0 - s * (g2 + g0 * tau);
            D.m[1][0] = D.m[0][1];
            D.m[1][2] = g2 + s * (g0 - g2 * tau);
            D.m[2][1] = D.m[1][2];

            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                float v0 = eigenvectors.m[k][0];
                float v2 = eigenvectors.m[k][2];
                eigenvectors.m[k][0] = v0 - s * (v2 + v0 * tau);
                eigenvectors.m[k][2] = v2 + s * (v0 - v2 * tau);
            }
        }

        // Pair (1, 2)
        if (fabsf(D.m[1][2]) > 1e-7f) {
            float theta = (D.m[2][2] - D.m[1][1]) / (2.0f * D.m[1][2]);
            float t = (theta >= 0.0f) ? (1.0f / (theta + sqrtf(1.0f + theta * theta)))
                                      : (-1.0f / (-theta + sqrtf(1.0f + theta * theta)));
            float c = 1.0f / sqrtf(1.0f + t * t);
            float s = t * c;
            float tau = s / (1.0f + c);

            float h = t * D.m[1][2];
            D.m[1][1] -= h;
            D.m[2][2] += h;
            D.m[1][2] = 0.0f;
            D.m[2][1] = 0.0f;

            float g1 = D.m[0][1];
            float g2 = D.m[0][2];
            D.m[0][1] = g1 - s * (g2 + g1 * tau);
            D.m[1][0] = D.m[0][1];
            D.m[0][2] = g2 + s * (g1 - g2 * tau);
            D.m[2][0] = D.m[0][2];

            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                float v1 = eigenvectors.m[k][1];
                float v2 = eigenvectors.m[k][2];
                eigenvectors.m[k][1] = v1 - s * (v2 + v1 * tau);
                eigenvectors.m[k][2] = v2 + s * (v1 - v2 * tau);
            }
        }
    }

    eigenvalues = make_float3(D.m[0][0], D.m[1][1], D.m[2][2]);
}

/**
 * @brief Quadratic Error Function (QEF) Solver for Dual Contouring.
 * 
 * Minimizes E(x) = sum_{i=0}^{K-1} (n_i . (x - p_i))^2 subject to x in [cell_min, cell_max].
 * Uses mass-point origin shifting and truncated SVD/pseudoinverse for numerical stability.
 * 
 * @param pts Array of K intersection points on voxel edges.
 * @param normals Array of K corresponding unit surface normal vectors.
 * @param count Number of intersection points (K).
 * @param cell_min Minimum AABB coordinate of the voxel cell.
 * @param cell_max Maximum AABB coordinate of the voxel cell.
 * @param svd_tolerance Relative eigenvalue threshold for pseudoinverse (default: 0.01).
 * @param cell_expand Scale applied to the cell half-extents before clamping (default: 2.0).
 * @return Optimal inner vertex position x*.
 */
__host__ __device__ __forceinline__ float3 solve_qef(
    const float3 *pts,
    const float3 *normals,
    int count,
    const float3 &cell_min,
    const float3 &cell_max,
    float svd_tolerance = 0.01f,
    float cell_expand = 2.0f
) {
    if (count <= 0) {
        return (cell_min + cell_max) * 0.5f;
    }

    const float3 cell_centre = (cell_min + cell_max) * 0.5f;
    const float3 cell_half = (cell_max - cell_min) * (0.5f * cell_expand);
    const float3 clamp_min = cell_centre - cell_half;
    const float3 clamp_max = cell_centre + cell_half;

    // 1. Compute Mass Point (Centroid)
    float3 mass_point = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < count; ++i) {
        mass_point = mass_point + pts[i];
    }
    mass_point = mass_point * (1.0f / (float)count);

    if (count == 1) {
        return maths::clamp(pts[0], clamp_min, clamp_max);
    }

    // 2. Build Shifted Normal Equation System: (A^T A) y = b_tilde
    // where y = x - mass_point, b_tilde = sum_i n_i (n_i . (p_i - mass_point))
    float3x3 ATA = make_float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    float3 b_tilde = make_float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < count; ++i) {
        float3 n = normals[i];
        float3 d = pts[i] - mass_point;
        float dot_val = maths::dot(n, d);

        ATA.m[0][0] += n.x * n.x;
        ATA.m[0][1] += n.x * n.y;
        ATA.m[0][2] += n.x * n.z;

        ATA.m[1][0] += n.y * n.x;
        ATA.m[1][1] += n.y * n.y;
        ATA.m[1][2] += n.y * n.z;

        ATA.m[2][0] += n.z * n.x;
        ATA.m[2][1] += n.z * n.y;
        ATA.m[2][2] += n.z * n.z;

        b_tilde = b_tilde + n * dot_val;
    }

    // 3. Compute Eigendecomposition of ATA via Jacobi rotations
    float3 eigenvalues;
    float3x3 V;
    symmetric_eigen_3x3(ATA, eigenvalues, V, 6);

    float max_eigenval = fmaxf(eigenvalues.x, fmaxf(eigenvalues.y, eigenvalues.z));
    float threshold = max_eigenval * svd_tolerance;

    // 4. Compute Moore-Penrose Pseudoinverse Solution
    float3 y = make_float3(0.0f, 0.0f, 0.0f);

    // Eigenvalue 0
    if (eigenvalues.x > threshold && eigenvalues.x > 1e-6f) {
        float3 v0 = make_float3(V.m[0][0], V.m[1][0], V.m[2][0]);
        float proj = maths::dot(v0, b_tilde) / eigenvalues.x;
        y = y + v0 * proj;
    }
    // Eigenvalue 1
    if (eigenvalues.y > threshold && eigenvalues.y > 1e-6f) {
        float3 v1 = make_float3(V.m[0][1], V.m[1][1], V.m[2][1]);
        float proj = maths::dot(v1, b_tilde) / eigenvalues.y;
        y = y + v1 * proj;
    }
    // Eigenvalue 2
    if (eigenvalues.z > threshold && eigenvalues.z > 1e-6f) {
        float3 v2 = make_float3(V.m[0][2], V.m[1][2], V.m[2][2]);
        float proj = maths::dot(v2, b_tilde) / eigenvalues.z;
        y = y + v2 * proj;
    }

    // 5. Unshift and Clamp to the (optionally widened) Voxel AABB
    float3 x = mass_point + y;
    x = maths::clamp(x, clamp_min, clamp_max);

    return x;
}

} // namespace maths

#endif // QEF_H
