/**
 * @file dmc.cu
 * @brief CUDA kernel implementations for Differentiable Dual Marching Cubes (DMC).
 * @details Extracts watertight 2-manifold surfaces with independent cell contour components
 * and Newton-Raphson level-set projections.
 */

#include "dmc.h"
#include "dmc_data.h"
#include "../maths/maths.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/unique.h>

#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDAGuard.h>

namespace conquer3d {
namespace ops {

namespace {

// Evaluate trilinear scalar value inside cell at (u, v, w) in [0, 1]^3
/**
 * @brief Trilinear interpolation of the eight corner values of a cell.
 * @details Evaluates the standard trilinear form. This is the exact field Dual Marching
 * Cubes projects its vertices onto, so extracted geometry agrees with the interpolant the
 * topology tables assume.
 * @param[in] u Local coordinate along $x$, in $[0, 1]$.
 * @param[in] v Local coordinate along $y$, in $[0, 1]$.
 * @param[in] w Local coordinate along $z$, in $[0, 1]$.
 * @param[in] s The cell's eight corner values.
 * @return The interpolated field value.
 */
__device__ __forceinline__ float trilinear_val(
    float u, float v, float w,
    const float s[8]
) {
    float u0 = (1.0f - u) * s[0] + u * s[1];
    float u1 = (1.0f - u) * s[3] + u * s[2];
    float u2 = (1.0f - u) * s[4] + u * s[5];
    float u3 = (1.0f - u) * s[7] + u * s[6];

    float v0 = (1.0f - v) * u0 + v * u1;
    float v1 = (1.0f - v) * u2 + v * u3;

    return (1.0f - w) * v0 + w * v1;
}

// Evaluate trilinear gradient (du, dv, dw)
/**
 * @brief Gradient of the trilinear field inside a unit cell.
 * @details Differentiates the trilinear interpolant analytically and divides by the cell
 * spacing to return a world-space gradient. The direction is the surface normal wherever
 * the field is a signed distance.
 * @param[in] u Local coordinate along $x$, in $[0, 1]$.
 * @param[in] v Local coordinate along $y$, in $[0, 1]$.
 * @param[in] w Local coordinate along $z$, in $[0, 1]$.
 * @param[in] s The cell's eight corner values.
 * @return The field gradient.
 * @note Paired with trilinear_val() to drive the Newton-Raphson projection of dual
 * vertices onto the level set.
 */
__device__ __forceinline__ float3 trilinear_grad(
    float u, float v, float w,
    const float s[8]
) {
    float du = (1.0f - w) * ((1.0f - v) * (s[1] - s[0]) + v * (s[2] - s[3])) +
                      w   * ((1.0f - v) * (s[5] - s[4]) + v * (s[6] - s[7]));

    float dv = (1.0f - w) * ((1.0f - u) * (s[3] - s[0]) + u * (s[2] - s[1])) +
                      w   * ((1.0f - u) * (s[7] - s[4]) + u * (s[6] - s[5]));

    float dw = (1.0f - v) * ((1.0f - u) * (s[4] - s[0]) + u * (s[5] - s[1])) +
                      v   * ((1.0f - u) * (s[7] - s[3]) + u * (s[6] - s[2]));

    return make_float3(du, dv, dw);
}

// Compute minimum angle of a 3D triangle in radians
/**
 * @brief Smallest interior angle of a triangle.
 * @details Used to choose between the two possible diagonals when splitting a quad: the
 * split with the larger minimum angle produces better-conditioned triangles and avoids
 * slivers.
 * @param[in] p0 First vertex.
 * @param[in] p1 Second vertex.
 * @param[in] p2 Third vertex.
 * @return The smallest interior angle in radians.
 */
__device__ __forceinline__ float triangle_min_angle(
    const float3 &p0, const float3 &p1, const float3 &p2
) {
    float3 e0 = p1 - p0;
    float3 e1 = p2 - p1;
    float3 e2 = p0 - p2;

    float l0 = maths::norm(e0);
    float l1 = maths::norm(e1);
    float l2 = maths::norm(e2);

    if (l0 < 1e-8f || l1 < 1e-8f || l2 < 1e-8f) return 0.0f;

    float cos0 = -maths::dot(e2, e0) / (l2 * l0);
    float cos1 = -maths::dot(e0, e1) / (l0 * l1);
    float cos2 = -maths::dot(e1, e2) / (l1 * l2);

    cos0 = fmaxf(-1.0f, fminf(1.0f, cos0));
    cos1 = fmaxf(-1.0f, fminf(1.0f, cos1));
    cos2 = fmaxf(-1.0f, fminf(1.0f, cos2));

    float a0 = acosf(cos0);
    float a1 = acosf(cos1);
    float a2 = acosf(cos2);

    return fminf(a0, fminf(a1, a2));
}

// Extract independent MC polygon contours for a single voxel cell
/**
 * @brief Decomposes a cell into its independent surface contours.
 * @details Reads the packed ::dmc_r_pattern row for the case, or invokes the asymptotic
 * decider when ::dmc_ambig_table marks it ambiguous. Returning several contours where a
 * cell contains disconnected surface sheets is precisely what lets Dual Marching Cubes emit
 * multiple dual vertices and so avoid the pinch points a single-vertex-per-cell method
 * produces.
 * @param[in] iso Isolevel separating inside from outside.
 * @param[in] i_case The cell's 8-bit corner sign code.
 * @param[in] s The cell's eight corner values.
 * @param[out] contour_sizes Edge count of each contour.
 * @param[out] contour_edges Local edge indices of each contour.
 * @return Number of contours, 0 for a fully inside or outside cell.
 * @note At most four contours fit in a cell, which bounds the output arrays.
 */
__device__ inline int extract_cell_contours(
    float iso,
    int i_case,
    const float s[8],
    int contour_sizes[4],
    int contour_edges[4][12]
) {
    if (i_case == 0 || i_case == 255) return 0;

    if (dmc_ambig_table[i_case] != DMC_AMBIGUOUS) {
        const char *c_lt = &dmc_r_pattern[17 * i_case];
        int num_cnt = (int)c_lt[0];
        int pos = 1 + num_cnt;
        for (int c = 0; c < num_cnt; ++c) {
            int sz = (int)c_lt[1 + c];
            contour_sizes[c] = sz;
            for (int i = 0; i < sz; ++i) {
                contour_edges[c][i] = (int)c_lt[pos++];
            }
        }
        return num_cnt;
    }

    // Ambiguous case: solve on-the-fly using asymptotic decider
    unsigned char segm_[12] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

    auto set_segm = [](int ei, int eo, unsigned char segm[12]) {
        segm[ei] &= 0xF0;
        segm[ei] |= ((unsigned char)eo) & 0xF;
        segm[eo] &= 0x0F;
        segm[eo] |= ((unsigned char)ei) << 4;
    };

    auto get_face_e = [](int f, int e) { return (int)((dmc_e_face[f] >> (4 * e)) & 0xF); };
    auto get_face_v = [](int f, int e) { return (int)((dmc_v_face[f] >> (4 * e)) & 0xF); };

    for (int f = 0; f < 6; ++f) {
        int v0 = get_face_v(f, 0);
        int v1 = get_face_v(f, 1);
        int v2 = get_face_v(f, 2);
        int v3 = get_face_v(f, 3);
        int e0 = get_face_e(f, 0);
        int e1 = get_face_e(f, 1);
        int e2 = get_face_e(f, 2);
        int e3 = get_face_e(f, 3);

        float f0 = s[v0];
        float f1 = s[v1];
        float f2 = s[v2];
        float f3 = s[v3];

        unsigned int f_case = 0;
        if (f0 >= iso) f_case |= 1;
        if (f1 >= iso) f_case |= 2;
        if (f2 >= iso) f_case |= 4;
        if (f3 >= iso) f_case |= 8;

        switch (f_case) {
            case 1:  set_segm(e0, e3, segm_); break;
            case 2:  set_segm(e1, e0, segm_); break;
            case 3:  set_segm(e1, e3, segm_); break;
            case 4:  set_segm(e3, e2, segm_); break;
            case 5:  set_segm(e0, e2, segm_); break;
            case 6: {
                float denom = f0 + f3 - f1 - f2;
                float val = (fabsf(denom) > 1e-7f) ? ((f0 * f3 - f1 * f2) / denom) : (0.5f * (f0 + f3));
                if (val >= iso) {
                    set_segm(e3, e0, segm_);
                    set_segm(e1, e2, segm_);
                } else {
                    set_segm(e1, e0, segm_);
                    set_segm(e3, e2, segm_);
                }
                break;
            }
            case 7:  set_segm(e1, e2, segm_); break;
            case 8:  set_segm(e2, e1, segm_); break;
            case 9: {
                float denom = f0 + f3 - f1 - f2;
                float val = (fabsf(denom) > 1e-7f) ? ((f0 * f3 - f1 * f2) / denom) : (0.5f * (f0 + f3));
                if (val >= iso) {
                    set_segm(e0, e1, segm_);
                    set_segm(e2, e3, segm_);
                } else {
                    set_segm(e0, e3, segm_);
                    set_segm(e2, e1, segm_);
                }
                break;
            }
            case 10: set_segm(e2, e0, segm_); break;
            case 11: set_segm(e2, e3, segm_); break;
            case 12: set_segm(e3, e1, segm_); break;
            case 13: set_segm(e0, e1, segm_); break;
            case 14: set_segm(e3, e0, segm_); break;
            default: break;
        }
    }

    // Connect segments into closed contours
    int cnt = 0;
    for (int e = 0; e < 12; ++e) {
        if (segm_[e] != 0xFF) {
            int eTo = (int)(segm_[e] & 0xF);
            int eStart = e;
            int pos = 0;
            contour_edges[cnt][pos++] = eStart;
            while (eTo != eStart && pos < 12) {
                contour_edges[cnt][pos++] = eTo;
                int eIn = eTo;
                eTo = (int)(segm_[eIn] & 0xF);
                segm_[eIn] = 0xFF; // unset
            }
            segm_[eStart] = 0xFF;
            contour_sizes[cnt] = pos;
            cnt++;
            if (cnt >= 4) break;
        }
    }
    return cnt;
}

} // anonymous namespace

// -----------------------------------------------------------------------------------------
// Kernel 1: Count Contours and Edge Crossings per Voxel
// -----------------------------------------------------------------------------------------
/**
 * @brief Counts the independent contours and edge instances of every voxel.
 * @details Sizing pass for Dual Marching Cubes. One thread per voxel: the corner sign code
 * selects a row of ::dmc_r_pattern giving how many separate contours the cell contains,
 * and ambiguous cases are resolved on the fly by the asymptotic decider. Cells that Dual
 * Contouring would collapse to one vertex may hold several contours here -- that
 * multiplicity is exactly what prevents pinch points and guarantees manifold output.
 * Counting first lets the host prefix-sum exact offsets so the extraction pass writes
 * without atomics.
 * @param[in] voxels Device array of eight corner indices per voxel.
 * @param[in] sdf Device array of scalar field values at grid vertices.
 * @param[in] iso Isolevel separating inside from outside.
 * @param[in] num_voxels Number of voxels.
 * @param[out] contour_counts Device array of per-voxel contour counts, one dual vertex each.
 * @param[out] edge_instance_counts Device array of per-voxel (contour, edge) instance counts.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void dmc_count_contours_kernel(
    const int *__restrict__ voxels,
    const float *__restrict__ sdf,
    float iso,
    int num_voxels,
    int *__restrict__ contour_counts,
    int *__restrict__ edge_instance_counts
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_voxels) return;

    float s[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        s[i] = sdf[voxels[m * 8 + i]];
    }

    float u[8];
    u[0] = s[0];
    u[1] = s[1];
    u[2] = s[3];
    u[3] = s[2];
    u[4] = s[4];
    u[5] = s[5];
    u[6] = s[7];
    u[7] = s[6];

    int i_case = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (u[i] >= iso) {
            i_case |= (1 << i);
        }
    }

    int c_sizes[4];
    int c_edges[4][12];
    int num_cnt = extract_cell_contours(iso, i_case, u, c_sizes, c_edges);

    contour_counts[m] = num_cnt;

    int total_edges = 0;
    for (int c = 0; c < num_cnt; ++c) {
        total_edges += c_sizes[c];
    }
    edge_instance_counts[m] = total_edges;
}

// -----------------------------------------------------------------------------------------
// Kernel 2: Extract Dual Vertices and Emit Edge Incidences
// -----------------------------------------------------------------------------------------
/**
 * @brief Emits every contour's dual vertex and its bipolar edge keys.
 * @details One thread per voxel, writing at the offsets computed by the counting pass. For
 * each contour the vertex starts at the centroid of its edge crossings and is then pushed
 * onto the exact trilinear level set by @p project_iters Newton-Raphson steps, using the
 * analytic trilinear value and gradient. Projection is what removes the faceting a plain
 * centroid leaves on curved surfaces. Supplying @p precomputed_vertices skips both the
 * centroid and the projection.
 * @param[in] grid_vertices Device array of grid vertex coordinates.
 * @param[in] voxels Device array of eight corner indices per voxel.
 * @param[in] sdf Device array of scalar field values at grid vertices.
 * @param[in] colors Device array of per-grid-vertex colours, or `nullptr`.
 * @param[in] precomputed_vertices Device array of externally supplied dual vertices, or
 *     `nullptr` to solve them here.
 * @param[in] num_channels Colour channels per vertex.
 * @param[in] vert_offsets Device array of per-voxel vertex write offsets.
 * @param[in] edge_offsets Device array of per-voxel edge instance write offsets.
 * @param[in] iso Isolevel separating inside from outside.
 * @param[in] project_iters Newton-Raphson iterations; 0 leaves the centroid unprojected.
 * @param[in] num_voxels Number of voxels.
 * @param[out] out_vertices Device array receiving dual vertices.
 * @param[out] out_colors Device array receiving interpolated colours.
 * @param[out] out_edge_keys Device array of 64-bit shared-edge keys.
 * @param[out] out_dual_vert_and_edge Device array packing the source vertex and local edge.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Vertices are clamped to their cell, so a diverging Newton step cannot place
 * geometry outside the voxel that produced it.
 * @warning Each iteration costs a trilinear value and gradient evaluation. Beyond a handful
 * of steps the surface has converged and further iterations only cost time.
 */
__global__ void dmc_extract_dual_vertices_and_edges_kernel(
    const float3 *__restrict__ grid_vertices,
    const int *__restrict__ voxels,
    const float *__restrict__ sdf,
    const float *__restrict__ colors,
    const float3 *__restrict__ precomputed_vertices,
    int num_channels,
    const int *__restrict__ vert_offsets,
    const int *__restrict__ edge_offsets,
    float iso,
    int project_iters,
    int num_voxels,
    float3 *__restrict__ out_vertices,
    float *__restrict__ out_colors,
    uint64_t *__restrict__ out_edge_keys,
    uint32_t *__restrict__ out_dual_vert_and_edge
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_voxels) return;

    int c_idx[8];
    float s[8];
    float3 p[8];
    int i_case = 0;

    float3 c_min = make_float3(1e30f, 1e30f, 1e30f);
    float3 c_max = make_float3(-1e30f, -1e30f, -1e30f);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        c_idx[i] = voxels[m * 8 + i];
        s[i] = sdf[c_idx[i]];
        p[i] = grid_vertices[c_idx[i]];
        c_min = maths::min(c_min, p[i]);
        c_max = maths::max(c_max, p[i]);
    }

    float u_arr[8];
    u_arr[0] = s[0];
    u_arr[1] = s[1];
    u_arr[2] = s[3];
    u_arr[3] = s[2];
    u_arr[4] = s[4];
    u_arr[5] = s[5];
    u_arr[6] = s[7];
    u_arr[7] = s[6];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (u_arr[i] >= iso) {
            i_case |= (1 << i);
        }
    }

    int c_sizes[4];
    int c_edges[4][12];
    int num_cnt = extract_cell_contours(iso, i_case, u_arr, c_sizes, c_edges);
    if (num_cnt == 0) return;

    float dx = fmaxf(c_max.x - c_min.x, 1e-6f);
    float dy = fmaxf(c_max.y - c_min.y, 1e-6f);
    float dz = fmaxf(c_max.z - c_min.z, 1e-6f);

    int v_base = vert_offsets[m];
    int e_base = edge_offsets[m];
    int curr_edge_offset = 0;

    for (int c = 0; c < num_cnt; ++c) {
        int sz = c_sizes[c];
        if (sz < 3) continue;

        int dual_vert_idx = v_base + c;
        float u_sum = 0.0f, v_sum = 0.0f, w_sum = 0.0f;

        for (int i = 0; i < sz; ++i) {
            int e = c_edges[c][i];
            int v0 = dmc_edge_corners[e][0];
            int v1 = dmc_edge_corners[e][1];
            float s0 = s[v0];
            float s1 = s[v1];
            float t = (iso - s0) / (s1 - s0);
            t = fmaxf(0.0f, fminf(1.0f, t));

            float u_e = dmc_corner_uvw[v0][0] + (dmc_corner_uvw[v1][0] - dmc_corner_uvw[v0][0]) * t;
            float v_e = dmc_corner_uvw[v0][1] + (dmc_corner_uvw[v1][1] - dmc_corner_uvw[v0][1]) * t;
            float w_e = dmc_corner_uvw[v0][2] + (dmc_corner_uvw[v1][2] - dmc_corner_uvw[v0][2]) * t;

            u_sum += u_e;
            v_sum += v_e;
            w_sum += w_e;

            // Emit 64-bit edge key
            int g0 = c_idx[v0];
            int g1 = c_idx[v1];
            int g_min = g0 < g1 ? g0 : g1;
            int g_max = g0 < g1 ? g1 : g0;
            uint64_t key = ((uint64_t)g_min << 32) | (uint64_t)g_max;

            out_edge_keys[e_base + curr_edge_offset] = key;
            out_dual_vert_and_edge[e_base + curr_edge_offset] = ((uint32_t)dual_vert_idx << 4) | (uint32_t)e;
            curr_edge_offset++;
        }

        float u = u_sum / sz;
        float v = v_sum / sz;
        float w = w_sum / sz;

        if (precomputed_vertices != nullptr) {
            // Fast path: use precomputed inside-voxel vertex directly
            out_vertices[dual_vert_idx] = precomputed_vertices[m];
            float3 pt = precomputed_vertices[m];
            u = fmaxf(0.0f, fminf(1.0f, (pt.x - c_min.x) / dx));
            v = fmaxf(0.0f, fminf(1.0f, (pt.y - c_min.y) / dy));
            w = fmaxf(0.0f, fminf(1.0f, (pt.z - c_min.z) / dz));
        } else {
            // Newton-Raphson Level-Set Projection
            for (int iter = 0; iter < project_iters; ++iter) {
                float f_val = trilinear_val(u, v, w, s);
                float3 grad = trilinear_grad(u, v, w, s);
                float len_sq = grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
                if (len_sq > 1e-8f) {
                    float step = 0.5f * (iso - f_val) / len_sq;
                    u = fmaxf(0.0f, fminf(1.0f, u + step * grad.x));
                    v = fmaxf(0.0f, fminf(1.0f, v + step * grad.y));
                    w = fmaxf(0.0f, fminf(1.0f, w + step * grad.z));
                }
            }

            float3 pt = make_float3(
                c_min.x + u * dx,
                c_min.y + v * dy,
                c_min.z + w * dz
            );
            out_vertices[dual_vert_idx] = pt;
        }

        // Interpolate colors if provided
        if (colors != nullptr && out_colors != nullptr) {
            float w_corners[8];
            w_corners[0] = (1.0f - u) * (1.0f - v) * (1.0f - w);
            w_corners[1] = u * (1.0f - v) * (1.0f - w);
            w_corners[2] = u * v * (1.0f - w);
            w_corners[3] = (1.0f - u) * v * (1.0f - w);
            w_corners[4] = (1.0f - u) * (1.0f - v) * w;
            w_corners[5] = u * (1.0f - v) * w;
            w_corners[6] = u * v * w;
            w_corners[7] = (1.0f - u) * v * w;

            for (int ch = 0; ch < num_channels; ++ch) {
                float c_interp = 0.0f;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    c_interp += w_corners[k] * colors[c_idx[k] * num_channels + ch];
                }
                out_colors[dual_vert_idx * num_channels + ch] = c_interp;
            }
        }
    }
}

// -----------------------------------------------------------------------------------------
// Kernel 3: Gather Dual Quads from Sorted Edge Incidences
// -----------------------------------------------------------------------------------------
/**
 * @brief Builds one dual face per unique shared edge.
 * @details One thread per edge instance; only the thread beginning a run of equal keys
 * proceeds, electing a single owner per unique edge. That owner gathers the dual vertices
 * of the contours meeting at the edge and emits a quad, or a triangle at the domain
 * boundary. Because Dual Marching Cubes may contribute several contours from one voxel,
 * the vertices gathered here are per-contour rather than per-cell, which is what keeps the
 * surface manifold where cells are ambiguous.
 * @param[in] sorted_edge_keys Device array of edge keys in ascending order.
 * @param[in] sorted_dual_vert_and_edge Device array of packed vertex and local edge
 *     identifiers, permuted alongside the keys.
 * @param[in] sdf Device array of scalar field values at grid vertices.
 * @param[in] total_instances Number of (contour, edge) instances.
 * @param[out] out_quads Device array receiving dual faces.
 * @param[in,out] out_quad_count Device counter, atomically incremented per face.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Requires @p sorted_edge_keys to be sorted so equal keys form contiguous runs.
 * @warning The output slot is claimed with `atomicAdd`, so face ordering varies between
 * runs. Geometry is unaffected, but a byte-identical mesh is not guaranteed.
 */
__global__ void dmc_gather_quads_kernel(
    const uint64_t *__restrict__ sorted_edge_keys,
    const uint32_t *__restrict__ sorted_dual_vert_and_edge,
    const float *__restrict__ sdf,
    int total_instances,
    int4 *__restrict__ out_quads,
    int *__restrict__ out_quad_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_instances) return;

    uint64_t key = sorted_edge_keys[i];
    if (i > 0 && sorted_edge_keys[i - 1] == key) {
        return; // Process unique edge clusters only once
    }

    int count = 1;
    while (i + count < total_instances && sorted_edge_keys[i + count] == key) {
        count++;
    }

    if (count < 3) return;

    int v0_id = (int)(key >> 32);
    int v1_id = (int)(key & 0xFFFFFFFFULL);
    float s0 = sdf[v0_id];
    float s1 = sdf[v1_id];

    int quad_v[4] = {-1, -1, -1, -1};
    int present_mask = 0;

    for (int k = 0; k < count && k < 4; ++k) {
        uint32_t val = sorted_dual_vert_and_edge[i + k];
        int d_idx = (int)(val >> 4);
        int e = (int)(val & 0xF);
        int slot = dmc_edge_quadrant[e];
        quad_v[slot] = d_idx;
        present_mask |= (1 << slot);
    }

    if (count >= 4) {
        int4 q;
        if (s0 < s1) {
            q = make_int4(quad_v[0], quad_v[1], quad_v[2], quad_v[3]);
        } else {
            q = make_int4(quad_v[0], quad_v[3], quad_v[2], quad_v[1]);
        }
        int q_idx = atomicAdd(out_quad_count, 1);
        out_quads[q_idx] = q;
    } else if (count == 3) {
        int missing_slot = 0;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (!(present_mask & (1 << s))) {
                missing_slot = s;
                break;
            }
        }
        int va = quad_v[(missing_slot + 1) & 3];
        int vb = quad_v[(missing_slot + 2) & 3];
        int vc = quad_v[(missing_slot + 3) & 3];

        int4 q;
        if (s0 < s1) {
            q = make_int4(va, vb, vc, va);
        } else {
            q = make_int4(va, vc, vb, va);
        }
        int q_idx = atomicAdd(out_quad_count, 1);
        out_quads[q_idx] = q;
    }
}

// -----------------------------------------------------------------------------------------
// Kernel 4: Optimal Quad-to-Triangle Splitting
// -----------------------------------------------------------------------------------------
/**
 * @brief Splits dual quads into triangles along the better-conditioned diagonal.
 * @details One thread per quad, choosing the split that maximises the minimum triangle
 * angle so the output avoids slivers. Boundary faces already emitted as triangles pass
 * through unchanged. Skipping this kernel entirely leaves a pure quad mesh.
 * @param[in] quads Device array of dual faces.
 * @param[in] vertices Device array of dual vertices.
 * @param[in] num_quads Number of faces.
 * @param[out] out_triangles Device array receiving triangle vertex index triples.
 * @param[in,out] out_tri_count Device counter, atomically incremented per triangle.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning The output slot is claimed with `atomicAdd`, so face ordering varies between
 * runs. Geometry is unaffected, but a byte-identical mesh is not guaranteed.
 */
__global__ void dmc_quad_to_triangle_kernel(
    const int4 *__restrict__ quads,
    const float3 *__restrict__ vertices,
    int num_quads,
    int3 *__restrict__ out_triangles,
    int *__restrict__ out_tri_count
) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_quads) return;

    int4 quad = quads[q];
    if (quad.x == quad.w) {
        int idx = atomicAdd(out_tri_count, 1);
        out_triangles[idx] = make_int3(quad.x, quad.y, quad.z);
        return;
    }

    float3 p0 = vertices[quad.x];
    float3 p1 = vertices[quad.y];
    float3 p2 = vertices[quad.z];
    float3 p3 = vertices[quad.w];

    float min_angle_A = fminf(triangle_min_angle(p0, p1, p2), triangle_min_angle(p0, p2, p3));
    float min_angle_B = fminf(triangle_min_angle(p1, p2, p3), triangle_min_angle(p1, p3, p0));

    int idx = atomicAdd(out_tri_count, 2);
    if (min_angle_A >= min_angle_B) {
        out_triangles[idx]     = make_int3(quad.x, quad.y, quad.z);
        out_triangles[idx + 1] = make_int3(quad.x, quad.z, quad.w);
    } else {
        out_triangles[idx]     = make_int3(quad.y, quad.z, quad.w);
        out_triangles[idx + 1] = make_int3(quad.y, quad.w, quad.x);
    }
}

// -----------------------------------------------------------------------------------------
// Kernel 5: Differentiable Backward Kernel
// -----------------------------------------------------------------------------------------
/**
 * @brief Backpropagates dual vertex gradients into the scalar field and colours.
 * @details One thread per voxel, recomputing the forward pass's contour decomposition so
 * the adjoint follows the same path. Each dual vertex's gradient is distributed to the
 * corner values through the derivatives of its edge crossings; colour gradients follow the
 * interpolation weights.
 * @param[in] grad_vertices Device array of incoming dual vertex gradients.
 * @param[in] grad_colors Device array of incoming colour gradients, or `nullptr`.
 * @param[in] grid_vertices Device array of grid vertex coordinates.
 * @param[in] voxels Device array of eight corner indices per voxel.
 * @param[in] sdf Device array of scalar field values at grid vertices.
 * @param[in] colors Device array of per-grid-vertex colours, or `nullptr`.
 * @param[in] num_channels Colour channels per vertex.
 * @param[in] vert_offsets Device array of per-voxel vertex offsets from the forward pass.
 * @param[in] iso Isolevel used in the forward pass.
 * @param[in] num_voxels Number of voxels.
 * @param[out] grad_sdf Device array accumulating scalar field gradients.
 * @param[out] grad_colors_in Device array accumulating colour gradients.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Accumulation into shared grid vertices uses `atomicAdd`, so results are not
 * bitwise reproducible between runs.
 * @warning The Newton projection applied in the forward pass is not differentiated; the
 * adjoint treats the projected vertex as if it came from the unprojected centroid, so
 * gradients are approximate when @p project_iters was large.
 */
__global__ void dmc_backward_kernel(
    const float3 *__restrict__ grad_vertices,
    const float *__restrict__ grad_colors,
    const float3 *__restrict__ grid_vertices,
    const int *__restrict__ voxels,
    const float *__restrict__ sdf,
    const float *__restrict__ colors,
    int num_channels,
    const int *__restrict__ vert_offsets,
    float iso,
    int num_voxels,
    float *__restrict__ grad_sdf,
    float *__restrict__ grad_colors_in
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_voxels) return;

    int c_idx[8];
    float s[8];
    float u_arr[8];
    int i_case = 0;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        c_idx[i] = voxels[m * 8 + i];
        s[i] = sdf[c_idx[i]];
    }

    u_arr[0] = s[0];
    u_arr[1] = s[1];
    u_arr[2] = s[3];
    u_arr[3] = s[2];
    u_arr[4] = s[4];
    u_arr[5] = s[5];
    u_arr[6] = s[7];
    u_arr[7] = s[6];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (u_arr[i] >= iso) {
            i_case |= (1 << i);
        }
    }

    int c_sizes[4];
    int c_edges[4][12];
    int num_cnt = extract_cell_contours(iso, i_case, u_arr, c_sizes, c_edges);
    if (num_cnt == 0) return;

    int v_base = vert_offsets[m];

    for (int c = 0; c < num_cnt; ++c) {
        int sz = c_sizes[c];
        if (sz < 3) continue;

        int dual_vert_idx = v_base + c;
        float3 gv = grad_vertices[dual_vert_idx];
        float inv_sz = 1.0f / (float)sz;

        for (int i = 0; i < sz; ++i) {
            int e = c_edges[c][i];
            int v0 = c_idx[dmc_edge_corners[e][0]];
            int v1 = c_idx[dmc_edge_corners[e][1]];
            float s0 = sdf[v0];
            float s1 = sdf[v1];
            float denom = s1 - s0;

            if (fabsf(denom) > 1e-7f) {
                float3 p0 = grid_vertices[v0];
                float3 p1 = grid_vertices[v1];
                float3 dp = p1 - p0;

                float dt_ds0 = (iso - s1) / (denom * denom);
                float dt_ds1 = -(iso - s0) / (denom * denom);

                float g_dot = maths::dot(gv, dp) * inv_sz;
                atomicAdd(&grad_sdf[v0], g_dot * dt_ds0);
                atomicAdd(&grad_sdf[v1], g_dot * dt_ds1);
            }
        }

        if (grad_colors != nullptr && grad_colors_in != nullptr) {
            for (int ch = 0; ch < num_channels; ++ch) {
                float gc = grad_colors[dual_vert_idx * num_channels + ch] * 0.125f;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    atomicAdd(&grad_colors_in[c_idx[k] * num_channels + ch], gc);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------------------
// Host Implementation
// -----------------------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> dual_marching_cubes(
    const at::Tensor &grid_vertices,
    const at::Tensor &voxels,
    const at::Tensor &sdf,
    const c10::optional<at::Tensor> &colors,
    const c10::optional<at::Tensor> &voxel_vertices,
    float iso,
    bool quad_split,
    int project_iters
) {
    TORCH_CHECK(grid_vertices.is_cuda(), "grid_vertices must be a CUDA tensor");
    TORCH_CHECK(voxels.is_cuda(), "voxels must be a CUDA tensor");
    TORCH_CHECK(sdf.is_cuda(), "sdf must be a CUDA tensor");

    at::cuda::CUDAGuard device_guard(grid_vertices.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto policy = thrust::cuda::par(at::cuda::ThrustAllocator()).on(stream);

    int num_voxels = voxels.size(0);
    if (num_voxels == 0) {
        int face_dim = quad_split ? 3 : 4;
        return std::make_tuple(
            at::empty({0, 3}, grid_vertices.options()),
            at::empty({0, face_dim}, voxels.options()),
            colors.has_value() ? c10::make_optional(at::empty({0, colors->size(1)}, colors->options())) : c10::nullopt
        );
    }

    // Step 1: Count contours and edge crossings per voxel
    auto contour_counts = at::empty({num_voxels}, voxels.options());
    auto edge_counts = at::empty({num_voxels}, voxels.options());

    int block = 256;
    int grid = (num_voxels + block - 1) / block;

    dmc_count_contours_kernel<<<grid, block, 0, stream>>>(
        voxels.data_ptr<int>(),
        sdf.data_ptr<float>(),
        iso,
        num_voxels,
        contour_counts.data_ptr<int>(),
        edge_counts.data_ptr<int>()
    );

    // Exclusive scan for vertex and edge offsets
    auto vert_offsets = at::empty({num_voxels}, voxels.options());
    auto edge_offsets = at::empty({num_voxels}, voxels.options());

    thrust::exclusive_scan(policy, contour_counts.data_ptr<int>(), contour_counts.data_ptr<int>() + num_voxels, vert_offsets.data_ptr<int>());
    thrust::exclusive_scan(policy, edge_counts.data_ptr<int>(), edge_counts.data_ptr<int>() + num_voxels, edge_offsets.data_ptr<int>());

    int total_vertices = 0;
    int total_edge_instances = 0;
    if (num_voxels > 0) {
        int last_v_count = 0, last_v_off = 0;
        int last_e_count = 0, last_e_off = 0;
        cudaMemcpyAsync(&last_v_count, contour_counts.data_ptr<int>() + (num_voxels - 1), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&last_v_off, vert_offsets.data_ptr<int>() + (num_voxels - 1), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&last_e_count, edge_counts.data_ptr<int>() + (num_voxels - 1), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&last_e_off, edge_offsets.data_ptr<int>() + (num_voxels - 1), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        total_vertices = last_v_off + last_v_count;
        total_edge_instances = last_e_off + last_e_count;
    }

    if (total_vertices == 0 || total_edge_instances == 0) {
        int face_dim = quad_split ? 3 : 4;
        return std::make_tuple(
            at::empty({0, 3}, grid_vertices.options()),
            at::empty({0, face_dim}, voxels.options()),
            colors.has_value() ? c10::make_optional(at::empty({0, colors->size(1)}, colors->options())) : c10::nullopt
        );
    }

    // Step 2: Extract dual vertices and emit edge instances
    auto out_vertices = at::empty({total_vertices, 3}, grid_vertices.options());
    c10::optional<at::Tensor> out_colors = c10::nullopt;
    float *colors_ptr = nullptr;
    float *out_colors_ptr = nullptr;
    int num_channels = 0;

    if (colors.has_value() && colors->defined()) {
        num_channels = colors->size(1);
        out_colors = at::empty({total_vertices, num_channels}, colors->options());
        colors_ptr = colors->data_ptr<float>();
        out_colors_ptr = out_colors->data_ptr<float>();
    }

    auto edge_keys = at::empty({total_edge_instances}, grid_vertices.options().dtype(at::kLong));
    auto dual_vert_and_edge = at::empty({total_edge_instances}, voxels.options());

    const float3 *precomputed_ptr = (voxel_vertices.has_value() && voxel_vertices->defined() && voxel_vertices->numel() > 0)
        ? reinterpret_cast<const float3*>(voxel_vertices->data_ptr<float>())
        : nullptr;

    dmc_extract_dual_vertices_and_edges_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>()),
        voxels.data_ptr<int>(),
        sdf.data_ptr<float>(),
        colors_ptr,
        precomputed_ptr,
        num_channels,
        vert_offsets.data_ptr<int>(),
        edge_offsets.data_ptr<int>(),
        iso,
        project_iters,
        num_voxels,
        reinterpret_cast<float3*>(out_vertices.data_ptr<float>()),
        out_colors_ptr,
        reinterpret_cast<uint64_t*>(edge_keys.data_ptr<int64_t>()),
        reinterpret_cast<uint32_t*>(dual_vert_and_edge.data_ptr<int>())
    );

    // Step 3: Thrust Radix Sort by 64-bit Edge Key
    thrust::sort_by_key(
        policy,
        reinterpret_cast<uint64_t*>(edge_keys.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(edge_keys.data_ptr<int64_t>()) + total_edge_instances,
        reinterpret_cast<uint32_t*>(dual_vert_and_edge.data_ptr<int>())
    );

    // Step 4: Gather Dual Quads
    auto out_quads = at::empty({total_edge_instances, 4}, voxels.options());
    auto quad_count_tensor = at::zeros({1}, voxels.options());

    int edge_grid = (total_edge_instances + block - 1) / block;
    dmc_gather_quads_kernel<<<edge_grid, block, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(edge_keys.data_ptr<int64_t>()),
        reinterpret_cast<const uint32_t*>(dual_vert_and_edge.data_ptr<int>()),
        sdf.data_ptr<float>(),
        total_edge_instances,
        reinterpret_cast<int4*>(out_quads.data_ptr<int>()),
        quad_count_tensor.data_ptr<int>()
    );

    int num_quads = 0;
    cudaMemcpyAsync(&num_quads, quad_count_tensor.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    out_quads = out_quads.slice(0, 0, num_quads);

    if (!quad_split) {
        return std::make_tuple(out_vertices, out_quads, out_colors);
    }

    // Step 5: Triangulate Quads
    auto out_triangles = at::empty({num_quads * 2, 3}, voxels.options());
    auto tri_count_tensor = at::zeros({1}, voxels.options());
    int q_grid = (num_quads + block - 1) / block;

    if (num_quads > 0) {
        dmc_quad_to_triangle_kernel<<<q_grid, block, 0, stream>>>(
            reinterpret_cast<const int4*>(out_quads.data_ptr<int>()),
            reinterpret_cast<const float3*>(out_vertices.data_ptr<float>()),
            num_quads,
            reinterpret_cast<int3*>(out_triangles.data_ptr<int>()),
            tri_count_tensor.data_ptr<int>()
        );
    }

    int num_triangles = 0;
    cudaMemcpyAsync(&num_triangles, tri_count_tensor.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    out_triangles = out_triangles.slice(0, 0, num_triangles);

    return std::make_tuple(out_vertices, out_triangles, out_colors);
}

std::tuple<at::Tensor, c10::optional<at::Tensor>> dual_marching_cubes_backward(
    const at::Tensor &grad_vertices,
    const c10::optional<at::Tensor> &grad_colors,
    const at::Tensor &grid_vertices,
    const at::Tensor &voxels,
    const at::Tensor &sdf,
    const c10::optional<at::Tensor> &colors,
    float iso,
    int project_iters
) {
    at::cuda::CUDAGuard device_guard(grid_vertices.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto policy = thrust::cuda::par(at::cuda::ThrustAllocator()).on(stream);

    int num_voxels = voxels.size(0);
    auto grad_sdf = at::zeros_like(sdf);
    c10::optional<at::Tensor> grad_colors_in = c10::nullopt;

    if (num_voxels == 0 || grad_vertices.size(0) == 0) {
        return std::make_tuple(grad_sdf, grad_colors_in);
    }

    auto contour_counts = at::empty({num_voxels}, voxels.options());
    auto edge_counts = at::empty({num_voxels}, voxels.options());

    int block = 256;
    int grid = (num_voxels + block - 1) / block;

    dmc_count_contours_kernel<<<grid, block, 0, stream>>>(
        voxels.data_ptr<int>(),
        sdf.data_ptr<float>(),
        iso,
        num_voxels,
        contour_counts.data_ptr<int>(),
        edge_counts.data_ptr<int>()
    );

    auto vert_offsets = at::empty({num_voxels}, voxels.options());
    thrust::exclusive_scan(policy, contour_counts.data_ptr<int>(), contour_counts.data_ptr<int>() + num_voxels, vert_offsets.data_ptr<int>());

    const float *grad_colors_ptr = nullptr;
    float *grad_colors_in_ptr = nullptr;
    const float *colors_ptr = nullptr;
    int num_channels = 0;

    if (grad_colors.has_value() && grad_colors->defined() && colors.has_value() && colors->defined()) {
        grad_colors_in = at::zeros_like(*colors);
        grad_colors_ptr = grad_colors->data_ptr<float>();
        grad_colors_in_ptr = grad_colors_in->data_ptr<float>();
        colors_ptr = colors->data_ptr<float>();
        num_channels = colors->size(1);
    }

    dmc_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float3*>(grad_vertices.data_ptr<float>()),
        grad_colors_ptr,
        reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>()),
        voxels.data_ptr<int>(),
        sdf.data_ptr<float>(),
        colors_ptr,
        num_channels,
        vert_offsets.data_ptr<int>(),
        iso,
        num_voxels,
        grad_sdf.data_ptr<float>(),
        grad_colors_in_ptr
    );

    return std::make_tuple(grad_sdf, grad_colors_in);
}

} // namespace ops
} // namespace conquer3d
