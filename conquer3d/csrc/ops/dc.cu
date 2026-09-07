/**
 * @file dc.cu
 * @brief CUDA kernel implementations for GPU-accelerated Dual Contouring with Jacobi QEF solvers.
 * @details Extracts sharp features and corner geometry using Quadratic Error Function (QEF)
 * minimization with 3x3 cyclic Jacobi SVD diagonalization on registers.
 */

#include "dc.h"
#include "dc_data.h"
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
 * @param[in] dx Cell spacing along $x$.
 * @param[in] dy Cell spacing along $y$.
 * @param[in] dz Cell spacing along $z$.
 * @note Supplies the QEF plane normals when the caller provides no explicit normal field.
 */
__device__ __forceinline__ float3 compute_trilinear_normal(
    float u, float v, float w,
    const float s[8],
    float dx, float dy, float dz
) {
    float du = (1.0f - w) * ((1.0f - v) * (s[1] - s[0]) + v * (s[2] - s[3])) +
                      w   * ((1.0f - v) * (s[5] - s[4]) + v * (s[6] - s[7]));

    float dv = (1.0f - w) * ((1.0f - u) * (s[3] - s[0]) + u * (s[2] - s[1])) +
                      w   * ((1.0f - u) * (s[7] - s[4]) + u * (s[6] - s[5]));

    float dw = (1.0f - v) * ((1.0f - u) * (s[4] - s[0]) + u * (s[5] - s[1])) +
                      v   * ((1.0f - u) * (s[7] - s[3]) + u * (s[6] - s[2]));

    float3 grad = make_float3(du / dx, dv / dy, dw / dz);
    float len = maths::norm(grad);
    if (len > 1e-8f) {
        return grad * (1.0f / len);
    }
    return make_float3(0.0f, 0.0f, 1.0f);
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

    return fminf(acosf(cos0), fminf(acosf(cos1), acosf(cos2)));
}

} // namespace

// -----------------------------------------------------------------------------------------
// Kernel 1: Dual Vertex Generation via QEF
// -----------------------------------------------------------------------------------------
/**
 * @brief Solves one dual vertex per active voxel by minimising a quadratic error function.
 * @details The heart of Dual Contouring. One thread per voxel. Each bipolar edge
 * contributes a plane through its crossing point with the local surface normal, and the
 * vertex is placed at the point minimising $\sum_i (\mathbf{n}_i \cdot (\mathbf{v} -
 * \mathbf{p}_i))^2$. Because the planes reconstruct the surface's own tangent structure,
 * the minimiser lands exactly on creases and corners -- which is why Dual Contouring keeps
 * mechanical CAD edges that vertex-on-edge methods round away.
 *
 * The normal matrix is diagonalised by cyclic Jacobi entirely in registers, and its
 * pseudoinverse is truncated at a relative eigenvalue tolerance so that under-constrained
 * cells fall back towards the cell centroid instead of shooting off along a null direction.
 *
 * @param[in] grid_vertices Device array of grid vertex coordinates.
 * @param[in] voxels Device array of eight corner indices per voxel.
 * @param[in] sdf Device array of scalar field values at grid vertices.
 * @param[in] grid_normals Device array of per-vertex normals, or `nullptr` to derive them
 *     from the trilinear field gradient.
 * @param[in] iso Isolevel separating inside from outside.
 * @param[in] num_voxels Number of voxels.
 * @param[out] dual_vertices Device array of one solved vertex per voxel.
 * @param[out] voxel_is_active Device array of per-voxel activity flags.
 * @param[out] bipolar_edge_counts Device array of per-voxel bipolar edge counts.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note The solved vertex is clamped to the cell's bounding box, so a badly conditioned
 * QEF can never place geometry outside the voxel that produced it.
 * @warning The Jacobi solve is register resident. Raising the sweep count increases
 * register pressure and can spill to local memory, which costs far more than the extra
 * precision is worth.
 */
__global__ void compute_dual_vertices_kernel(
    const float3 *__restrict__ grid_vertices,
    const int *__restrict__ voxels,
    const float *__restrict__ sdf,
    const float3 *__restrict__ grid_normals,
    float iso,
    int num_voxels,
    float3 *__restrict__ dual_vertices,
    int *__restrict__ voxel_is_active,
    int *__restrict__ bipolar_edge_counts
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_voxels) return;

    // Load 8 corner indices
    int c_idx[8];
    float s[8];
    float3 p[8];
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

    float dx = fmaxf(c_max.x - c_min.x, 1e-6f);
    float dy = fmaxf(c_max.y - c_min.y, 1e-6f);
    float dz = fmaxf(c_max.z - c_min.z, 1e-6f);

    float3 pts[12];
    float3 normals[12];
    int count = 0;

    // Check all 12 edges
    #pragma unroll
    for (int e = 0; e < 12; ++e) {
        int v0 = dc_edge_corners[e][0];
        int v1 = dc_edge_corners[e][1];
        float s0 = s[v0];
        float s1 = s[v1];

        // Bipolar test
        if ((s0 < iso && s1 >= iso) || (s0 >= iso && s1 < iso)) {
            float t = (iso - s0) / (s1 - s0);
            t = fmaxf(0.0f, fminf(1.0f, t));

            float3 pt = p[v0] + (p[v1] - p[v0]) * t;
            pts[count] = pt;

            float3 n;
            if (grid_normals != nullptr) {
                float3 n0 = grid_normals[c_idx[v0]];
                float3 n1 = grid_normals[c_idx[v1]];
                float3 n_interp = n0 + (n1 - n0) * t;
                float len = maths::norm(n_interp);
                n = (len > 1e-8f) ? (n_interp * (1.0f / len)) : make_float3(0, 0, 1);
            } else {
                float u = dc_corner_uvw[v0][0] + (dc_corner_uvw[v1][0] - dc_corner_uvw[v0][0]) * t;
                float v = dc_corner_uvw[v0][1] + (dc_corner_uvw[v1][1] - dc_corner_uvw[v0][1]) * t;
                float w = dc_corner_uvw[v0][2] + (dc_corner_uvw[v1][2] - dc_corner_uvw[v0][2]) * t;
                n = compute_trilinear_normal(u, v, w, s, dx, dy, dz);
            }
            normals[count] = n;
            count++;
        }
    }

    bipolar_edge_counts[m] = count;

    if (count > 0) {
        voxel_is_active[m] = 1;
        float3 dual_v = maths::solve_qef(pts, normals, count, c_min, c_max, 0.01f);
        dual_vertices[m] = dual_v;
    } else {
        voxel_is_active[m] = 0;
        dual_vertices[m] = (c_min + c_max) * 0.5f;
    }
}

// -----------------------------------------------------------------------------------------
// Kernel 2: Emit Bipolar Edge Instances
// -----------------------------------------------------------------------------------------
/**
 * @brief Emits a sortable key for every (voxel, bipolar edge) instance.
 * @details One thread per voxel, writing a 64-bit key identifying the shared grid edge
 * alongside the voxel and local edge that produced it. Sorting these keys groups the
 * (up to four) voxels around each edge together, which is what lets the next stage build a
 * dual face from an edge's incident cells without any neighbour lookup structure.
 * @param[in] voxels Device array of eight corner indices per voxel.
 * @param[in] sdf Device array of scalar field values at grid vertices.
 * @param[in] edge_offsets Device array of per-voxel write offsets from the prefix sum.
 * @param[in] iso Isolevel separating inside from outside.
 * @param[in] num_voxels Number of voxels.
 * @param[out] out_edge_keys Device array of 64-bit shared-edge keys.
 * @param[out] out_voxel_and_edge Device array packing the source voxel and local edge.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Writes go to precomputed offsets, so this kernel needs no atomics.
 */
__global__ void emit_bipolar_edges_kernel(
    const int *__restrict__ voxels,
    const float *__restrict__ sdf,
    const int *__restrict__ edge_offsets,
    float iso,
    int num_voxels,
    uint64_t *__restrict__ out_edge_keys,
    uint32_t *__restrict__ out_voxel_and_edge
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_voxels) return;

    int offset = edge_offsets[m];

    #pragma unroll
    for (int e = 0; e < 12; ++e) {
        int v0 = voxels[m * 8 + dc_edge_corners[e][0]];
        int v1 = voxels[m * 8 + dc_edge_corners[e][1]];
        float s0 = sdf[v0];
        float s1 = sdf[v1];

        if ((s0 < iso && s1 >= iso) || (s0 >= iso && s1 < iso)) {
            int g_min = v0 < v1 ? v0 : v1;
            int g_max = v0 < v1 ? v1 : v0;
            uint64_t key = ((uint64_t)g_min << 32) | (uint64_t)g_max;

            out_edge_keys[offset] = key;
            out_voxel_and_edge[offset] = ((uint32_t)m << 4) | (uint32_t)e;
            offset++;
        }
    }
}

// -----------------------------------------------------------------------------------------
// Kernel 3: Gather Dual Quads from Edge Incidences
// -----------------------------------------------------------------------------------------
/**
 * @brief Builds one dual face from the voxels surrounding each shared edge.
 * @details One thread per key instance, but only the thread at the start of each run of
 * equal keys proceeds -- a lightweight way to elect one owner per unique edge without a
 * separate compaction pass. That owner collects the dual vertices of the incident voxels
 * and emits a quad, or a triangle where the edge lies on the domain boundary and only
 * three cells exist. Winding is chosen from the sign of the edge so faces are consistently
 * oriented.
 * @param[in] sorted_edge_keys Device array of edge keys in ascending order.
 * @param[in] sorted_voxel_and_edge Device array of packed voxel and local edge identifiers,
 *     permuted alongside the keys.
 * @param[in] grid_vertices Device array of grid vertex coordinates.
 * @param[in] sdf Device array of scalar field values at grid vertices.
 * @param[in] dual_vertices Device array of per-voxel solved vertices.
 * @param[in] voxel_to_compact_idx Device array mapping voxel index to compacted vertex index.
 * @param[in] total_instances Number of (voxel, edge) instances.
 * @param[out] out_quads Device array receiving dual faces; a boundary triangle repeats its
 *     last index.
 * @param[in,out] out_quad_count Device counter, atomically incremented per face.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Requires @p sorted_edge_keys to be sorted, otherwise runs are not contiguous and
 * faces are emitted several times or not at all.
 * @warning The output slot is claimed with `atomicAdd`, so face ordering varies between
 * runs. Geometry is unaffected, but a byte-identical mesh is not guaranteed.
 */
__global__ void gather_dual_quads_kernel(
    const uint64_t *__restrict__ sorted_edge_keys,
    const uint32_t *__restrict__ sorted_voxel_and_edge,
    const float3 *__restrict__ grid_vertices,
    const float *__restrict__ sdf,
    const float3 *__restrict__ dual_vertices,
    const int *__restrict__ voxel_to_compact_idx,
    int total_instances,
    int4 *__restrict__ out_quads,
    int *__restrict__ out_quad_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_instances) return;

    // We only process the start of a multi-voxel cluster sharing the same unique edge key
    uint64_t key = sorted_edge_keys[i];
    if (i > 0 && sorted_edge_keys[i - 1] == key) {
        return; // Not the first instance of this edge
    }

    // Count how many incident voxels share this edge key
    int count = 1;
    while (i + count < total_instances && sorted_edge_keys[i + count] == key) {
        count++;
    }

    if (count < 3) {
        return; // Need at least 3 cells to form a closed surface face
    }

    // Extract edge endpoints
    int v0_id = (int)(key >> 32);
    int v1_id = (int)(key & 0xFFFFFFFFULL);
    float s0 = sdf[v0_id];
    float s1 = sdf[v1_id];

    int vox_ids[4] = {-1, -1, -1, -1};
    int present_mask = 0;
    for (int k = 0; k < count && k < 4; ++k) {
        uint32_t val = sorted_voxel_and_edge[i + k];
        int vox_m = (int)(val >> 4);
        int e = (int)(val & 0xF);
        int slot = dc_edge_quadrant[e];
        vox_ids[slot] = voxel_to_compact_idx[vox_m];
        present_mask |= (1 << slot);
    }

    if (count >= 4) {
        // Orient quad: if s0 < s1, CCW order around edge: (0, 1, 2, 3)
        // if s0 > s1, reverse order: (0, 3, 2, 1)
        int4 quad;
        if (s0 < s1) {
            quad = make_int4(vox_ids[0], vox_ids[1], vox_ids[2], vox_ids[3]);
        } else {
            quad = make_int4(vox_ids[0], vox_ids[3], vox_ids[2], vox_ids[1]);
        }

        int quad_idx = atomicAdd(out_quad_count, 1);
        out_quads[quad_idx] = quad;
    } else if (count == 3) {
        // 3 slots present, 1 missing at boundary of sparse grid
        int missing_slot = 0;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (!(present_mask & (1 << s))) {
                missing_slot = s;
                break;
            }
        }
        int va = vox_ids[(missing_slot + 1) & 3];
        int vb = vox_ids[(missing_slot + 2) & 3];
        int vc = vox_ids[(missing_slot + 3) & 3];

        int4 quad;
        if (s0 < s1) {
            quad = make_int4(va, vb, vc, va);
        } else {
            quad = make_int4(va, vc, vb, va);
        }

        int quad_idx = atomicAdd(out_quad_count, 1);
        out_quads[quad_idx] = quad;
    }
}

// -----------------------------------------------------------------------------------------
// Kernel 4: Optimal Quad-to-Triangle Splitting (Max-Min Angle Criterion)
// -----------------------------------------------------------------------------------------
/**
 * @brief Splits dual quads into triangles along the shorter diagonal.
 * @details One thread per quad. The diagonal is chosen by comparing the two candidate
 * splits, which keeps the resulting triangles better conditioned than always cutting the
 * same way. Faces already emitted as boundary triangles -- marked by a repeated final
 * index -- are passed through unchanged.
 * @param[in] quads Device array of dual faces.
 * @param[in] compact_vertices Device array of compacted dual vertices.
 * @param[in] num_quads Number of faces.
 * @param[out] out_triangles Device array receiving triangle vertex index triples.
 * @param[in,out] out_tri_count Device counter, atomically incremented per triangle.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning The output slot is claimed with `atomicAdd`, so face ordering varies between
 * runs. Geometry is unaffected, but a byte-identical mesh is not guaranteed.
 */
__global__ void quad_to_triangle_kernel(
    const int4 *__restrict__ quads,
    const float3 *__restrict__ compact_vertices,
    int num_quads,
    int3 *__restrict__ out_triangles,
    int *__restrict__ out_tri_count
) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_quads) return;

    int4 quad = quads[q];
    if (quad.x == quad.w) {
        // Single boundary triangle (quad.x, quad.y, quad.z)
        int idx = atomicAdd(out_tri_count, 1);
        out_triangles[idx] = make_int3(quad.x, quad.y, quad.z);
        return;
    }

    float3 p0 = compact_vertices[quad.x];
    float3 p1 = compact_vertices[quad.y];
    float3 p2 = compact_vertices[quad.z];
    float3 p3 = compact_vertices[quad.w];

    // Split A: (0, 1, 2) & (0, 2, 3)
    float min_angle_A = fminf(triangle_min_angle(p0, p1, p2), triangle_min_angle(p0, p2, p3));

    // Split B: (1, 2, 3) & (1, 3, 0)
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
// Kernel 5: Compact Dual Vertices & Feature Colors
// -----------------------------------------------------------------------------------------
/**
 * @brief Compacts per-voxel dual vertices and colours into dense output arrays.
 * @details One thread per voxel; inactive voxels exit immediately. Active ones copy their
 * vertex to the slot given by the prefix-summed index map and, when colours are present,
 * trilinearly interpolate the field colour at the dual vertex position. Writes target
 * disjoint slots, so no atomics are required.
 * @param[in] dual_vertices Device array of one solved vertex per voxel.
 * @param[in] voxel_is_active Device array of per-voxel activity flags.
 * @param[in] voxel_to_compact_idx Device array mapping voxel index to output slot.
 * @param[in] voxels Device array of eight corner indices per voxel.
 * @param[in] colors Device array of per-grid-vertex colours, or `nullptr`.
 * @param[in] grid_vertices Device array of grid vertex coordinates.
 * @param[in] num_voxels Number of voxels.
 * @param[in] num_channels Colour channels per vertex.
 * @param[out] compact_vertices Device array of dense output vertices.
 * @param[out] compact_colors Device array of dense output colours.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void compact_dual_vertices_and_colors_kernel(
    const float3 *__restrict__ dual_vertices,
    const int *__restrict__ voxel_is_active,
    const int *__restrict__ voxel_to_compact_idx,
    const int *__restrict__ voxels,
    const float *__restrict__ colors,
    const float3 *__restrict__ grid_vertices,
    int num_voxels,
    int num_channels,
    float3 *__restrict__ compact_vertices,
    float *__restrict__ compact_colors
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_voxels) return;

    if (!voxel_is_active[m]) return;

    int dst_idx = voxel_to_compact_idx[m];
    float3 v = dual_vertices[m];
    compact_vertices[dst_idx] = v;

    if (colors != nullptr && compact_colors != nullptr) {
        // Trilinearly interpolate corner colors to dual vertex
        float3 c_min = grid_vertices[voxels[m * 8 + 0]];
        float3 c_max = grid_vertices[voxels[m * 8 + 6]];
        float dx = fmaxf(c_max.x - c_min.x, 1e-6f);
        float dy = fmaxf(c_max.y - c_min.y, 1e-6f);
        float dz = fmaxf(c_max.z - c_min.z, 1e-6f);

        float u = fmaxf(0.0f, fminf(1.0f, (v.x - c_min.x) / dx));
        float val_v = fmaxf(0.0f, fminf(1.0f, (v.y - c_min.y) / dy));
        float w = fmaxf(0.0f, fminf(1.0f, (v.z - c_min.z) / dz));

        #pragma unroll
        for (int ch = 0; ch < num_channels; ++ch) {
            float c0 = colors[voxels[m * 8 + 0] * num_channels + ch];
            float c1 = colors[voxels[m * 8 + 1] * num_channels + ch];
            float c2 = colors[voxels[m * 8 + 2] * num_channels + ch];
            float c3 = colors[voxels[m * 8 + 3] * num_channels + ch];
            float c4 = colors[voxels[m * 8 + 4] * num_channels + ch];
            float c5 = colors[voxels[m * 8 + 5] * num_channels + ch];
            float c6 = colors[voxels[m * 8 + 6] * num_channels + ch];
            float c7 = colors[voxels[m * 8 + 7] * num_channels + ch];

            float c_interp = 
                (1 - u) * (1 - val_v) * (1 - w) * c0 +
                u       * (1 - val_v) * (1 - w) * c1 +
                u       * val_v       * (1 - w) * c2 +
                (1 - u) * val_v       * (1 - w) * c3 +
                (1 - u) * (1 - val_v) * w       * c4 +
                u       * (1 - val_v) * w       * c5 +
                u       * val_v       * w       * c6 +
                (1 - u) * val_v       * w       * c7;

            compact_colors[dst_idx * num_channels + ch] = c_interp;
        }
    }
}

// -----------------------------------------------------------------------------------------
// Kernel 6: Analytical Differentiable Backward Kernel
// -----------------------------------------------------------------------------------------
/**
 * @brief Backpropagates dual vertex gradients into the scalar field and colours.
 * @details One thread per voxel. The incoming gradient on a dual vertex is redistributed
 * across the bipolar edges that constrained its QEF, each contributing through the
 * derivative of its crossing point with respect to the two corner values. Colour gradients
 * follow the trilinear interpolation weights used in the forward pass.
 * @param[in] grad_verts Device array of incoming dual vertex gradients.
 * @param[in] grad_colors Device array of incoming colour gradients, or `nullptr`.
 * @param[in] grid_vertices Device array of grid vertex coordinates.
 * @param[in] voxels Device array of eight corner indices per voxel.
 * @param[in] sdf Device array of scalar field values at grid vertices.
 * @param[in] voxel_is_active Device array of per-voxel activity flags.
 * @param[in] voxel_to_compact_idx Device array mapping voxel index to output slot.
 * @param[in] iso Isolevel used in the forward pass.
 * @param[in] num_voxels Number of voxels.
 * @param[in] num_channels Colour channels per vertex.
 * @param[out] grad_sdf Device array accumulating scalar field gradients.
 * @param[out] grad_colors_in Device array accumulating colour gradients.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Grid vertices are shared between voxels, so accumulation goes through
 * `atomicAdd` and the reduction order -- and hence the last bits of the result -- varies
 * between runs.
 * @warning This adjoint covers the QEF solution's dependence on the edge crossings, not
 * the discrete choice of which voxels are active. Gradients are therefore undefined
 * exactly at topology changes, where a voxel switches between active and inactive.
 */
__global__ void backward_dual_contouring_kernel(
    const float3 *__restrict__ grad_verts,
    const float *__restrict__ grad_colors,
    const float3 *__restrict__ grid_vertices,
    const int *__restrict__ voxels,
    const float *__restrict__ sdf,
    const int *__restrict__ voxel_is_active,
    const int *__restrict__ voxel_to_compact_idx,
    float iso,
    int num_voxels,
    int num_channels,
    float *__restrict__ grad_sdf,
    float *__restrict__ grad_colors_in
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_voxels) return;

    if (!voxel_is_active[m]) return;

    int compact_idx = voxel_to_compact_idx[m];
    float3 gv = grad_verts[compact_idx];

    // Distribute gradient to corner SDF values
    #pragma unroll
    for (int e = 0; e < 12; ++e) {
        int v0 = voxels[m * 8 + dc_edge_corners[e][0]];
        int v1 = voxels[m * 8 + dc_edge_corners[e][1]];
        float s0 = sdf[v0];
        float s1 = sdf[v1];

        if ((s0 < iso && s1 >= iso) || (s0 >= iso && s1 < iso)) {
            float denom = s1 - s0;
            if (fabsf(denom) > 1e-7f) {
                float3 p0 = grid_vertices[v0];
                float3 p1 = grid_vertices[v1];
                float3 dp = p1 - p0;

                // dt / ds0 = (iso - s1) / (s1 - s0)^2
                // dt / ds1 = -(iso - s0) / (s1 - s0)^2
                float dt_ds0 = (iso - s1) / (denom * denom);
                float dt_ds1 = -(iso - s0) / (denom * denom);

                float g_dot = maths::dot(gv, dp);
                atomicAdd(&grad_sdf[v0], g_dot * dt_ds0 * 0.25f);
                atomicAdd(&grad_sdf[v1], g_dot * dt_ds1 * 0.25f);
            }
        }
    }

    if (grad_colors != nullptr && grad_colors_in != nullptr) {
        #pragma unroll
        for (int ch = 0; ch < num_channels; ++ch) {
            float gc = grad_colors[compact_idx * num_channels + ch] * 0.125f;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                atomicAdd(&grad_colors_in[voxels[m * 8 + k] * num_channels + ch], gc);
            }
        }
    }
}

/**
 * @brief Flags voxels the surface crosses and counts their bipolar edges.
 * @details Cheap first pass. One thread per voxel: a voxel is active when its eight corner
 * signs are not all equal, and the number of bipolar edges is recorded so the host can
 * prefix-sum exact output offsets before any heavy work runs. Separating this from the QEF
 * solve means the expensive stage only ever launches over active voxels.
 * @param[in] voxels Device array of eight corner indices per voxel.
 * @param[in] sdf Device array of scalar field values at grid vertices.
 * @param[in] iso Isolevel separating inside from outside.
 * @param[in] num_voxels Number of voxels.
 * @param[out] voxel_is_active Device array of per-voxel activity flags.
 * @param[out] bipolar_edge_counts Device array of per-voxel bipolar edge counts.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void detect_active_voxels_kernel(
    const int *__restrict__ voxels,
    const float *__restrict__ sdf,
    float iso,
    int num_voxels,
    int *__restrict__ voxel_is_active,
    int *__restrict__ bipolar_edge_counts
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_voxels) return;

    float s[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        s[i] = sdf[voxels[m * 8 + i]];
    }

    int count = 0;
    #pragma unroll
    for (int e = 0; e < 12; ++e) {
        float s0 = s[dc_edge_corners[e][0]];
        float s1 = s[dc_edge_corners[e][1]];
        if ((s0 < iso && s1 >= iso) || (s0 >= iso && s1 < iso)) {
            count++;
        }
    }

    bipolar_edge_counts[m] = count;
    voxel_is_active[m] = (count > 0) ? 1 : 0;
}

// -----------------------------------------------------------------------------------------
// Host Implementation
// -----------------------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> dual_contouring(
    const at::Tensor &grid_vertices,
    const at::Tensor &voxels,
    const at::Tensor &sdf,
    const c10::optional<at::Tensor> &grid_normals,
    const c10::optional<at::Tensor> &colors,
    const c10::optional<at::Tensor> &voxel_vertices,
    float iso,
    bool quad_split
) {
    at::cuda::CUDAGuard device_guard(grid_vertices.device());
    auto allocator = at::cuda::ThrustAllocator();
    auto policy = thrust::cuda::par(allocator).on(at::cuda::getCurrentCUDAStream());

    int num_voxels = voxels.size(0);
    if (num_voxels == 0) {
        int face_dim = quad_split ? 3 : 4;
        return {
            at::zeros({0, 3}, grid_vertices.options()),
            at::zeros({0, face_dim}, voxels.options()),
            colors.has_value() ? c10::optional<at::Tensor>(at::zeros({0, colors.value().size(1)}, colors.value().options())) : c10::nullopt
        };
    }

    int threads = 256;
    int blocks = (num_voxels + threads - 1) / threads;

    at::Tensor voxel_is_active = at::zeros({num_voxels}, voxels.options());
    at::Tensor bipolar_edge_counts = at::empty({num_voxels}, voxels.options());
    const float3 *source_dual_vertices_ptr = nullptr;
    at::Tensor dual_vertices;

    if (voxel_vertices.has_value() && voxel_vertices.value().defined() && voxel_vertices.value().numel() > 0) {
        // Fast path: use precomputed inside-voxel vertices directly (zero QEF overhead)
        source_dual_vertices_ptr = reinterpret_cast<const float3*>(voxel_vertices.value().data_ptr<float>());
        detect_active_voxels_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            voxels.data_ptr<int>(),
            sdf.data_ptr<float>(),
            iso,
            num_voxels,
            voxel_is_active.data_ptr<int>(),
            bipolar_edge_counts.data_ptr<int>()
        );
    } else {
        // Standard path: compute dual vertices via GPU QEF minimization
        dual_vertices = at::empty({num_voxels, 3}, grid_vertices.options());
        source_dual_vertices_ptr = reinterpret_cast<const float3*>(dual_vertices.data_ptr<float>());
        const float3 *normals_ptr = grid_normals.has_value() ? reinterpret_cast<const float3*>(grid_normals.value().data_ptr<float>()) : nullptr;

        compute_dual_vertices_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>()),
            voxels.data_ptr<int>(),
            sdf.data_ptr<float>(),
            normals_ptr,
            iso,
            num_voxels,
            reinterpret_cast<float3*>(dual_vertices.data_ptr<float>()),
            voxel_is_active.data_ptr<int>(),
            bipolar_edge_counts.data_ptr<int>()
        );
    }

    // 2. Prefix sum for active voxels
    at::Tensor voxel_to_compact_idx = at::empty({num_voxels}, voxels.options());
    thrust::exclusive_scan(
        policy,
        voxel_is_active.data_ptr<int>(),
        voxel_is_active.data_ptr<int>() + num_voxels,
        voxel_to_compact_idx.data_ptr<int>(),
        0
    );

    int total_active_voxels = 0;
    int last_active = 0;
    int last_compact = 0;
    cudaMemcpyAsync(&last_active, voxel_is_active.data_ptr<int>() + num_voxels - 1, sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
    cudaMemcpyAsync(&last_compact, voxel_to_compact_idx.data_ptr<int>() + num_voxels - 1, sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
    total_active_voxels = last_compact + last_active;

    if (total_active_voxels == 0) {
        int face_dim = quad_split ? 3 : 4;
        return {
            at::zeros({0, 3}, grid_vertices.options()),
            at::zeros({0, face_dim}, voxels.options()),
            colors.has_value() ? c10::optional<at::Tensor>(at::zeros({0, colors.value().size(1)}, colors.value().options())) : c10::nullopt
        };
    }

    // 3. Compact Dual Vertices & Feature Colors
    at::Tensor compact_vertices = at::empty({total_active_voxels, 3}, grid_vertices.options());
    c10::optional<at::Tensor> compact_colors = c10::nullopt;
    int num_channels = 0;
    float *compact_colors_ptr = nullptr;
    const float *colors_ptr = nullptr;

    if (colors.has_value()) {
        num_channels = colors.value().size(1);
        compact_colors = at::empty({total_active_voxels, num_channels}, colors.value().options());
        compact_colors_ptr = compact_colors.value().data_ptr<float>();
        colors_ptr = colors.value().data_ptr<float>();
    }

    compact_dual_vertices_and_colors_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        source_dual_vertices_ptr,
        voxel_is_active.data_ptr<int>(),
        voxel_to_compact_idx.data_ptr<int>(),
        voxels.data_ptr<int>(),
        colors_ptr,
        reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>()),
        num_voxels,
        num_channels,
        reinterpret_cast<float3*>(compact_vertices.data_ptr<float>()),
        compact_colors_ptr
    );

    // 4. Pass 2: Prefix sum for bipolar edge instances
    at::Tensor edge_offsets = at::empty({num_voxels}, voxels.options());
    thrust::exclusive_scan(
        policy,
        bipolar_edge_counts.data_ptr<int>(),
        bipolar_edge_counts.data_ptr<int>() + num_voxels,
        edge_offsets.data_ptr<int>(),
        0
    );

    int total_edge_instances = 0;
    int last_edge_count = 0;
    int last_edge_offset = 0;
    cudaMemcpyAsync(&last_edge_count, bipolar_edge_counts.data_ptr<int>() + num_voxels - 1, sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
    cudaMemcpyAsync(&last_edge_offset, edge_offsets.data_ptr<int>() + num_voxels - 1, sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
    total_edge_instances = last_edge_offset + last_edge_count;

    if (total_edge_instances == 0) {
        int face_dim = quad_split ? 3 : 4;
        return {
            compact_vertices,
            at::zeros({0, face_dim}, voxels.options()),
            compact_colors
        };
    }

    // 5. Emit bipolar edges
    at::Tensor edge_keys = at::empty({total_edge_instances}, voxels.options().dtype(at::kLong));
    at::Tensor voxel_and_edge = at::empty({total_edge_instances}, voxels.options());

    emit_bipolar_edges_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        voxels.data_ptr<int>(),
        sdf.data_ptr<float>(),
        edge_offsets.data_ptr<int>(),
        iso,
        num_voxels,
        reinterpret_cast<uint64_t*>(edge_keys.data_ptr<int64_t>()),
        reinterpret_cast<uint32_t*>(voxel_and_edge.data_ptr<int>())
    );

    // 6. Thrust Radix Sort by Edge Key
    thrust::sort_by_key(
        policy,
        reinterpret_cast<uint64_t*>(edge_keys.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(edge_keys.data_ptr<int64_t>()) + total_edge_instances,
        reinterpret_cast<uint32_t*>(voxel_and_edge.data_ptr<int>())
    );

    // 7. Pass 3: Gather Dual Quads
    int max_possible_quads = total_edge_instances / 3 + 1;
    at::Tensor raw_quads = at::empty({max_possible_quads, 4}, voxels.options());
    at::Tensor quad_count_tensor = at::zeros({1}, voxels.options());

    int edge_blocks = (total_edge_instances + threads - 1) / threads;
    gather_dual_quads_kernel<<<edge_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint64_t*>(edge_keys.data_ptr<int64_t>()),
        reinterpret_cast<const uint32_t*>(voxel_and_edge.data_ptr<int>()),
        reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>()),
        sdf.data_ptr<float>(),
        source_dual_vertices_ptr,
        voxel_to_compact_idx.data_ptr<int>(),
        total_edge_instances,
        reinterpret_cast<int4*>(raw_quads.data_ptr<int>()),
        quad_count_tensor.data_ptr<int>()
    );

    int num_quads = 0;
    cudaMemcpyAsync(&num_quads, quad_count_tensor.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());

    at::Tensor out_quads = raw_quads.slice(0, 0, num_quads);

    // 8. Optional Quad-to-Triangle Splitting
    if (quad_split) {
        at::Tensor raw_triangles = at::empty({num_quads * 2, 3}, voxels.options());
        at::Tensor tri_count_tensor = at::zeros({1}, voxels.options());
        if (num_quads > 0) {
            int quad_blocks = (num_quads + threads - 1) / threads;
            quad_to_triangle_kernel<<<quad_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const int4*>(out_quads.data_ptr<int>()),
                reinterpret_cast<const float3*>(compact_vertices.data_ptr<float>()),
                num_quads,
                reinterpret_cast<int3*>(raw_triangles.data_ptr<int>()),
                tri_count_tensor.data_ptr<int>()
            );
        }
        int num_triangles = 0;
        cudaMemcpyAsync(&num_triangles, tri_count_tensor.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
        cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
        at::Tensor triangles = raw_triangles.slice(0, 0, num_triangles);
        return {compact_vertices, triangles, compact_colors};
    }

    return {compact_vertices, out_quads, compact_colors};
}

std::tuple<at::Tensor, c10::optional<at::Tensor>> dual_contouring_backward(
    const at::Tensor &grad_verts,
    const c10::optional<at::Tensor> &grad_colors,
    const at::Tensor &grid_vertices,
    const at::Tensor &voxels,
    const at::Tensor &sdf,
    const c10::optional<at::Tensor> &grid_normals,
    const c10::optional<at::Tensor> &colors,
    float iso
) {
    at::cuda::CUDAGuard device_guard(grid_vertices.device());
    int num_voxels = voxels.size(0);

    at::Tensor grad_sdf = at::zeros_like(sdf);
    c10::optional<at::Tensor> grad_colors_in = c10::nullopt;
    float *grad_colors_in_ptr = nullptr;
    const float *grad_colors_ptr = nullptr;
    int num_channels = 0;

    if (colors.has_value() && grad_colors.has_value()) {
        grad_colors_in = at::zeros_like(colors.value());
        grad_colors_in_ptr = grad_colors_in.value().data_ptr<float>();
        grad_colors_ptr = grad_colors.value().data_ptr<float>();
        num_channels = colors.value().size(1);
    }

    // Reconstruct active status and compact index
    at::Tensor voxel_is_active = at::zeros({num_voxels}, voxels.options());
    at::Tensor bipolar_edge_counts = at::empty({num_voxels}, voxels.options());
    at::Tensor dual_vertices = at::empty({num_voxels, 3}, grid_vertices.options());

    int threads = 256;
    int blocks = (num_voxels + threads - 1) / threads;
    const float3 *normals_ptr = grid_normals.has_value() ? reinterpret_cast<const float3*>(grid_normals.value().data_ptr<float>()) : nullptr;

    compute_dual_vertices_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>()),
        voxels.data_ptr<int>(),
        sdf.data_ptr<float>(),
        normals_ptr,
        iso,
        num_voxels,
        reinterpret_cast<float3*>(dual_vertices.data_ptr<float>()),
        voxel_is_active.data_ptr<int>(),
        bipolar_edge_counts.data_ptr<int>()
    );

    auto allocator = at::cuda::ThrustAllocator();
    auto policy = thrust::cuda::par(allocator).on(at::cuda::getCurrentCUDAStream());
    at::Tensor voxel_to_compact_idx = at::empty({num_voxels}, voxels.options());
    thrust::exclusive_scan(
        policy,
        voxel_is_active.data_ptr<int>(),
        voxel_is_active.data_ptr<int>() + num_voxels,
        voxel_to_compact_idx.data_ptr<int>(),
        0
    );

    backward_dual_contouring_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float3*>(grad_verts.data_ptr<float>()),
        grad_colors_ptr,
        reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>()),
        voxels.data_ptr<int>(),
        sdf.data_ptr<float>(),
        voxel_is_active.data_ptr<int>(),
        voxel_to_compact_idx.data_ptr<int>(),
        iso,
        num_voxels,
        num_channels,
        grad_sdf.data_ptr<float>(),
        grad_colors_in_ptr
    );

    return {grad_sdf, grad_colors_in};
}

} // namespace ops
} // namespace conquer3d
