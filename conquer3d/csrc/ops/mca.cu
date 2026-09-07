/**
 * @file mca.cu
 * @brief CUDA kernel implementations for Marching Cubes with Asymptotic Deciders (Nielson & Hamann 1991).
 * @details Resolves face and internal ambiguities on voxel cells to extract topologically consistent,
 * watertight 2-manifold surfaces.
 */

#include "mca.h"
#include "mca_data.h"
#include "../maths/maths.h"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <thrust/execution_policy.h>

namespace mca {

// Helper to evaluate oriented segments on a cube face with Asymptotic Decider
/**
 * @brief Resolves one cube face's connectivity with the asymptotic decider.
 * @details Evaluates the bilinear saddle value on the face and links its bipolar edges
 * accordingly, recording the result as a successor map. Both cells sharing the face compute
 * the same saddle from the same corner values, so they always agree -- which is exactly why
 * the decider eliminates the cracks plain Marching Cubes leaves on ambiguous faces.
 * @param[in] f Face index, in the ::v_face and ::e_face ordering.
 * @param[in] F The cell's eight corner values.
 * @param[in] iso Isolevel separating inside from outside.
 * @param[in,out] next_e Successor map over the 12 local edges, filled for this face.
 */
__device__ __forceinline__ void evaluate_face_segments(
    int f,
    const float F[8],
    float iso,
    uint8_t next_e[12]
) {
    int v0 = v_face[f][0];
    int v1 = v_face[f][1];
    int v2 = v_face[f][2];
    int v3 = v_face[f][3];

    int e0 = e_face[f][0];
    int e1 = e_face[f][1];
    int e2 = e_face[f][2];
    int e3 = e_face[f][3];

    float f0 = F[v0];
    float f1 = F[v1];
    float f2 = F[v2];
    float f3 = F[v3];

    uint8_t b0 = (f0 < iso) ? 1 : 0;
    uint8_t b1 = (f1 < iso) ? 1 : 0;
    uint8_t b2 = (f2 < iso) ? 1 : 0;
    uint8_t b3 = (f3 < iso) ? 1 : 0;

    uint8_t f_case = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3);

    switch (f_case) {
        case 1:
            next_e[e3] = e0;
            break;
        case 2:
            next_e[e0] = e1;
            break;
        case 3:
            next_e[e3] = e1;
            break;
        case 4:
            next_e[e1] = e2;
            break;
        case 5: {
            // Ambiguous diagonal (c0, c2 vs c1, c3)
            float denom = (f0 + f2) - (f1 + f3);
            float saddle = (fabsf(denom) > 1e-8f) ? (f0 * f2 - f1 * f3) / denom : 0.5f * (f0 + f2);
            if (saddle < iso) {
                // Negative corners connect across hyperbolic neck
                next_e[e3] = e2;
                next_e[e1] = e0;
            } else {
                // Positive corners connect, separating negative components
                next_e[e3] = e0;
                next_e[e1] = e2;
            }
            break;
        }
        case 6:
            next_e[e0] = e2;
            break;
        case 7:
            next_e[e3] = e2;
            break;
        case 8:
            next_e[e2] = e3;
            break;
        case 9:
            next_e[e2] = e0;
            break;
        case 10: {
            // Ambiguous diagonal (c1, c3 vs c0, c2)
            float denom = (f0 + f2) - (f1 + f3);
            float saddle = (fabsf(denom) > 1e-8f) ? (f0 * f2 - f1 * f3) / denom : 0.5f * (f1 + f3);
            if (saddle < iso) {
                // Negative corners connect across hyperbolic neck
                next_e[e0] = e3;
                next_e[e2] = e1;
            } else {
                // Positive corners connect
                next_e[e0] = e1;
                next_e[e2] = e3;
            }
            break;
        }
        case 11:
            next_e[e2] = e1;
            break;
        case 12:
            next_e[e1] = e3;
            break;
        case 13:
            next_e[e1] = e0;
            break;
        case 14:
            next_e[e0] = e3;
            break;
        default:
            break;
    }
}

// Compute closed polygon loops and count total triangles
/**
 * @brief Traces a cell's surface contours by following the resolved edge successors.
 * @details Runs the decider over all six faces to build a successor map, then walks it to
 * recover closed loops of edges. Each loop is one connected surface component within the
 * cell, and triangulating the loops rather than reading a fixed table is what lets the
 * method emit topologically correct geometry for ambiguous cases.
 * @param[in] F The cell's eight corner values.
 * @param[in] iso Isolevel separating inside from outside.
 * @param[out] out_loops Local edge indices of each traced loop.
 * @param[out] out_loop_lens Length of each loop.
 * @return Number of loops found.
 * @note At most four loops fit in a cell, which bounds the output arrays.
 */
__device__ __forceinline__ int trace_mca_loops(
    const float F[8],
    float iso,
    uint8_t out_loops[4][12],
    uint8_t out_loop_lens[4]
) {
    uint8_t next_e[12];
    #pragma unroll
    for (int i = 0; i < 12; ++i) {
        next_e[i] = 0xFF;
    }

    #pragma unroll
    for (int f = 0; f < 6; ++f) {
        evaluate_face_segments(f, F, iso, next_e);
    }

    uint16_t visited = 0;
    int num_loops = 0;
    int total_triangles = 0;

    for (int e = 0; e < 12; ++e) {
        if (next_e[e] != 0xFF && !(visited & (1 << e))) {
            uint8_t loop_len = 0;
            uint8_t curr = e;
            while (curr != 0xFF && !(visited & (1 << curr)) && loop_len < 12) {
                visited |= (1 << curr);
                out_loops[num_loops][loop_len++] = curr;
                curr = next_e[curr];
                if (curr == e) break;
            }
            if (loop_len >= 3) {
                out_loop_lens[num_loops] = loop_len;
                total_triangles += (loop_len - 2);
                num_loops++;
                if (num_loops >= 4) break;
            }
        }
    }
    return total_triangles;
}

/**
 * @brief Counts the triangles each voxel emits, resolving ambiguous faces on the fly.
 * @details Sizing pass. One thread per voxel. Unambiguous cases read their triangle count
 * straight from the Marching Cubes tables; cases flagged by ::cubeFaceAmbigMask evaluate
 * the bilinear saddle on each ambiguous face and pick the connection its sign implies.
 * Because both cells sharing a face compute the same saddle value, neighbours always agree
 * and the surface cannot crack -- the defect plain Marching Cubes exhibits on these cases.
 * @param[in] num_voxels Number of voxels.
 * @param[in] voxels Device array of eight corner indices per voxel.
 * @param[in] sdf Device array of scalar field values at grid vertices.
 * @param[in] iso Isolevel separating inside from outside.
 * @param[out] triangle_counts Device array of per-voxel triangle counts.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void compute_active_voxels_mca_kernel(
    const uint32_t num_voxels,
    const uint32_t *__restrict__ voxels,
    const float *__restrict__ sdf,
    const float iso,
    uint32_t *__restrict__ triangle_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_voxels) return;

    uint32_t v0 = voxels[idx * 8 + 0];
    uint32_t v1 = voxels[idx * 8 + 1];
    uint32_t v2 = voxels[idx * 8 + 2];
    uint32_t v3 = voxels[idx * 8 + 3];
    uint32_t v4 = voxels[idx * 8 + 4];
    uint32_t v5 = voxels[idx * 8 + 5];
    uint32_t v6 = voxels[idx * 8 + 6];
    uint32_t v7 = voxels[idx * 8 + 7];

    float F[8] = {
        sdf[v0], sdf[v1], sdf[v2], sdf[v3],
        sdf[v4], sdf[v5], sdf[v6], sdf[v7]
    };

    uint8_t voxel_code = 0;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (F[k] < iso) voxel_code |= (1 << k);
    }

    if (voxel_code == 0 || voxel_code == 255) {
        triangle_counts[idx] = 0;
        return;
    }

    // Approach B: Check cubeFaceAmbigMask for fast path
    uint8_t ambig_mask = cubeFaceAmbigMask[voxel_code];
    if (ambig_mask == 0) {
        triangle_counts[idx] = (uint32_t)((trinumTable[voxel_code + 1] - trinumTable[voxel_code]) / 3);
    } else {
        uint8_t loops[4][12];
        uint8_t loop_lens[4] = {0, 0, 0, 0};
        int num_tris = trace_mca_loops(F, iso, loops, loop_lens);
        triangle_counts[idx] = (uint32_t)num_tris;
    }
}

/**
 * @brief Emits triangles and their bipolar edge keys at precomputed offsets.
 * @details One thread per voxel, repeating the decider decisions from the counting pass so
 * both agree exactly. Triangles are written as edge keys rather than vertex indices, to be
 * remapped once the keys have been deduplicated; writes go to prefix-summed offsets and so
 * need no atomics.
 * @param[in] num_voxels Number of voxels.
 * @param[in] voxels Device array of eight corner indices per voxel.
 * @param[in] sdf Device array of scalar field values at grid vertices.
 * @param[in] iso Isolevel separating inside from outside.
 * @param[in] tri_offsets Device array of per-voxel triangle write offsets.
 * @param[out] out_edges Device array of 64-bit edge keys, three per triangle.
 * @param[out] out_triangles Device array of triangles, initially indexed by edge instance.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning The decider must reach the same conclusion as the counting pass. Any divergence
 * between the two -- for instance from a differently rounded saddle evaluation -- would
 * overrun the allotted output range.
 */
__global__ void generate_edges_and_triangles_mca_kernel(
    const uint32_t num_voxels,
    const uint32_t *__restrict__ voxels,
    const float *__restrict__ sdf,
    const float iso,
    const uint32_t *__restrict__ tri_offsets,
    uint64_t *__restrict__ out_edges,
    int3 *__restrict__ out_triangles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_voxels) return;

    uint32_t tri_start = tri_offsets[idx];
    uint32_t tri_end = tri_offsets[idx + 1];
    if (tri_start == tri_end) return;

    uint32_t v[8] = {
        voxels[idx * 8 + 0], voxels[idx * 8 + 1], voxels[idx * 8 + 2], voxels[idx * 8 + 3],
        voxels[idx * 8 + 4], voxels[idx * 8 + 5], voxels[idx * 8 + 6], voxels[idx * 8 + 7]
    };

    float F[8] = {
        sdf[v[0]], sdf[v[1]], sdf[v[2]], sdf[v[3]],
        sdf[v[4]], sdf[v[5]], sdf[v[6]], sdf[v[7]]
    };

    uint8_t voxel_code = 0;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (F[k] < iso) voxel_code |= (1 << k);
    }

    uint32_t current_tri = tri_start;
    uint8_t ambig_mask = cubeFaceAmbigMask[voxel_code];

    if (ambig_mask == 0) {
        // Fast-path: read directly from triTable
        const int *tri_edges = triTable[voxel_code];
        for (int t = 0; t < 5 && tri_edges[t * 3] != -1; ++t) {
            int local_edges[3] = {
                tri_edges[t * 3 + 0],
                tri_edges[t * 3 + 1],
                tri_edges[t * 3 + 2]
            };
            #pragma unroll
            for (int e_idx = 0; e_idx < 3; ++e_idx) {
                int le = local_edges[e_idx];
                uint32_t v_a = v[edgeConnection[le][0]];
                uint32_t v_b = v[edgeConnection[le][1]];
                uint32_t v_min = (v_a < v_b) ? v_a : v_b;
                uint32_t v_max = (v_a > v_b) ? v_a : v_b;

                uint64_t edge_key = (((uint64_t)v_min) << 32) | ((uint64_t)v_max);
                out_edges[current_tri * 3 + e_idx] = edge_key;
            }
            current_tri++;
        }
    } else {
        // Ambiguous path: trace loops with dynamic asymptotic decider
        uint8_t loops[4][12];
        uint8_t loop_lens[4] = {0, 0, 0, 0};
        trace_mca_loops(F, iso, loops, loop_lens);

        for (int l = 0; l < 4; ++l) {
            uint8_t len = loop_lens[l];
            if (len < 3) continue;

            // Fan triangulation of polygon loop
            for (int k = 0; k < len - 2; ++k) {
                int local_e0 = loops[l][0];
                int local_e1 = loops[l][k + 1];
                int local_e2 = loops[l][k + 2];

                int local_edges[3] = {local_e0, local_e1, local_e2};
                #pragma unroll
                for (int e_idx = 0; e_idx < 3; ++e_idx) {
                    int le = local_edges[e_idx];
                    uint32_t v_a = v[edgeConnection[le][0]];
                    uint32_t v_b = v[edgeConnection[le][1]];
                    uint32_t v_min = (v_a < v_b) ? v_a : v_b;
                    uint32_t v_max = (v_a > v_b) ? v_a : v_b;

                    uint64_t edge_key = (((uint64_t)v_min) << 32) | ((uint64_t)v_max);
                    out_edges[current_tri * 3 + e_idx] = edge_key;
                }
                current_tri++;
            }
        }
    }
}

/**
 * @brief Rewrites triangle indices from edge instances to deduplicated vertices.
 * @details One thread per triangle, substituting each of its three edge-instance slots for
 * the unique vertex index found by the host's sort-and-unique pass. This is the step that
 * welds the mesh: triangles from neighbouring voxels come to share vertices instead of
 * duplicating them.
 * @param[in] num_triangles Number of triangles.
 * @param[in] edge_indices Device array mapping edge instances to unique vertex indices.
 * @param[in,out] triangles Device array of triangles, remapped in place.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void remap_triangles_kernel(
    const uint32_t num_triangles,
    const uint32_t *__restrict__ edge_indices,
    int3 *__restrict__ triangles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_triangles) return;

    triangles[idx] = make_int3(
        edge_indices[idx * 3 + 0],
        edge_indices[idx * 3 + 1],
        edge_indices[idx * 3 + 2]
    );
}

/**
 * @brief Places one output vertex on each unique bipolar edge.
 * @details One thread per unique edge. The 64-bit key is unpacked into its two grid vertex
 * indices and the crossing is interpolated linearly, with colours following the same
 * parameter. Each edge is visited once, so no atomics are needed.
 * @param[in] num_edges Number of unique edges.
 * @param[in] unique_edge_keys Device array of deduplicated 64-bit edge keys.
 * @param[in] grid_vertices Device array of grid vertex coordinates.
 * @param[in] sdf Device array of scalar field values at grid vertices.
 * @param[in] in_colors Device array of per-grid-vertex colours, or `nullptr`.
 * @param[in] num_color_channels Colour channels per vertex.
 * @param[in] iso Isolevel being extracted.
 * @param[out] out_vertices Device array of interpolated surface vertices.
 * @param[out] out_colors Device array of interpolated colours.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void interpolate_vertices_and_features_kernel(
    const uint32_t num_edges,
    const uint64_t *__restrict__ unique_edge_keys,
    const float3 *__restrict__ grid_vertices,
    const float *__restrict__ sdf,
    const float *__restrict__ in_colors,
    const int num_color_channels,
    const float iso,
    float3 *__restrict__ out_vertices,
    float *__restrict__ out_colors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;

    uint64_t key = unique_edge_keys[idx];
    uint32_t v0 = (uint32_t)(key >> 32);
    uint32_t v1 = (uint32_t)(key & 0xFFFFFFFFULL);

    float3 p0 = grid_vertices[v0];
    float3 p1 = grid_vertices[v1];
    float s0 = sdf[v0];
    float s1 = sdf[v1];

    float t = (iso - s0) / (s1 - s0 + 1e-8f);
    t = fminf(fmaxf(t, 0.0f), 1.0f);

    out_vertices[idx] = p0 + (p1 - p0) * t;

    if (in_colors != nullptr && out_colors != nullptr) {
        for (int c = 0; c < num_color_channels; ++c) {
            float c0 = in_colors[v0 * num_color_channels + c];
            float c1 = in_colors[v1 * num_color_channels + c];
            out_colors[idx * num_color_channels + c] = c0 + (c1 - c0) * t;
        }
    }
}

/**
 * @brief Backpropagates vertex gradients into the scalar field and colours.
 * @details The analytical adjoint of the interpolation pass. One thread per unique edge,
 * applying the closed-form derivatives of $t = (\text{iso} - f_0) / (f_1 - f_0)$ with
 * respect to both corner values.
 * @param[in] num_edges Number of unique edges.
 * @param[in] unique_edge_keys Device array of deduplicated 64-bit edge keys.
 * @param[in] grad_vertices Device array of incoming vertex position gradients.
 * @param[in] grad_colors Device array of incoming colour gradients, or `nullptr`.
 * @param[in] grid_vertices Device array of grid vertex coordinates.
 * @param[in] in_colors Device array of per-grid-vertex colours, or `nullptr`.
 * @param[in] num_color_channels Colour channels per vertex.
 * @param[in] sdf Device array of scalar field values at grid vertices.
 * @param[in] iso Isolevel used in the forward pass.
 * @param[out] grad_sdf Device array accumulating scalar field gradients.
 * @param[out] grad_in_colors Device array accumulating colour gradients.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Accumulation into shared grid vertices uses `atomicAdd`, so the reduction order
 * varies between runs.
 */
__global__ void backward_dmca_kernel(
    const uint32_t num_edges,
    const uint64_t *__restrict__ unique_edge_keys,
    const float3 *__restrict__ grad_vertices,
    const float *__restrict__ grad_colors,
    const float3 *__restrict__ grid_vertices,
    const float *__restrict__ in_colors,
    const int num_color_channels,
    const float *__restrict__ sdf,
    const float iso,
    float *__restrict__ grad_sdf,
    float *__restrict__ grad_in_colors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;

    uint64_t key = unique_edge_keys[idx];
    uint32_t v0 = (uint32_t)(key >> 32);
    uint32_t v1 = (uint32_t)(key & 0xFFFFFFFFULL);

    float3 p0 = grid_vertices[v0];
    float3 p1 = grid_vertices[v1];
    float s0 = sdf[v0];
    float s1 = sdf[v1];

    float3 gv = grad_vertices[idx];
    float denom = s1 - s0;
    float denom2 = denom * denom + 1e-8f;

    float dt_ds0 = (s1 - iso) / denom2;
    float dt_ds1 = (iso - s0) / denom2;

    float3 dp = p1 - p0;
    float g_pos_dot = maths::dot(gv, dp);

    float g_feat_dot = 0.0f;
    float t = (iso - s0) / (denom + 1e-8f);
    t = fminf(fmaxf(t, 0.0f), 1.0f);

    if (grad_colors != nullptr && in_colors != nullptr && grad_in_colors != nullptr) {
        for (int c = 0; c < num_color_channels; ++c) {
            float gc = grad_colors[idx * num_color_channels + c];
            float c0 = in_colors[v0 * num_color_channels + c];
            float c1 = in_colors[v1 * num_color_channels + c];
            g_feat_dot += gc * (c1 - c0);

            atomicAdd(&grad_in_colors[v0 * num_color_channels + c], gc * (1.0f - t));
            atomicAdd(&grad_in_colors[v1 * num_color_channels + c], gc * t);
        }
    }

    float total_dot = g_pos_dot + g_feat_dot;
    atomicAdd(&grad_sdf[v0], total_dot * (-dt_ds0));
    atomicAdd(&grad_sdf[v1], total_dot * (-dt_ds1));
}

std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> marching_cubes_asymptotic(
    const torch::Tensor &grid_vertices,
    const torch::Tensor &voxels,
    const torch::Tensor &sdf,
    const std::optional<torch::Tensor> &colors,
    float iso
) {
    TORCH_CHECK(grid_vertices.is_cuda(), "grid_vertices must be a CUDA tensor");
    TORCH_CHECK(voxels.is_cuda(), "voxels must be a CUDA tensor");
    TORCH_CHECK(sdf.is_cuda(), "sdf must be a CUDA tensor");
    TORCH_CHECK(voxels.size(1) == 8, "voxels must have shape (N, 8)");

    const uint32_t num_voxels = voxels.size(0);
    const int block_size = 256;
    const int num_blocks = (num_voxels + block_size - 1) / block_size;

    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(voxels.device());
    auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(voxels.device());
    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(grid_vertices.device());

    torch::Tensor tri_counts = torch::empty({num_voxels}, options_i32);
    compute_active_voxels_mca_kernel<<<num_blocks, block_size>>>(
        num_voxels,
        reinterpret_cast<const uint32_t*>(voxels.data_ptr<int32_t>()),
        sdf.data_ptr<float>(),
        iso,
        reinterpret_cast<uint32_t*>(tri_counts.data_ptr<int32_t>())
    );

    torch::Tensor tri_offsets = torch::empty({num_voxels + 1}, options_i32);
    at::cuda::ThrustAllocator allocator;
    auto policy = thrust::cuda::par(allocator);

    auto d_counts = thrust::device_pointer_cast(reinterpret_cast<uint32_t*>(tri_counts.data_ptr<int32_t>()));
    auto d_offsets = thrust::device_pointer_cast(reinterpret_cast<uint32_t*>(tri_offsets.data_ptr<int32_t>()));

    thrust::exclusive_scan(policy, d_counts, d_counts + num_voxels, d_offsets, (uint32_t)0);
    uint32_t total_triangles = 0;
    cudaMemcpy(&total_triangles, tri_offsets.data_ptr<int32_t>() + num_voxels - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t last_count = 0;
    cudaMemcpy(&last_count, tri_counts.data_ptr<int32_t>() + num_voxels - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    total_triangles += last_count;

    if (total_triangles == 0) {
        return std::make_tuple(
            torch::empty({0, 3}, options_f32),
            torch::empty({0, 3}, options_i32),
            colors.has_value() ? std::make_optional(torch::empty({0, colors.value().size(1)}, options_f32)) : std::nullopt
        );
    }

    torch::Tensor raw_edges = torch::empty({total_triangles * 3}, options_i64);
    torch::Tensor triangles = torch::empty({total_triangles, 3}, options_i32);

    generate_edges_and_triangles_mca_kernel<<<num_blocks, block_size>>>(
        num_voxels,
        reinterpret_cast<const uint32_t*>(voxels.data_ptr<int32_t>()),
        sdf.data_ptr<float>(),
        iso,
        reinterpret_cast<const uint32_t*>(tri_offsets.data_ptr<int32_t>()),
        reinterpret_cast<uint64_t*>(raw_edges.data_ptr<int64_t>()),
        reinterpret_cast<int3*>(triangles.data_ptr<int32_t>())
    );

    torch::Tensor unique_edges = raw_edges.clone();
    auto d_edges = thrust::device_pointer_cast(reinterpret_cast<uint64_t*>(unique_edges.data_ptr<int64_t>()));
    thrust::sort(policy, d_edges, d_edges + total_triangles * 3);
    auto end_unique = thrust::unique(policy, d_edges, d_edges + total_triangles * 3);
    uint32_t num_unique_edges = end_unique - d_edges;
    unique_edges = unique_edges.slice(0, 0, num_unique_edges);

    torch::Tensor edge_indices = torch::empty({total_triangles * 3}, options_i32);
    auto d_raw = thrust::device_pointer_cast(reinterpret_cast<uint64_t*>(raw_edges.data_ptr<int64_t>()));
    auto d_out_indices = thrust::device_pointer_cast(reinterpret_cast<uint32_t*>(edge_indices.data_ptr<int32_t>()));
    auto d_uniq = thrust::device_pointer_cast(reinterpret_cast<uint64_t*>(unique_edges.data_ptr<int64_t>()));

    thrust::lower_bound(policy, d_uniq, d_uniq + num_unique_edges, d_raw, d_raw + total_triangles * 3, d_out_indices);

    int tri_blocks = (total_triangles + block_size - 1) / block_size;
    remap_triangles_kernel<<<tri_blocks, block_size>>>(
        total_triangles,
        reinterpret_cast<const uint32_t*>(edge_indices.data_ptr<int32_t>()),
        reinterpret_cast<int3*>(triangles.data_ptr<int32_t>())
    );

    torch::Tensor out_vertices = torch::empty({num_unique_edges, 3}, options_f32);
    std::optional<torch::Tensor> out_colors = std::nullopt;
    float *out_colors_ptr = nullptr;
    const float *in_colors_ptr = nullptr;
    int num_color_channels = 0;

    if (colors.has_value() && colors.value().defined()) {
        const auto &c_tensor = colors.value();
        num_color_channels = c_tensor.size(1);
        out_colors = torch::empty({num_unique_edges, num_color_channels}, options_f32);
        out_colors_ptr = out_colors.value().data_ptr<float>();
        in_colors_ptr = c_tensor.data_ptr<float>();
    }

    int edge_blocks = (num_unique_edges + block_size - 1) / block_size;
    interpolate_vertices_and_features_kernel<<<edge_blocks, block_size>>>(
        num_unique_edges,
        reinterpret_cast<const uint64_t*>(unique_edges.data_ptr<int64_t>()),
        reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>()),
        sdf.data_ptr<float>(),
        in_colors_ptr,
        num_color_channels,
        iso,
        reinterpret_cast<float3*>(out_vertices.data_ptr<float>()),
        out_colors_ptr
    );

    return std::make_tuple(out_vertices, triangles, out_colors);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> marching_cubes_asymptotic_backward(
    const torch::Tensor &grad_vertices,
    const std::optional<torch::Tensor> &grad_colors,
    const torch::Tensor &grid_vertices,
    const torch::Tensor &unique_edges,
    const torch::Tensor &sdf,
    const std::optional<torch::Tensor> &colors,
    float iso
) {
    uint32_t num_edges = unique_edges.size(0);
    auto grad_sdf = torch::zeros_like(sdf);

    std::optional<torch::Tensor> grad_in_colors = std::nullopt;
    float *grad_in_colors_ptr = nullptr;
    const float *grad_colors_ptr = nullptr;
    const float *in_colors_ptr = nullptr;
    int num_color_channels = 0;

    if (colors.has_value() && colors.value().defined() && grad_colors.has_value() && grad_colors.value().defined()) {
        grad_in_colors = torch::zeros_like(colors.value());
        grad_in_colors_ptr = grad_in_colors.value().data_ptr<float>();
        grad_colors_ptr = grad_colors.value().data_ptr<float>();
        in_colors_ptr = colors.value().data_ptr<float>();
        num_color_channels = colors.value().size(1);
    }

    if (num_edges > 0) {
        int block_size = 256;
        int num_blocks = (num_edges + block_size - 1) / block_size;

        backward_dmca_kernel<<<num_blocks, block_size>>>(
            num_edges,
            reinterpret_cast<const uint64_t*>(unique_edges.data_ptr<int64_t>()),
            reinterpret_cast<const float3*>(grad_vertices.data_ptr<float>()),
            grad_colors_ptr,
            reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>()),
            in_colors_ptr,
            num_color_channels,
            sdf.data_ptr<float>(),
            iso,
            grad_sdf.data_ptr<float>(),
            grad_in_colors_ptr
        );
    }

    return std::make_tuple(grad_sdf, grad_in_colors);
}

} // namespace mca
