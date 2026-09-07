/**
 * @file mtg.cu
 * @brief CUDA kernel implementations for Marching Tetrahedra on structured cubic grids (6-tet decomposition).
 */

#include "mtg.h"
#include "../maths/maths.h"
#include <cuda_runtime.h>
#include <thrust/distance.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>

namespace mtg
{
    /**
     * @brief Thrust predicate selecting voxels the surface crosses.
     * @details A voxel is active when its sign code is neither all-inside nor
     * all-outside. Used to stream-compact the active set, so later stages launch only
     * over voxels that actually contribute geometry.
     */
    struct is_active_voxel
    {
        /**
         * @brief Tests whether a sign code marks an active voxel.
         * @param[in] code The voxel's corner sign code.
         * @return 1 when the surface crosses it, 0 otherwise.
         */
        __host__ __device__ int operator()(const uint8_t code) const
        {
            return (code > 0 && code < 255) ? 1 : 0;
        }
    };

    /**
     * @brief Thrust functor mapping a sign code to its triangle count.
     * @details Feeds the exclusive scan that assigns each active voxel a disjoint output
     * range, so triangle assembly needs no atomics.
     */
    struct num_triangles_functor
    {
        /**
         * @brief Looks up the triangle count implied by a sign code.
         * @param[in] code The voxel's corner sign code.
         * @return Number of triangles the voxel emits.
         */
        __device__ uint32_t operator()(const uint8_t code) const {
            uint32_t num = 0;
            for (int t = 0; t < 6; t++) {
                uint8_t tet_code = 0;
                for (int i = 0; i < 4; i++) {
                    if (code & (1 << mtg_tets[t][i])) tet_code |= (1 << i);
                }
                num += mtg_num_tris[tet_code];
            }
            return num;
        }
    };

    /**
     * @brief Packs eight corner signs into a Marching Cubes case index.
     * @details Sets bit $i$ when corner $i$ lies below the isolevel, producing the 8-bit code
     * that indexes every topology table. Branch-free enough to compile to predicated
     * instructions, so it costs nothing in warp divergence.
     * @param[in] sv0 Value at corner 0.
     * @param[in] sv1 Value at corner 1.
     * @param[in] sv2 Value at corner 2.
     * @param[in] sv3 Value at corner 3.
     * @param[in] sv4 Value at corner 4.
     * @param[in] sv5 Value at corner 5.
     * @param[in] sv6 Value at corner 6.
     * @param[in] sv7 Value at corner 7.
     * @param[in] iso Isolevel separating inside from outside.
     * @param[out] voxel_code The resulting 8-bit case index.
     * @note Codes 0 and 255 mean the cell is entirely outside or inside and emits nothing.
     */
    __device__ __forceinline__ void compute_voxel_code(
        float sv0, float sv1, float sv2, float sv3,
        float sv4, float sv5, float sv6, float sv7,
        float iso, uint8_t &voxel_code)
    {
        voxel_code = 0;
        if (sv0 < iso) voxel_code |= 1;
        if (sv1 < iso) voxel_code |= 2;
        if (sv2 < iso) voxel_code |= 4;
        if (sv3 < iso) voxel_code |= 8;
        if (sv4 < iso) voxel_code |= 16;
        if (sv5 < iso) voxel_code |= 32;
        if (sv6 < iso) voxel_code |= 64;
        if (sv7 < iso) voxel_code |= 128;
    }

/**
 * @brief Classifies every voxel by the sign pattern of its eight corners.
 * @details Stage 1 of extraction. One thread per voxel; each compares its corner values
 * against the isolevel and packs the results into a 256-case code. The code alone
 * determines the voxel's surface topology, so all later stages are table lookups rather
 * than searches. Inactive voxels -- entirely inside or outside -- receive code 0 and are
 * compacted away by the host before stage 2, which is what makes cost scale with surface
 * area rather than volume.
 *
 * Each voxel is subsequently split into the six tetrahedra of ::mtg_tets, inheriting
 * Marching Tetrahedra's freedom from ambiguity while keeping a structured grid.
 * @param[in] num_voxels Number of voxels.
 * @param[in] voxels Device array of corner indices, eight per voxel.
 * @param[in] sdf Device array of scalar field values at the grid vertices.
 * @param[in] iso Isolevel separating inside from outside.
 * @param[out] voxel_codes Device array of one sign code per voxel.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void compute_active_voxels_kernel(
        const uint32_t num_voxels,
        const uint32_t *__restrict__ voxels,
        const float *__restrict__ sdf,
        const float iso,
        uint8_t *__restrict__ voxel_codes)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_voxels)
            return;

        uint32_t v0 = voxels[idx * 8 + 0];
        uint32_t v1 = voxels[idx * 8 + 1];
        uint32_t v2 = voxels[idx * 8 + 2];
        uint32_t v3 = voxels[idx * 8 + 3];
        uint32_t v4 = voxels[idx * 8 + 4];
        uint32_t v5 = voxels[idx * 8 + 5];
        uint32_t v6 = voxels[idx * 8 + 6];
        uint32_t v7 = voxels[idx * 8 + 7];

        uint8_t voxel_code = 0;
        compute_voxel_code(
            sdf[v0], sdf[v1], sdf[v2], sdf[v3],
            sdf[v4], sdf[v5], sdf[v6], sdf[v7],
            iso, voxel_code);

        voxel_codes[idx] = voxel_code;
    }

/**
 * @brief Emits the bipolar edge keys of every active voxel.
 * @details Stage 2. One thread per active voxel, writing an ::Edge key -- the sorted pair
 * of grid vertex indices -- for each edge the surface crosses. Keys are deliberately
 * emitted with duplicates: an edge shared by several voxels produces one key per voxel,
 * and the host then sorts and uniques them. Deduplicating this way welds the mesh, so a
 * vertex on a shared edge exists exactly once and the result is watertight rather than a
 * soup of unconnected voxel patches.
 * @param[in] num_active_voxels Number of active voxels after compaction.
 * @param[in] voxels Device array of corner indices, eight per voxel.
 * @param[in] used_voxel_index Device array mapping compacted index to original voxel index.
 * @param[in] used_voxel_codes Device array of sign codes for the active voxels.
 * @param[out] active_edges Device array receiving the emitted edge keys, with duplicates.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Uses the ::mtg_triTable / ::mtg_num_tris table to look up which edges a case activates.
 */
__global__ void compute_active_edges_kernel(
        const uint32_t num_active_voxels,
        const uint32_t *voxels,
        const uint32_t *used_voxel_index,
        const uint8_t *used_voxel_codes,
        Edge *active_edges)
    {
        uint32_t active_voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (active_voxel_idx >= num_active_voxels)
            return;

        uint32_t voxel_idx = used_voxel_index[active_voxel_idx];
        uint8_t voxel_code = used_voxel_codes[active_voxel_idx];
        const uint32_t *vertices_indices = &voxels[voxel_idx * 8];

        #pragma unroll
        for (int t = 0; t < 6; t++) {
            #pragma unroll
            for (int e = 0; e < 6; e++) {
                int corner_idx0 = mtg_tets[t][mtg_edgeConnection[e][0]];
                int corner_idx1 = mtg_tets[t][mtg_edgeConnection[e][1]];
                
                bool sign0 = (voxel_code & (1 << corner_idx0)) != 0;
                bool sign1 = (voxel_code & (1 << corner_idx1)) != 0;
                
                if (sign0 != sign1) {
                    uint32_t v0 = vertices_indices[corner_idx0];
                    uint32_t v1 = vertices_indices[corner_idx1];
                    active_edges[active_voxel_idx * 36 + t * 6 + e] = Edge(v0, v1);
                } else {
                    active_edges[active_voxel_idx * 36 + t * 6 + e] = Edge(0xFFFFFFFF, 0xFFFFFFFF);
                }
            }
        }
    }

/**
 * @brief Maps each active voxel's local edges onto global output vertex indices.
 * @details Stage 3. After the host has sorted and uniqued the edge keys, this kernel binds
 * voxel-local edge slots to positions in the deduplicated vertex array. One thread per
 * active voxel, binary-searching the unique key array for each of its bipolar edges. The
 * resulting map is what lets stage 5 emit triangle indices without any further searching.
 * @param[in] num_active_voxels Number of active voxels.
 * @param[in] num_unique_edges Number of deduplicated edge keys.
 * @param[in] voxels Device array of corner indices, eight per voxel.
 * @param[in] used_voxel_index Device array mapping compacted index to original voxel index.
 * @param[in] used_voxel_codes Device array of sign codes for the active voxels.
 * @param[in] unique_edges Device array of sorted, deduplicated edge keys.
 * @param[out] voxel_edge_to_vert_idx Device array mapping each voxel-local edge slot to a
 *     global vertex index.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Requires @p unique_edges to be sorted; the lookup is a binary search.
 */
__global__ void build_edge_map_kernel(
        const uint32_t num_active_voxels,
        const uint32_t num_unique_edges,
        const uint32_t *voxels,
        const uint32_t *used_voxel_index,
        const uint8_t *used_voxel_codes,
        const Edge *unique_edges,
        uint32_t *voxel_edge_to_vert_idx)
    {
        uint32_t active_voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (active_voxel_idx >= num_active_voxels)
            return;

        uint32_t global_voxel_idx = used_voxel_index[active_voxel_idx];
        uint8_t voxel_code = used_voxel_codes[active_voxel_idx];
        const uint32_t *vertices_indices = &voxels[global_voxel_idx * 8];

        #pragma unroll
        for (int t = 0; t < 6; t++) {
            #pragma unroll
            for (int e = 0; e < 6; e++) {
                int corner_idx0 = mtg_tets[t][mtg_edgeConnection[e][0]];
                int corner_idx1 = mtg_tets[t][mtg_edgeConnection[e][1]];
                
                bool sign0 = (voxel_code & (1 << corner_idx0)) != 0;
                bool sign1 = (voxel_code & (1 << corner_idx1)) != 0;
                
                if (sign0 != sign1) {
                    uint32_t v0 = vertices_indices[corner_idx0];
                    uint32_t v1 = vertices_indices[corner_idx1];
                    Edge edge(v0, v1);

                    uint32_t left = 0;
                    uint32_t right = num_unique_edges - 1;
                    uint32_t unique_id = 0xFFFFFFFF;

                    while (left <= right)
                    {
                        uint32_t mid = left + (right - left) / 2;
                        if (unique_edges[mid] == edge)
                        {
                            unique_id = mid;
                            break;
                        }
                        else if (unique_edges[mid] < edge)
                        {
                            left = mid + 1;
                        }
                        else
                        {
                            right = mid - 1;
                        }
                    }
                    voxel_edge_to_vert_idx[active_voxel_idx * 36 + t * 6 + e] = unique_id;
                } else {
                    voxel_edge_to_vert_idx[active_voxel_idx * 36 + t * 6 + e] = 0xFFFFFFFF;
                }
            }
        }
    }

/**
 * @brief Places one output vertex on each unique bipolar edge.
 * @details Stage 4. One thread per unique edge, linearly interpolating the crossing point
 * $\mathbf{p} = \mathbf{p}_0 + t(\mathbf{p}_1 - \mathbf{p}_0)$ with
 * $t = (\text{iso} - f_0) / (f_1 - f_0)$. Normals and colours defined on the grid are
 * interpolated with the same $t$, so all attributes stay consistent with the geometry.
 * Because each edge is visited once, every vertex is written exactly once and no atomics
 * are needed.
 * @param[in] num_out_vertices Number of unique edges, one output vertex each.
 * @param[in] unique_edges Device array of deduplicated edge keys.
 * @param[in] grid_vertices Device array of grid vertex coordinates.
 * @param[in] values Device array of scalar field values at grid vertices.
 * @param[in] grid_normals Device array of per-vertex normals, or `nullptr`.
 * @param[in] grid_colors Device array of per-vertex colours, or `nullptr`.
 * @param[in] iso Isolevel being extracted.
 * @param[out] out_verts Device array of interpolated surface vertices.
 * @param[out] out_normals Device array of interpolated normals, when requested.
 * @param[out] out_colors Device array of interpolated colours, when requested.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning The interpolation denominator $f_1 - f_0$ is non-zero for any genuinely bipolar
 * edge, but a field with exactly equal corner values either side of the isolevel would
 * divide by zero. Such edges are excluded upstream by the sign test.
 */
__global__ void interpolate_vertices_kernel(
        const uint32_t num_out_vertices,
        const Edge* unique_edges,
        const float3* grid_vertices,
        const float* values,
        const float3* grid_normals,
        const float3* grid_colors,
        const float iso,
        float3* out_verts,
        float3* out_normals,
        float3* out_colors
    )
    {
        uint32_t v_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (v_idx >= num_out_vertices)
            return;

        Edge edge_sig = unique_edges[v_idx];
        uint32_t v0_idx = edge_sig.v0;
        uint32_t v1_idx = edge_sig.v1;

        float3 p0 = grid_vertices[v0_idx];
        float3 p1 = grid_vertices[v1_idx];
        float val0 = values[v0_idx];
        float val1 = values[v1_idx];

        float3 p, n, c;
        bool has_normals = grid_normals != nullptr;
        bool has_colors = grid_colors != nullptr;
        float3 n0 = has_normals ? grid_normals[v0_idx] : make_float3(0, 0, 0);
        float3 n1 = has_normals ? grid_normals[v1_idx] : make_float3(0, 0, 0);
        float3 c0 = has_colors ? grid_colors[v0_idx] : make_float3(0, 0, 0);
        float3 c1 = has_colors ? grid_colors[v1_idx] : make_float3(0, 0, 0);

        if (fabsf(iso - val0) < EPS) {
            p = p0; if (has_normals) n = n0; if (has_colors) c = c0;
        } else if (fabsf(iso - val1) < EPS) {
            p = p1; if (has_normals) n = n1; if (has_colors) c = c1;
        } else if (fabsf(val0 - val1) < EPS) {
            p = p0; if (has_normals) n = n0; if (has_colors) c = c0;
        } else {
            float t = (val1 != val0) ? fmaxf(0.0f, fminf(1.0f, (iso - val0) / (val1 - val0))) : 0.5f;
            p = p0 + (p1 - p0) * t;
            if (has_normals) {
                n = n0 + (n1 - n0) * t;
                n = maths::normalize(n);
            }
            if (has_colors) c = c0 + (c1 - c0) * t;
        }

        out_verts[v_idx] = p;
        if (has_normals) out_normals[v_idx] = n;
        if (has_colors) out_colors[v_idx] = c;
    }

    /**
     * @brief Launches stage 1: classify every voxel by its corner signs.
     * @details Host wrapper sizing a 1D grid over all voxels and launching the classification
     * kernel. See the kernel for the parallel decomposition.
     * @param[in] num_voxels Number of voxels.
     * @param[in] voxels Device array of corner indices.
     * @param[in] sdf Device array of scalar field values.
     * @param[in] iso Isolevel separating inside from outside.
     * @param[out] voxel_codes Device array of per-voxel sign codes.
     */
    void compute_active_voxels(
        const uint32_t num_voxels,
        const uint32_t *voxels,
        const float *sdf,
        const float iso,
        uint8_t *voxel_codes)
    {
        int block_size = NTHREADS;
        int grid_size = (num_voxels + block_size - 1) / block_size;
        compute_active_voxels_kernel<<<grid_size, block_size>>>(
            num_voxels, voxels, sdf, iso, voxel_codes);
    }

    /**
     * @brief Launches stage 2: emit bipolar edge keys for the active voxels.
     * @details Host wrapper over the edge emission kernel. Keys are emitted with duplicates;
     * the caller sorts and uniques them to weld shared vertices.
     * @param[in] num_active_voxels Number of active voxels.
     * @param[in] voxels Device array of corner indices.
     * @param[in] used_voxel_index Device array mapping compacted index to original index.
     * @param[in] used_voxel_codes Device array of active sign codes.
     * @param[out] active_edges Device array of emitted edge keys.
     */
    void compute_active_edges(
        const uint32_t num_active_voxels,
        const uint32_t *voxels,
        const uint32_t *used_voxel_index,
        const uint8_t *used_voxel_codes,
        Edge *active_edges)
    {
        int block_size = NTHREADS;
        int grid_size = (num_active_voxels + block_size - 1) / block_size;
        compute_active_edges_kernel<<<grid_size, block_size>>>(
            num_active_voxels, voxels, used_voxel_index, used_voxel_codes, active_edges);
    }

    /**
     * @brief Counts the voxels the surface crosses.
     * @details Reduces the sign codes on device, treating any code other than all-inside or
     * all-outside as active. The host needs this count before it can size the compacted arrays.
     * @param[in] num_voxels Number of voxels.
     * @param[in] voxel_codes Device array of per-voxel sign codes.
     * @param[out] num_active_voxels Receives the active count.
     * @note Synchronises to bring the count back to the host.
     */
    void compute_number_active_voxels(
        const uint32_t num_voxels,
        uint8_t *voxel_codes,
        uint32_t &num_active_voxels)
    {
        thrust::device_ptr<uint8_t> d_codes(voxel_codes);
        auto active_flag_iter = thrust::make_transform_iterator(d_codes, is_active_voxel());

        uint32_t *__restrict__ temp_buffer;
        CHECK_CUDA_INTERNAL(cudaMalloc((void **)&temp_buffer, num_voxels * sizeof(uint32_t)));
        thrust::device_ptr<uint32_t> d_prefix_sum(temp_buffer);

        thrust::exclusive_scan(active_flag_iter, active_flag_iter + num_voxels, d_prefix_sum);

        uint8_t last_flag;
        uint32_t last_prefix_sum;
        CHECK_CUDA_INTERNAL(cudaMemcpy(&last_flag, voxel_codes + num_voxels - 1, sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA_INTERNAL(cudaMemcpy(&last_prefix_sum, temp_buffer + num_voxels - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        num_active_voxels = last_prefix_sum + ((last_flag > 0 && last_flag < 255) ? 1 : 0);
        CHECK_CUDA_INTERNAL(cudaFree(temp_buffer));
    }

    /**
     * @brief Compacts active voxels into dense arrays.
     * @details Stream-compacts indices and codes so every later stage launches over active
     * voxels only. This is what makes cost scale with surface area rather than volume.
     * @param[in] num_voxels Number of voxels.
     * @param[in] voxel_codes Device array of per-voxel sign codes.
     * @param[out] used_voxel_index Device array mapping compacted index to original index.
     * @param[out] used_voxel_code Device array of the active voxels' sign codes.
     */
    void compact_active_voxels(
        const uint32_t num_voxels,
        const uint8_t *voxel_codes,
        uint32_t *used_voxel_index,
        uint8_t *used_voxel_code)
    {
        thrust::device_ptr<const uint8_t> d_codes(voxel_codes);
        auto counting_iter = thrust::make_counting_iterator<uint32_t>(0);

        auto zip_in = thrust::make_zip_iterator(thrust::make_tuple(counting_iter, d_codes));

        thrust::device_ptr<uint32_t> d_out_idx(used_voxel_index);
        thrust::device_ptr<uint8_t> d_out_code(used_voxel_code);
        auto zip_out = thrust::make_zip_iterator(thrust::make_tuple(d_out_idx, d_out_code));

        thrust::copy_if(
            zip_in, zip_in + num_voxels, d_codes, zip_out, is_active_voxel());
    }

    /**
     * @brief Sorts and deduplicates the emitted edge keys.
     * @details Device sort followed by a unique pass. Deduplication is what welds the mesh:
     * an edge shared by several cells yields exactly one output vertex, so the result is
     * watertight rather than a soup of disconnected patches.
     * @return The number of unique edges.
     * @note Synchronises to bring the unique count back to the host.
     */
    Edge* compute_unique_active_edges(
        const uint32_t num_active_voxels,
        Edge *active_edges,
        uint32_t &num_unique_edges)
    {
        thrust::device_ptr<Edge> d_active_edges(active_edges);
        thrust::sort(d_active_edges, d_active_edges + (num_active_voxels * 36));

        Edge empty_edge = Edge(0xFFFFFFFF, 0xFFFFFFFF);
        auto valid_end = thrust::lower_bound(d_active_edges, d_active_edges + (num_active_voxels * 36), empty_edge);
        auto unique_end = thrust::unique(d_active_edges, valid_end);

        num_unique_edges = thrust::distance(d_active_edges, unique_end);

        Edge *__restrict__ unique_edges;
        CHECK_CUDA_INTERNAL(cudaMalloc((void **)&unique_edges, num_unique_edges * sizeof(Edge)));

        thrust::device_ptr<Edge> d_unique_edges(unique_edges);
        thrust::copy(d_active_edges, unique_end, d_unique_edges);
        return unique_edges;
    }

    /**
     * @brief Launches stage 3: bind voxel-local edge slots to global vertex indices.
     * @details Host wrapper over the edge-map kernel, run after deduplication so the binary
     * search has a sorted key array to work against.
     * @param[in] num_active_voxels Number of active voxels.
     * @param[in] num_unique_edges Number of deduplicated edges.
     * @param[out] voxel_edge_to_vert_idx Device array mapping local edge slots to vertices.
     */
    void build_edge_map(
        const uint32_t num_active_voxels,
        const uint32_t num_unique_edges,
        const uint32_t *voxels,
        const uint32_t *used_voxel_index,
        const uint8_t *used_voxel_codes,
        const Edge *unique_edges,
        uint32_t *voxel_edge_to_vert_idx)
    {
        if (num_active_voxels == 0 || num_unique_edges == 0) return;
        int block_size = NTHREADS;
        int grid_size = (num_active_voxels + block_size - 1) / block_size;
        build_edge_map_kernel<<<grid_size, block_size>>>(
            num_active_voxels, num_unique_edges, voxels, used_voxel_index, used_voxel_codes, unique_edges, voxel_edge_to_vert_idx);
    }

    /**
     * @brief Launches stage 4: place one vertex on each unique bipolar edge.
     * @details Host wrapper sizing a 1D grid over the unique edges.
     * @param[in] num_unique_edges Number of unique edges, one output vertex each.
     * @param[in] iso Isolevel being extracted.
     * @param[out] out_verts Device array of interpolated surface vertices.
     */
    void interpolate_vertices(
        const uint32_t num_unique_edges,
        const Edge* unique_edges,
        const float3* grid_vertices,
        const float* values,
        const float3* grid_normals,
        const float3* grid_colors,
        const float iso,
        float3* out_verts,
        float3* out_normals,
        float3* out_colors
    )
    {
        if (num_unique_edges == 0) return;
        int block_size = NTHREADS;
        int grid_size = (num_unique_edges + block_size - 1) / block_size;
        interpolate_vertices_kernel<<<grid_size, block_size>>>(
            num_unique_edges, unique_edges, grid_vertices, values, grid_normals, grid_colors, iso, out_verts, out_normals, out_colors);
    }

    /**
     * @brief Counts output triangles and prefix-sums their per-voxel offsets.
     * @details Each sign code implies a fixed triangle count, so an exclusive scan over the
     * active voxels gives every voxel a disjoint output range. Assembly then writes without
     * atomics or contention.
     * @param[in] num_active_voxels Number of active voxels.
     * @param[in] used_voxel_codes Device array of active sign codes.
     * @param[out] num_triangles Receives the total triangle count.
     * @param[out] voxel_triangle_prefix_sums Device array of per-voxel output offsets.
     * @note Synchronises to bring the total back to the host.
     */
    void compute_number_triangles(
        const uint32_t num_active_voxels,
        const uint8_t *used_voxel_codes,
        uint32_t &num_triangles,
        uint32_t *voxel_triangle_prefix_sums)
    {
        thrust::device_ptr<const uint8_t> d_codes(used_voxel_codes);
        auto num_tris_iter = thrust::make_transform_iterator(d_codes, num_triangles_functor());

        thrust::device_ptr<uint32_t> d_prefix_sum(voxel_triangle_prefix_sums);
        
        num_triangles = thrust::reduce(num_tris_iter, num_tris_iter + num_active_voxels);
        thrust::exclusive_scan(num_tris_iter, num_tris_iter + num_active_voxels, d_prefix_sum);
    }

/**
 * @brief Writes the triangle index triples for every active voxel.
 * @details Stage 5. One thread per active voxel. The sign code selects a row of the
 * ::mtg_triTable / ::mtg_num_tris triangle table, whose voxel-local edge slots are translated into global vertex
 * indices through the stage 3 map. Each voxel's output begins at a precomputed prefix-sum
 * offset, so threads write to disjoint ranges without atomics or contention.
 * @param[in] num_active_voxels Number of active voxels.
 * @param[in] used_voxel_codes Device array of sign codes for the active voxels.
 * @param[in] voxel_edge_to_vert_idx Device array mapping voxel-local edge slots to global
 *     vertex indices.
 * @param[in] voxel_triangle_prefix_sums Device array of per-voxel output offsets.
 * @param[out] out_triangles Device array receiving triangle vertex index triples.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Winding follows the table, giving outward-facing normals for a field that is
 * negative inside.
 */
__global__ void assemble_triangles_kernel(
        const uint32_t num_active_voxels,
        const uint8_t *used_voxel_codes,
        const uint32_t *voxel_edge_to_vert_idx,
        const uint32_t *voxel_triangle_prefix_sums,
        uint32_t *out_triangles)
    {
        uint32_t active_voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (active_voxel_idx >= num_active_voxels)
            return;

        uint8_t voxel_code = used_voxel_codes[active_voxel_idx];
        uint32_t start_tri_idx = voxel_triangle_prefix_sums[active_voxel_idx];

        int tri_count = 0;
        for (int t = 0; t < 6; t++) {
            uint8_t tet_code = 0;
            for (int i = 0; i < 4; i++) {
                if (voxel_code & (1 << mtg_tets[t][i])) tet_code |= (1 << i);
            }
            
            for (int i = 0; i < 6; i += 3)
            {
                int edge0 = mtg_triTable[tet_code][i];
                if (edge0 == -1) break;

                int edge1 = mtg_triTable[tet_code][i + 1];
                int edge2 = mtg_triTable[tet_code][i + 2];

                uint32_t v0 = voxel_edge_to_vert_idx[active_voxel_idx * 36 + t * 6 + edge0];
                uint32_t v1 = voxel_edge_to_vert_idx[active_voxel_idx * 36 + t * 6 + edge1];
                uint32_t v2 = voxel_edge_to_vert_idx[active_voxel_idx * 36 + t * 6 + edge2];

                out_triangles[(start_tri_idx + tri_count) * 3 + 0] = v0;
                out_triangles[(start_tri_idx + tri_count) * 3 + 1] = v1;
                out_triangles[(start_tri_idx + tri_count) * 3 + 2] = v2;
                
                tri_count++;
            }
        }
    }

    /**
     * @brief Launches stage 5: write the triangle index triples.
     * @details Host wrapper over the assembly kernel; each voxel writes into the disjoint
     * range given by its prefix-sum offset.
     * @param[in] num_active_voxels Number of active voxels.
     * @param[out] out_triangles Device array of triangle vertex index triples.
     */
    void assemble_triangles(
        const uint32_t num_active_voxels,
        const uint8_t *used_voxel_codes,
        const uint32_t *voxel_edge_to_vert_idx,
        const uint32_t *voxel_triangle_prefix_sums,
        uint32_t *out_triangles)
    {
        if (num_active_voxels == 0) return;
        int block_size = NTHREADS;
        int grid_size = (num_active_voxels + block_size - 1) / block_size;
        assemble_triangles_kernel<<<grid_size, block_size>>>(
            num_active_voxels, used_voxel_codes, voxel_edge_to_vert_idx, voxel_triangle_prefix_sums, out_triangles);
    }

    /**
     * @brief End-to-end grid Marching Tetrahedra isosurface extraction on the GPU.
     * @details The host dispatcher orchestrating the whole pipeline: classify, compact, emit
     * edges, deduplicate, map, interpolate, and assemble. Output buffers are allocated through
     * PyTorch, so the extracted mesh needs no copy to be returned to Python.
     * @return A tuple of vertices, triangles, and the optional interpolated attributes.
     * @note Three device synchronisations are unavoidable -- after the active count, the unique
     * edge count, and the triangle count -- because each sizes the next allocation.
     */
    std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>> marching_tetrahedra_grid(
        const uint32_t num_voxels,
        const float3* __restrict__ grid_vertices,
        const uint32_t* __restrict__ voxels,
        const float* __restrict__ voxel_values,
        const float3* __restrict__ grid_normals,
        const float3* __restrict__ grid_colors,
        const float iso,
        torch::TensorOptions vert_options,
        torch::TensorOptions tri_options,
        bool return_unique_edges
    )
    {
        uint8_t *__restrict__ voxel_codes;
        CHECK_CUDA_INTERNAL(cudaMalloc((void **)&voxel_codes, num_voxels * sizeof(uint8_t)));
        
        compute_active_voxels(num_voxels, voxels, voxel_values, iso, voxel_codes);

        uint32_t num_active_voxels;
        compute_number_active_voxels(num_voxels, voxel_codes, num_active_voxels);

        if (num_active_voxels == 0) {
            CHECK_CUDA_INTERNAL(cudaFree(voxel_codes));
            std::optional<torch::Tensor> out_n = std::nullopt;
            std::optional<torch::Tensor> out_c = std::nullopt;
            if (grid_normals != nullptr) out_n = torch::empty({0, 3}, vert_options);
            if (grid_colors != nullptr) out_c = torch::empty({0, 3}, vert_options);
            return std::make_tuple(
                torch::empty({0, 3}, vert_options),
                torch::empty({0, 3}, tri_options),
                out_n,
                out_c,
                std::nullopt
            );
        }

        uint32_t *__restrict__ used_voxel_index;
        uint8_t *__restrict__ used_voxel_codes;
        CHECK_CUDA_INTERNAL(cudaMalloc((void **)&used_voxel_index, num_active_voxels * sizeof(uint32_t)));
        CHECK_CUDA_INTERNAL(cudaMalloc((void **)&used_voxel_codes, num_active_voxels * sizeof(uint8_t)));
        compact_active_voxels(num_voxels, voxel_codes, used_voxel_index, used_voxel_codes);

        Edge *__restrict__ active_edges;
        CHECK_CUDA_INTERNAL(cudaMalloc((void **)&active_edges, num_active_voxels * 36 * sizeof(Edge)));
        compute_active_edges(num_active_voxels, voxels, used_voxel_index, used_voxel_codes, active_edges);

        uint32_t out_num_vertices;
        Edge *__restrict__ unique_edges = compute_unique_active_edges(num_active_voxels, active_edges, out_num_vertices);

        uint32_t *__restrict__ voxel_edge_to_vert_idx;
        CHECK_CUDA_INTERNAL(cudaMalloc((void **)&voxel_edge_to_vert_idx, num_active_voxels * 36 * sizeof(uint32_t)));
        build_edge_map(num_active_voxels, out_num_vertices, voxels, used_voxel_index, used_voxel_codes, unique_edges, voxel_edge_to_vert_idx);

        torch::Tensor out_vertices = torch::empty({out_num_vertices, 3}, vert_options);
        float3* __restrict__ p_out_vertices = (float3*)out_vertices.data_ptr<float>();

        std::optional<torch::Tensor> out_normals_opt = std::nullopt;
        float3* __restrict__ p_out_normals = nullptr;
        if (grid_normals != nullptr) {
            torch::Tensor out_normals = torch::empty({out_num_vertices, 3}, vert_options);
            p_out_normals = (float3*)out_normals.data_ptr<float>();
            out_normals_opt = out_normals;
        }

        std::optional<torch::Tensor> out_colors_opt = std::nullopt;
        float3* __restrict__ p_out_colors = nullptr;
        if (grid_colors != nullptr) {
            torch::Tensor out_colors = torch::empty({out_num_vertices, 3}, vert_options);
            p_out_colors = (float3*)out_colors.data_ptr<float>();
            out_colors_opt = out_colors;
        }

        interpolate_vertices(out_num_vertices, unique_edges, grid_vertices, voxel_values, grid_normals, grid_colors, iso, p_out_vertices, p_out_normals, p_out_colors);

        uint32_t out_num_triangles;
        uint32_t *__restrict__ voxel_triangle_prefix_sums;
        CHECK_CUDA_INTERNAL(cudaMalloc((void **)&voxel_triangle_prefix_sums, num_active_voxels * sizeof(uint32_t)));
        compute_number_triangles(num_active_voxels, used_voxel_codes, out_num_triangles, voxel_triangle_prefix_sums);

        torch::Tensor out_triangles = torch::empty({out_num_triangles, 3}, tri_options);
        uint32_t* __restrict__ p_out_triangles = (uint32_t*)out_triangles.data_ptr<int32_t>();

        assemble_triangles(num_active_voxels, used_voxel_codes, voxel_edge_to_vert_idx, voxel_triangle_prefix_sums, p_out_triangles);

        std::optional<torch::Tensor> out_unique_edges_opt = std::nullopt;
        if (return_unique_edges) {
            torch::Tensor out_unique_edges = torch::empty({out_num_vertices, 2}, torch::dtype(torch::kInt32).device(vert_options.device()));
            CHECK_CUDA_INTERNAL(cudaMemcpy(out_unique_edges.data_ptr(), unique_edges, out_num_vertices * sizeof(Edge), cudaMemcpyDeviceToDevice));
            out_unique_edges_opt = out_unique_edges;
        }

        // Cleanup
        CHECK_CUDA_INTERNAL(cudaFree(voxel_codes));
        CHECK_CUDA_INTERNAL(cudaFree(used_voxel_index));
        CHECK_CUDA_INTERNAL(cudaFree(used_voxel_codes));
        CHECK_CUDA_INTERNAL(cudaFree(active_edges));
        CHECK_CUDA_INTERNAL(cudaFree(unique_edges));
        CHECK_CUDA_INTERNAL(cudaFree(voxel_edge_to_vert_idx));
        CHECK_CUDA_INTERNAL(cudaFree(voxel_triangle_prefix_sums));

        return std::make_tuple(out_vertices, out_triangles, out_normals_opt, out_colors_opt, out_unique_edges_opt);
    }

/**
 * @brief Backpropagates vertex gradients into the scalar field and colours.
 * @details The analytical adjoint of stage 4. A vertex sits at
 * $\mathbf{p} = \mathbf{p}_0 + t(\mathbf{p}_1 - \mathbf{p}_0)$ with
 * $t = (\text{iso} - f_0) / (f_1 - f_0)$, so $\partial t / \partial f_0$ and
 * $\partial t / \partial f_1$ are available in closed form and the chain rule gives the
 * field gradient directly -- no finite differences, no autograd graph over the extraction
 * itself. One thread per output vertex.
 * @param[in] n_verts Number of output vertices.
 * @param[in] unique_edges Device array of the edge keys that produced them.
 * @param[in] grid_values Device array of scalar field values at grid vertices.
 * @param[in] grid_coords Device array of grid vertex coordinates.
 * @param[in] grid_colors Device array of per-vertex colours, or `nullptr`.
 * @param[in] adj_verts Device array of incoming vertex position gradients.
 * @param[in] adj_colors Device array of incoming colour gradients, or `nullptr`.
 * @param[in] iso Isolevel used in the forward pass.
 * @param[out] adj_values Device array accumulating scalar field gradients.
 * @param[out] adj_grid_colors Device array accumulating colour gradients.
 * @param[in] with_colors Whether colour gradients are propagated.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Grid vertices are shared between edges, so several threads accumulate into the
 * same slot. Writes go through `atomicAdd`, which makes the reduction order
 * nondeterministic and the result bitwise non-reproducible between runs.
 */
__global__ void backward_kernel(
        const uint32_t n_verts,
        const Edge *unique_edges,
        const float *grid_values,
        const float3 *grid_coords,
        const float3 *grid_colors,
        const float3 *adj_verts,
        const float3 *adj_colors,
        const float iso,
        float *adj_values,
        float3 *adj_grid_colors,
        bool with_colors)
    {
        uint32_t v_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (v_idx >= n_verts)
            return;

        Edge edge_sig = unique_edges[v_idx];
        uint32_t v0_idx = edge_sig.v0;
        uint32_t v1_idx = edge_sig.v1;

        float v0_val = grid_values[v0_idx];
        float v1_val = grid_values[v1_idx];
        float3 p0 = grid_coords[v0_idx];
        float3 p1 = grid_coords[v1_idx];

        float3 grad_p_out = adj_verts[v_idx];

        float3 c0, c1, grad_c_out;
        if (with_colors)
        {
            c0 = grid_colors[v0_idx];
            c1 = grid_colors[v1_idx];
            grad_c_out = adj_colors[v_idx];
        }

        float diff = v1_val - v0_val;

        if (with_colors)
        {
            float t = 0.5f;
            if (fabsf(iso - v0_val) < EPS) t = 0.0f;
            else if (fabsf(iso - v1_val) < EPS) t = 1.0f;
            else if (fabsf(diff) >= EPS) t = fmaxf(0.0f, fminf(1.0f, (iso - v0_val) / diff));

            float t0 = 1.0f - t;
            float t1 = t;
            
            atomicAdd(&(adj_grid_colors[v0_idx].x), grad_c_out.x * t0);
            atomicAdd(&(adj_grid_colors[v0_idx].y), grad_c_out.y * t0);
            atomicAdd(&(adj_grid_colors[v0_idx].z), grad_c_out.z * t0);

            atomicAdd(&(adj_grid_colors[v1_idx].x), grad_c_out.x * t1);
            atomicAdd(&(adj_grid_colors[v1_idx].y), grad_c_out.y * t1);
            atomicAdd(&(adj_grid_colors[v1_idx].z), grad_c_out.z * t1);
        }

        if (diff * diff < 1e-14f)
            return;
            
        float dot_prod = (p1.x - p0.x) * grad_p_out.x + (p1.y - p0.y) * grad_p_out.y + (p1.z - p0.z) * grad_p_out.z;
        float common = dot_prod / (diff * diff);
        float grad_v0 = common * (iso - v1_val);
        float grad_v1 = common * (v0_val - iso);

        atomicAdd(&adj_values[v0_idx], grad_v0);
        atomicAdd(&adj_values[v1_idx], grad_v1);
    }

    /**
     * @brief Launches the analytical backward pass.
     * @details Host wrapper over the adjoint kernel, which differentiates the edge interpolation
     * in closed form.
     * @param[in] n_verts Number of output vertices.
     * @param[in] iso Isolevel used in the forward pass.
     * @param[out] adj_values Device array accumulating scalar field gradients.
     * @warning Accumulation into shared grid vertices is atomic, so results are not bitwise
     * reproducible between runs.
     */
    void backward(
        const uint32_t n_verts,
        const Edge *unique_edges,
        const float3 *grid_vertices,
        const float3 *grid_colors,
        const float *values,
        const float3 *adj_verts,
        const float3 *adj_colors,
        float *adj_values,
        float3 *adj_grid_colors,
        const float iso)
    {
        if (n_verts == 0) return;

        bool with_colors = (grid_colors != nullptr && adj_colors != nullptr && adj_grid_colors != nullptr);
        int block_size = NTHREADS;
        int grid_size = (n_verts + block_size - 1) / block_size;
        
        backward_kernel<<<grid_size, block_size>>>(
            n_verts,
            unique_edges,
            values,
            grid_vertices,
            grid_colors,
            adj_verts,
            adj_colors,
            iso,
            adj_values,
            adj_grid_colors,
            with_colors);
    }
}
