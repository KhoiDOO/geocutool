/**
 * @file mt.cu
 * @brief CUDA kernel implementations for Marching Tetrahedra on unstructured meshes and analytical backward pass.
 */

#include "mt.h"
#include "../maths/maths.h"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAFunctions.h>
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
#include <ATen/cuda/ThrustAllocator.h>
#include <thrust/execution_policy.h>

namespace mt
{
    /**
     * @brief Thrust predicate selecting tets the surface crosses.
     * @details A tet is active when its sign code is neither all-inside nor
     * all-outside. Used to stream-compact the active set, so later stages launch only
     * over tets that actually contribute geometry.
     */
    struct is_active_tet
    {
        /**
         * @brief Tests whether a sign code marks an active tetrahedron.
         * @param[in] code The tetrahedron's corner sign code.
         * @return 1 when the surface crosses it, 0 otherwise.
         */
        __host__ __device__ int operator()(const uint8_t code) const
        {
            return (code > 0 && code < 15) ? 1 : 0;
        }
    };

    /**
     * @brief Thrust functor mapping a sign code to its triangle count.
     * @details Feeds the exclusive scan that assigns each active tet a disjoint output
     * range, so triangle assembly needs no atomics.
     */
    struct num_triangles_functor
    {
        /**
         * @brief Looks up the triangle count implied by a sign code.
         * @param[in] code The tetrahedron's corner sign code.
         * @return Number of triangles the tetrahedron emits.
         */
        __device__ uint32_t operator()(const uint8_t code) const {
            return (tetTriNumTable[code + 1] - tetTriNumTable[code]) / 3;
        }
    };

    /**
     * @brief Packs four corner signs into a Marching Tetrahedra case index.
     * @details Sets bit $i$ when corner $i$ lies below the isolevel, giving a 4-bit code with
     * 16 possible cases -- none of them ambiguous, which is why tetrahedral extraction needs no
     * decider.
     * @param[in] sv0 Value at corner 0.
     * @param[in] sv1 Value at corner 1.
     * @param[in] sv2 Value at corner 2.
     * @param[in] sv3 Value at corner 3.
     * @param[in] iso Isolevel separating inside from outside.
     * @param[out] tet_code The resulting 4-bit case index.
     * @note Codes 0 and 15 mean the tetrahedron is entirely outside or inside.
     */
    __device__ __forceinline__ void compute_tet_code(
        float sv0, float sv1, float sv2, float sv3,
        float iso, uint8_t &tet_code)
    {
        tet_code = 0;
        if (sv0 < iso)
            tet_code |= 1;
        if (sv1 < iso)
            tet_code |= 2;
        if (sv2 < iso)
            tet_code |= 4;
        if (sv3 < iso)
            tet_code |= 8;
    }

/**
 * @brief Classifies every tet by the sign pattern of its four corners.
 * @details Stage 1 of extraction. One thread per tet; each compares its corner values
 * against the isolevel and packs the results into a 16-case code. The code alone
 * determines the tet's surface topology, so all later stages are table lookups rather
 * than searches. Inactive tets -- entirely inside or outside -- receive code 0 and are
 * compacted away by the host before stage 2, which is what makes cost scale with surface
 * area rather than volume.
 *
 * A tetrahedron has only 16 cases and none of them is ambiguous, which is why Marching
 * Tetrahedra needs no asymptotic decider and cannot produce cracks.
 * @param[in] num_tets Number of tets.
 * @param[in] tets Device array of corner indices, four per tet.
 * @param[in] vert_values Device array of scalar field values at the grid vertices.
 * @param[in] iso Isolevel separating inside from outside.
 * @param[out] tet_codes Device array of one sign code per tet.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void compute_active_tets_kernel(
        const uint32_t num_tets,
        const uint32_t *__restrict__ tets,
        const float *__restrict__ vert_values,
        const float iso,
        uint8_t *__restrict__ tet_codes)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_tets)
            return;

        uint32_t v0 = tets[idx * 4 + 0];
        uint32_t v1 = tets[idx * 4 + 1];
        uint32_t v2 = tets[idx * 4 + 2];
        uint32_t v3 = tets[idx * 4 + 3];

        uint8_t tet_code = 0;
        compute_tet_code(
            vert_values[v0], vert_values[v1], vert_values[v2], vert_values[v3],
            iso, tet_code);

        tet_codes[idx] = tet_code;
    }

/**
 * @brief Emits the bipolar edge keys of every active tet.
 * @details Stage 2. One thread per active tet, writing an ::Edge key -- the sorted pair
 * of grid vertex indices -- for each edge the surface crosses. Keys are deliberately
 * emitted with duplicates: an edge shared by several tets produces one key per tet,
 * and the host then sorts and uniques them. Deduplicating this way welds the mesh, so a
 * vertex on a shared edge exists exactly once and the result is watertight rather than a
 * soup of unconnected tet patches.
 * @param[in] num_active_tets Number of active tets after compaction.
 * @param[in] tets Device array of corner indices, four per tet.
 * @param[in] used_tet_index Device array mapping compacted index to original tet index.
 * @param[in] used_tet_codes Device array of sign codes for the active tets.
 * @param[out] active_edges Device array receiving the emitted edge keys, with duplicates.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Uses the ::tetEdgeTable / ::tetTriTable table to look up which edges a case activates.
 */
__global__ void compute_active_edges_kernel(
        const uint32_t num_active_tets,
        const uint32_t *tets,
        const uint32_t *used_tet_index,
        const uint8_t *used_tet_codes,
        Edge *active_edges)
    {
        uint32_t active_tet_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (active_tet_idx >= num_active_tets)
            return;

        uint32_t tet_idx = used_tet_index[active_tet_idx];
        uint8_t tet_code = used_tet_codes[active_tet_idx];
        const uint32_t *vertices_indices = &tets[tet_idx * 4];

        int edgeFlags = tetEdgeTable[tet_code];

        #pragma unroll
        for (int i = 0; i < 6; i++)
        {
            if (edgeFlags & (1 << i))
            {
                uint32_t v0 = vertices_indices[tetEdgeConnection[i][0]];
                uint32_t v1 = vertices_indices[tetEdgeConnection[i][1]];
                active_edges[active_tet_idx * 6 + i] = Edge(v0, v1);
            }
            else
            {
                active_edges[active_tet_idx * 6 + i] = Edge(0xFFFFFFFF, 0xFFFFFFFF);
            }
        }
    }

/**
 * @brief Maps each active tet's local edges onto global output vertex indices.
 * @details Stage 3. After the host has sorted and uniqued the edge keys, this kernel binds
 * tet-local edge slots to positions in the deduplicated vertex array. One thread per
 * active tet, binary-searching the unique key array for each of its bipolar edges. The
 * resulting map is what lets stage 5 emit triangle indices without any further searching.
 * @param[in] num_active_tets Number of active tets.
 * @param[in] num_unique_edges Number of deduplicated edge keys.
 * @param[in] tets Device array of corner indices, four per tet.
 * @param[in] used_tet_index Device array mapping compacted index to original tet index.
 * @param[in] used_tet_codes Device array of sign codes for the active tets.
 * @param[in] unique_edges Device array of sorted, deduplicated edge keys.
 * @param[out] tet_edge_to_vert_idx Device array mapping each tet-local edge slot to a
 *     global vertex index.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Requires @p unique_edges to be sorted; the lookup is a binary search.
 */
__global__ void build_edge_map_kernel(
        const uint32_t num_active_tets,
        const uint32_t num_unique_edges,
        const uint32_t *tets,
        const uint32_t *used_tet_index,
        const uint8_t *used_tet_codes,
        const Edge *unique_edges,
        uint32_t *tet_edge_to_vert_idx)
    {
        uint32_t active_tet_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (active_tet_idx >= num_active_tets)
            return;

        uint32_t global_tet_idx = used_tet_index[active_tet_idx];
        uint8_t tet_code = used_tet_codes[active_tet_idx];
        const uint32_t *vertices_indices = &tets[global_tet_idx * 4];

        int edgeFlags = tetEdgeTable[tet_code];

        #pragma unroll
        for (int i = 0; i < 6; i++)
        {
            if (edgeFlags & (1 << i)) {
                uint32_t v0 = vertices_indices[tetEdgeConnection[i][0]];
                uint32_t v1 = vertices_indices[tetEdgeConnection[i][1]];
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
                        if (mid == 0) break;
                        right = mid - 1;
                    }
                }
                tet_edge_to_vert_idx[active_tet_idx * 6 + i] = unique_id;
            } else {
                tet_edge_to_vert_idx[active_tet_idx * 6 + i] = 0xFFFFFFFF;
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
 * @param[in] num_unique_edges Number of unique edges, one output vertex each.
 * @param[in] unique_edges Device array of deduplicated edge keys.
 * @param[in] grid_vertices Device array of grid vertex coordinates.
 * @param[in] vert_values Device array of scalar field values at grid vertices.
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
        const uint32_t num_unique_edges,
        const Edge *__restrict__ unique_edges,
        const float3 *__restrict__ grid_vertices,
        const float *__restrict__ vert_values,
        const float3 *__restrict__ grid_normals,
        const float3 *__restrict__ grid_colors,
        const float iso,
        float3 *__restrict__ out_verts,
        float3 *__restrict__ out_normals,
        float3 *__restrict__ out_colors)
    {
        uint32_t v_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (v_idx >= num_unique_edges)
            return;

        Edge edge = unique_edges[v_idx];
        uint32_t v0_idx = edge.v0;
        uint32_t v1_idx = edge.v1;

        float3 p0 = grid_vertices[v0_idx];
        float3 p1 = grid_vertices[v1_idx];

        float val0 = vert_values[v0_idx];
        float val1 = vert_values[v1_idx];

        float3 p;
        float3 n;
        float3 c;
        bool has_normals = (grid_normals != nullptr && out_normals != nullptr);
        bool has_colors = (grid_colors != nullptr && out_colors != nullptr);
        float3 n0 = has_normals ? grid_normals[v0_idx] : make_float3(0, 0, 0);
        float3 n1 = has_normals ? grid_normals[v1_idx] : make_float3(0, 0, 0);
        float3 c0 = has_colors ? grid_colors[v0_idx] : make_float3(0, 0, 0);
        float3 c1 = has_colors ? grid_colors[v1_idx] : make_float3(0, 0, 0);

        if (fabsf(iso - val0) < EPS)
        {
            p = p0;
            if (has_normals) n = n0;
            if (has_colors) c = c0;
        }
        else if (fabsf(iso - val1) < EPS)
        {
            p = p1;
            if (has_normals) n = n1;
            if (has_colors) c = c1;
        }
        else if (fabsf(val0 - val1) < EPS)
        {
            p = p0;
            if (has_normals) n = n0;
            if (has_colors) c = c0;
        }
        else
        {
            float t = (val1 != val0) ? maths::saturate((iso - val0) / (val1 - val0)) : 0.5f;

            p = maths::lerp(p0, p1, t);

            if (has_normals) {
                n = maths::lerp(n0, n1, t);
                n = maths::normalize(n);
            }
            if (has_colors) {
                c = maths::lerp(c0, c1, t);
            }
        }

        out_verts[v_idx] = p;
        if (has_normals) {
            out_normals[v_idx] = n;
        }
        if (has_colors) {
            out_colors[v_idx] = c;
        }
    }

/**
 * @brief Writes the triangle index triples for every active tet.
 * @details Stage 5. One thread per active tet. The sign code selects a row of the
 * ::tetEdgeTable / ::tetTriTable triangle table, whose tet-local edge slots are translated into global vertex
 * indices through the stage 3 map. Each tet's output begins at a precomputed prefix-sum
 * offset, so threads write to disjoint ranges without atomics or contention.
 * @param[in] num_active_tets Number of active tets.
 * @param[in] used_tet_codes Device array of sign codes for the active tets.
 * @param[in] tet_edge_to_vert_idx Device array mapping tet-local edge slots to global
 *     vertex indices.
 * @param[in] tet_triangle_prefix_sums Device array of per-tet output offsets.
 * @param[out] out_triangles Device array receiving triangle vertex index triples.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Winding follows the table, giving outward-facing normals for a field that is
 * negative inside.
 */
__global__ void assemble_triangles_kernel(
        const uint32_t num_active_tets,
        const uint8_t *__restrict__ used_tet_codes,
        const uint32_t *__restrict__ tet_edge_to_vert_idx,
        const uint32_t *__restrict__ tet_triangle_prefix_sums,
        uint32_t *__restrict__ out_triangles)
    {
        uint32_t active_tet_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (active_tet_idx >= num_active_tets)
            return;

        uint8_t tet_code = used_tet_codes[active_tet_idx];
        uint32_t start_tri_idx = tet_triangle_prefix_sums[active_tet_idx];

        int tri_count = 0;
        #pragma unroll
        for (int i = 0; i < 7; i += 3)
        {
            int edge0 = tetTriTable[tet_code][i];
            if (edge0 == -1)
                break;

            int edge1 = tetTriTable[tet_code][i + 1];
            int edge2 = tetTriTable[tet_code][i + 2];

            uint32_t v0 = tet_edge_to_vert_idx[active_tet_idx * 6 + edge0];
            uint32_t v1 = tet_edge_to_vert_idx[active_tet_idx * 6 + edge1];
            uint32_t v2 = tet_edge_to_vert_idx[active_tet_idx * 6 + edge2];

            out_triangles[(start_tri_idx + tri_count) * 3 + 0] = v0;
            out_triangles[(start_tri_idx + tri_count) * 3 + 1] = v1;
            out_triangles[(start_tri_idx + tri_count) * 3 + 2] = v2;

            tri_count++;
        }
    }

    /**
     * @brief Launches stage 1: classify every tet by its corner signs.
     * @details Host wrapper sizing a 1D grid over all tets and launching the classification
     * kernel. See the kernel for the parallel decomposition.
     * @param[in] num_tets Number of tets.
     * @param[in] tets Device array of corner indices.
     * @param[in] vert_values Device array of scalar field values.
     * @param[in] iso Isolevel separating inside from outside.
     * @param[out] tet_codes Device array of per-tet sign codes.
     */
    void compute_active_tets(
        const uint32_t num_tets,
        const uint32_t *tets,
        const float *vert_values,
        const float iso,
        uint8_t *tet_codes)
    {
        int block_size = NTHREADS;
        int grid_size = (num_tets + block_size - 1) / block_size;
        compute_active_tets_kernel<<<grid_size, block_size>>>(
            num_tets, tets, vert_values, iso, tet_codes);
    }

    /**
     * @brief Counts the tets the surface crosses.
     * @details Reduces the sign codes on device, treating any code other than all-inside or
     * all-outside as active. The host needs this count before it can size the compacted arrays.
     * @param[in] num_tets Number of tets.
     * @param[in] tet_codes Device array of per-tet sign codes.
     * @param[out] num_active_tets Receives the active count.
     * @note Synchronises to bring the count back to the host.
     */
    void compute_number_active_tets(
        const uint32_t num_tets,
        uint8_t *tet_codes,
        uint32_t &num_active_tets)
    {
        at::cuda::ThrustAllocator allocator;
        auto policy = thrust::cuda::par(allocator);

        thrust::device_ptr<uint8_t> d_codes(tet_codes);
        auto active_flag_iter = thrust::make_transform_iterator(d_codes, is_active_tet());

        auto temp_buffer_t = torch::empty({(int64_t)num_tets}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, c10::cuda::current_device()));
        uint32_t *__restrict__ temp_buffer = (uint32_t*)temp_buffer_t.data_ptr<int32_t>();
        thrust::device_ptr<uint32_t> d_prefix_sum(temp_buffer);

        thrust::exclusive_scan(policy, active_flag_iter, active_flag_iter + num_tets, d_prefix_sum);

        uint8_t last_flag;
        uint32_t last_prefix_sum;
        CHECK_CUDA_INTERNAL(cudaMemcpy(&last_flag, tet_codes + num_tets - 1, sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA_INTERNAL(cudaMemcpy(&last_prefix_sum, temp_buffer + num_tets - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        num_active_tets = last_prefix_sum + ((last_flag > 0 && last_flag < 15) ? 1 : 0);
    }

    /**
     * @brief Compacts active tets into dense arrays.
     * @details Stream-compacts indices and codes so every later stage launches over active
     * tets only. This is what makes cost scale with surface area rather than volume.
     * @param[in] num_tets Number of tets.
     * @param[in] tet_codes Device array of per-tet sign codes.
     * @param[out] used_tet_index Device array mapping compacted index to original index.
     * @param[out] used_tet_code Device array of the active tets' sign codes.
     */
    void compact_active_tets(
        const uint32_t num_tets,
        const uint8_t *tet_codes,
        uint32_t *used_tet_index,
        uint8_t *used_tet_code)
    {
        at::cuda::ThrustAllocator allocator;
        auto policy = thrust::cuda::par(allocator);

        thrust::device_ptr<const uint8_t> d_codes(tet_codes);
        auto counting_iter = thrust::make_counting_iterator<uint32_t>(0);

        auto zip_in = thrust::make_zip_iterator(thrust::make_tuple(counting_iter, d_codes));

        thrust::device_ptr<uint32_t> d_out_idx(used_tet_index);
        thrust::device_ptr<uint8_t> d_out_code(used_tet_code);
        auto zip_out = thrust::make_zip_iterator(thrust::make_tuple(d_out_idx, d_out_code));

        thrust::copy_if(
            policy,
            zip_in,
            zip_in + num_tets,
            d_codes,
            zip_out,
            is_active_tet()
        );
    }

    /**
     * @brief Launches stage 2: emit bipolar edge keys for the active tets.
     * @details Host wrapper over the edge emission kernel. Keys are emitted with duplicates;
     * the caller sorts and uniques them to weld shared vertices.
     * @param[in] num_active_tets Number of active tets.
     * @param[in] tets Device array of corner indices.
     * @param[in] used_tet_index Device array mapping compacted index to original index.
     * @param[in] used_tet_codes Device array of active sign codes.
     * @param[out] active_edges Device array of emitted edge keys.
     */
    void compute_active_edges(
        const uint32_t num_active_tets,
        const uint32_t *tets,
        const uint32_t *used_tet_index,
        const uint8_t *used_tet_codes,
        Edge *active_edges)
    {
        if (num_active_tets == 0) return;
        int block_size = NTHREADS;
        int grid_size = (num_active_tets + block_size - 1) / block_size;
        compute_active_edges_kernel<<<grid_size, block_size>>>(
            num_active_tets, tets, used_tet_index, used_tet_codes, active_edges);
    }

    /**
     * @brief Sorts and deduplicates the emitted edge keys.
     * @details Device sort followed by a unique pass. Deduplication is what welds the mesh:
     * an edge shared by several cells yields exactly one output vertex, so the result is
     * watertight rather than a soup of disconnected patches.
     * @return The unique edge keys as a PyTorch tensor.
     * @note Synchronises to bring the unique count back to the host.
     */
    torch::Tensor compute_unique_active_edges(
        const uint32_t num_active_tets,
        Edge *active_edges,
        uint32_t &num_unique_edges)
    {
        at::cuda::ThrustAllocator allocator;
        auto policy = thrust::cuda::par(allocator);

        thrust::device_ptr<Edge> d_active_edges(active_edges);
        thrust::sort(policy, d_active_edges, d_active_edges + (num_active_tets * 6));

        Edge empty_edge = Edge(0xFFFFFFFF, 0xFFFFFFFF);
        auto valid_end = thrust::lower_bound(policy, d_active_edges, d_active_edges + (num_active_tets * 6), empty_edge);
        auto unique_end = thrust::unique(policy, d_active_edges, valid_end);

        num_unique_edges = thrust::distance(d_active_edges, unique_end);

        auto unique_edges_t = torch::empty({(int64_t)(num_unique_edges * sizeof(Edge))}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, c10::cuda::current_device()));
        Edge *__restrict__ unique_edges = (Edge*)unique_edges_t.data_ptr<uint8_t>();

        thrust::device_ptr<Edge> d_unique_edges(unique_edges);
        thrust::copy(policy, d_active_edges, unique_end, d_unique_edges);
        return unique_edges_t;
    }

    /**
     * @brief Launches stage 3: bind tet-local edge slots to global vertex indices.
     * @details Host wrapper over the edge-map kernel, run after deduplication so the binary
     * search has a sorted key array to work against.
     * @param[in] num_active_tets Number of active tets.
     * @param[in] num_unique_edges Number of deduplicated edges.
     * @param[out] tet_edge_to_vert_idx Device array mapping local edge slots to vertices.
     */
    void build_edge_map(
        const uint32_t num_active_tets,
        const uint32_t num_unique_edges,
        const uint32_t *tets,
        const uint32_t *used_tet_index,
        const uint8_t *used_tet_codes,
        const Edge *unique_edges,
        uint32_t *tet_edge_to_vert_idx)
    {
        if (num_active_tets == 0 || num_unique_edges == 0) return;
        int block_size = NTHREADS;
        int grid_size = (num_active_tets + block_size - 1) / block_size;
        build_edge_map_kernel<<<grid_size, block_size>>>(
            num_active_tets, num_unique_edges, tets, used_tet_index, used_tet_codes, unique_edges, tet_edge_to_vert_idx);
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
        const Edge *unique_edges,
        const float3 *grid_vertices,
        const float *vert_values,
        const float3 *grid_normals,
        const float3 *grid_colors,
        const float iso,
        float3 *out_verts,
        float3 *out_normals,
        float3 *out_colors)
    {
        if (num_unique_edges == 0) return;
        int block_size = NTHREADS;
        int grid_size = (num_unique_edges + block_size - 1) / block_size;
        interpolate_vertices_kernel<<<grid_size, block_size>>>(
            num_unique_edges, unique_edges, grid_vertices, vert_values, grid_normals, grid_colors, iso, out_verts, out_normals, out_colors);
    }

    /**
     * @brief Counts output triangles and prefix-sums their per-tet offsets.
     * @details Each sign code implies a fixed triangle count, so an exclusive scan over the
     * active tets gives every tet a disjoint output range. Assembly then writes without
     * atomics or contention.
     * @param[in] num_active_tets Number of active tets.
     * @param[in] used_tet_codes Device array of active sign codes.
     * @param[out] num_triangles Receives the total triangle count.
     * @param[out] tet_triangle_prefix_sums Device array of per-tet output offsets.
     * @note Synchronises to bring the total back to the host.
     */
    void compute_number_triangles(
        const uint32_t num_active_tets,
        const uint8_t *used_tet_codes,
        uint32_t &num_triangles,
        uint32_t *tet_triangle_prefix_sums)
    {
        at::cuda::ThrustAllocator allocator;
        auto policy = thrust::cuda::par(allocator);

        thrust::device_ptr<const uint8_t> d_codes(used_tet_codes);
        auto num_tris_iter = thrust::make_transform_iterator(d_codes, num_triangles_functor());

        thrust::device_ptr<uint32_t> d_prefix_sum(tet_triangle_prefix_sums);
        
        num_triangles = thrust::reduce(policy, num_tris_iter, num_tris_iter + num_active_tets);
        thrust::exclusive_scan(policy, num_tris_iter, num_tris_iter + num_active_tets, d_prefix_sum);
    }

    /**
     * @brief Launches stage 5: write the triangle index triples.
     * @details Host wrapper over the assembly kernel; each tet writes into the disjoint
     * range given by its prefix-sum offset.
     * @param[in] num_active_tets Number of active tets.
     * @param[out] out_triangles Device array of triangle vertex index triples.
     */
    void assemble_triangles(
        const uint32_t num_active_tets,
        const uint8_t *used_tet_codes,
        const uint32_t *tet_edge_to_vert_idx,
        const uint32_t *tet_triangle_prefix_sums,
        uint32_t *out_triangles)
    {
        if (num_active_tets == 0) return;
        int block_size = NTHREADS;
        int grid_size = (num_active_tets + block_size - 1) / block_size;
        assemble_triangles_kernel<<<grid_size, block_size>>>(
            num_active_tets, used_tet_codes, tet_edge_to_vert_idx, tet_triangle_prefix_sums, out_triangles);
    }

    /**
     * @brief End-to-end Marching Tetrahedra isosurface extraction on the GPU.
     * @details The host dispatcher orchestrating the whole pipeline: classify, compact, emit
     * edges, deduplicate, map, interpolate, and assemble. Output buffers are allocated through
     * PyTorch, so the extracted mesh needs no copy to be returned to Python.
     * @return A tuple of vertices, triangles, and the optional interpolated attributes.
     * @note Three device synchronisations are unavoidable -- after the active count, the unique
     * edge count, and the triangle count -- because each sizes the next allocation.
     */
    std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>> marching_tetrahedra(
        const uint32_t num_tets,
        const float3* __restrict__ grid_vertices,
        const uint32_t* __restrict__ tets,
        const float* __restrict__ vert_values,
        const float3* __restrict__ grid_normals,
        const float3* __restrict__ grid_colors,
        const float iso,
        torch::TensorOptions vert_options,
        torch::TensorOptions tri_options,
        bool return_unique_edges
    )
    {
        auto tet_codes_t = torch::empty({(int64_t)num_tets}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, c10::cuda::current_device()));
        uint8_t *__restrict__ tet_codes = tet_codes_t.data_ptr<uint8_t>();

        compute_active_tets(num_tets, tets, vert_values, iso, tet_codes);

        uint32_t num_active_tets;
        compute_number_active_tets(num_tets, tet_codes, num_active_tets);

        if (num_active_tets == 0) {
            std::optional<torch::Tensor> out_n = std::nullopt;
            std::optional<torch::Tensor> out_c = std::nullopt;
            if (grid_normals != nullptr) out_n = torch::empty({0, 3}, vert_options);
            if (grid_colors != nullptr) out_c = torch::empty({0, 3}, vert_options);
            return std::make_tuple(
                torch::empty({0, 3}, vert_options),
                torch::empty({0, 3}, tri_options),
                out_n,
                out_c,
                std::optional<torch::Tensor>(std::nullopt)
            );
        }

        auto used_tet_index_t = torch::empty({(int64_t)num_active_tets}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, c10::cuda::current_device()));
        uint32_t *__restrict__ used_tet_index = (uint32_t*)used_tet_index_t.data_ptr<int32_t>();

        auto used_tet_codes_t = torch::empty({(int64_t)num_active_tets}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, c10::cuda::current_device()));
        uint8_t *__restrict__ used_tet_codes = used_tet_codes_t.data_ptr<uint8_t>();
        compact_active_tets(num_tets, tet_codes, used_tet_index, used_tet_codes);

        auto active_edges_t = torch::empty({(int64_t)(num_active_tets * 6 * sizeof(Edge))}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, c10::cuda::current_device()));
        Edge *__restrict__ active_edges = (Edge*)active_edges_t.data_ptr<uint8_t>();
        compute_active_edges(num_active_tets, tets, used_tet_index, used_tet_codes, active_edges);

        uint32_t out_num_vertices;
        auto unique_edges_t = compute_unique_active_edges(num_active_tets, active_edges, out_num_vertices);
        Edge *__restrict__ unique_edges = (Edge*)unique_edges_t.data_ptr<uint8_t>();

        auto tet_edge_to_vert_idx_t = torch::empty({(int64_t)(num_active_tets * 6)}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, c10::cuda::current_device()));
        uint32_t *__restrict__ tet_edge_to_vert_idx = (uint32_t*)tet_edge_to_vert_idx_t.data_ptr<int32_t>();
        build_edge_map(num_active_tets, out_num_vertices, tets, used_tet_index, used_tet_codes, unique_edges, tet_edge_to_vert_idx);

        torch::Tensor out_vertices = torch::empty({(int64_t)out_num_vertices, 3}, vert_options);
        float3* __restrict__ p_out_vertices = (float3*)out_vertices.data_ptr<float>();

        std::optional<torch::Tensor> out_normals_opt = std::nullopt;
        float3* __restrict__ p_out_normals = nullptr;
        if (grid_normals != nullptr) {
            torch::Tensor out_normals = torch::empty({(int64_t)out_num_vertices, 3}, vert_options);
            p_out_normals = (float3*)out_normals.data_ptr<float>();
            out_normals_opt = out_normals;
        }

        std::optional<torch::Tensor> out_colors_opt = std::nullopt;
        float3* __restrict__ p_out_colors = nullptr;
        if (grid_colors != nullptr) {
            torch::Tensor out_colors = torch::empty({(int64_t)out_num_vertices, 3}, vert_options);
            p_out_colors = (float3*)out_colors.data_ptr<float>();
            out_colors_opt = out_colors;
        }

        interpolate_vertices(out_num_vertices, unique_edges, grid_vertices, vert_values, grid_normals, grid_colors, iso, p_out_vertices, p_out_normals, p_out_colors);

        uint32_t out_num_triangles;
        auto tet_triangle_prefix_sums_t = torch::empty({(int64_t)num_active_tets}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, c10::cuda::current_device()));
        uint32_t *__restrict__ tet_triangle_prefix_sums = (uint32_t*)tet_triangle_prefix_sums_t.data_ptr<int32_t>();
        compute_number_triangles(num_active_tets, used_tet_codes, out_num_triangles, tet_triangle_prefix_sums);

        torch::Tensor out_triangles = torch::empty({(int64_t)out_num_triangles, 3}, tri_options);
        uint32_t* __restrict__ p_out_triangles = (uint32_t*)out_triangles.data_ptr<int32_t>();

        assemble_triangles(num_active_tets, used_tet_codes, tet_edge_to_vert_idx, tet_triangle_prefix_sums, p_out_triangles);

        std::optional<torch::Tensor> out_unique_edges_opt = std::nullopt;
        if (return_unique_edges) {
            torch::Tensor out_unique_edges = torch::empty({(int64_t)out_num_vertices, 2}, torch::dtype(torch::kInt32).device(vert_options.device()));
            CHECK_CUDA_INTERNAL(cudaMemcpy(out_unique_edges.data_ptr(), unique_edges, out_num_vertices * sizeof(Edge), cudaMemcpyDeviceToDevice));
            out_unique_edges_opt = out_unique_edges;
        }

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
__global__ void backward_dmt_kernel(
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

        float3 grad_p_out = (adj_verts != nullptr) ? adj_verts[v_idx] : make_float3(0, 0, 0);

        float diff = v1_val - v0_val;
        float3 grad_c_out = make_float3(0, 0, 0);

        if (with_colors)
        {
            grad_c_out = (adj_colors != nullptr) ? adj_colors[v_idx] : make_float3(0, 0, 0);
            float t = 0.5f;
            if (fabsf(iso - v0_val) < EPS)
                t = 0.0f;
            else if (fabsf(iso - v1_val) < EPS)
                t = 1.0f;
            else if (fabsf(diff) >= EPS)
            {
                t = maths::saturate((iso - v0_val) / diff);
            }

            float t0 = 1.0f - t;
            float t1 = t;

            if (adj_grid_colors != nullptr) {
                atomicAdd(&(adj_grid_colors[v0_idx]), grad_c_out * t0);
                atomicAdd(&(adj_grid_colors[v1_idx]), grad_c_out * t1);
            }
        }

        if (diff * diff < 1e-14f)
            return;

        float dot_prod_p = (adj_verts != nullptr) ? maths::dot(p1 - p0, grad_p_out) : 0.0f;
        float dot_prod_c = 0.0f;
        if (with_colors && adj_colors != nullptr && grid_colors != nullptr) {
            float3 c0 = grid_colors[v0_idx];
            float3 c1 = grid_colors[v1_idx];
            dot_prod_c = maths::dot(c1 - c0, grad_c_out);
        }

        float dot_prod = dot_prod_p + dot_prod_c;
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

        bool with_colors = (grid_colors != nullptr);
        int block_size = NTHREADS;
        int grid_size = (n_verts + block_size - 1) / block_size;

        backward_dmt_kernel<<<grid_size, block_size>>>(
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
