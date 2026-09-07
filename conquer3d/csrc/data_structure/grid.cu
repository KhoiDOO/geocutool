/**
 * @file grid.cu
 * @brief CUDA kernel implementations for depth map unprojection and active voxel identification.
 */

#include "grid.h"
#include <stdio.h>
#include <math.h>

namespace grid {

/**
 * @brief Back-projects a depth map into the voxel indices its surface touches.
 * @details One thread per pixel. Each unprojects its depth sample through the inverse
 * intrinsics into camera space, transforms to world space, and quantises the result to a
 * voxel index -- carving a surface band directly out of a depth image without ever
 * allocating the dense volume. With @p activate_neighbor set, the $3 \times 3 \times 3$
 * neighbourhood is activated too, widening the band so later extraction has samples on
 * both sides of the surface.
 *
 * @param[in] num_pixels Number of depth samples $W \times H$.
 * @param[in] depth_image Row-major depth buffer.
 * @param[in] c2w Camera-to-world transform.
 * @param[in] intrinsics_inv Inverse camera intrinsics.
 * @param[in] image_width Depth image width in pixels.
 * @param[in] image_height Depth image height in pixels.
 * @param[in] grid_min World coordinate of the grid's lower corner.
 * @param[in] grid_max World coordinate of the grid's upper corner.
 * @param[in] res Per-axis voxel resolution.
 * @param[out] out_voxel_ids Device array receiving activated linear voxel indices.
 * @param[in,out] valid_counter Device counter, atomically incremented per emission.
 * @param[in] activate_neighbor Whether to also activate the surrounding 26 voxels.
 * @param[in] trunc_margin Band half-width in world units.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Pixels with non-positive depth are treated as missing and skipped.
 * @warning Output order is nondeterministic because slots are claimed by atomics, and the
 * same voxel may be emitted by many pixels. Callers must sort and deduplicate.
 */
__global__ void get_active_voxel_ids_from_depth_kernel(
        const int num_pixels,
        const float* depth_image,
        const float4x4 c2w,
        const float3x3 intrinsics_inv,
        const int image_width,
        const int image_height,
        const float3 grid_min,
        const float3 grid_max,
        const int3 res,
        int64_t* out_voxel_ids,
        unsigned long long* valid_counter,
        bool activate_neighbor,
        float trunc_margin
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_pixels) return;

        float d = depth_image[idx];
        if (d <= 0.0f) return; // Ignore invalid depths

        int ui = idx % image_width;
        int vi = idx / image_width;

        // 1. Unproject to camera coordinates using inverse intrinsics
        float3 p_cam;
        p_cam.x = d * (intrinsics_inv.m[0][0] * ui + intrinsics_inv.m[0][1] * vi + intrinsics_inv.m[0][2]);
        p_cam.y = d * (intrinsics_inv.m[1][0] * ui + intrinsics_inv.m[1][1] * vi + intrinsics_inv.m[1][2]);
        p_cam.z = d;

        // 2. Transform to world coordinates using Camera-To-World matrix
        float4 p_cam4 = make_float4(p_cam.x, p_cam.y, p_cam.z, 1.0f);
        float4 p_world4 = c2w * p_cam4;
        
        // 3. Compute 3D grid cell indices (0-indexed)
        float spacing_x = (grid_max.x - grid_min.x) / fmaxf(1.0f, (float)(res.x - 1));
        float spacing_y = (grid_max.y - grid_min.y) / fmaxf(1.0f, (float)(res.y - 1));
        float spacing_z = (grid_max.z - grid_min.z) / fmaxf(1.0f, (float)(res.z - 1));

        int i_min = floorf((p_world4.x - grid_min.x) / spacing_x);
        int i_max = i_min;
        int j_min = floorf((p_world4.y - grid_min.y) / spacing_y);
        int j_max = j_min;
        int k_min = floorf((p_world4.z - grid_min.z) / spacing_z);
        int k_max = k_min;

        if (activate_neighbor) {
            i_min = floorf((p_world4.x - trunc_margin - grid_min.x) / spacing_x);
            i_max = floorf((p_world4.x + trunc_margin - grid_min.x) / spacing_x);
            j_min = floorf((p_world4.y - trunc_margin - grid_min.y) / spacing_y);
            j_max = floorf((p_world4.y + trunc_margin - grid_min.y) / spacing_y);
            k_min = floorf((p_world4.z - trunc_margin - grid_min.z) / spacing_z);
            k_max = floorf((p_world4.z + trunc_margin - grid_min.z) / spacing_z);
        }

        for (int i = i_min; i <= i_max; ++i) {
            for (int j = j_min; j <= j_max; ++j) {
                for (int k = k_min; k <= k_max; ++k) {
                    if (i >= 0 && i < res.x - 1 && j >= 0 && j < res.y - 1 && k >= 0 && k < res.z - 1) {
                        int64_t voxel_id = (int64_t)i * (res.y - 1) * (res.z - 1) + (int64_t)j * (res.z - 1) + (int64_t)k;
                        unsigned long long write_idx = atomicAdd(valid_counter, 1ULL);
                        out_voxel_ids[write_idx] = voxel_id;
                    }
                }
            }
        }
    }

    void get_active_voxel_ids_from_depth(
        const int num_pixels,
        const float* depth_image,
        const float4x4 c2w,
        const float3x3 intrinsics_inv,
        const int image_width,
        const int image_height,
        const float3 grid_min,
        const float3 grid_max,
        const int3 res,
        int64_t* out_voxel_ids,
        unsigned long long* valid_counter,
        bool activate_neighbor,
        float trunc_margin
    ) {
        int block_size = 256;
        int grid_size = (num_pixels + block_size - 1) / block_size;

        get_active_voxel_ids_from_depth_kernel<<<grid_size, block_size>>>(
            num_pixels,
            depth_image,
            c2w,
            intrinsics_inv,
            image_width,
            image_height,
            grid_min,
            grid_max,
            res,
            out_voxel_ids,
            valid_counter,
            activate_neighbor,
            trunc_margin
        );
    }

/**
 * @brief Maps each vertex to the linear index of the voxel containing it.
 * @details One thread per vertex, computing $\lfloor (\mathbf{v} - \mathbf{g}_{min}) /
 * \mathbf{s} \rfloor$ and flattening the result in $x$-major order. Vertices outside the
 * grid are dropped rather than clamped, so out-of-bounds geometry cannot fold onto the
 * boundary and create phantom occupancy.
 * @param[in] vertices Device array of `num_vertices` coordinates.
 * @param[in] num_vertices Number of vertices.
 * @param[in] grid_min World coordinate of the grid's lower corner.
 * @param[in] grid_spacing Per-axis voxel size.
 * @param[in] num_cells Per-axis cell counts.
 * @param[out] out_voxel_ids Device array of `num_vertices` linear voxel indices.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Indices are `int64_t`; a $1024^3$ grid exceeds the 32-bit range.
 */
__global__ void quantize_vertices_to_voxel_ids_kernel(
        const float3* __restrict__ vertices,
        int num_vertices,
        float3 grid_min,
        float3 grid_spacing,
        int3 num_cells,
        int64_t* __restrict__ out_voxel_ids
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_vertices) return;

        float3 v = vertices[idx];
        int vi = floorf((v.x - grid_min.x) / grid_spacing.x);
        int vj = floorf((v.y - grid_min.y) / grid_spacing.y);
        int vk = floorf((v.z - grid_min.z) / grid_spacing.z);

        if (vi >= 0 && vi < num_cells.x && vj >= 0 && vj < num_cells.y && vk >= 0 && vk < num_cells.z) {
            int64_t voxel_id = (int64_t)vi * num_cells.y * num_cells.z + (int64_t)vj * num_cells.z + (int64_t)vk;
            out_voxel_ids[idx] = voxel_id;
        } else {
            out_voxel_ids[idx] = -1;
        }
    }

    torch::Tensor filter_voxels_containing_vertices(
        const torch::Tensor& active_voxel_ids,
        const torch::Tensor& vertices,
        std::vector<float> grid_min,
        std::vector<float> grid_max,
        std::vector<int64_t> res
    ) {
        if (active_voxel_ids.size(0) == 0 || vertices.size(0) == 0) {
            return active_voxel_ids;
        }

        int num_vertices = static_cast<int>(vertices.size(0));
        int64_t rx = res[0];
        int64_t ry = res[1];
        int64_t rz = res[2];

        float3 f_min = make_float3(grid_min[0], grid_min[1], grid_min[2]);
        float3 f_spacing = make_float3(
            (rx > 1) ? (grid_max[0] - grid_min[0]) / (rx - 1) : 1.0f,
            (ry > 1) ? (grid_max[1] - grid_min[1]) / (ry - 1) : 1.0f,
            (rz > 1) ? (grid_max[2] - grid_min[2]) / (rz - 1) : 1.0f
        );
        int3 num_cells = make_int3(
            static_cast<int>(rx - 1),
            static_cast<int>(ry - 1),
            static_cast<int>(rz - 1)
        );

        auto options = torch::TensorOptions().device(vertices.device()).dtype(torch::kInt64);
        auto raw_vids = torch::empty({num_vertices}, options);

        int threads = 256;
        int blocks = (num_vertices + threads - 1) / threads;

        quantize_vertices_to_voxel_ids_kernel<<<blocks, threads>>>(
            (const float3*)vertices.data_ptr<float>(),
            num_vertices,
            f_min,
            f_spacing,
            num_cells,
            raw_vids.data_ptr<int64_t>()
        );

        auto valid_vids = raw_vids.masked_select(raw_vids >= 0);
        if (valid_vids.size(0) == 0) {
            return torch::empty({0}, options);
        }

        auto unique_vids = std::get<0>(torch::_unique2(valid_vids, true, false, false));
        auto is_contained = torch::isin(active_voxel_ids, unique_vids);
        return active_voxel_ids.masked_select(is_contained);
    }

/**
 * @brief Expands each voxel centre into its eight corner coordinates.
 * @details One thread per voxel centre, writing eight corners at a fixed stride so the
 * output is a dense `(N, 8, 3)` array requiring no compaction. Corner order follows the
 * library's counter-clockwise convention, matching the topology tables the extraction
 * kernels index into.
 * @param[in] vertices Device array of `num_vertices` voxel centre coordinates.
 * @param[in] num_vertices Number of voxel centres.
 * @param[in] voxel_spacing Per-axis voxel size; corners sit half a spacing from the centre.
 * @param[out] out_corners Device array of `num_vertices * 8` corner coordinates.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Corners shared between neighbouring voxels are duplicated; deduplicate downstream
 * if a welded vertex set is required.
 */
__global__ void create_voxel_cloud_corners_kernel(
        const float3* __restrict__ vertices,
        int num_vertices,
        float3 voxel_spacing,
        float3* __restrict__ out_corners
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_vertices) return;

        float3 v = vertices[idx];
        float hx = 0.5f * voxel_spacing.x;
        float hy = 0.5f * voxel_spacing.y;
        float hz = 0.5f * voxel_spacing.z;

        int base_idx = idx * 8;
        out_corners[base_idx + 0] = make_float3(v.x - hx, v.y - hy, v.z - hz);
        out_corners[base_idx + 1] = make_float3(v.x + hx, v.y - hy, v.z - hz);
        out_corners[base_idx + 2] = make_float3(v.x + hx, v.y + hy, v.z - hz);
        out_corners[base_idx + 3] = make_float3(v.x - hx, v.y + hy, v.z - hz);
        out_corners[base_idx + 4] = make_float3(v.x - hx, v.y - hy, v.z + hz);
        out_corners[base_idx + 5] = make_float3(v.x + hx, v.y - hy, v.z + hz);
        out_corners[base_idx + 6] = make_float3(v.x + hx, v.y + hy, v.z + hz);
        out_corners[base_idx + 7] = make_float3(v.x - hx, v.y + hy, v.z + hz);
    }

    std::tuple<torch::Tensor, torch::Tensor> create_voxel_cloud_corners(
        const torch::Tensor& vertices,
        std::vector<float> grid_min,
        std::vector<float> grid_max,
        std::vector<int64_t> res
    ) {
        int num_vertices = static_cast<int>(vertices.size(0));
        int64_t rx = res[0];
        int64_t ry = res[1];
        int64_t rz = res[2];

        float3 f_spacing = make_float3(
            (rx > 1) ? (grid_max[0] - grid_min[0]) / (rx - 1) : 1.0f,
            (ry > 1) ? (grid_max[1] - grid_min[1]) / (ry - 1) : 1.0f,
            (rz > 1) ? (grid_max[2] - grid_min[2]) / (rz - 1) : 1.0f
        );

        auto options = torch::TensorOptions().device(vertices.device()).dtype(torch::kFloat32);
        auto raw_corners = torch::empty({num_vertices * 8, 3}, options);

        if (num_vertices > 0) {
            int threads = 256;
            int blocks = (num_vertices + threads - 1) / threads;

            create_voxel_cloud_corners_kernel<<<blocks, threads>>>(
                (const float3*)vertices.data_ptr<float>(),
                num_vertices,
                f_spacing,
                (float3*)raw_corners.data_ptr<float>()
            );
        }

        auto spacing_tensor = torch::tensor({f_spacing.x, f_spacing.y, f_spacing.z}, options);
        return std::make_tuple(raw_corners, spacing_tensor);
    }
}
