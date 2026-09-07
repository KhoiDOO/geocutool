#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <string>
#include <vector>
#include "../../data_structure/mesh_bvh.h"
#include "../../data_structure/triangle_mesh.h"
#include "../../data_structure/grid.h"
#include <cuda_runtime.h>

namespace py = pybind11;

/**
 * @brief Builds a dense structured voxel grid over the requested bounds and resolution.
 * @details Validates its tensor arguments, then dispatches to the CUDA implementation.
 * @return The constructed grid arrays as PyTorch tensors.
 */
std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> create_voxel_grid(
    std::vector<float> grid_min,
    std::vector<float> grid_max,
    std::vector<int64_t> res,
    std::string device_str,
    bool return_idx_grids
) {
    TORCH_CHECK(grid_min.size() == 3, "grid_min must have 3 elements.");
    TORCH_CHECK(grid_max.size() == 3, "grid_max must have 3 elements.");
    TORCH_CHECK(res.size() == 3, "res must have 3 elements.");

    c10::Device device(device_str);
    auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);

    int64_t rx = res[0];
    int64_t ry = res[1];
    int64_t rz = res[2];

    auto x = torch::linspace(grid_min[0], grid_max[0], rx, options);
    auto y = torch::linspace(grid_min[1], grid_max[1], ry, options);
    auto z = torch::linspace(grid_min[2], grid_max[2], rz, options);

    auto grids = torch::meshgrid({x, y, z}, "ij");
    auto grid_x = grids[0].flatten();
    auto grid_y = grids[1].flatten();
    auto grid_z = grids[2].flatten();
    
    auto grid_vertices = torch::stack({grid_x, grid_y, grid_z}, 1).contiguous();

    auto options_int = options.dtype(torch::kInt64);
    auto i = torch::arange(rx - 1, options_int);
    auto j = torch::arange(ry - 1, options_int);
    auto k = torch::arange(rz - 1, options_int);

    auto idx_grids = torch::meshgrid({i, j, k}, "ij");
    auto vi = idx_grids[0];
    auto vj = idx_grids[1];
    auto vk = idx_grids[2];

    // Voxel connectivity indexing mapping matching mc.cu
    auto v0 = vi * ry * rz + vj * rz + vk;
    auto v1 = (vi + 1) * ry * rz + vj * rz + vk;
    auto v2 = (vi + 1) * ry * rz + (vj + 1) * rz + vk;
    auto v3 = vi * ry * rz + (vj + 1) * rz + vk;
    auto v4 = vi * ry * rz + vj * rz + (vk + 1);
    auto v5 = (vi + 1) * ry * rz + vj * rz + (vk + 1);
    auto v6 = (vi + 1) * ry * rz + (vj + 1) * rz + (vk + 1);
    auto v7 = vi * ry * rz + (vj + 1) * rz + (vk + 1);

    auto voxels = torch::stack({v0, v1, v2, v3, v4, v5, v6, v7}, -1).view({-1, 8}).contiguous().to(torch::kInt32);

    std::optional<torch::Tensor> opt_idx_grids = std::nullopt;
    if (return_idx_grids) {
        auto i_v = torch::arange(rx, options_int);
        auto j_v = torch::arange(ry, options_int);
        auto k_v = torch::arange(rz, options_int);
        auto idx_grids_v = torch::meshgrid({i_v, j_v, k_v}, "ij");
        opt_idx_grids = torch::stack({idx_grids_v[0].flatten(), idx_grids_v[1].flatten(), idx_grids_v[2].flatten()}, -1).contiguous();
    }

    return std::make_tuple(grid_vertices, voxels, opt_idx_grids);
}

/**
 * @brief Evaluates per-grid-vertex normals from face normals, smooth normals, or the field gradient.
 * @details Validates its tensor arguments, then dispatches to the CUDA implementation.
 * @return The constructed grid arrays as PyTorch tensors.
 */
torch::Tensor compute_grid_normal(torch::Tensor sdf, torch::Tensor grid_vertices, torch::Tensor idx_grids, std::vector<int64_t> res) {
    TORCH_CHECK(res.size() == 3, "res must have 3 elements.");
    
    auto i = idx_grids.select(1, 0);
    auto j = idx_grids.select(1, 1);
    auto k = idx_grids.select(1, 2);
    
    auto rx = res[0];
    auto ry = res[1];
    auto rz = res[2];
    
    auto i_prev = torch::clamp(i - 1, 0, rx - 1);
    auto i_next = torch::clamp(i + 1, 0, rx - 1);
    
    auto j_prev = torch::clamp(j - 1, 0, ry - 1);
    auto j_next = torch::clamp(j + 1, 0, ry - 1);
    
    auto k_prev = torch::clamp(k - 1, 0, rz - 1);
    auto k_next = torch::clamp(k + 1, 0, rz - 1);
    
    auto idx_x_prev = i_prev * ry * rz + j * rz + k;
    auto idx_x_next = i_next * ry * rz + j * rz + k;
    
    auto idx_y_prev = i * ry * rz + j_prev * rz + k;
    auto idx_y_next = i * ry * rz + j_next * rz + k;
    
    auto idx_z_prev = i * ry * rz + j * rz + k_prev;
    auto idx_z_next = i * ry * rz + j * rz + k_next;
    
    auto sdf_flat = sdf.flatten();
    
    auto diff_x = sdf_flat.index_select(0, idx_x_next) - sdf_flat.index_select(0, idx_x_prev);
    auto diff_y = sdf_flat.index_select(0, idx_y_next) - sdf_flat.index_select(0, idx_y_prev);
    auto diff_z = sdf_flat.index_select(0, idx_z_next) - sdf_flat.index_select(0, idx_z_prev);

    auto vx = grid_vertices.select(1, 0);
    auto vy = grid_vertices.select(1, 1);
    auto vz = grid_vertices.select(1, 2);

    auto dx = vx.index_select(0, idx_x_next) - vx.index_select(0, idx_x_prev);
    auto dy = vy.index_select(0, idx_y_next) - vy.index_select(0, idx_y_prev);
    auto dz = vz.index_select(0, idx_z_next) - vz.index_select(0, idx_z_prev);

    dx = torch::where(dx == 0, torch::tensor(1e-5f, dx.options()), dx);
    dy = torch::where(dy == 0, torch::tensor(1e-5f, dy.options()), dy);
    dz = torch::where(dz == 0, torch::tensor(1e-5f, dz.options()), dz);

    auto grad_x = diff_x / dx;
    auto grad_y = diff_y / dy;
    auto grad_z = diff_z / dz;
    
    auto normals = torch::stack({grad_x, grad_y, grad_z}, 1);
    return torch::nn::functional::normalize(normals, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
}

/**
 * @brief Selects the voxels a surface actually crosses, discarding the rest.
 * @details Validates its tensor arguments, then dispatches to the CUDA implementation.
 * @return The constructed grid arrays as PyTorch tensors.
 */
torch::Tensor compute_active_voxels(torch::Tensor voxels, torch::Tensor sdf, float iso) {
    TORCH_CHECK(voxels.device() == sdf.device(), "voxels and sdf must be on the same device.");
    TORCH_CHECK(voxels.dtype() == torch::kInt32, "voxels must be int32.");
    TORCH_CHECK(sdf.dtype() == torch::kFloat32, "sdf must be float32.");
    
    auto voxels_flat = voxels.flatten().to(torch::kInt64);
    auto voxel_sdfs = sdf.index_select(0, voxels_flat).view({-1, 8});
    
    auto below_iso = voxel_sdfs < iso;
    auto any_below = below_iso.any(1);
    auto any_above = (~below_iso).any(1);
    
    auto active_mask = any_below & any_above;
    auto active_indices = torch::nonzero(active_mask).squeeze(1);
    
    return active_indices;
}

/**
 * @brief Builds a narrow-band sparse voxel grid hugging a triangle mesh surface.
 * @details Validates its tensor arguments, then dispatches to the CUDA implementation.
 * @return The constructed grid arrays as PyTorch tensors.
 */
std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>> create_voxel_grid_from_tmesh(
    std::vector<float> grid_min,
    std::vector<float> grid_max,
    std::vector<int64_t> res,
    TriangleMesh &tmesh,
    bool return_unique_vert_ids = true,
    int pad = 0,
    bool return_normals = false,
    int normal_mode = 0,
    bool drop_empty_vertex_voxels = false
) {
    TORCH_CHECK(grid_min.size() == 3, "grid_min must have 3 elements.");
    TORCH_CHECK(grid_max.size() == 3, "grid_max must have 3 elements.");
    TORCH_CHECK(res.size() == 3, "res must have 3 elements.");

    int64_t rx = res[0];
    int64_t ry = res[1];
    int64_t rz = res[2];

    MeshBVH bvh = tmesh.build_bvh();
    auto vertices = tmesh.get_vertices();
    auto triangles = tmesh.get_triangles();

    auto active_voxel_ids = bvh.get_active_voxel_ids_from_grid(grid_min, grid_max, res, vertices, triangles);

    if (active_voxel_ids.size(0) == 0) {
        return std::make_tuple(
            torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device())),
            torch::empty({0, 8}, torch::TensorOptions().dtype(torch::kInt32).device(vertices.device())),
            return_unique_vert_ids ? std::make_optional(torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(vertices.device()))) : std::nullopt,
            return_normals ? std::make_optional(torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device()))) : std::nullopt
        );
    }

    if (pad > 0) {
        int64_t nrx = rx - 1;
        int64_t nry = ry - 1;
        int64_t nrz = rz - 1;

        auto vi = active_voxel_ids.div(nry * nrz, "trunc");
        auto rem = active_voxel_ids.remainder(nry * nrz);
        auto vj = rem.div(nrz, "trunc");
        auto vk = rem.remainder(nrz);

        std::vector<torch::Tensor> dilated_list;
        for (int dx = -pad; dx <= pad; ++dx) {
            for (int dy = -pad; dy <= pad; ++dy) {
                for (int dz = -pad; dz <= pad; ++dz) {
                    auto n_vi = vi + dx;
                    auto n_vj = vj + dy;
                    auto n_vk = vk + dz;
                    auto mask = (n_vi >= 0) & (n_vi < nrx) & (n_vj >= 0) & (n_vj < nry) & (n_vk >= 0) & (n_vk < nrz);
                    auto valid_vi = n_vi.masked_select(mask);
                    auto valid_vj = n_vj.masked_select(mask);
                    auto valid_vk = n_vk.masked_select(mask);
                    auto n_ids = valid_vi * (nry * nrz) + valid_vj * nrz + valid_vk;
                    dilated_list.push_back(n_ids);
                }
            }
        }
        auto all_dilated = torch::cat(dilated_list);
        active_voxel_ids = std::get<0>(torch::_unique2(all_dilated, true, false, false));
    }

    if (drop_empty_vertex_voxels) {
        active_voxel_ids = grid::filter_voxels_containing_vertices(active_voxel_ids, vertices, grid_min, grid_max, res);
        if (active_voxel_ids.size(0) == 0) {
            return std::make_tuple(
                torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device())),
                torch::empty({0, 8}, torch::TensorOptions().dtype(torch::kInt32).device(vertices.device())),
                return_unique_vert_ids ? std::make_optional(torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(vertices.device()))) : std::nullopt,
                return_normals ? std::make_optional(torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device()))) : std::nullopt
            );
        }
    }

    auto vi = active_voxel_ids.div((ry - 1) * (rz - 1), "trunc");
    auto rem = active_voxel_ids.remainder((ry - 1) * (rz - 1));
    auto vj = rem.div(rz - 1, "trunc");
    auto vk = rem.remainder(rz - 1);

    auto v0 = vi * ry * rz + vj * rz + vk;
    auto v1 = (vi + 1) * ry * rz + vj * rz + vk;
    auto v2 = (vi + 1) * ry * rz + (vj + 1) * rz + vk;
    auto v3 = vi * ry * rz + (vj + 1) * rz + vk;
    auto v4 = vi * ry * rz + vj * rz + (vk + 1);
    auto v5 = (vi + 1) * ry * rz + vj * rz + (vk + 1);
    auto v6 = (vi + 1) * ry * rz + (vj + 1) * rz + (vk + 1);
    auto v7 = vi * ry * rz + (vj + 1) * rz + (vk + 1);

    auto active_voxels = torch::stack({v0, v1, v2, v3, v4, v5, v6, v7}, -1);

    torch::Tensor unique_vert_ids, inverse_indices, counts;
    std::tie(unique_vert_ids, inverse_indices, counts) = torch::_unique2(active_voxels.flatten(), true, true, false);

    auto remapped_voxels = inverse_indices.view({-1, 8}).to(torch::kInt32);

    auto u_i = unique_vert_ids.div(ry * rz, "trunc");
    auto u_rem = unique_vert_ids.remainder(ry * rz);
    auto u_j = u_rem.div(rz, "trunc");
    auto u_k = u_rem.remainder(rz);

    float spacing_x = (grid_max[0] - grid_min[0]) / (rx - 1);
    float spacing_y = (grid_max[1] - grid_min[1]) / (ry - 1);
    float spacing_z = (grid_max[2] - grid_min[2]) / (rz - 1);

    auto x = grid_min[0] + u_i.to(torch::kFloat32) * spacing_x;
    auto y = grid_min[1] + u_j.to(torch::kFloat32) * spacing_y;
    auto z = grid_min[2] + u_k.to(torch::kFloat32) * spacing_z;

    auto sparse_grid_vertices = torch::stack({x, y, z}, 1).contiguous();

    std::optional<torch::Tensor> grid_normals = std::nullopt;
    if (return_normals) {
        if (sparse_grid_vertices.size(0) > 0) {
            if (normal_mode == 0) {
                // Mode 0: Closest Triangle Face Normal (Exact for sharp CAD creases)
                auto query_res = bvh.query_point(
                    sparse_grid_vertices, vertices, triangles,
                    false, false, 0
                );
                auto closest_tri_ids = std::get<1>(query_res);
                auto tri_normals = tmesh.get_triangle_normals();
                grid_normals = tri_normals.index_select(0, closest_tri_ids);
            } else if (normal_mode == 1) {
                // Mode 1: Barycentric Interpolated Vertex Normal (Smooth shading)
                auto query_res = bvh.query_point(
                    sparse_grid_vertices, vertices, triangles,
                    false, true, 0
                );
                auto closest_tri_ids = std::get<1>(query_res);
                auto prj_pts = std::get<2>(query_res);
                auto vert_normals = tmesh.get_vertex_normals(1);

                auto tri_indices = triangles.index_select(0, closest_tri_ids);
                auto i0 = tri_indices.select(1, 0).to(torch::kInt64);
                auto i1 = tri_indices.select(1, 1).to(torch::kInt64);
                auto i2 = tri_indices.select(1, 2).to(torch::kInt64);

                auto p0 = vertices.index_select(0, i0);
                auto p1 = vertices.index_select(0, i1);
                auto p2 = vertices.index_select(0, i2);

                auto n0 = vert_normals.index_select(0, i0);
                auto n1 = vert_normals.index_select(0, i1);
                auto n2 = vert_normals.index_select(0, i2);

                auto e0 = p1 - p0;
                auto e1 = p2 - p0;
                auto e2 = prj_pts - p0;

                auto d00 = (e0 * e0).sum(-1, true);
                auto d01 = (e0 * e1).sum(-1, true);
                auto d11 = (e1 * e1).sum(-1, true);
                auto d20 = (e2 * e0).sum(-1, true);
                auto d21 = (e2 * e1).sum(-1, true);

                auto denom = (d00 * d11 - d01 * d01).clamp_min(1e-8f);
                auto v_coord = ((d11 * d20 - d01 * d21) / denom).clamp(0.0f, 1.0f);
                auto w_coord = ((d00 * d21 - d01 * d20) / denom).clamp(0.0f, 1.0f);
                auto u_coord = (1.0f - v_coord - w_coord).clamp(0.0f, 1.0f);
                auto sum_coord = (u_coord + v_coord + w_coord).clamp_min(1e-8f);
                u_coord = u_coord / sum_coord;
                v_coord = v_coord / sum_coord;
                w_coord = w_coord / sum_coord;

                auto interp_n = u_coord * n0 + v_coord * n1 + w_coord * n2;
                auto n_len = torch::norm(interp_n, 2, -1, true).clamp_min(1e-8f);
                grid_normals = interp_n / n_len;
            } else if (normal_mode == 2) {
                // Mode 2: Normalized Displacement Vector (SDF Gradient)
                auto query_res = bvh.query_point(
                    sparse_grid_vertices, vertices, triangles,
                    false, true, 0
                );
                auto prj_pts = std::get<2>(query_res);
                auto diff = sparse_grid_vertices - prj_pts;
                auto dist = torch::norm(diff, 2, -1, true).clamp_min(1e-8f);
                grid_normals = diff / dist;
            } else {
                throw std::runtime_error("Unknown normal_mode. Supported modes are 0 (closest triangle face normal), 1 (interpolated vertex normal), 2 (displacement gradient vector).");
            }
        } else {
            grid_normals = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device()));
        }
    }

    return std::make_tuple(
        sparse_grid_vertices,
        remapped_voxels,
        return_unique_vert_ids ? std::make_optional(unique_vert_ids) : std::nullopt,
        return_normals ? grid_normals : std::nullopt
    );
}

/**
 * @brief Builds a point cloud of voxel centres covering a triangle mesh surface.
 * @details Validates its tensor arguments, then dispatches to the CUDA implementation.
 * @return The constructed grid arrays as PyTorch tensors.
 */
std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>> create_voxel_cloud_from_tmesh(
    std::vector<float> grid_min,
    std::vector<float> grid_max,
    std::vector<int64_t> res,
    TriangleMesh &tmesh,
    bool return_unique_vert_ids = true,
    bool return_normals = false,
    int normal_mode = 0
) {
    TORCH_CHECK(grid_min.size() == 3, "grid_min must have 3 elements.");
    TORCH_CHECK(grid_max.size() == 3, "grid_max must have 3 elements.");
    TORCH_CHECK(res.size() == 3, "res must have 3 elements.");

    auto vertices = tmesh.get_vertices();
    auto triangles = tmesh.get_triangles();

    if (vertices.size(0) == 0) {
        return std::make_tuple(
            torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device())),
            torch::empty({0, 8}, torch::TensorOptions().dtype(torch::kInt32).device(vertices.device())),
            return_unique_vert_ids ? std::make_optional(torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(vertices.device()))) : std::nullopt,
            return_normals ? std::make_optional(torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device()))) : std::nullopt
        );
    }

    // 1. Generate 8 vertex-centered corner coordinates on GPU using custom CUDA kernel
    torch::Tensor raw_corners, spacing_tensor;
    std::tie(raw_corners, spacing_tensor) = grid::create_voxel_cloud_corners(vertices, grid_min, grid_max, res);

    // 2. Deduplicate 3D corner positions and re-index voxel cell corner connectivity
    torch::Tensor unique_grid_vertices, inverse_indices, counts;
    std::tie(unique_grid_vertices, inverse_indices, counts) = at::unique_dim(raw_corners, 0, true, true, false);

    auto remapped_voxels = inverse_indices.view({-1, 8}).to(torch::kInt32);

    // 3. Optional Surface Normal Evaluation
    std::optional<torch::Tensor> grid_normals = std::nullopt;
    if (return_normals) {
        if (unique_grid_vertices.size(0) > 0) {
            MeshBVH bvh = tmesh.build_bvh();
            if (normal_mode == 0) {
                auto query_res = bvh.query_point(
                    unique_grid_vertices, vertices, triangles,
                    false, false, 0
                );
                auto closest_tri_ids = std::get<1>(query_res);
                auto tri_normals = tmesh.get_triangle_normals();
                grid_normals = tri_normals.index_select(0, closest_tri_ids);
            } else if (normal_mode == 1) {
                auto query_res = bvh.query_point(
                    unique_grid_vertices, vertices, triangles,
                    false, true, 0
                );
                auto closest_tri_ids = std::get<1>(query_res);
                auto prj_pts = std::get<2>(query_res);
                auto vert_normals = tmesh.get_vertex_normals(1);

                auto tri_indices = triangles.index_select(0, closest_tri_ids);
                auto i0 = tri_indices.select(1, 0).to(torch::kInt64);
                auto i1 = tri_indices.select(1, 1).to(torch::kInt64);
                auto i2 = tri_indices.select(1, 2).to(torch::kInt64);

                auto p0 = vertices.index_select(0, i0);
                auto p1 = vertices.index_select(0, i1);
                auto p2 = vertices.index_select(0, i2);

                auto n0 = vert_normals.index_select(0, i0);
                auto n1 = vert_normals.index_select(0, i1);
                auto n2 = vert_normals.index_select(0, i2);

                auto e0 = p1 - p0;
                auto e1 = p2 - p0;
                auto e2 = prj_pts - p0;

                auto d00 = (e0 * e0).sum(-1, true);
                auto d01 = (e0 * e1).sum(-1, true);
                auto d11 = (e1 * e1).sum(-1, true);
                auto d20 = (e2 * e0).sum(-1, true);
                auto d21 = (e2 * e1).sum(-1, true);

                auto denom = (d00 * d11 - d01 * d01).clamp_min(1e-8f);
                auto v_coord = ((d11 * d20 - d01 * d21) / denom).clamp(0.0f, 1.0f);
                auto w_coord = ((d00 * d21 - d01 * d20) / denom).clamp(0.0f, 1.0f);
                auto u_coord = (1.0f - v_coord - w_coord).clamp(0.0f, 1.0f);
                auto sum_coord = (u_coord + v_coord + w_coord).clamp_min(1e-8f);
                u_coord = u_coord / sum_coord;
                v_coord = v_coord / sum_coord;
                w_coord = w_coord / sum_coord;

                auto interp_normals = u_coord * n0 + v_coord * n1 + w_coord * n2;
                auto norm_len = interp_normals.norm(2, -1, true).clamp_min(1e-8f);
                grid_normals = interp_normals / norm_len;
            } else if (normal_mode == 2) {
                auto query_res = bvh.query_point(
                    unique_grid_vertices, vertices, triangles,
                    false, true, 0
                );
                auto prj_pts = std::get<2>(query_res);
                auto disp = prj_pts - unique_grid_vertices;
                auto dist = disp.norm(2, -1, true).clamp_min(1e-8f);
                grid_normals = disp / dist;
            } else {
                throw std::runtime_error("Unknown normal_mode. Supported modes are 0, 1, 2.");
            }
        } else {
            grid_normals = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device()));
        }
    }

    auto vert_ids = torch::arange(vertices.size(0), torch::TensorOptions().device(vertices.device()).dtype(torch::kInt64));

    return std::make_tuple(
        unique_grid_vertices,
        remapped_voxels,
        return_unique_vert_ids ? std::make_optional(vert_ids) : std::nullopt,
        return_normals ? grid_normals : std::nullopt
    );
}

/**
 * @brief Tensor-level entry point for the depth-map voxel activation query.
 * @details Sits between pybind11 and the host dispatcher: it applies the `CHECK_INPUT`
 * contract -- CUDA device, contiguous layout, expected dtype -- then unwraps
 * `data_ptr` and calls the kernel launcher. Validating here keeps the launch path
 * free of checks and gives Python callers a clear error instead of a device fault.
 * @return The operator's results as PyTorch tensors.
 */
torch::Tensor get_active_voxel_ids_from_depth_py(
    torch::Tensor depth_image,
    torch::Tensor c2w_tensor,
    torch::Tensor intrinsics_inv_tensor,
    std::vector<float> grid_min,
    std::vector<float> grid_max,
    std::vector<int64_t> res,
    bool activate_neighbor = false,
    float trunc_margin = 0.0f
) {
    TORCH_CHECK(depth_image.is_cuda(), "depth_image must be a CUDA tensor");
    TORCH_CHECK(depth_image.dim() == 2, "depth_image must be 2D (H, W)");
    
    int image_height = depth_image.size(0);
    int image_width = depth_image.size(1);
    int num_pixels = image_height * image_width;
    
    float4x4 c2w;
    auto c2w_cpu = c2w_tensor.to(torch::kFloat32).cpu().contiguous();
    auto c2w_a = c2w_cpu.accessor<float, 2>();
    for(int r=0; r<4; ++r) for(int c=0; c<4; ++c) c2w.m[r][c] = c2w_a[r][c];
    
    float3x3 intrinsics_inv;
    auto int_cpu = intrinsics_inv_tensor.to(torch::kFloat32).cpu().contiguous();
    auto int_a = int_cpu.accessor<float, 2>();
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) intrinsics_inv.m[r][c] = int_a[r][c];
    
    float3 g_min; g_min.x = grid_min[0]; g_min.y = grid_min[1]; g_min.z = grid_min[2];
    float3 g_max; g_max.x = grid_max[0]; g_max.y = grid_max[1]; g_max.z = grid_max[2];
    int3 g_res; g_res.x = res[0]; g_res.y = res[1]; g_res.z = res[2];
    
    int max_pad_x = std::ceil(trunc_margin / ((grid_max[0] - grid_min[0]) / (res[0] - 1)));
    int max_pad_y = std::ceil(trunc_margin / ((grid_max[1] - grid_min[1]) / (res[1] - 1)));
    int max_pad_z = std::ceil(trunc_margin / ((grid_max[2] - grid_min[2]) / (res[2] - 1)));
    int max_neighbors = activate_neighbor ? ((2 * max_pad_x + 1) * (2 * max_pad_y + 1) * (2 * max_pad_z + 1)) : 1;
    
    int64_t max_voxels = (int64_t)num_pixels * max_neighbors;
    auto options = torch::TensorOptions().device(depth_image.device()).dtype(torch::kInt64);
    auto out_voxel_ids = torch::empty({max_voxels}, options);
    
    auto valid_counter_tensor = torch::zeros({1}, options);

    grid::get_active_voxel_ids_from_depth(
        num_pixels,
        depth_image.data_ptr<float>(),
        c2w,
        intrinsics_inv,
        image_width,
        image_height,
        g_min,
        g_max,
        g_res,
        out_voxel_ids.data_ptr<int64_t>(),
        reinterpret_cast<unsigned long long*>(valid_counter_tensor.data_ptr<int64_t>()),
        activate_neighbor,
        trunc_margin
    );
    
    int64_t valid_count = valid_counter_tensor.item<int64_t>();
    return out_voxel_ids.slice(0, 0, valid_count);
}

/**
 * @brief Assembles a sparse grid's vertex and voxel arrays from a set of active voxel indices.
 * @details Validates its tensor arguments, then dispatches to the CUDA implementation.
 * @return The constructed grid arrays as PyTorch tensors.
 */
std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> build_sparse_grid_from_active_voxels(
    torch::Tensor active_voxel_ids,
    std::vector<float> grid_min,
    std::vector<float> grid_max,
    std::vector<int64_t> res,
    bool return_unique_vert_ids = true
) {
    TORCH_CHECK(grid_min.size() == 3, "grid_min must have 3 elements.");
    TORCH_CHECK(grid_max.size() == 3, "grid_max must have 3 elements.");
    TORCH_CHECK(res.size() == 3, "res must have 3 elements.");

    int64_t rx = res[0];
    int64_t ry = res[1];
    int64_t rz = res[2];

    if (active_voxel_ids.size(0) == 0) {
        return std::make_tuple(
            torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(active_voxel_ids.device())),
            torch::empty({0, 8}, torch::TensorOptions().dtype(torch::kInt32).device(active_voxel_ids.device())),
            return_unique_vert_ids ? std::make_optional(torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(active_voxel_ids.device()))) : std::nullopt
        );
    }

    auto vi = active_voxel_ids.div((ry - 1) * (rz - 1), "trunc");
    auto rem = active_voxel_ids.remainder((ry - 1) * (rz - 1));
    auto vj = rem.div(rz - 1, "trunc");
    auto vk = rem.remainder(rz - 1);

    auto v0 = vi * ry * rz + vj * rz + vk;
    auto v1 = (vi + 1) * ry * rz + vj * rz + vk;
    auto v2 = (vi + 1) * ry * rz + (vj + 1) * rz + vk;
    auto v3 = vi * ry * rz + (vj + 1) * rz + vk;
    auto v4 = vi * ry * rz + vj * rz + (vk + 1);
    auto v5 = (vi + 1) * ry * rz + vj * rz + (vk + 1);
    auto v6 = (vi + 1) * ry * rz + (vj + 1) * rz + (vk + 1);
    auto v7 = vi * ry * rz + (vj + 1) * rz + (vk + 1);

    auto active_voxels = torch::stack({v0, v1, v2, v3, v4, v5, v6, v7}, -1);

    torch::Tensor unique_vert_ids, inverse_indices, counts;
    std::tie(unique_vert_ids, inverse_indices, counts) = torch::_unique2(active_voxels.flatten(), true, true, false);

    auto remapped_voxels = inverse_indices.view({-1, 8}).to(torch::kInt32);

    auto u_i = unique_vert_ids.div(ry * rz, "trunc");
    auto u_rem = unique_vert_ids.remainder(ry * rz);
    auto u_j = u_rem.div(rz, "trunc");
    auto u_k = u_rem.remainder(rz);

    float spacing_x = (grid_max[0] - grid_min[0]) / (rx - 1);
    float spacing_y = (grid_max[1] - grid_min[1]) / (ry - 1);
    float spacing_z = (grid_max[2] - grid_min[2]) / (rz - 1);

    auto x = grid_min[0] + u_i.to(torch::kFloat32) * spacing_x;
    auto y = grid_min[1] + u_j.to(torch::kFloat32) * spacing_y;
    auto z = grid_min[2] + u_k.to(torch::kFloat32) * spacing_z;

    auto sparse_grid_vertices = torch::stack({x, y, z}, 1).contiguous();

    return std::make_tuple(sparse_grid_vertices, remapped_voxels, return_unique_vert_ids ? std::make_optional(unique_vert_ids) : std::nullopt);
}

/**
 * @brief Registers voxel grid construction and depth-map carving functions on the extension module.
 * @details Called once from `pybind.cpp` with the root module, so every symbol
 * defined here lands directly on `conquer3d._C`.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ds_grid(py::module_& m) {
    m.def("create_voxel_grid", &create_voxel_grid,
          py::arg("grid_min"), py::arg("grid_max"), py::arg("res"),
          py::arg("device_str") = "cuda", py::arg("return_idx_grids") = true,
          R"pbdoc(
          Creates a dense structured 3D voxel grid.

          Args:
              grid_min (List[float]): Lower bounding coordinates [x_min, y_min, z_min].
              grid_max (List[float]): Upper bounding coordinates [x_max, y_max, z_max].
              res (List[int]): Grid resolution [rx, ry, rz].
              device_str (str, optional): Target compute device. Defaults to "cuda".
              return_idx_grids (bool, optional): Return discrete 3D index grids. Defaults to True.

          Returns:
              Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                  - grid_vertices (torch.Tensor): (rx*ry*rz, 3) float32 coordinates.
                  - voxels (torch.Tensor): ((rx-1)*(ry-1)*(rz-1), 8) int64 corner index mapping.
                  - [idx_grids] (torch.Tensor, optional): (3, rx, ry, rz) int64 index tensor.

          Example:
              >>> verts, voxels, idx_grids = create_voxel_grid([-1,-1,-1], [1,1,1], [64,64,64])
          )pbdoc");
    m.def("create_voxel_grid_from_tmesh", &create_voxel_grid_from_tmesh,
          py::arg("grid_min"), py::arg("grid_max"), py::arg("res"), py::arg("tmesh"),
          py::arg("return_unique_vert_ids") = true, py::arg("pad") = 0,
          py::arg("return_normals") = false, py::arg("normal_mode") = 0,
          py::arg("drop_empty_vertex_voxels") = false,
          R"pbdoc(
          Creates a memory-efficient sparse 3D voxel grid strictly intersecting or bordering the input triangle mesh.

          Args:
              grid_min (List[float]): Lower bounding coordinates [x_min, y_min, z_min].
              grid_max (List[float]): Upper bounding coordinates [x_max, y_max, z_max].
              res (List[int]): Grid resolution [rx, ry, rz].
              tmesh (TriangleMesh): Input TriangleMesh GPU structure.
              return_unique_vert_ids (bool, optional): Return original linear vertex IDs. Defaults to True.
              pad (int, optional): Voxel layer dilation radius. Defaults to 0.
              return_normals (bool, optional): Return surface normals at sparse vertices. Defaults to False.
              normal_mode (int, optional): Normal mode (0: face normals, 1: vertex normals, 2: displacement vector). Defaults to 0.
              drop_empty_vertex_voxels (bool, optional): If True, drops voxels containing no mesh vertices inside their 3D bounding box. Defaults to False.

          Returns:
              Tuple: Sparse grid vertices, remapped voxels, and optional vertex IDs/normals.

          Example:
              >>> sparse_verts, voxels, vert_ids = create_voxel_grid_from_tmesh([-1,-1,-1], [1,1,1], [128,128,128], tmesh)
          )pbdoc");
    m.def("create_voxel_cloud_from_tmesh", &create_voxel_cloud_from_tmesh,
          py::arg("grid_min"), py::arg("grid_max"), py::arg("res"), py::arg("tmesh"),
          py::arg("return_unique_vert_ids") = true,
          py::arg("return_normals") = false, py::arg("normal_mode") = 0,
          R"pbdoc(
          Creates an overlapping 3D voxel cloud where each 3D voxel cell is centered directly at a mesh vertex.

          Args:
              grid_min (List[float]): Lower bounding coordinates [x_min, y_min, z_min].
              grid_max (List[float]): Upper bounding coordinates [x_max, y_max, z_max].
              res (List[int]): Grid resolution [rx, ry, rz].
              tmesh (TriangleMesh): Input TriangleMesh GPU structure.
              return_unique_vert_ids (bool, optional): Return original linear vertex IDs. Defaults to True.
              return_normals (bool, optional): Return surface normals at sparse vertices. Defaults to False.
              normal_mode (int, optional): Normal mode (0: face normals, 1: vertex normals, 2: displacement vector). Defaults to 0.

          Returns:
              Tuple: Sparse grid vertices, remapped voxels, and optional vertex IDs/normals.

          Example:
              >>> cloud_verts, voxels, vert_ids = create_voxel_cloud_from_tmesh([-1,-1,-1], [1,1,1], [128,128,128], tmesh)
          )pbdoc");
    m.def("compute_grid_normal", &compute_grid_normal,
          py::arg("sdf"), py::arg("grid_vertices"), py::arg("idx_grids"), py::arg("res"),
          R"pbdoc(
          Computes surface normal vectors on a 3D scalar grid via central finite differences.

          Args:
              sdf (torch.Tensor): (N,) float32 scalar values on CUDA.
              grid_vertices (torch.Tensor): (N, 3) float32 coordinates on CUDA.
              idx_grids (torch.Tensor): (3, rx, ry, rz) int64 index map.
              res (List[int]): Grid resolution [rx, ry, rz].

          Returns:
              torch.Tensor: (N, 3) float32 normalized outward unit normal vectors.

          Example:
              >>> normals = compute_grid_normal(sdf, verts, idx_grids, [64, 64, 64])
          )pbdoc");
    m.def("compute_active_voxels", &compute_active_voxels,
          py::arg("voxels"), py::arg("sdf"), py::arg("iso"),
          R"pbdoc(
          Computes boolean active mask for voxels intersecting the zero-crossing isosurface.

          Args:
              voxels (torch.Tensor): (M, 8) int32/int64 voxel corner indices.
              sdf (torch.Tensor): (N,) float32 scalar field values.
              iso (float): Isosurface extraction threshold.

          Returns:
              torch.Tensor: (M,) bool mask where True indicates voxel has sign changes.

          Example:
              >>> active_mask = compute_active_voxels(voxels, sdf, iso=0.0)
          )pbdoc");
    m.def("get_active_voxel_ids_from_depth", &get_active_voxel_ids_from_depth_py,
          py::arg("depth_image"), py::arg("c2w_tensor"), py::arg("intrinsics_inv_tensor"),
          py::arg("grid_min"), py::arg("grid_max"), py::arg("res"),
          py::arg("activate_neighbor") = false, py::arg("trunc_margin") = 0.0f,
          R"pbdoc(
          Extracts active voxel IDs by unprojecting an RGB-D depth map on GPU.

          Args:
              depth_image (torch.Tensor): (H, W) float32 depth map in meters on CUDA.
              c2w_tensor (torch.Tensor): (4, 4) float32 Camera-to-World matrix.
              intrinsics_inv_tensor (torch.Tensor): (3, 3) float32 inverse intrinsics matrix.
              grid_min (List[float]): Lower grid extents [x_min, y_min, z_min].
              grid_max (List[float]): Upper grid extents [x_max, y_max, z_max].
              res (List[int]): Grid resolution [rx, ry, rz].
              activate_neighbor (bool, optional): Dilation into adjacent voxel cells. Defaults to False.
              trunc_margin (float, optional): Truncation band thickness in meters. Defaults to 0.0.

          Returns:
              torch.Tensor: (K,) int64 sorted unique active voxel IDs.

          Example:
              >>> active_vids = get_active_voxel_ids_from_depth(depth, c2w, k_inv, [-1,-1,-1], [1,1,1], [128,128,128])
          )pbdoc");
    m.def("build_sparse_grid_from_active_voxels", &build_sparse_grid_from_active_voxels,
          py::arg("active_voxel_ids"), py::arg("grid_min"), py::arg("grid_max"), py::arg("res"),
          py::arg("return_unique_vert_ids") = true,
          R"pbdoc(
          Reconstructs sparse grid vertices and re-indexed voxel connectivity from 1D active voxel IDs.

          Args:
              active_voxel_ids (torch.Tensor): (K,) int64 active voxel cell IDs.
              grid_min (List[float]): Lower bounding coordinates [x_min, y_min, z_min].
              grid_max (List[float]): Upper bounding coordinates [x_max, y_max, z_max].
              res (List[int]): Grid resolution [rx, ry, rz].
              return_unique_vert_ids (bool, optional): Return original linear vertex IDs. Defaults to True.

          Returns:
              Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                  - sparse_grid_vertices (torch.Tensor): (V_sparse, 3) float32 coordinates.
                  - remapped_voxels (torch.Tensor): (K, 8) int32 re-indexed voxel corners.
                  - [unique_vert_ids] (torch.Tensor, optional): (V_sparse,) int64 original vertex IDs.

          Example:
              >>> s_verts, s_voxels, u_vids = build_sparse_grid_from_active_voxels(vids, [-1,-1,-1], [1,1,1], [128,128,128])
          )pbdoc");
}
