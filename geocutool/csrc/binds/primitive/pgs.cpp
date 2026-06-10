#include <torch/extension.h>
#include "../../primitive/pgs.h"
#include "../../check.h"

#include <cuda_fp16.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>> query_pgs_voxel_pair_intersection_brute_force_wrapper(
    const torch::Tensor &vx_aabb_mins,
    const torch::Tensor &vx_aabb_maxs,
    const torch::Tensor &means,
    const torch::Tensor &normals,
    const torch::Tensor &covis,
    const torch::Tensor &gs_aabb_mins,
    const torch::Tensor &gs_aabb_maxs,
    const float iso,
    const bool return_centroids,
    const bool return_centroid_densities,
    const int64_t max_capacity)
{
    // 1. Enforce memory contiguity and CUDA residency
    CHECK_INPUT(vx_aabb_mins);
    CHECK_INPUT(vx_aabb_maxs);
    CHECK_INPUT(means);
    CHECK_INPUT(normals);
    CHECK_INPUT(covis);
    CHECK_INPUT(gs_aabb_mins);
    CHECK_INPUT(gs_aabb_maxs);

    // 2. Enforce Data Types (float3 maps to 3x float32)
    TORCH_CHECK(vx_aabb_mins.scalar_type() == torch::kFloat32, "vx_aabb_mins must be float32");
    TORCH_CHECK(vx_aabb_maxs.scalar_type() == torch::kFloat32, "vx_aabb_maxs must be float32");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(normals.scalar_type() == torch::kFloat32, "normals must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");
    TORCH_CHECK(gs_aabb_mins.scalar_type() == torch::kFloat32, "gs_aabb_mins must be float32");
    TORCH_CHECK(gs_aabb_maxs.scalar_type() == torch::kFloat32, "gs_aabb_maxs must be float32");

    // 3. Extract dimensions and validate shapes
    const uint32_t num_gaussians = means.size(0);
    const uint32_t num_voxels = vx_aabb_mins.size(0);

    TORCH_CHECK(vx_aabb_mins.size(1) == 3, "vx_aabb_mins must have shape (M, 3)");
    TORCH_CHECK(vx_aabb_maxs.size(0) == num_voxels && vx_aabb_maxs.size(1) == 3, "vx_aabb_maxs must match vx_aabb_mins shape");
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(normals.size(0) == num_gaussians && normals.size(1) == 3, "normals must match means shape");
    TORCH_CHECK(covis.size(0) == num_gaussians && covis.size(1) == 6, "covis must have shape (N, 6)");
    TORCH_CHECK(gs_aabb_mins.size(0) == num_gaussians && gs_aabb_mins.size(1) == 3, "gs_aabb_mins must match means shape");
    TORCH_CHECK(gs_aabb_maxs.size(0) == num_gaussians && gs_aabb_maxs.size(1) == 3, "gs_aabb_maxs must match means shape");

    // 4. Allocate Output Tensors directly on the same GPU as the inputs
    auto options = means.options();
    torch::Tensor hit_mask = torch::zeros({num_voxels}, options.dtype(torch::kBool));
    torch::Tensor out_voxel_ids = torch::empty({max_capacity}, options.dtype(torch::kInt64));
    torch::Tensor out_gaus_ids = torch::empty({max_capacity}, options.dtype(torch::kInt64));
    torch::Tensor global_counter = torch::zeros({1}, options.dtype(torch::kInt64));
    
    torch::Tensor centroids;
    torch::Tensor densities;
    float3* centroids_ptr = nullptr;
    float* densities_ptr = nullptr;
    
    if (return_centroids) {
        // Only allocate the [max_capacity, 3] float tensor if requested
        centroids = torch::empty({max_capacity, 3}, options.dtype(torch::kFloat32));
        centroids_ptr = reinterpret_cast<float3*>(centroids.data_ptr<float>());
        if (return_centroid_densities) {
            densities = torch::empty({max_capacity}, options.dtype(torch::kFloat32));
            densities_ptr = densities.data_ptr<float>();
        }
    }
    
    // 5. Launch the CUDA Kernel
    pgs_aabb::query_pgs_voxel_pair_intersection_brute_force(
        num_voxels,
        num_gaussians,
        reinterpret_cast<const float3 *>(vx_aabb_mins.data_ptr<float>()),
        reinterpret_cast<const float3 *>(vx_aabb_maxs.data_ptr<float>()),
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float3 *>(normals.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        reinterpret_cast<const float3 *>(gs_aabb_mins.data_ptr<float>()),
        reinterpret_cast<const float3 *>(gs_aabb_maxs.data_ptr<float>()),
        iso,
        return_centroids,
        return_centroid_densities,
        reinterpret_cast<bool *>(hit_mask.data_ptr<bool>()),
        reinterpret_cast<int64_t *>(out_voxel_ids.data_ptr<int64_t>()),
        reinterpret_cast<int64_t *>(out_gaus_ids.data_ptr<int64_t>()),
        centroids_ptr,
        densities_ptr,
        reinterpret_cast<int64_t *>(global_counter.data_ptr<int64_t>()),
        max_capacity);

    // 6. Return as a Python Tuple (also return the count of valid intersections)
    int64_t num_intersections = global_counter.item<int64_t>();
    if (num_intersections > max_capacity) {
        TORCH_WARN("Exceeded max capacity! Found ", num_intersections, " hits but capacity was ", max_capacity);
    }
    int64_t valid_hits = std::min(num_intersections, max_capacity);
    if (return_centroids && return_centroid_densities) {
        return std::make_tuple(
            hit_mask,
            out_voxel_ids.slice(0, 0, valid_hits), 
            out_gaus_ids.slice(0, 0, valid_hits),
            centroids.slice(0, 0, valid_hits),
            densities.slice(0, 0, valid_hits)
        );
    } else if (return_centroids) {
        return std::make_tuple(
            hit_mask,
            out_voxel_ids.slice(0, 0, valid_hits), 
            out_gaus_ids.slice(0, 0, valid_hits),
            centroids.slice(0, 0, valid_hits),
            std::nullopt
        );
    } else {
        // Return std::nullopt for the optional tensor
        return std::make_tuple(
            hit_mask,
            out_voxel_ids.slice(0, 0, valid_hits), 
            out_gaus_ids.slice(0, 0, valid_hits),
            std::nullopt,
            std::nullopt
        );
    }
}

std::tuple<torch::Tensor, torch::Tensor> query_pgs_edge_intersection_brute_force_wrapper(
    const torch::Tensor &edge_starts,
    const torch::Tensor &edge_ends,
    const torch::Tensor &means,
    const torch::Tensor &normals,
    const torch::Tensor &opacities,
    const torch::Tensor &covis,
    const float iso)
{
     // 1. Enforce memory contiguity and CUDA residency
    CHECK_INPUT(edge_starts);
    CHECK_INPUT(edge_ends);
    CHECK_INPUT(means);
    CHECK_INPUT(normals);
    CHECK_INPUT(opacities);
    CHECK_INPUT(covis);

    // 2. Enforce Data Types
    TORCH_CHECK(edge_starts.scalar_type() == torch::kFloat32, "edge_starts must be float32");
    TORCH_CHECK(edge_ends.scalar_type() == torch::kFloat32, "edge_ends must be float32");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(normals.scalar_type() == torch::kFloat32, "normals must be float32");
    TORCH_CHECK(opacities.scalar_type() == torch::kFloat32, "opacities must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");

    // 3. Extract dimensions and validate shapes
    const uint32_t num_edges = edge_starts.size(0);
    const uint32_t num_gaussians = means.size(0);

    TORCH_CHECK(edge_starts.size(1) == 3, "edge_starts must have shape (E, 3)");
    TORCH_CHECK(edge_ends.size(0) == num_edges && edge_ends.size(1) == 3, "edge_ends must match edge_starts");
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(normals.size(0) == num_gaussians && normals.size(1) == 3, "normals must have shape (N, 3)");
    TORCH_CHECK(covis.size(0) == num_gaussians && covis.size(1) == 6, "covis must have shape (N, 6)");
    TORCH_CHECK(opacities.size(0) == num_gaussians, "opacities must have shape (N)");

    // Allocate Output Tensors (Exact 1-to-1 mapping with edges)
    auto options = means.options();
    torch::Tensor hit_mask = torch::zeros({num_edges}, options.dtype(torch::kBool));
    torch::Tensor out_gaus_ids = torch::full({num_edges}, -1, options.dtype(torch::kInt64)); // Initialize with -1

    // Launch the CUDA Kernel
    pgs_aabb::query_pgs_edge_intersection_brute_force(
        num_edges,
        num_gaussians,
        reinterpret_cast<const float3 *>(edge_starts.data_ptr<float>()),
        reinterpret_cast<const float3 *>(edge_ends.data_ptr<float>()),
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float3 *>(normals.data_ptr<float>()),
        reinterpret_cast<const float *>(opacities.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        iso,
        reinterpret_cast<bool *>(hit_mask.data_ptr<bool>()),
        reinterpret_cast<int64_t *>(out_gaus_ids.data_ptr<int64_t>()));

    return std::make_tuple(hit_mask, out_gaus_ids);
}

void bind_primitive_pgs(py::module_ &m)
{
    m.def("query_pgs_voxel_pair_intersection_brute_force", &query_pgs_voxel_pair_intersection_brute_force_wrapper, 
        py::arg("vx_aabb_mins"),
        py::arg("vx_aabb_maxs"),
        py::arg("means"),
        py::arg("normals"),
        py::arg("covis"),
        py::arg("gs_aabb_mins"),
        py::arg("gs_aabb_maxs"),
        py::arg("iso"),
        py::arg("return_centroids") = false,
        py::arg("return_centroid_densities") = false,
        py::arg("max_capacity") = 1000000
    );

    m.def("query_pgs_edge_intersection_brute_force", &query_pgs_edge_intersection_brute_force_wrapper, 
        py::arg("edge_starts"),
        py::arg("edge_ends"),
        py::arg("means"),
        py::arg("normals"),
        py::arg("opacities"),
        py::arg("covis"),
        py::arg("iso")
    );
}