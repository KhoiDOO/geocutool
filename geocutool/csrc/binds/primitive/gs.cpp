#include "../../primitive/gs.h"

#include <cuda_fp16.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;

// Helper macros to check tensor properties
#define CHECK_CUDA(x) \
    AT_ASSERTM((x).options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    AT_ASSERTM((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> compute_aabb_wrapper(
    const torch::Tensor &means,
    const torch::Tensor &rotations,
    const torch::Tensor &scales,
    const float iso,
    const float tol,
    const uint32_t level,
    const bool rotnorm)
{
    // 1. Enforce memory contiguity and CUDA residency
    CHECK_INPUT(means);
    CHECK_INPUT(rotations);
    CHECK_INPUT(scales);

    // 2. Enforce Data Types (float3 maps to 3x float32)
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(rotations.scalar_type() == torch::kFloat32, "rotations must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");

    // 3. Extract dimensions and validate shapes
    const uint32_t num_gaussians = means.size(0);
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(rotations.size(1) == 4, "rotations must have shape (N, 4)");
    TORCH_CHECK(scales.size(1) == 3, "scales must have shape (N, 3)");

    // 4. Allocate Output Tensors directly on the same GPU as the inputs
    auto options = means.options();
    torch::Tensor aabb_min = torch::empty({num_gaussians, 3}, options);
    torch::Tensor aabb_max = torch::empty({num_gaussians, 3}, options);
    torch::Tensor contact_points = torch::empty({num_gaussians, 9}, options);
    torch::Tensor covi = torch::empty({num_gaussians, 6}, options);

    // 5. Launch the CUDA Kernel
    // We safely cast the flat float arrays into float3/float4 structs.
    gs_aabb::compute_aabb(
        num_gaussians,
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float4 *>(rotations.data_ptr<float>()),
        reinterpret_cast<const float3 *>(scales.data_ptr<float>()),
        iso,
        tol,
        level,
        rotnorm,
        reinterpret_cast<float3 *>(aabb_min.data_ptr<float>()),
        reinterpret_cast<float3 *>(aabb_max.data_ptr<float>()),
        reinterpret_cast<float3 *>(contact_points.data_ptr<float>()),
        reinterpret_cast<float *>(covi.data_ptr<float>()));

    // 6. Return as a Python Tuple
    return std::make_tuple(aabb_min, aabb_max, contact_points, covi);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> query_gs_voxel_intersection_brute_force_wrapper(
    const torch::Tensor &vx_aabb_mins,
    const torch::Tensor &vx_aabb_maxs,
    const torch::Tensor &means,
    const torch::Tensor &covis,
    const torch::Tensor &gs_aabb_mins,
    const torch::Tensor &gs_aabb_maxs,
    const torch::Tensor &contact_points,
    const float iso,
    const int64_t max_capacity)
{
    // 1. Enforce memory contiguity and CUDA residency
    CHECK_INPUT(vx_aabb_mins);
    CHECK_INPUT(vx_aabb_maxs);
    CHECK_INPUT(means);
    CHECK_INPUT(covis);
    CHECK_INPUT(gs_aabb_mins);
    CHECK_INPUT(gs_aabb_maxs);
    CHECK_INPUT(contact_points);

    // 2. Enforce Data Types (float3 maps to 3x float32)
    TORCH_CHECK(vx_aabb_mins.scalar_type() == torch::kFloat32, "vx_aabb_mins must be float32");
    TORCH_CHECK(vx_aabb_maxs.scalar_type() == torch::kFloat32, "vx_aabb_maxs must be float32");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");
    TORCH_CHECK(gs_aabb_mins.scalar_type() == torch::kFloat32, "gs_aabb_mins must be float32");
    TORCH_CHECK(gs_aabb_maxs.scalar_type() == torch::kFloat32, "gs_aabb_maxs must be float32");
    TORCH_CHECK(contact_points.scalar_type() == torch::kFloat32, "contact_points must be float32");

    // 3. Extract dimensions and validate shapes
    const uint32_t num_gaussians = means.size(0);
    const uint32_t num_voxels = vx_aabb_mins.size(0);
    TORCH_CHECK(vx_aabb_mins.size(1) == 3, "vx_aabb_mins must have shape (M, 3)");
    TORCH_CHECK(vx_aabb_maxs.size(0) == num_voxels, "vx_aabb_maxs must have the same number of voxels as vx_aabb_mins");
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(covis.size(0) == num_gaussians, "covis must have the same number of gaussians as means");
    TORCH_CHECK(vx_aabb_mins.size(1) == 3, "vx_aabb_mins must have shape (M, 3)");
    TORCH_CHECK(vx_aabb_maxs.size(1) == 3, "vx_aabb_maxs must have shape (M, 3)");
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(covis.size(1) == 6, "covis must have shape (N, 6)");
    TORCH_CHECK(gs_aabb_mins.size(1) == 3, "gs_aabb_mins must have shape (N, 3)");
    TORCH_CHECK(gs_aabb_maxs.size(1) == 3, "gs_aabb_maxs must have shape (N, 3)");
    TORCH_CHECK(contact_points.size(1) == 9, "contact_points must have shape (N, 9)");

    // 4. Allocate Output Tensors directly on the same GPU as the inputs
    auto options = means.options();
    torch::Tensor hit_mask = torch::zeros({num_voxels}, options.dtype(torch::kBool));
    torch::Tensor out_voxel_ids = torch::empty({max_capacity}, options.dtype(torch::kInt64));
    torch::Tensor out_gaus_ids = torch::empty({max_capacity}, options.dtype(torch::kInt64));
    torch::Tensor global_counter = torch::zeros({1}, options.dtype(torch::kInt64));
    
    // 5. Launch the CUDA Kernel
    gs_aabb::query_gs_voxel_intersection_brute_force(
        num_voxels,
        num_gaussians,
        reinterpret_cast<const float3 *>(vx_aabb_mins.data_ptr<float>()),
        reinterpret_cast<const float3 *>(vx_aabb_maxs.data_ptr<float>()),
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        reinterpret_cast<const float3 *>(gs_aabb_mins.data_ptr<float>()),
        reinterpret_cast<const float3 *>(gs_aabb_maxs.data_ptr<float>()),
        reinterpret_cast<const float3 *>(contact_points.data_ptr<float>()),
        iso,
        reinterpret_cast<bool *>(hit_mask.data_ptr<bool>()),
        reinterpret_cast<int64_t *>(out_voxel_ids.data_ptr<int64_t>()),
        reinterpret_cast<int64_t *>(out_gaus_ids.data_ptr<int64_t>()),
        reinterpret_cast<int64_t *>(global_counter.data_ptr<int64_t>()),
        max_capacity);

    // 6. Return as a Python Tuple (also return the count of valid intersections)
    int64_t num_intersections = global_counter.item<int64_t>();
    if (num_intersections > max_capacity) {
        TORCH_WARN("Exceeded max capacity! Found ", num_intersections, " hits but capacity was ", max_capacity);
    }
    int64_t valid_hits = std::min(num_intersections, max_capacity);
    return std::make_tuple(
        hit_mask,
        out_voxel_ids.slice(0, 0, valid_hits), 
        out_gaus_ids.slice(0, 0, valid_hits));
}

void bind_primitive_gs(py::module_ &m)
{
    m.def("compute_aabb_wrapper", &compute_aabb_wrapper, "Compute AABB for 3D Gaussians",
          py::arg("means"),
          py::arg("rotations"),
          py::arg("scales"),
          py::arg("iso"),
          py::arg("tol"),
          py::arg("level"),
          py::arg("rotnorm") = false);
    m.def("query_gs_voxel_intersection_brute_force_wrapper", &query_gs_voxel_intersection_brute_force_wrapper, "Query Gaussian-Voxel Intersections (Brute Force)",
          py::arg("vx_aabb_mins"),
          py::arg("vx_aabb_maxs"),
          py::arg("means"),
          py::arg("covis"),
          py::arg("gs_aabb_mins"),
          py::arg("gs_aabb_maxs"),
          py::arg("contact_points"),
          py::arg("iso"),
          py::arg("max_capacity") = 10000000);
}