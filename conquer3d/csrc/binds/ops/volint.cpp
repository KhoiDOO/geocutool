#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../ops/volint.h"
#include "../../maths/maths.h"

namespace py = pybind11;

/**
 * @brief Tensor-level entry point for the single-view volume integration in-place update.
 * @details Sits between pybind11 and the host dispatcher: it applies the `CHECK_INPUT`
 * contract -- CUDA device, contiguous layout, expected dtype -- then unwraps
 * `data_ptr` and calls the kernel launcher. Validating here keeps the launch path
 * free of checks and gives Python callers a clear error instead of a device fault.
 * @return The operator's results as PyTorch tensors.
 */
void single_view_volume_integral_bind(
    torch::Tensor grid_vertices,
    torch::Tensor sdf,
    torch::Tensor weight,
    std::optional<torch::Tensor> color_opt,
    torch::Tensor depth_image,
    std::optional<torch::Tensor> color_image_opt,
    torch::Tensor extrinsics,
    torch::Tensor intrinsics,
    float trunc_margin,
    int mode
) {
    TORCH_CHECK(grid_vertices.is_cuda() && grid_vertices.is_contiguous(), "grid_vertices must be CUDA and contiguous");
    TORCH_CHECK(sdf.is_cuda() && sdf.is_contiguous(), "sdf must be CUDA and contiguous");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "weight must be CUDA and contiguous");
    TORCH_CHECK(depth_image.is_cuda() && depth_image.is_contiguous(), "depth_image must be CUDA and contiguous");

    int num_vertices = grid_vertices.size(0);
    int image_height = depth_image.size(0);
    int image_width = depth_image.size(1);

    const float3* grid_vertices_ptr = reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>());
    float* sdf_ptr = sdf.data_ptr<float>();
    float* weight_ptr = weight.data_ptr<float>();
    const float* depth_image_ptr = depth_image.data_ptr<float>();

    float3* color_ptr = nullptr;
    if (color_opt.has_value() && color_opt.value().defined() && color_opt.value().numel() > 0) {
        auto color = color_opt.value();
        TORCH_CHECK(color.is_cuda() && color.is_contiguous(), "color must be CUDA and contiguous");
        color_ptr = reinterpret_cast<float3*>(color.data_ptr<float>());
    }

    const float3* color_image_ptr = nullptr;
    if (color_image_opt.has_value() && color_image_opt.value().defined() && color_image_opt.value().numel() > 0) {
        auto color_image = color_image_opt.value();
        TORCH_CHECK(color_image.is_cuda() && color_image.is_contiguous(), "color_image must be CUDA and contiguous");
        color_image_ptr = reinterpret_cast<const float3*>(color_image.data_ptr<float>());
    }

    torch::Tensor ex_cpu = extrinsics.cpu().contiguous();
    torch::Tensor in_cpu = intrinsics.cpu().contiguous();
    
    const float* ex_data = ex_cpu.data_ptr<float>();
    float4x4 ex = make_float4x4(
        ex_data[0], ex_data[1], ex_data[2], ex_data[3],
        ex_data[4], ex_data[5], ex_data[6], ex_data[7],
        ex_data[8], ex_data[9], ex_data[10], ex_data[11],
        ex_data[12], ex_data[13], ex_data[14], ex_data[15]
    );

    const float* in_data = in_cpu.data_ptr<float>();
    float3x3 in = make_float3x3(
        in_data[0], in_data[1], in_data[2],
        in_data[3], in_data[4], in_data[5],
        in_data[6], in_data[7], in_data[8]
    );

    single_view_volume_integral(
        num_vertices,
        grid_vertices_ptr,
        sdf_ptr,
        weight_ptr,
        color_ptr,
        depth_image_ptr,
        color_image_ptr,
        image_width,
        image_height,
        ex,
        in,
        trunc_margin,
        mode
    );
}

/**
 * @brief Registers the single-view volume integration operator on the extension module.
 * @details Called once from `pybind.cpp` with the root module, so every symbol
 * defined here lands directly on `conquer3d._C`.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_volint(py::module_& m) {
    m.def("single_view_volume_integral", &single_view_volume_integral_bind,
          py::arg("grid_vertices"), py::arg("sdf"), py::arg("weight"), py::arg("color"),
          py::arg("depth_image"), py::arg("color_image"), py::arg("extrinsics"),
          py::arg("intrinsics"), py::arg("trunc_margin"), py::arg("mode") = 1,
          R"pbdoc(
          Integrates a single depth map and optional RGB frame into a 3D volumetric TSDF grid in-place (CUDA).

          Args:
              grid_vertices (torch.Tensor): (N, 3) float32 coordinates on CUDA.
              sdf (torch.Tensor): (N,) float32 running TSDF field updated in-place on CUDA.
              weight (torch.Tensor): (N,) float32 running sample weights updated in-place on CUDA.
              color (torch.Tensor, optional): (N, 3) float32 running RGB colors updated in-place on CUDA.
              depth_image (torch.Tensor): (H, W) float32 depth map in meters on CUDA.
              color_image (torch.Tensor, optional): (H, W, 3) float32 RGB image on CUDA.
              extrinsics (torch.Tensor): (4, 4) float32 World-to-Camera transformation matrix.
              intrinsics (torch.Tensor): (3, 3) float32 camera intrinsic calibration matrix.
              trunc_margin (float): TSDF truncation distance $\mu$ in meters.
              mode (int, optional): Integration mode (1: Euclidean distance, 0: projective). Defaults to 1.

          Example:
              >>> import torch
              >>> from conquer3d._C import single_view_volume_integral
              >>> single_view_volume_integral(verts, sdf, weight, color, depth, rgb, c2w, k, trunc_margin=0.05, mode=1)
          )pbdoc");
}
