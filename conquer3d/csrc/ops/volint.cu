/**
 * @file volint.cu
 * @brief CUDA kernel implementations for real-time single-view TSDF volumetric depth/color integration.
 */

#include "volint.h"
#include <stdio.h>
#include <math.h>

/**
 * @brief Integrates one RGB-D frame into a truncated signed distance volume.
 * @details One thread per grid vertex. Each transforms its world position into camera
 * space, projects it through the intrinsics, samples the depth map at the nearest pixel,
 * and folds the resulting signed distance into a running weighted average
 * $\text{tsdf} \leftarrow (\text{tsdf} \cdot w + d) / (w + 1)$. Colour, when supplied,
 * is averaged with the same weights.
 *
 * Two distance conventions are available. `mode == 1` scales the depth residual by
 * $\|\mathbf{p}_{cam}\| / z_c$ to give a true Euclidean distance along the viewing ray
 * (matching Open3D's `UniformTSDFVolume`); any other value keeps the cheaper projective
 * residual $d - z_c$, which is accurate near the optical axis and increasingly
 * approximate towards the image corners.
 *
 * @param[in] num_vertices Number of grid vertices $N$.
 * @param[in] grid_vertices Device array of $N$ world-space vertex coordinates.
 * @param[in,out] sdf Device array of $N$ truncated signed distances, updated in place.
 * @param[in,out] weight Device array of $N$ accumulated observation counts.
 * @param[in,out] color Device array of $N$ RGB values, or `nullptr` to skip colour.
 * @param[in] depth_image Row-major depth buffer of `image_width * image_height` samples.
 * @param[in] color_image Row-major RGB buffer, or `nullptr`.
 * @param[in] image_width Depth and colour image width in pixels.
 * @param[in] image_height Depth and colour image height in pixels.
 * @param[in] extrinsics World-to-camera transform.
 * @param[in] intrinsics Camera projection matrix.
 * @param[in] trunc_margin Truncation band half-width; distances are normalised by it and
 *     clamped to 1.
 * @param[in] mode Distance convention: 1 for Euclidean, otherwise projective.
 * @note Launched with 256 threads per block over a 1D grid.
 * @note Vertices behind the camera ($z_c \le 0$), projecting outside the image, landing on
 * invalid depth ($d \le 0$), or falling behind the truncation band are skipped, leaving
 * their state untouched.
 * @warning Each vertex is written by exactly one thread, so no atomics are needed --
 * but this also means integrating several frames concurrently into one volume would
 * race. Call once per frame.
 */
__global__ void single_view_volume_integral_kernel(
    const int num_vertices,
    const float3* grid_vertices,
    float* sdf,
    float* weight,
    float3* color,
    const float* depth_image,
    const float3* color_image,
    const int image_width,
    const int image_height,
    const float4x4 extrinsics,
    const float3x3 intrinsics,
    const float trunc_margin,
    const int mode
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;

    // 1. Fetch world position
    float3 p_world = grid_vertices[idx];

    // 2. Transform to camera coordinates using Extrinsics (World-to-Camera)
    float4 p_world4 = make_float4(p_world.x, p_world.y, p_world.z, 1.0f);
    float4 p_cam4 = extrinsics * p_world4;
    float3 p_cam = make_float3(p_cam4.x, p_cam4.y, p_cam4.z);
    
    float xc = p_cam.x, yc = p_cam.y, zc = p_cam.z;

    // Check if the vertex is behind the camera
    if (zc <= 0.0f) return;

    // 3. Project to image plane using Intrinsics
    float3 p_uvw = intrinsics * p_cam;
    float u = p_uvw.x / p_uvw.z;
    float v = p_uvw.y / p_uvw.z;

    int ui = (int)roundf(u);
    int vi = (int)roundf(v);

    // 4. Boundary check
    if (ui >= 0 && ui < image_width && vi >= 0 && vi < image_height) {
        int pixel_idx = vi * image_width + ui;
        float d = depth_image[pixel_idx];

        if (d > 0.0f) {
            // 5. Calculate SDF based on mode
            float sdf_val;
            if (mode == 1) {
                // True Euclidean SDF (Open3D UniformTSDFVolume convention)
                float ray_length = maths::norm(p_cam);
                float depth_to_camera_distance_multiplier = ray_length / zc;
                sdf_val = (d - zc) * depth_to_camera_distance_multiplier;
            } else {
                // Projective SDF shortcut
                sdf_val = d - zc;
            }
            
            // 6. Truncate and update running average if within margin
            if (sdf_val > -trunc_margin) {
                float tsdf = fminf(1.0f, sdf_val / trunc_margin);
                
                float old_sdf = sdf[idx];
                float old_weight = weight[idx];
                
                float inv_wsum = 1.0f / (old_weight + 1.0f);
                float new_sdf = (old_sdf * old_weight + tsdf) * inv_wsum;
                
                sdf[idx] = new_sdf;
                
                // Color integration
                if (color != nullptr && color_image != nullptr) {
                    float3 old_color = color[idx];
                    float3 incoming_color = color_image[pixel_idx];
                    
                    color[idx] = (old_color * old_weight + incoming_color) * inv_wsum;
                }
                
                weight[idx] = old_weight + 1.0f;
            }
        }
    }
}

void single_view_volume_integral(
    const int num_vertices,
    const float3* grid_vertices,
    float* sdf,
    float* weight,
    float3* color,
    const float* depth_image,
    const float3* color_image,
    const int image_width,
    const int image_height,
    const float4x4 extrinsics,
    const float3x3 intrinsics,
    const float trunc_margin,
    const int mode
) {
    int block_size = 256;
    int grid_size = (num_vertices + block_size - 1) / block_size;

    single_view_volume_integral_kernel<<<grid_size, block_size>>>(
        num_vertices, 
        grid_vertices, 
        sdf, 
        weight, 
        color,
        depth_image, 
        color_image, 
        image_width, 
        image_height, 
        extrinsics, 
        intrinsics, 
        trunc_margin,
        mode
    );
}