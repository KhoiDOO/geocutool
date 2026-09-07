/**
 * @file triangle_mesh.cu
 * @brief CUDA kernel implementations for Discrete Differential Geometry (DDG) operators and topological mesh analysis.
 */

#include "triangle_mesh.h"
#include "../primitive/triangle.h"
#include "../primitive/edge.h"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace triangle_mesh
{
/**
 * @brief Computes the geometric normal of every triangle.
 * @details One thread per triangle, taking the normalised cross product of two edge
 * vectors. Orientation follows the winding, so a consistently wound mesh yields outward
 * normals.
 * @param[in] num_triangles Number of triangles.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[out] triangle_normals Device array of per-triangle unit normals.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Degenerate triangles have a zero-length cross product and produce NaNs. Screen
 * them with compute_triangle_areas_kernel() first.
 */
__global__ void compute_triangle_normals_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float3 *__restrict__ triangle_normals)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            float3 v0 = vertices[tri.x];
            float3 v1 = vertices[tri.y];
            float3 v2 = vertices[tri.z];
            
            triangle_normals[idx] = triangle::compute_normal(v0, v1, v2);
        }
    }

/**
 * @brief Computes the area of every triangle.
 * @details One thread per triangle: half the magnitude of the edge cross product. Also the
 * cheapest way to find degenerate faces, whose area collapses to zero.
 * @param[in] num_triangles Number of triangles.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[out] triangle_areas Device array of per-triangle areas.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void compute_triangle_areas_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ triangle_areas)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            float3 v0 = vertices[tri.x];
            float3 v1 = vertices[tri.y];
            float3 v2 = vertices[tri.z];
            
            triangle_areas[idx] = triangle::compute_area(v0, v1, v2);
        }
    }

/**
 * @brief Computes a normalised shape quality score for every triangle.
 * @details One thread per triangle. The score approaches 1 for an equilateral triangle and
 * 0 for a degenerate sliver, giving a single scalar for mesh quality assessment.
 * @param[in] num_triangles Number of triangles.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[out] qualities Device array of per-triangle quality scores in $[0, 1]$.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void compute_quality_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ qualities)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
            qualities[idx] = T.compute_quality();
        }
    }

/**
 * @brief Computes the aspect ratio of every triangle.
 * @details One thread per triangle. @p mode selects the ratio definition -- longest edge to
 * shortest, or longest edge to inradius -- since different meshing literature uses
 * different conventions.
 * @param[in] num_triangles Number of triangles.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] mode Aspect ratio definition selector.
 * @param[out] aspect_ratios Device array of per-triangle aspect ratios.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Larger values indicate worse-conditioned triangles; an equilateral triangle
 * attains the minimum.
 */
__global__ void compute_aspect_ratio_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        int mode,
        float *__restrict__ aspect_ratios)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
            aspect_ratios[idx] = T.compute_ar(mode);
        }
    }

/**
 * @brief Computes the circumradius-to-inradius ratio of every triangle.
 * @details One thread per triangle. The ratio reaches its minimum of 2 for an equilateral
 * triangle and grows without bound as a triangle degenerates, making it a sensitive
 * conditioning measure.
 * @param[in] num_triangles Number of triangles.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[out] ratios Device array of per-triangle radii ratios.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void compute_radii_ratio_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ ratios)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
            ratios[idx] = T.compute_radii_ratio();
        }
    }

/**
 * @brief Computes how close each triangle is to equilateral.
 * @details One thread per triangle, scoring 1 for a perfectly regular triangle and falling
 * towards 0 as edge lengths diverge.
 * @param[in] num_triangles Number of triangles.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[out] regularities Device array of per-triangle regularity scores.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void compute_triangle_regularity_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ regularities)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
            regularities[idx] = T.compute_triangle_regularity();
        }
    }

/**
 * @brief Computes the circumradius-to-shortest-edge ratio of every triangle.
 * @details One thread per triangle. This is the quantity Delaunay refinement bounds, so it
 * is the natural measure when assessing a mesh produced by such an algorithm.
 * @param[in] num_triangles Number of triangles.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[out] ratios Device array of per-triangle radius-edge ratios.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void compute_radius_edge_ratio_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ ratios)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
            ratios[idx] = T.compute_radius_edge_ratio();
        }
    }

/**
 * @brief Computes each triangle's maximum deviation from 60 degrees.
 * @details One thread per triangle, reporting the largest absolute difference between an
 * interior angle and the equilateral ideal. Unlike area-based scores this catches a
 * triangle that is small yet badly shaped.
 * @param[in] num_triangles Number of triangles.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[out] deviations Device array of per-triangle angle deviations, in radians.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void compute_angle_deviation_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ deviations)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
            deviations[idx] = T.compute_angle_deviation();
        }
    }

/**
 * @brief Computes the axis-aligned bounding box of every triangle.
 * @details One thread per triangle. These boxes are the leaf primitives the BVH builder
 * consumes.
 * @param[in] num_triangles Number of triangles.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[out] aabb_mins Device array of per-triangle lower bounds.
 * @param[out] aabb_maxs Device array of per-triangle upper bounds.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void compute_triangle_aabbs_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float3 *__restrict__ aabb_mins,
        float3 *__restrict__ aabb_maxs)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            float3 v0 = vertices[tri.x];
            float3 v1 = vertices[tri.y];
            float3 v2 = vertices[tri.z];
            
            triangle::compute_aabb(v0, v1, v2, aabb_mins[idx], aabb_maxs[idx]);
        }
    }

    /**
     * @brief Launches the per-triangle unit geometric normal computation.
     * @details Host wrapper sizing a 1D grid over the triangles; see the kernel for the
     * parallel decomposition.
     * @param[in] num_triangles Number of triangles.
     * @param[in] vertices Device array of mesh vertex coordinates.
     * @param[in] triangles Device array of triangle vertex indices.
     */
    __host__ void compute_triangle_normals(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float3 *__restrict__ triangle_normals)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_triangle_normals_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, triangle_normals);
    }

    /**
     * @brief Launches the per-triangle area computation.
     * @details Host wrapper sizing a 1D grid over the triangles; see the kernel for the
     * parallel decomposition.
     * @param[in] num_triangles Number of triangles.
     * @param[in] vertices Device array of mesh vertex coordinates.
     * @param[in] triangles Device array of triangle vertex indices.
     */
    __host__ void compute_triangle_areas(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ triangle_areas)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_triangle_areas_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, triangle_areas);
    }

    /**
     * @brief Launches the per-triangle normalised shape quality score computation.
     * @details Host wrapper sizing a 1D grid over the triangles; see the kernel for the
     * parallel decomposition.
     * @param[in] num_triangles Number of triangles.
     * @param[in] vertices Device array of mesh vertex coordinates.
     * @param[in] triangles Device array of triangle vertex indices.
     */
    __host__ void compute_quality(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ qualities)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_quality_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, qualities);
    }

    /**
     * @brief Launches the per-triangle aspect ratio computation.
     * @details Host wrapper sizing a 1D grid over the triangles; see the kernel for the
     * parallel decomposition.
     * @param[in] num_triangles Number of triangles.
     * @param[in] vertices Device array of mesh vertex coordinates.
     * @param[in] triangles Device array of triangle vertex indices.
     */
    __host__ void compute_aspect_ratio(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        int mode,
        float *__restrict__ aspect_ratios)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_aspect_ratio_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, mode, aspect_ratios);
    }

    /**
     * @brief Launches the per-triangle circumradius-to-inradius ratio computation.
     * @details Host wrapper sizing a 1D grid over the triangles; see the kernel for the
     * parallel decomposition.
     * @param[in] num_triangles Number of triangles.
     * @param[in] vertices Device array of mesh vertex coordinates.
     * @param[in] triangles Device array of triangle vertex indices.
     */
    __host__ void compute_radii_ratio(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ ratios)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_radii_ratio_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, ratios);
    }

    /**
     * @brief Launches the per-triangle regularity score computation.
     * @details Host wrapper sizing a 1D grid over the triangles; see the kernel for the
     * parallel decomposition.
     * @param[in] num_triangles Number of triangles.
     * @param[in] vertices Device array of mesh vertex coordinates.
     * @param[in] triangles Device array of triangle vertex indices.
     */
    __host__ void compute_triangle_regularity(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ regularities)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_triangle_regularity_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, regularities);
    }

    /**
     * @brief Launches the per-triangle circumradius-to-shortest-edge ratio computation.
     * @details Host wrapper sizing a 1D grid over the triangles; see the kernel for the
     * parallel decomposition.
     * @param[in] num_triangles Number of triangles.
     * @param[in] vertices Device array of mesh vertex coordinates.
     * @param[in] triangles Device array of triangle vertex indices.
     */
    __host__ void compute_radius_edge_ratio(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ ratios)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_radius_edge_ratio_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, ratios);
    }

    /**
     * @brief Launches the per-triangle maximum angle deviation from 60 degrees computation.
     * @details Host wrapper sizing a 1D grid over the triangles; see the kernel for the
     * parallel decomposition.
     * @param[in] num_triangles Number of triangles.
     * @param[in] vertices Device array of mesh vertex coordinates.
     * @param[in] triangles Device array of triangle vertex indices.
     */
    __host__ void compute_angle_deviation(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ deviations)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_angle_deviation_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, deviations);
    }

    /**
     * @brief Launches the per-triangle axis-aligned bounding box computation.
     * @details Host wrapper sizing a 1D grid over the triangles; see the kernel for the
     * parallel decomposition.
     * @param[in] num_triangles Number of triangles.
     * @param[in] vertices Device array of mesh vertex coordinates.
     * @param[in] triangles Device array of triangle vertex indices.
     */
    __host__ void compute_triangle_aabbs(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float3 *__restrict__ aabb_mins,
        float3 *__restrict__ aabb_maxs)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_triangle_aabbs_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, aabb_mins, aabb_maxs);
    }

/**
 * @brief Accumulates face normals onto their incident vertices.
 * @details One thread per triangle, scattering each face normal to its three corners.
 * @p mode selects the weighting -- uniform, area-weighted, or angle-weighted. Angle
 * weighting is the choice that makes the result an exact pseudonormal, which is what the
 * signed-distance queries rely on for correct inside/outside classification.
 * @param[in] num_triangles Number of triangles.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] triangle_normals Device array of per-triangle normals.
 * @param[out] vertex_normals Device array accumulating unnormalised vertex normals.
 * @param[in] mode Weighting selector.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Results are unnormalised; follow with normalize_vertex_normals_kernel().
 * @warning Vertices are shared between triangles, so accumulation uses `atomicAdd`; the
 * reduction order, and hence the last bits of the result, varies between runs.
 */
__global__ void compute_vertex_normals_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ triangle_normals,
        float3 *__restrict__ vertex_normals,
        int mode)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            float3 n = triangle_normals[idx];
            
            if (mode == 0) {
                atomicAdd(&vertex_normals[tri.x].x, n.x);
                atomicAdd(&vertex_normals[tri.x].y, n.y);
                atomicAdd(&vertex_normals[tri.x].z, n.z);
                
                atomicAdd(&vertex_normals[tri.y].x, n.x);
                atomicAdd(&vertex_normals[tri.y].y, n.y);
                atomicAdd(&vertex_normals[tri.y].z, n.z);
                
                atomicAdd(&vertex_normals[tri.z].x, n.x);
                atomicAdd(&vertex_normals[tri.z].y, n.y);
                atomicAdd(&vertex_normals[tri.z].z, n.z);
            } else if (mode == 1) {
                float3 v0 = vertices[tri.x];
                float3 v1 = vertices[tri.y];
                float3 v2 = vertices[tri.z];
                float3 e0 = maths::normalize(v1 - v0);
                float3 e1 = maths::normalize(v2 - v1);
                float3 e2 = maths::normalize(v0 - v2);

                float a0 = acosf(fminf(fmaxf(-maths::dot(e0, e2), -1.0f), 1.0f));
                float a1 = acosf(fminf(fmaxf(-maths::dot(e1, e0), -1.0f), 1.0f));
                float a2 = acosf(fminf(fmaxf(-maths::dot(e2, e1), -1.0f), 1.0f));

                if (isfinite(a0)) {
                    atomicAdd(&vertex_normals[tri.x].x, a0 * n.x);
                    atomicAdd(&vertex_normals[tri.x].y, a0 * n.y);
                    atomicAdd(&vertex_normals[tri.x].z, a0 * n.z);
                }
                if (isfinite(a1)) {
                    atomicAdd(&vertex_normals[tri.y].x, a1 * n.x);
                    atomicAdd(&vertex_normals[tri.y].y, a1 * n.y);
                    atomicAdd(&vertex_normals[tri.y].z, a1 * n.z);
                }
                if (isfinite(a2)) {
                    atomicAdd(&vertex_normals[tri.z].x, a2 * n.x);
                    atomicAdd(&vertex_normals[tri.z].y, a2 * n.y);
                    atomicAdd(&vertex_normals[tri.z].z, a2 * n.z);
                }
            }
        }
    }

/**
 * @brief Normalises accumulated vertex normals to unit length.
 * @details One thread per vertex, completing the two-pass accumulate-then-normalise scheme.
 * @param[in] num_vertices Number of vertices.
 * @param[in,out] vertex_normals Device array of vertex normals, normalised in place.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning An isolated vertex accumulates a zero vector, whose normalisation is undefined
 * and yields NaNs.
 */
__global__ void normalize_vertex_normals_kernel(
        const uint32_t num_vertices,
        float3 *__restrict__ vertex_normals)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_vertices)
        {
            float3 n = vertex_normals[idx];
            float length = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
            if (length > 1e-8f) {
                vertex_normals[idx] = make_float3(n.x / length, n.y / length, n.z / length);
            }
        }
    }

    /**
     * @brief Computes per-vertex normals from the incident faces.
     * @details Two launches: scatter weighted face normals onto their vertices, then normalise.
     * The weighting mode decides whether the result is a true angle-weighted pseudonormal, which
     * is what the signed-distance queries require for correct sign determination.
     */
    __host__ void compute_vertex_normals(
        const uint32_t num_vertices,
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ triangle_normals,
        float3 *__restrict__ vertex_normals,
        int mode)
    {
        if (num_triangles == 0 || num_vertices == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_vertex_normals_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, triangle_normals, vertex_normals, mode);
            
        int blocks_vert = (num_vertices + threads - 1) / threads;
        normalize_vertex_normals_kernel<<<blocks_vert, threads>>>(
            num_vertices, vertex_normals);
    }

/**
 * @brief Emits an edge key and source slot for each of a triangle's three edges.
 * @details One thread per triangle, writing sorted vertex-index pairs. Duplicates are
 * intentional: sorting the keys groups the triangles sharing an edge together, which is how
 * the adjacency structures are built without a hash table.
 * @param[in] num_triangles Number of triangles.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[out] edge_keys Device array of $3N$ edge keys, with duplicates.
 * @param[out] edge_indices Device array of $3N$ source triangle slots.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void extract_edge_slots_kernel(
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        Edge *__restrict__ edge_keys,
        int *__restrict__ edge_indices)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            edge_keys[3 * idx + 0] = Edge(tri.x, tri.y);
            edge_indices[3 * idx + 0] = 3 * idx + 0;

            edge_keys[3 * idx + 1] = Edge(tri.y, tri.z);
            edge_indices[3 * idx + 1] = 3 * idx + 1;

            edge_keys[3 * idx + 2] = Edge(tri.z, tri.x);
            edge_indices[3 * idx + 2] = 3 * idx + 2;
        }
    }

/**
 * @brief Averages the normals of the triangles meeting at each edge.
 * @details One thread per unique edge, reading the run of sorted entries that share its
 * key. Angle-weighted edge pseudonormals are required alongside the vertex ones for exact
 * sign determination on a watertight mesh.
 * @param[in] num_edges Number of unique edges.
 * @param[in] sorted_edge_keys Device array of edge keys in ascending order.
 * @param[in] sorted_edge_indices Device array of source triangle slots, permuted alongside.
 * @param[in] triangle_normals Device array of per-triangle normals.
 * @param[out] edge_normals Device array of per-edge normals.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note A boundary edge has a single incident triangle and simply inherits its normal.
 */
__global__ void compute_edge_normals_kernel(
        const uint32_t num_edges,
        const Edge *__restrict__ sorted_edge_keys,
        const int *__restrict__ sorted_edge_indices,
        const float3 *__restrict__ triangle_normals,
        float3 *__restrict__ edge_normals)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_edges)
        {
            Edge my_key = sorted_edge_keys[idx];
            int my_orig_idx = sorted_edge_indices[idx];

            float3 N = triangle_normals[my_orig_idx / 3];

            int j = idx - 1;
            while (j >= 0 && sorted_edge_keys[j] == my_key)
            {
                N = N + triangle_normals[sorted_edge_indices[j] / 3];
                j--;
            }

            j = idx + 1;
            while (j < num_edges && sorted_edge_keys[j] == my_key)
            {
                N = N + triangle_normals[sorted_edge_indices[j] / 3];
                j++;
            }

            edge_normals[my_orig_idx] = maths::normalize(N);
        }
    }

    /**
     * @brief Computes per-edge normals by averaging the incident faces.
     * @details Needed alongside the vertex pseudonormals for exact sign determination on a
     * watertight mesh.
     */
    __host__ void compute_edge_normals(
        const uint32_t num_triangles,
        const torch::Tensor &triangles,
        const float3 *__restrict__ triangle_normals,
        float3 *__restrict__ edge_normals)
    {
        if (num_triangles == 0) return;

        at::cuda::CUDAGuard device_guard(triangles.device());
        auto allocator = at::cuda::ThrustAllocator();
        auto policy = thrust::cuda::par(allocator).on(at::cuda::getCurrentCUDAStream());

        int threads = NTHREADS;
        int blocks_tri = (num_triangles + threads - 1) / threads;

        uint32_t num_edges = num_triangles * 3;
        int blocks_edge = (num_edges + threads - 1) / threads;

        auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(triangles.device());
        auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(triangles.device());

        torch::Tensor edge_keys = torch::empty({static_cast<int64_t>(num_edges)}, options_i64);
        torch::Tensor edge_indices = torch::empty({static_cast<int64_t>(num_edges)}, options_i32);

        extract_edge_slots_kernel<<<blocks_tri, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_triangles, (const int3*)triangles.data_ptr<int>(),
            reinterpret_cast<Edge *>(edge_keys.data_ptr<int64_t>()),
            edge_indices.data_ptr<int>());

        thrust::sort_by_key(
            policy,
            reinterpret_cast<Edge *>(edge_keys.data_ptr<int64_t>()),
            reinterpret_cast<Edge *>(edge_keys.data_ptr<int64_t>()) + num_edges,
            edge_indices.data_ptr<int>());

        compute_edge_normals_kernel<<<blocks_edge, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_edges,
            reinterpret_cast<const Edge *>(edge_keys.data_ptr<int64_t>()),
            edge_indices.data_ptr<int>(),
            triangle_normals,
            edge_normals);
    }

/**
 * @brief Emits edge keys paired with their source triangle index.
 * @details One thread per triangle. A lighter variant of extract_edge_slots_kernel() used
 * where only the owning triangle, not the local edge slot, is needed downstream.
 * @param[in] num_triangles Number of triangles.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[out] edge_keys Device array of $3N$ edge keys, with duplicates.
 * @param[out] triangle_indices Device array of $3N$ source triangle indices.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void extract_edges_kernel(
        const uint32_t num_triangles,
        const int3* triangles,
        Edge* edge_keys,
        int* triangle_indices)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles) {
            int3 tri = triangles[idx];
            
            edge_keys[3*idx + 0] = Edge(tri.x, tri.y);
            triangle_indices[3*idx + 0] = idx;
            
            edge_keys[3*idx + 1] = Edge(tri.y, tri.z);
            triangle_indices[3*idx + 1] = idx;
            
            edge_keys[3*idx + 2] = Edge(tri.z, tri.x);
            triangle_indices[3*idx + 2] = idx;
        }
    }
    
/**
 * @brief Expands deduplicated edge keys into an index pair array.
 * @details One thread per unique edge, unpacking each key into the `(V, 2)` layout the
 * Python API exposes.
 * @param[in] num_unique_edges Number of unique edges.
 * @param[in] unique_edge_keys Device array of deduplicated edge keys.
 * @param[out] unique_edges_out Device array of $2E$ vertex indices.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void unpack_edges_kernel(
        const uint32_t num_unique_edges,
        const Edge* unique_edge_keys,
        int* unique_edges_out)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_unique_edges) {
            Edge key = unique_edge_keys[idx];
            unique_edges_out[2*idx + 0] = key.v0;
            unique_edges_out[2*idx + 1] = key.v1;
        }
    }

    /**
     * @brief Builds the edge-to-triangle CSR connectivity map.
     * @details Emits an edge key per triangle corner, sorts, uniques, and derives CSR offsets.
     * The per-edge incident count doubles as a manifoldness test: 2 is manifold, 1 a boundary,
     * more than 2 non-manifold.
     */
    __host__ void compute_edges_to_triangle_map(
        const uint32_t num_triangles,
        const torch::Tensor &triangles,
        torch::Tensor &out_unique_edges,
        torch::Tensor &out_offsets,
        torch::Tensor &out_counts,
        torch::Tensor &out_sorted_triangle_indices)
    {
        if (num_triangles == 0) return;

        at::cuda::CUDAGuard device_guard(triangles.device());
        auto allocator = at::cuda::ThrustAllocator();
        auto policy = thrust::cuda::par(allocator).on(at::cuda::getCurrentCUDAStream());

        auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(triangles.device());
        auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(triangles.device());

        uint32_t num_edges = num_triangles * 3;
        
        torch::Tensor edge_keys = torch::empty({num_edges}, options_i64);
        out_sorted_triangle_indices = torch::empty({num_edges}, options_i32);

        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        extract_edges_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_triangles, (const int3*)triangles.data_ptr<int>(), 
            (Edge*)edge_keys.data_ptr<int64_t>(), 
            out_sorted_triangle_indices.data_ptr<int>());

        thrust::sort_by_key(
            policy,
            (Edge*)edge_keys.data_ptr<int64_t>(),
            (Edge*)edge_keys.data_ptr<int64_t>() + num_edges,
            out_sorted_triangle_indices.data_ptr<int>()
        );

        torch::Tensor unique_keys = torch::empty({num_edges}, options_i64);
        out_counts = torch::empty({num_edges}, options_i32);
        torch::Tensor ones = torch::ones({num_edges}, options_i32);

        auto new_end = thrust::reduce_by_key(
            policy,
            (Edge*)edge_keys.data_ptr<int64_t>(),
            (Edge*)edge_keys.data_ptr<int64_t>() + num_edges,
            ones.data_ptr<int>(),
            (Edge*)unique_keys.data_ptr<int64_t>(),
            out_counts.data_ptr<int>()
        );

        int num_unique_edges = new_end.first - (Edge*)unique_keys.data_ptr<int64_t>();

        unique_keys = unique_keys.slice(0, 0, num_unique_edges);
        out_counts = out_counts.slice(0, 0, num_unique_edges);

        out_offsets = torch::zeros({num_unique_edges}, options_i32);
        if (num_unique_edges > 1) {
            torch::Tensor cumsum = torch::cumsum(out_counts.slice(0, 0, num_unique_edges - 1), 0, torch::kInt32);
            out_offsets.slice(0, 1, num_unique_edges).copy_(cumsum);
        }

        out_unique_edges = torch::empty({num_unique_edges, 2}, options_i32);
        int blocks2 = (num_unique_edges + threads - 1) / threads;
        if (blocks2 > 0) {
            unpack_edges_kernel<<<blocks2, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_unique_edges,
                (Edge*)unique_keys.data_ptr<int64_t>(),
                out_unique_edges.data_ptr<int>()
            );
        }
    }

/**
 * @brief Counts the triangles incident on each vertex.
 * @details One thread per triangle, atomically incrementing the counter of each corner.
 * The counts are prefix-summed by the host into the CSR offsets of the vertex-to-triangle
 * map.
 * @param[in] num_triangles Number of triangles.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[out] counts Device array of per-vertex incidence counts; must be zeroed first.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning @p counts must be zeroed before launch, and the increments are atomic.
 */
__global__ void compute_vertex_triangle_counts_kernel(
        const uint32_t num_triangles,
        const int3* triangles,
        int* counts)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles) {
            int3 tri = triangles[idx];
            atomicAdd(&counts[tri.x], 1);
            atomicAdd(&counts[tri.y], 1);
            atomicAdd(&counts[tri.z], 1);
        }
    }

/**
 * @brief Fills the vertex-to-triangle CSR index array.
 * @details One thread per triangle. Each corner claims the next free slot in its vertex's
 * CSR range through an atomic bump of a per-vertex cursor.
 * @param[in] num_triangles Number of triangles.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in,out] current_offsets Device array of per-vertex write cursors, initialised to
 *     the CSR offsets.
 * @param[out] indices Device array receiving incident triangle indices.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Triangles appear in nondeterministic order within each vertex's range. Sort per
 * vertex if a canonical ordering matters.
 */
__global__ void compute_vertex_triangle_indices_kernel(
        const uint32_t num_triangles,
        const int3* triangles,
        int* current_offsets,
        int* indices)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles) {
            int3 tri = triangles[idx];
            int pos_x = atomicAdd(&current_offsets[tri.x], 1);
            indices[pos_x] = idx;
            int pos_y = atomicAdd(&current_offsets[tri.y], 1);
            indices[pos_y] = idx;
            int pos_z = atomicAdd(&current_offsets[tri.z], 1);
            indices[pos_z] = idx;
        }
    }

    /**
     * @brief Builds the vertex-to-triangle CSR connectivity map.
     * @details Counts incidences, prefix-sums them into offsets, then scatters triangle indices
     * into each vertex's range.
     * @warning Triangles land in nondeterministic order within a vertex's range.
     */
    void build_vertices_to_triangle_map(
        const uint32_t num_vertices,
        const uint32_t num_triangles,
        const torch::Tensor& triangles,
        torch::Tensor& out_counts,
        torch::Tensor& out_offsets,
        torch::Tensor& out_indices)
    {
        if (num_triangles == 0 || num_vertices == 0) return;

        at::cuda::CUDAGuard device_guard(triangles.device());
        auto allocator = at::cuda::ThrustAllocator();
        auto policy = thrust::cuda::par(allocator).on(at::cuda::getCurrentCUDAStream());

        auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(triangles.device());
        out_counts = torch::zeros({num_vertices}, options_i32);
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;

        compute_vertex_triangle_counts_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_triangles,
            (const int3*)triangles.data_ptr<int>(),
            out_counts.data_ptr<int>()
        );

        out_offsets = torch::zeros({num_vertices}, options_i32);
        if (num_vertices > 1) {
            torch::Tensor cumsum = torch::cumsum(out_counts.slice(0, 0, num_vertices - 1), 0, torch::kInt32);
            out_offsets.slice(0, 1, num_vertices).copy_(cumsum);
        }

        // We need a temporary copy of offsets to use as sliding pointers
        torch::Tensor current_offsets = out_offsets.clone();
        out_indices = torch::empty({num_triangles * 3}, options_i32);

        compute_vertex_triangle_indices_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_triangles,
            (const int3*)triangles.data_ptr<int>(),
            current_offsets.data_ptr<int>(),
            out_indices.data_ptr<int>()
        );
    }

/**
 * @brief Flags vertices whose incident triangles do not form a single fan.
 * @details One thread per vertex, walking its incident triangles through the CSR map and
 * checking that they link edge to edge into one connected fan (or a single open strip at a
 * boundary). A vertex where two otherwise separate surface sheets meet at a point yields
 * several fans -- geometrically valid, topologically non-manifold, and a case most
 * downstream algorithms cannot handle.
 * @param[in] num_vertices Number of vertices.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] v2t_offsets Device array of per-vertex CSR offsets.
 * @param[in] v2t_counts Device array of per-vertex incidence counts.
 * @param[in] v2t_indices Device array of incident triangle indices.
 * @param[out] out_is_non_manifold Device array of per-vertex flags.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Detects only vertex-based non-manifoldness; edges shared by more than two
 * triangles are a separate check.
 */
__global__ void get_non_manifold_vertices_kernel(
        const uint32_t num_vertices,
        const int3* triangles,
        const int* v2t_offsets,
        const int* v2t_counts,
        const int* v2t_indices,
        bool* out_is_non_manifold)
    {
        uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
        if (v >= num_vertices) return;
        
        int count = v2t_counts[v];
        if (count == 0) {
            out_is_non_manifold[v] = false;
            return;
        }
        
        if (count > 64) {
            out_is_non_manifold[v] = true; // Fallback for safely avoiding overflow
            return;
        }
        
        int offset = v2t_offsets[v];
        int neighbors[128]; // max 64 triangles * 2
        
        for (int i = 0; i < count; ++i) {
            int3 t = triangles[v2t_indices[offset + i]];
            int n1 = -1, n2 = -1;
            if (t.x != v) { n1 = t.x; }
            if (t.y != v) { if (n1 == -1) n1 = t.y; else n2 = t.y; }
            if (t.z != v) { n2 = t.z; }
            neighbors[2*i + 0] = n1;
            neighbors[2*i + 1] = n2;
        }
        
        // 1. Check for bad edges (spoke shared by >2 triangles)
        for (int i = 0; i < count * 2; ++i) {
            int target = neighbors[i];
            int occurrences = 0;
            for (int j = 0; j < count * 2; ++j) {
                if (neighbors[j] == target) occurrences++;
            }
            if (occurrences > 2) {
                out_is_non_manifold[v] = true;
                return;
            }
        }
        
        // 2. Check for bowtie (disconnected components) via bitmask BFS
        unsigned long long visited = 1ULL;
        unsigned long long frontier = 1ULL;
        
        while (frontier != 0) {
            int current_idx = __ffsll(frontier) - 1;
            frontier &= ~(1ULL << current_idx);
            
            int n1_current = neighbors[2*current_idx + 0];
            int n2_current = neighbors[2*current_idx + 1];
            
            for (int i = 0; i < count; ++i) {
                if ((visited & (1ULL << i)) == 0) {
                    int n1_other = neighbors[2*i + 0];
                    int n2_other = neighbors[2*i + 1];
                    if (n1_current == n1_other || n1_current == n2_other ||
                        n2_current == n1_other || n2_current == n2_other) {
                        visited |= (1ULL << i);
                        frontier |= (1ULL << i);
                    }
                }
            }
        }
        
        unsigned long long expected_visited = (count == 64) ? ~0ULL : ((1ULL << count) - 1);
        if (visited != expected_visited) {
            out_is_non_manifold[v] = true;
        } else {
            out_is_non_manifold[v] = false;
        }
    }

    /**
     * @brief Flags vertices whose incident triangles do not form a single fan.
     * @details Detects the case where two surface sheets meet at a single point -- geometrically
     * valid but topologically non-manifold, and unusable by most downstream algorithms.
     */
    torch::Tensor get_non_manifold_vertices(
        const uint32_t num_vertices,
        const torch::Tensor& triangles,
        const torch::Tensor& v2t_offsets,
        const torch::Tensor& v2t_counts,
        const torch::Tensor& v2t_indices)
    {
        auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(triangles.device());
        torch::Tensor out_is_non_manifold = torch::empty({num_vertices}, options_bool);
        
        if (num_vertices == 0) return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(triangles.device()));
        
        int threads = NTHREADS;
        int blocks = (num_vertices + threads - 1) / threads;
        
        get_non_manifold_vertices_kernel<<<blocks, threads>>>(
            num_vertices,
            (const int3*)triangles.data_ptr<int>(),
            v2t_offsets.data_ptr<int>(),
            v2t_counts.data_ptr<int>(),
            v2t_indices.data_ptr<int>(),
            out_is_non_manifold.data_ptr<bool>()
        );
        
        return torch::nonzero(out_is_non_manifold).squeeze(1);
    }
/**
 * @brief Samples points uniformly across the mesh surface.
 * @details One thread per sample. Barycentric coordinates come from the standard
 * square-root warp of two uniform variates, which maps them uniformly over the triangle.
 * Combined with area-proportional triangle selection by the caller, the result is uniform
 * over the whole surface rather than biased towards small faces. Normals and colours are
 * interpolated at the sample.
 * @param[in] num_points Number of samples.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] tri_indices Device array of the triangle chosen for each sample.
 * @param[in] r1_r2 Device array of two uniform variates per sample.
 * @param[in] vertex_normals Device array of per-vertex normals, or `nullptr`.
 * @param[in] triangle_normals Device array of per-triangle normals, or `nullptr`.
 * @param[in] vertex_colors Device array of per-vertex colours, or `nullptr`.
 * @param[out] out_points Device array of sampled positions.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Uniformity depends on @p tri_indices being drawn proportionally to triangle area.
 */
__global__ void sample_points_triangle_mesh_kernel(
        const int num_points,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const int64_t *__restrict__ tri_indices,
        const float2 *__restrict__ r1_r2,
        const float3 *__restrict__ vertex_normals,
        const float3 *__restrict__ triangle_normals,
        const float3 *__restrict__ vertex_colors,
        float3 *__restrict__ out_points,
        float3 *__restrict__ out_normals,
        float3 *__restrict__ out_colors)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points)
            return;

        int tri_idx = tri_indices[idx];
        int3 tri = triangles[tri_idx];
        float2 r = r1_r2[idx];

        Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
        out_points[idx] = T.sample_point(r.x, r.y);

        float sqrt_r1 = sqrtf(r.x);
        float u = 1.0f - sqrt_r1;
        float v = r.y * sqrt_r1;
        float w = 1.0f - u - v;

        if (out_normals) {
            if (triangle_normals) {
                out_normals[idx] = triangle_normals[tri_idx];
            } else if (vertex_normals) {
                float3 n0 = vertex_normals[tri.x];
                float3 n1 = vertex_normals[tri.y];
                float3 n2 = vertex_normals[tri.z];
                float3 n = make_float3(
                    n0.x * u + n1.x * v + n2.x * w,
                    n0.y * u + n1.y * v + n2.y * w,
                    n0.z * u + n1.z * v + n2.z * w);
                
                float length = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
                if (length > 1e-8f) {
                    out_normals[idx] = make_float3(n.x / length, n.y / length, n.z / length);
                } else {
                    out_normals[idx] = make_float3(0.0f, 0.0f, 0.0f);
                }
            }
        }

        if (out_colors && vertex_colors) {
            float3 c0 = vertex_colors[tri.x];
            float3 c1 = vertex_colors[tri.y];
            float3 c2 = vertex_colors[tri.z];
            out_colors[idx] = make_float3(
                c0.x * u + c1.x * v + c2.x * w,
                c0.y * u + c1.y * v + c2.y * w,
                c0.z * u + c1.z * v + c2.z * w);
        }
    }

    /**
     * @brief Samples points uniformly over the mesh surface.
     * @details Selects triangles with probability proportional to area, then samples uniformly
     * within each, so the result is uniform over the surface rather than biased towards small
     * faces.
     */
    __host__ void sample_points_triangle_mesh(
        const int num_points,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const int64_t *__restrict__ tri_indices,
        const float2 *__restrict__ r1_r2,
        const float3 *__restrict__ vertex_normals,
        const float3 *__restrict__ triangle_normals,
        const float3 *__restrict__ vertex_colors,
        float3 *__restrict__ out_points,
        float3 *__restrict__ out_normals,
        float3 *__restrict__ out_colors)
    {
        int threads = NTHREADS;
        int blocks = (num_points + threads - 1) / threads;

        sample_points_triangle_mesh_kernel<<<blocks, threads>>>(
            num_points,
            vertices,
            triangles,
            tri_indices,
            r1_r2,
            vertex_normals,
            triangle_normals,
            vertex_colors,
            out_points,
            out_normals,
            out_colors);
    }

/**
 * @brief Counts the edges incident on each vertex.
 * @details One thread per unique edge, atomically incrementing both endpoints. The degree
 * is the normalising denominator of the uniform Laplacian.
 * @param[in] num_unique_edges Number of unique edges.
 * @param[in] unique_edges Device array of $2E$ vertex indices.
 * @param[out] vertex_degrees Device array of per-vertex degrees; must be zeroed first.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning @p vertex_degrees must be zeroed before launch.
 */
__global__ void compute_vertex_degree_kernel(
        const uint32_t num_unique_edges,
        const int *__restrict__ unique_edges,
        int *__restrict__ vertex_degrees)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_unique_edges) {
            int v0 = unique_edges[2 * idx];
            int v1 = unique_edges[2 * idx + 1];
            atomicAdd(&vertex_degrees[v0], 1);
            atomicAdd(&vertex_degrees[v1], 1);
        }
    }

    /**
     * @brief Counts the edges incident on each vertex.
     * @details The normalising denominator of the uniform Laplacian.
     */
    __host__ void compute_vertex_degree(
        const uint32_t num_unique_edges,
        const int *__restrict__ unique_edges,
        int *__restrict__ vertex_degrees)
    {
        if (num_unique_edges == 0) return;

        int threads = NTHREADS;
        int blocks = (num_unique_edges + threads - 1) / threads;

        compute_vertex_degree_kernel<<<blocks, threads>>>(
            num_unique_edges,
            unique_edges,
            vertex_degrees);
    }

/**
 * @brief Accumulates the unnormalised uniform (graph) Laplacian.
 * @details One thread per unique edge, adding each endpoint's displacement to the other.
 * The uniform Laplacian ignores edge lengths entirely, which makes it cheap and robust on
 * irregular triangulations but geometrically biased where triangle sizes vary.
 * @param[in] num_unique_edges Number of unique edges.
 * @param[in] unique_edges Device array of $2E$ vertex indices.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[out] vertex_lb_uniform Device array accumulating Laplacian vectors.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Vertices are shared between triangles, so accumulation uses `atomicAdd`; the
 * reduction order, and hence the last bits of the result, varies between runs.
 */
__global__ void compute_uniform_laplacian_kernel(
        const uint32_t num_unique_edges,
        const int *__restrict__ unique_edges,
        const float3 *__restrict__ vertices,
        float3 *__restrict__ vertex_lb_uniform)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_unique_edges) {
            int u = unique_edges[2 * idx];
            int v = unique_edges[2 * idx + 1];
            
            float3 pos_u = vertices[u];
            float3 pos_v = vertices[v];
            
            atomicAdd(&vertex_lb_uniform[u], pos_v - pos_u);
            atomicAdd(&vertex_lb_uniform[v], pos_u - pos_v);
        }
    }

/**
 * @brief Divides the accumulated uniform Laplacian by each vertex's degree.
 * @details One thread per vertex, completing the accumulate-then-normalise pair.
 * @param[in] num_vertices Number of vertices.
 * @param[in] vertex_degrees Device array of per-vertex degrees.
 * @param[in,out] vertex_lb_uniform Device array of Laplacian vectors, normalised in place.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Isolated vertices have degree zero; guard against division by zero.
 */
__global__ void normalize_uniform_laplacian_kernel(
        const uint32_t num_vertices,
        const int *__restrict__ vertex_degrees,
        float3 *__restrict__ vertex_lb_uniform)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_vertices) {
            int degree = vertex_degrees[idx];
            if (degree > 0) {
                vertex_lb_uniform[idx] /= degree;
            }
        }
    }

    /**
     * @brief Computes the uniform (graph) Laplacian at every vertex.
     * @details Accumulates neighbour displacements, then divides by degree. Cheap and robust on
     * irregular triangulations, but geometrically biased where triangle sizes vary -- prefer the
     * cotangent form when geometry matters.
     */
    __host__ void compute_uniform_laplacian(
        const uint32_t num_vertices,
        const uint32_t num_unique_edges,
        const int *__restrict__ unique_edges,
        const int *__restrict__ vertex_degrees,
        const float3 *__restrict__ vertices,
        float3 *__restrict__ vertex_lb_uniform)
    {
        if (num_unique_edges == 0 || num_vertices == 0) return;

        int threads = NTHREADS;
        int blocks = (num_unique_edges + threads - 1) / threads;

        compute_uniform_laplacian_kernel<<<blocks, threads>>>(
            num_unique_edges,
            unique_edges,
            vertices,
            vertex_lb_uniform);

        int blocks_vert = (num_vertices + threads - 1) / threads;
        normalize_uniform_laplacian_kernel<<<blocks_vert, threads>>>(
            num_vertices,
            vertex_degrees,
            vertex_lb_uniform);
    }

/**
 * @brief Accumulates the mixed Voronoi area of each vertex.
 * @details One thread per triangle, distributing area to its corners by the Meyer et al.
 * (2003) mixed rule: the circumcentric Voronoi region for a well-shaped triangle, and a
 * barycentric fallback for obtuse ones whose circumcentre falls outside. Without that
 * fallback obtuse triangles would contribute negative area and the curvature operators
 * built on it would diverge.
 * @param[in] num_triangles Number of triangles.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[out] voronoi_areas Device array of per-vertex areas; must be zeroed first.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Vertices are shared between triangles, so accumulation uses `atomicAdd`; the
 * reduction order, and hence the last bits of the result, varies between runs.
 */
__global__ void compute_voronoi_areas_kernel(
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        float *__restrict__ voronoi_areas)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles) {
            int3 tri = triangles[idx];
            int v0 = tri.x;
            int v1 = tri.y;
            int v2 = tri.z;
            
            float3 p0 = vertices[v0];
            float3 p1 = vertices[v1];
            float3 p2 = vertices[v2];
            
            float area = triangle::compute_area(p0, p1, p2);
            float area0, area1, area2;
            
            if (triangle::is_obtuse(p0, p1, p2)) {
                float3 e01 = p1 - p0;
                float3 e12 = p2 - p1;
                float3 e20 = p0 - p2;
                
                float d0 = maths::dot(e01, -e20);
                float d1 = maths::dot(e12, -e01);
                
                if (d0 < 0.0f) {
                    area0 = area * 0.5f;
                    area1 = area * 0.25f;
                    area2 = area * 0.25f;
                } else if (d1 < 0.0f) {
                    area0 = area * 0.25f;
                    area1 = area * 0.5f;
                    area2 = area * 0.25f;
                } else {
                    area0 = area * 0.25f;
                    area1 = area * 0.25f;
                    area2 = area * 0.5f;
                }
            } else {
                float cot0, cot1, cot2;
                Triangle(p0, p1, p2).compute_cotangents(cot0, cot1, cot2);
                
                float3 e01 = p1 - p0;
                float3 e12 = p2 - p1;
                float3 e20 = p0 - p2;
                
                float l01 = maths::dot(e01, e01);
                float l12 = maths::dot(e12, e12);
                float l20 = maths::dot(e20, e20);
                
                area0 = 0.125f * (l01 * cot2 + l20 * cot1);
                area1 = 0.125f * (l01 * cot2 + l12 * cot0);
                area2 = 0.125f * (l20 * cot1 + l12 * cot0);
            }
            
            atomicAdd(&voronoi_areas[v0], area0);
            atomicAdd(&voronoi_areas[v1], area1);
            atomicAdd(&voronoi_areas[v2], area2);
        }
    }

    /**
     * @brief Computes the mixed Voronoi area of every vertex.
     * @details Uses the Meyer et al. (2003) mixed rule, falling back to barycentric areas on
     * obtuse triangles whose circumcentre lies outside. Without that fallback the areas could go
     * negative and the operators built on them would diverge.
     */
    __host__ void compute_voronoi_areas(
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        float *__restrict__ voronoi_areas)
    {
        if (num_triangles == 0) return;
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        compute_voronoi_areas_kernel<<<blocks, threads>>>(
            num_triangles, triangles, vertices, voronoi_areas);
    }

/**
 * @brief Accumulates the unnormalised cotangent Laplace-Beltrami operator.
 * @details One thread per triangle, adding the cotangent weights
 * $\tfrac{1}{2}(\cot \alpha + \cot \beta)$ of its edges. Unlike the uniform
 * Laplacian these weights encode the surface's actual geometry, which is what makes the
 * operator converge to the true Laplace-Beltrami operator under refinement.
 * @param[in] num_triangles Number of triangles.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[out] vertex_lb_cot Device array accumulating Laplacian vectors.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Cotangent weights become unbounded as a triangle degenerates and are negative
 * for obtuse angles, so a poor-quality mesh can produce a non-positive-definite operator.
 * @warning Vertices are shared between triangles, so accumulation uses `atomicAdd`; the
 * reduction order, and hence the last bits of the result, varies between runs.
 */
__global__ void compute_cotangent_laplacian_kernel(
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        float3 *__restrict__ vertex_lb_cot)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles) {
            int3 tri = triangles[idx];
            int v0 = tri.x;
            int v1 = tri.y;
            int v2 = tri.z;
            
            float3 p0 = vertices[v0];
            float3 p1 = vertices[v1];
            float3 p2 = vertices[v2];
            
            float cot0, cot1, cot2;
            Triangle(p0, p1, p2).compute_cotangents(cot0, cot1, cot2);
            
            // Contribution to edge (v1, v2) from v0
            float3 w0 = cot0 * (p2 - p1);
            atomicAdd(&vertex_lb_cot[v1], w0);
            atomicAdd(&vertex_lb_cot[v2], -w0);
            
            // Contribution to edge (v2, v0) from v1
            float3 w1 = cot1 * (p0 - p2);
            atomicAdd(&vertex_lb_cot[v2], w1);
            atomicAdd(&vertex_lb_cot[v0], -w1);
            
            // Contribution to edge (v0, v1) from v2
            float3 w2 = cot2 * (p1 - p0);
            atomicAdd(&vertex_lb_cot[v0], w2);
            atomicAdd(&vertex_lb_cot[v1], -w2);
        }
    }

/**
 * @brief Divides the cotangent Laplacian by each vertex's mixed Voronoi area.
 * @details One thread per vertex. The area normalisation converts the accumulated
 * integrated quantity into a pointwise one, completing the discrete Laplace-Beltrami
 * operator.
 * @param[in] num_vertices Number of vertices.
 * @param[in] voronoi_areas Device array of per-vertex mixed Voronoi areas.
 * @param[in,out] vertex_lb_cot Device array of Laplacian vectors, normalised in place.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Vertices with near-zero area amplify the result; clamp or filter degenerate
 * neighbourhoods first.
 */
__global__ void normalize_cotangent_laplacian_kernel(
        const uint32_t num_vertices,
        const float *__restrict__ voronoi_areas,
        float3 *__restrict__ vertex_lb_cot)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_vertices) {
            float area = voronoi_areas[idx];
            if (area > 1e-8f) {
                float inv_area = 1.0f / (2.0f * area);
                vertex_lb_cot[idx] *= inv_area;
            }
        }
    }

    /**
     * @brief Computes the discrete Laplace-Beltrami operator at every vertex.
     * @details Accumulates cotangent weights per triangle, then normalises by mixed Voronoi area.
     * Unlike the uniform Laplacian these weights encode the surface's actual geometry, so the
     * operator converges under refinement.
     * @warning Cotangent weights are unbounded for degenerate triangles and negative for obtuse
     * angles, so a poor-quality mesh can yield a non-positive-definite operator.
     */
    __host__ void compute_cotangent_laplacian(
        const uint32_t num_vertices,
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        float *__restrict__ voronoi_areas,
        float3 *__restrict__ vertex_lb_cot)
    {
        if (num_triangles == 0 || num_vertices == 0) return;

        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;

        compute_cotangent_laplacian_kernel<<<blocks, threads>>>(
            num_triangles,
            triangles,
            vertices,
            vertex_lb_cot);

        int blocks_vert = (num_vertices + threads - 1) / threads;
        normalize_cotangent_laplacian_kernel<<<blocks_vert, threads>>>(
            num_vertices,
            voronoi_areas,
            vertex_lb_cot);
    }

/**
 * @brief Accumulates the interior angles meeting at each vertex.
 * @details One thread per triangle, adding each of its three interior angles to the
 * corresponding corner. The total is the angle sum entering the Gaussian curvature
 * formula.
 * @param[in] num_triangles Number of triangles.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[out] vertex_angle_sum Device array of per-vertex angle sums; must be zeroed first.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Vertices are shared between triangles, so accumulation uses `atomicAdd`; the
 * reduction order, and hence the last bits of the result, varies between runs.
 */
__global__ void compute_incident_angles_kernel(
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        float *__restrict__ vertex_angle_sum)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles) {
            int3 tri = triangles[idx];
            float a0, a1, a2;
            Triangle(vertices[tri.x], vertices[tri.y], vertices[tri.z]).compute_angles(a0, a1, a2);
            
            atomicAdd(&vertex_angle_sum[tri.x], a0);
            atomicAdd(&vertex_angle_sum[tri.y], a1);
            atomicAdd(&vertex_angle_sum[tri.z], a2);
        }
    }

/**
 * @brief Computes discrete Gaussian curvature from angle deficit.
 * @details One thread per vertex, evaluating $K = (2\pi - \sum \theta_i) / A$ -- the
 * discrete Gauss-Bonnet form, where curvature is the failure of the incident angles to sum
 * to a full turn, normalised by the mixed Voronoi area.
 * @param[in] num_vertices Number of vertices.
 * @param[in] voronoi_areas Device array of per-vertex mixed Voronoi areas.
 * @param[in] vertex_angle_sum Device array of per-vertex angle sums.
 * @param[out] gaussian_curvature Device array of per-vertex curvature values.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Boundary vertices have no full angular neighbourhood, so the $2\pi$ deficit is
 * meaningless there; treat their values as invalid.
 * @warning Near-zero Voronoi areas amplify the deficit without bound.
 */
__global__ void finalize_gaussian_curvature_kernel(
        const uint32_t num_vertices,
        const float *__restrict__ voronoi_areas,
        const float *__restrict__ vertex_angle_sum,
        float *__restrict__ gaussian_curvature)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_vertices) {
            float area = voronoi_areas[idx];
            float angle_sum = vertex_angle_sum[idx];
            // 2.0f * M_PI = 6.28318530718f
            gaussian_curvature[idx] = (6.28318530718f - angle_sum) / area;
        }
    }

    /**
     * @brief Computes discrete Gaussian curvature at every vertex.
     * @details Evaluates the angle deficit $K = (2\pi - \sum \theta_i) / A$ of the discrete
     * Gauss-Bonnet theorem.
     * @note Boundary vertices lack a full angular neighbourhood, so their values are not
     * meaningful.
     */
    __host__ void compute_gaussian_curvature(
        const uint32_t num_vertices,
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        const float *__restrict__ voronoi_areas,
        float *__restrict__ vertex_angle_sum,
        float *__restrict__ gaussian_curvature)
    {
        if (num_triangles == 0 || num_vertices == 0) return;
        
        int threads = NTHREADS;
        int blocks_tri = (num_triangles + threads - 1) / threads;
        compute_incident_angles_kernel<<<blocks_tri, threads>>>(
            num_triangles, triangles, vertices, vertex_angle_sum);
            
        int blocks_vert = (num_vertices + threads - 1) / threads;
        finalize_gaussian_curvature_kernel<<<blocks_vert, threads>>>(
            num_vertices, voronoi_areas, vertex_angle_sum, gaussian_curvature);
    }
/**
 * @brief Finds any triangle not yet reached by the winding-repair traversal.
 * @details One thread per triangle, racing to claim a seed for the next connected
 * component. A mesh in several disconnected pieces needs one traversal per piece, and this
 * is how the host discovers the next starting point.
 * @param[in] num_triangles Number of triangles.
 * @param[in] visited Device array of per-triangle visit flags.
 * @param[out] seed Device slot receiving an unvisited triangle index.
 * @param[out] found Device flag set when a seed was written.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Which unvisited triangle wins the race is unspecified; any is a valid seed.
 */
__global__ void find_unvisited_kernel(
        const int num_triangles,
        const int *__restrict__ visited,
        int *__restrict__ seed,
        int *__restrict__ found)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_triangles) return;
        
        if (visited[idx] == 0) {
            if (atomicCAS(found, 0, 1) == 0) {
                *seed = idx;
            }
        }
    }

/**
 * @brief Propagates consistent winding across the mesh, one frontier layer per launch.
 * @details One thread per frontier triangle. Each visits its edge-adjacent neighbours
 * through the vertex-to-triangle map and, where a shared edge is traversed in the same
 * direction by both -- the signature of opposed winding -- flips the neighbour before
 * enqueueing it. The host relaunches until the frontier empties, so a whole connected
 * component ends up consistently oriented.
 * @param[in] num_triangles Number of triangles.
 * @param[in,out] triangles Device array of triangle vertex indices, flipped in place.
 * @param[in] v2t_offsets Device array of per-vertex CSR offsets.
 * @param[in] v2t_counts Device array of per-vertex incidence counts.
 * @param[in] v2t_indices Device array of incident triangle indices.
 * @param[in,out] visited Device array of per-triangle visit flags.
 * @param[in] frontier Device array of triangle indices to expand.
 * @param[in] frontier_size Number of entries in @p frontier.
 * @param[out] next_frontier Device array receiving the following layer.
 * @param[in,out] next_frontier_size Device counter, atomically incremented per insertion.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Only makes winding *consistent*, not outward-facing; a whole component may end
 * up inverted. component_signed_volume_kernel() decides that separately.
 * @warning Non-manifold edges shared by more than two triangles have no well-defined
 * propagation and may be left inconsistent.
 */
__global__ void fix_winding_bfs_kernel(
        const int num_triangles,
        int3 *__restrict__ triangles,
        const int *__restrict__ v2t_offsets,
        const int *__restrict__ v2t_counts,
        const int *__restrict__ v2t_indices,
        int *__restrict__ visited,
        const int *__restrict__ frontier,
        const int frontier_size,
        int *__restrict__ next_frontier,
        int *__restrict__ next_frontier_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= frontier_size) return;

        int tri_id = frontier[idx];
        int3 tri = triangles[tri_id];

        int edge_v[3][2] = {{tri.x, tri.y}, {tri.y, tri.z}, {tri.z, tri.x}};
        
        for (int e = 0; e < 3; ++e) {
            int v0 = edge_v[e][0];
            int v1 = edge_v[e][1];
            
            int shared_faces[10];
            int num_shared = 0;
            
            int start0 = v2t_offsets[v0];
            int end0 = start0 + v2t_counts[v0];
            int start1 = v2t_offsets[v1];
            int end1 = start1 + v2t_counts[v1];
            for (int i = start0; i < end0; ++i) {
                int f0 = v2t_indices[i];
                for (int j = start1; j < end1; ++j) {
                    if (f0 == v2t_indices[j]) {
                        if (num_shared < 10) {
                            shared_faces[num_shared++] = f0;
                        }
                    }
                }
            }
            
            for (int k = 0; k < num_shared; ++k) {
                int neighbor_id = shared_faces[k];
                if (neighbor_id != tri_id) {
                    if (atomicCAS(&visited[neighbor_id], 0, 1) == 0) {
                        int3 n_tri = triangles[neighbor_id];
                        bool n_has_v0_v1 = (n_tri.x == v0 && n_tri.y == v1) || 
                                           (n_tri.y == v0 && n_tri.z == v1) || 
                                           (n_tri.z == v0 && n_tri.x == v1);
                        if (n_has_v0_v1) {
                            triangles[neighbor_id] = make_int3(n_tri.x, n_tri.z, n_tri.y);
                        }
                        int push_idx = atomicAdd(next_frontier_size, 1);
                        next_frontier[push_idx] = neighbor_id;
                    }
                }
            }
        }
    }

/**
 * @brief Accumulates the signed volume of one connected component.
 * @details One thread per face, summing the signed volumes of the tetrahedra formed with
 * the origin by the divergence theorem. A closed, outward-wound component gives a positive
 * total; a negative one means the component is consistently but inwardly wound.
 * @param[in] num_component_faces Number of faces in the component.
 * @param[in] component_faces Device array of the component's triangle indices.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[out] volumes Device accumulator for the signed volume.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Meaningful only for a closed component; an open surface gives an
 * origin-dependent value with no orientation interpretation.
 * @warning Vertices are shared between triangles, so accumulation uses `atomicAdd`; the
 * reduction order, and hence the last bits of the result, varies between runs.
 */
__global__ void component_signed_volume_kernel(
        const int num_component_faces,
        const int *__restrict__ component_faces,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ volumes)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_component_faces) return;
        
        int tri_id = component_faces[idx];
        int3 tri = triangles[tri_id];
        float3 a = vertices[tri.x];
        float3 b = vertices[tri.y];
        float3 c = vertices[tri.z];
        
        float3 cross_bc = maths::cross(b, c);
        float vol = maths::dot(a, cross_bc) / 6.0f;
        volumes[idx] = vol;
    }

/**
 * @brief Reverses the winding of every face in a component.
 * @details One thread per face, swapping two vertex indices. Applied to components whose
 * signed volume came out negative, flipping them to outward-facing.
 * @param[in] num_component_faces Number of faces in the component.
 * @param[in] component_faces Device array of the component's triangle indices.
 * @param[in,out] triangles Device array of triangle vertex indices, flipped in place.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
__global__ void invert_component_kernel(
        const int num_component_faces,
        const int *__restrict__ component_faces,
        int3 *__restrict__ triangles)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_component_faces) return;
        
        int tri_id = component_faces[idx];
        int3 tri = triangles[tri_id];
        triangles[tri_id] = make_int3(tri.x, tri.z, tri.y);
    }

    /**
     * @brief Makes triangle winding consistent and outward-facing.
     * @details Breadth-first propagation orients each connected component consistently, then the
     * component's signed volume decides whether it needs inverting wholesale. The two steps are
     * separate because consistency alone cannot distinguish inward from outward.
     * @warning Non-manifold edges have no well-defined propagation and may remain inconsistent.
     */
    __host__ void fix_normals(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const torch::Tensor &v2t_offsets,
        const torch::Tensor &v2t_counts,
        const torch::Tensor &v2t_indices,
        int3 *__restrict__ triangles)
    {
        at::cuda::CUDAGuard device_guard(v2t_offsets.device());
        auto options = torch::TensorOptions().device(v2t_offsets.device()).dtype(torch::kInt32);
        torch::Tensor visited = torch::zeros({num_triangles}, options);
        torch::Tensor frontier = torch::empty({num_triangles}, options);
        torch::Tensor next_frontier = torch::empty({num_triangles}, options);
        torch::Tensor component_faces = torch::empty({num_triangles}, options);
        
        int *d_visited = visited.data_ptr<int>();
        int *d_frontier = frontier.data_ptr<int>();
        int *d_next_frontier = next_frontier.data_ptr<int>();
        int *d_component_faces = component_faces.data_ptr<int>();
        
        torch::Tensor seed_t = torch::zeros({1}, options);
        torch::Tensor found_t = torch::zeros({1}, options);
        torch::Tensor next_frontier_size_t = torch::zeros({1}, options);
        int *d_seed = seed_t.data_ptr<int>();
        int *d_found = found_t.data_ptr<int>();
        int *d_next_frontier_size = next_frontier_size_t.data_ptr<int>();
        
        int h_found = 0;
        int h_seed = 0;
        
        while (true) {
            cudaMemsetAsync(d_found, 0, sizeof(int), at::cuda::getCurrentCUDAStream());
            int blocks = (num_triangles + NTHREADS - 1) / NTHREADS;
            find_unvisited_kernel<<<blocks, NTHREADS, 0, at::cuda::getCurrentCUDAStream()>>>(num_triangles, d_visited, d_seed, d_found);
            cudaMemcpyAsync(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
            c10::cuda::getCurrentCUDAStream().synchronize();
            
            if (h_found == 0) break;
            
            cudaMemcpyAsync(&h_seed, d_seed, sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
            c10::cuda::getCurrentCUDAStream().synchronize();
            
            int h_one = 1;
            cudaMemcpyAsync(&d_visited[h_seed], &h_one, sizeof(int), cudaMemcpyHostToDevice, at::cuda::getCurrentCUDAStream());
            cudaMemcpyAsync(&d_frontier[0], &h_seed, sizeof(int), cudaMemcpyHostToDevice, at::cuda::getCurrentCUDAStream());
            
            int frontier_size = 1;
            int component_size = 0;
            
            while (frontier_size > 0) {
                cudaMemcpyAsync(&d_component_faces[component_size], d_frontier, frontier_size * sizeof(int), cudaMemcpyDeviceToDevice, at::cuda::getCurrentCUDAStream());
                component_size += frontier_size;
                
                cudaMemsetAsync(d_next_frontier_size, 0, sizeof(int), at::cuda::getCurrentCUDAStream());
                
                int bfs_blocks = (frontier_size + NTHREADS - 1) / NTHREADS;
                fix_winding_bfs_kernel<<<bfs_blocks, NTHREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                    num_triangles, triangles, 
                    v2t_offsets.data_ptr<int>(), v2t_counts.data_ptr<int>(), v2t_indices.data_ptr<int>(),
                    d_visited, d_frontier, frontier_size, d_next_frontier, d_next_frontier_size);
                    
                cudaMemcpyAsync(&frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
                c10::cuda::getCurrentCUDAStream().synchronize();
                
                int *tmp = d_frontier;
                d_frontier = d_next_frontier;
                d_next_frontier = tmp;
            }
            
            auto vol_options = torch::TensorOptions().device(v2t_offsets.device()).dtype(torch::kFloat32);
            torch::Tensor volumes = torch::empty({component_size}, vol_options);
            
            int vol_blocks = (component_size + NTHREADS - 1) / NTHREADS;
            component_signed_volume_kernel<<<vol_blocks, NTHREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                component_size, d_component_faces, vertices, triangles, volumes.data_ptr<float>());
                
            float comp_volume = volumes.sum().item<float>();
            
            if (comp_volume < 0.0f) {
                invert_component_kernel<<<vol_blocks, NTHREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                    component_size, d_component_faces, triangles);
            }
        }
    }
}
