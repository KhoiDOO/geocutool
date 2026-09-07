/**
 * @file triangle_mesh.h
 * @brief High-performance GPU discrete differential geometry and topological analysis engine for 3D triangle meshes.
 */

#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include "../maths/maths.h"
#include "../constants.h"
#include "../primitive/aabb.h"
#include "mesh_bvh.h"

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

/**
 * @brief GPU-accelerated Triangle Mesh representation supporting differential geometry and topology operators.
 */
class TriangleMesh
{
protected:
    uint32_t num_triangles; ///< Total number of triangle faces ($M$).

    torch::Tensor vertices;       ///< Vertex coordinates of shape [N, 3] (float32, CUDA).
    torch::Tensor vertex_normals; ///< Surface normals of shape [N, 3] (float32, CUDA).
    std::optional<int> vertex_normals_mode = std::nullopt; ///< Weighting used for the cached vertex normals: uniform, area, or angle.
    torch::Tensor vertex_colors;  ///< Per-vertex colors of shape [N, 3] (float32, CUDA).
    torch::Tensor vertex_degrees; ///< Discrete topological degrees of shape [N] (int32, CUDA).

    torch::Tensor triangles;        ///< Triangle corner indices of shape [M, 3] (int32, CUDA).
    torch::Tensor triangle_normals; ///< Per-triangle face normals of shape [M, 3] (float32, CUDA).
    torch::Tensor edge_normals;     ///< Edge pseudonormals of shape [3*M, 3] for SDF sign evaluation.
    torch::Tensor triangle_areas;   ///< Individual face surface areas of shape [M] (float32, CUDA).
    torch::Tensor surface_area;     ///< Total integrated mesh surface area scalar.

    std::optional<MeshBVH> bvh; ///< Lazily built hierarchy over the mesh triangles.
    std::optional<torch::Tensor> flood_fill_mask = std::nullopt; ///< Cached single-level flood fill occupancy labels.
    std::optional<std::vector<float>> flood_grid_min = std::nullopt; ///< Lower corner of the grid the flood fill was computed on.
    std::optional<std::vector<float>> flood_grid_max = std::nullopt; ///< Upper corner of the grid the flood fill was computed on.
    std::optional<std::vector<int64_t>> flood_grid_res = std::nullopt; ///< Resolution of the grid the flood fill was computed on.

    std::optional<torch::Tensor> cf_coarse_mask = std::nullopt; ///< Coarse-level labels from the coarse-fine flood fill.
    std::optional<torch::Tensor> cf_boundary_lookup = std::nullopt; ///< Dense coarse-index to boundary-slot lookup, -1 outside the boundary set.
    std::optional<torch::Tensor> cf_fine_masks = std::nullopt; ///< Per-boundary-block fine voxel labels.
    std::optional<std::vector<int64_t>> cf_block_size = std::nullopt; ///< Fine voxels per coarse block.
    std::optional<std::vector<int64_t>> cf_coarse_res = std::nullopt; ///< Coarse grid resolution.

    std::optional<bool> opt_edge_manifold; ///< Cached edge-manifoldness result, ignoring boundaries.
    std::optional<bool> opt_edge_manifold_w_boundary; ///< Cached edge-manifoldness result, allowing boundaries.
    std::optional<bool> opt_vertex_manifold; ///< Cached vertex-manifoldness result.
    std::optional<bool> opt_self_intersected; ///< Cached self-intersection result.

    torch::Tensor vertex_lb_uniform;   ///< Uniform Laplace-Beltrami vector of shape [N, 3].
    torch::Tensor vertex_lb_cotangent; ///< Cotangent Laplace-Beltrami vector of shape [N, 3].
    torch::Tensor voronoi_areas;       ///< Meyer et al. mixed Voronoi cell areas of shape [N].
    torch::Tensor gaussian_curvature;  ///< Discrete Gaussian curvature $K$ of shape [N].
    
    torch::Tensor edges;                       ///< Unique edges of shape [E, 2].
    torch::Tensor edge_to_triangle_offsets;    ///< CSR offsets for edge-to-triangle map.
    torch::Tensor edge_to_triangle_counts;     ///< Number of incident triangles per edge.
    torch::Tensor edge_to_triangle_indices;    ///< Flattened incident triangle indices.

    torch::Tensor vertex_to_triangle_offsets;  ///< CSR offsets for vertex-to-triangle map.
    torch::Tensor vertex_to_triangle_counts;   ///< Number of incident triangles per vertex.
    torch::Tensor vertex_to_triangle_indices;  ///< Flattened incident triangle indices.

public:
    /**
     * @brief Constructs a TriangleMesh from vertex and triangle face index tensors.
     * 
     * @param[in] in_vertices        (N, 3) float32 coordinates on CUDA device.
     * @param[in] in_triangles       (M, 3) int32 face indices on CUDA device.
     * @param[in] in_vertex_normals  Optional (N, 3) float32 initial normals.
     * @param[in] in_vertex_colors   Optional (N, 3) float32 initial colors.
     */
    TriangleMesh(
        const torch::Tensor &in_vertices,
        const torch::Tensor &in_triangles,
        std::optional<torch::Tensor> in_vertex_normals = std::nullopt,
        std::optional<torch::Tensor> in_vertex_colors = std::nullopt);

    /** @brief Returns the total number of triangle faces. */
    uint32_t get_num_triangles() const { return num_triangles; }
    /** @brief Returns the (N, 3) float32 vertex coordinates tensor. */
    torch::Tensor get_vertices() const { return vertices; }
    /** @brief Returns or computes (N, 3) vertex normals (mode 0: uniform, 1: area-weighted, 2: angle-weighted). */
    torch::Tensor get_vertex_normals(int mode = 0);
    /** @brief Returns the (N, 3) vertex colors tensor. */
    torch::Tensor get_vertex_colors() const { return vertex_colors; }
    /** @brief Returns the (M, 3) int32 triangle face index tensor. */
    torch::Tensor get_triangles() const { return triangles; }

    /** @brief Computes normalized outward face normals for all triangles on GPU. */
    void compute_triangle_normals();
    /** @brief Computes vertex normals with area or angle weighting on GPU. */
    void compute_vertex_normals(int mode = 0);
    /** @brief Computes angle-weighted edge pseudonormals for all half-edges on GPU. */
    void compute_edge_normals();
    /** @brief Alias for compute_edge_normals. */
    void compute_edge_normal() { compute_edge_normals(); }
    /** @brief Computes surface areas for all triangle faces on GPU. */
    void compute_triangle_areas();

    /** @brief Returns (M,) float32 tensor of individual triangle areas. */
    torch::Tensor get_triangle_areas();
    /** @brief Returns (M, 3) float32 tensor of triangle face normals. */
    torch::Tensor get_triangle_normals();
    /** @brief Returns (3*M, 3) float32 tensor of edge pseudonormals. */
    torch::Tensor get_edge_normals();
    /** @brief Alias for get_edge_normals. */
    torch::Tensor get_edge_normal() { return get_edge_normals(); }
    /** @brief Returns the integrated total surface area scalar tensor. */
    torch::Tensor get_surface_area();
    
    /** @brief Computes discrete topological vertex valences (degrees) on GPU. */
    void compute_vertex_degrees();
    /** @brief Returns (N,) int32 tensor of vertex degrees. */
    torch::Tensor get_vertex_degrees();
    /** @brief Returns percentage of vertices having regular 5, 6, or 7 valences. */
    float get_valence_567_percentage();
    
    /** @brief Computes Uniform Laplace-Beltrami operator vectors on GPU. */
    void compute_vertex_lb_uniform();
    /** @brief Returns (N, 3) Uniform Laplace-Beltrami vectors. */
    torch::Tensor get_vertex_lb_uniform();
    
    /** @brief Computes Cotangent-weighted Laplace-Beltrami operator vectors on GPU. */
    void compute_vertex_lb_cotangent();
    /** @brief Returns (N, 3) Cotangent Laplace-Beltrami vectors. */
    torch::Tensor get_vertex_lb_cotangent();
    
    /** @brief Computes Meyer et al. mixed Voronoi cell areas on GPU. */
    void compute_voronoi_areas();
    /** @brief Returns (N,) float32 tensor of mixed Voronoi cell areas. */
    torch::Tensor get_voronoi_areas();
    
    /** @brief Computes discrete Gaussian curvature $K = (2\pi - \sum \theta_j)/A_i$ on GPU. */
    void compute_gaussian_curvature();
    /** @brief Returns (N,) float32 tensor of Gaussian curvatures. */
    torch::Tensor get_gaussian_curvature();
    
    /**
     * @brief Computes Mean Curvature $H = \frac{1}{2}\|\Delta_{LB} x\|$.
     * @param[in] signed_curvature If true, preserves normal sign orientation.
     * @return (N,) float32 tensor of mean curvatures.
     */
    torch::Tensor get_mean_curvature(bool signed_curvature = false);

    /**
     * @brief Computes Principal Curvatures $k_1, k_2 = H \pm \sqrt{\max(0, H^2 - K)}$.
     * @param[in] signed_curvature If true, preserves normal sign.
     * @return (N, 2) float32 tensor of principal curvature pairs $(k_1, k_2)$.
     */
    torch::Tensor get_principal_curvatures(bool signed_curvature = true);
    
    /** @brief Computes Laplace-Beltrami tensor for specified mode (0: uniform, 1: cotangent). */
    torch::Tensor compute_laplacian(int mode);
    
    /** @brief Discovers vertices not referenced by any triangle face. */
    torch::Tensor get_isolated_vertices();
    /** @brief Returns the count of isolated unused vertices. */
    int32_t get_num_isolated_vertices();
    /** @brief Removes unreferenced vertices and compacts triangle indices in-place. */
    void remove_isolated_vertices();
    
    /** @brief Builds and returns a high-speed GPU Linear MeshBVH over triangle primitives. */
    MeshBVH build_bvh();

    /**
     * @brief Builds 3D volumetric flood-fill occupancy data on GPU for sign evaluation.
     */
    void build_flood_fill_data(
        std::optional<std::vector<float>> grid_min = std::nullopt,
        std::optional<std::vector<float>> grid_max = std::nullopt,
        std::optional<std::vector<int64_t>> res = std::nullopt,
        int connectivity = 6);
    /**
     * @brief Occupancy labels from the cached single-level flood fill.
     * @return `(Rx, Ry, Rz)` int8 labels.
     * @warning Requires build_flood_fill_data() to have been called.
     */
    torch::Tensor get_flood_fill_mask();
    /**
     * @brief Lower corner of the grid the flood fill was computed on.
     * @return `[x, y, z]`.
     */
    std::vector<float> get_flood_grid_min();
    /**
     * @brief Upper corner of the grid the flood fill was computed on.
     * @return `[x, y, z]`.
     */
    std::vector<float> get_flood_grid_max();
    /**
     * @brief Resolution of the grid the flood fill was computed on.
     * @return `[Rx, Ry, Rz]`.
     */
    std::vector<int64_t> get_flood_grid_res();

    /**
     * @brief Builds 2-Level Coarse-to-Fine (CF) Volumetric Flood Fill data (< 10 MB VRAM at 1024^3).
     */
    void build_flood_fill_cf_data(
        std::optional<std::vector<float>> grid_min = std::nullopt,
        std::optional<std::vector<float>> grid_max = std::nullopt,
        std::optional<std::vector<int64_t>> res = std::nullopt,
        std::optional<std::vector<int64_t>> block_size = std::nullopt,
        int connectivity = 6);
    /**
     * @brief Coarse-level labels from the cached coarse-fine flood fill.
     * @return `(Cx, Cy, Cz)` int8 labels.
     * @warning Requires the coarse-fine flood fill to have been built.
     */
    torch::Tensor get_cf_coarse_mask();
    /**
     * @brief Dense lookup from coarse index to boundary-block slot.
     * @return `(Cx, Cy, Cz)` int32 indices, -1 where the block is not on the boundary.
     */
    torch::Tensor get_cf_boundary_lookup();
    /**
     * @brief Per-boundary-block fine voxel labels.
     * @return `(N, Bx, By, Bz)` int8 labels.
     */
    torch::Tensor get_cf_fine_masks();
    /**
     * @brief Fine voxels per coarse block.
     * @return `[Bx, By, Bz]`.
     */
    std::vector<int64_t> get_cf_block_size();
    /**
     * @brief Coarse grid resolution.
     * @return `[Cx, Cy, Cz]`.
     */
    std::vector<int64_t> get_cf_coarse_res();

    /** @brief Returns colliding triangle index pairs $(N, 2)$. */
    torch::Tensor get_self_intersection();
    /** @brief Checks if the mesh contains any self-intersecting faces. */
    bool is_self_intersection();

    /**
     * @brief Queries closest point projections, distances, and SDF signs for arbitrary points.
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> query_points(
        const torch::Tensor &query_pts,
        bool return_sdf = false,
        bool return_prj_pts = true,
        int sign_mode = 0,
        int distance_mode = 0);

    /**
     * @brief Accelerated Ray-Mesh intersection queries (Möller-Trumbore).
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_ray_intersection(
        const torch::Tensor &ray_origins,
        const torch::Tensor &ray_dirs,
        bool return_distance = false);

    /**
     * @brief Uniform area-weighted surface point sampling on GPU.
     */
    std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>> sample_points(
        int num_points,
        bool uniform = false,
        bool return_normals = false,
        bool return_colors = false,
        bool use_triangle_normal = true);

    /** @brief Builds Compressed Sparse Row (CSR) edge-to-triangle incidence topology graph on GPU. */
    void compute_edges_to_triangle_map();
    /**
     * @brief The full edge-to-triangle connectivity map.
     * @details Returns the unique edges together with the CSR offsets, counts and indices
     * listing the triangles incident on each. Built on first use and cached.
     * @return A tuple of (edges, offsets, counts, indices).
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_edges_to_triangle_map();
    
    /**
     * @brief Unique edges of the mesh.
     * @return `(E, 2)` int32 vertex index pairs.
     */
    torch::Tensor get_edges();
    /**
     * @brief CSR offsets of the edge-to-triangle map.
     * @return `(E + 1,)` int32 offsets.
     */
    torch::Tensor get_edge_to_triangle_offsets();
    /**
     * @brief Number of triangles incident on each edge.
     * @return `(E,)` int32 counts; 1 marks a boundary edge and more than 2 a non-manifold one.
     */
    torch::Tensor get_edge_to_triangle_counts();
    /**
     * @brief Flattened incident triangle indices of the edge map.
     * @return `(sum(counts),)` int32 indices.
     */
    torch::Tensor get_edge_to_triangle_indices();

    /** @brief Builds CSR vertex-to-triangle incidence topology graph on GPU. */
    void compute_vertices_to_triangle_map();
    /**
     * @brief The full vertex-to-triangle connectivity map.
     * @details Returns the CSR offsets, counts and indices listing the triangles incident on
     * each vertex. Built on first use and cached.
     * @return A tuple of (offsets, counts, indices).
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_vertices_to_triangle_map();
    /**
     * @brief CSR offsets of the vertex-to-triangle map.
     * @return `(V + 1,)` int32 offsets.
     */
    torch::Tensor get_vertex_to_triangle_offsets();
    /**
     * @brief Number of triangles incident on each vertex.
     * @return `(V,)` int32 counts.
     */
    torch::Tensor get_vertex_to_triangle_counts();
    /**
     * @brief Flattened incident triangle indices of the vertex map.
     * @return `(sum(counts),)` int32 indices.
     */
    torch::Tensor get_vertex_to_triangle_indices();

    /** @brief Checks if every mesh edge is shared by at most 2 triangles (or exactly 2 if boundaries disallowed). */
    bool is_edge_manifold(bool allow_boundary_edge = true);
    /** @brief Checks if the vertex link is a single topological open/closed fan (umbrella property). */
    bool is_vertex_manifold();
    /** @brief Checks both edge and vertex 2-manifold conditions. */
    bool is_manifold(bool allow_boundary_edge = true);
    /** @brief Discovers non-manifold vertex indices. */
    torch::Tensor get_non_manifold_vertices();

    /** @brief Removes triangle faces specified by a boolean keep mask in-place. */
    void remove_triangles_by_mask(const torch::Tensor &keep_mask);
    
    /** @brief Reorients triangle normal winding order consistently across adjacent manifold faces. */
    void fix_normals();

    /** @brief Computes Euler characteristic $\chi = V - E + F$. */
    int32_t get_euler_characteristic();
    /** @brief Computes topological genus $g = 1 - \chi / 2$. */
    int32_t get_genus();

    /** @brief Computes mean and min triangle aspect ratios. */
    std::tuple<float, float> get_quality();
    /** @brief Computes per-triangle aspect ratio (mode 0: inradius/circumradius, mode 1: min edge/max edge). */
    torch::Tensor get_aspect_ratio(int mode);
    /** @brief Computes normalized inradius-to-circumradius ratio $2 \cdot r_{in} / r_{circ}$. */
    torch::Tensor get_radii_ratio();
    /** @brief Computes triangle regularity $4\sqrt{3} \cdot \text{Area} / \sum l_i^2$. */
    torch::Tensor get_triangle_regularity();
    /** @brief Computes radius-edge ratio. */
    torch::Tensor get_radius_edge_ratio();
    /** @brief Computes deviation of internal angles from equilateral $60^\circ$. */
    torch::Tensor get_angle_deviation();
};

namespace triangle_mesh
{
    __host__ void compute_triangle_normals(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float3 *__restrict__ triangle_normals);

    __host__ void compute_vertex_normals(
        const uint32_t num_vertices,
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ triangle_normals,
        float3 *__restrict__ vertex_normals,
        int mode = 0);

    __host__ void compute_edge_normals(
        const uint32_t num_triangles,
        const torch::Tensor &triangles,
        const float3 *__restrict__ triangle_normals,
        float3 *__restrict__ edge_normals);

    __host__ void compute_triangle_areas(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ triangle_areas);

    __host__ void compute_quality(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ qualities);

    __host__ void compute_aspect_ratio(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        int mode,
        float *__restrict__ aspect_ratios);

    __host__ void compute_radii_ratio(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ ratios);

    __host__ void compute_triangle_regularity(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ regularities);

    __host__ void compute_radius_edge_ratio(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ ratios);

    __host__ void compute_angle_deviation(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ deviations);

    __host__ void compute_triangle_aabbs(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float3 *__restrict__ aabb_mins,
        float3 *__restrict__ aabb_maxs);

    __host__ void compute_edges_to_triangle_map(
        const uint32_t num_triangles,
        const torch::Tensor &triangles,
        torch::Tensor &out_unique_edges,
        torch::Tensor &out_offsets,
        torch::Tensor &out_counts,
        torch::Tensor &out_sorted_triangle_indices);

    __host__ void build_vertices_to_triangle_map(
        const uint32_t num_vertices,
        const uint32_t num_triangles,
        const torch::Tensor& triangles,
        torch::Tensor& out_counts,
        torch::Tensor& out_offsets,
        torch::Tensor& out_indices);

    torch::Tensor get_non_manifold_vertices(
        const uint32_t num_vertices,
        const torch::Tensor& triangles,
        const torch::Tensor& v2t_offsets,
        const torch::Tensor& v2t_counts,
        const torch::Tensor& v2t_indices);

    __host__ void fix_normals(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const torch::Tensor &v2t_offsets,
        const torch::Tensor &v2t_counts,
        const torch::Tensor &v2t_indices,
        int3 *__restrict__ triangles);

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
        float3 *__restrict__ out_colors);
    
    __host__ void compute_vertex_degree(
        const uint32_t num_unique_edges,
        const int *__restrict__ unique_edges,
        int *__restrict__ vertex_degrees
    );

    __host__ void compute_uniform_laplacian(
        const uint32_t num_vertices,
        const uint32_t num_unique_edges,
        const int *__restrict__ unique_edges,
        const int *__restrict__ vertex_degrees,
        const float3 *__restrict__ vertices,
        float3 *__restrict__ vertex_lb_uniform
    );

    __host__ void compute_voronoi_areas(
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        float *__restrict__ voronoi_areas
    );

    __host__ void compute_cotangent_laplacian(
        const uint32_t num_vertices,
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        float *__restrict__ voronoi_areas,
        float3 *__restrict__ vertex_lb_cot
    );

    __host__ void compute_gaussian_curvature(
        const uint32_t num_vertices,
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        const float *__restrict__ voronoi_areas,
        float *__restrict__ vertex_angle_sum,
        float *__restrict__ gaussian_curvature
    );
}

#endif // TRIANGLE_MESH_H