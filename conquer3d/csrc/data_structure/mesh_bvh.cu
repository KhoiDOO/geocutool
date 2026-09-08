/**
 * @file mesh_bvh.cu
 * @brief CUDA kernel implementations for MeshBVH: Fast Winding Numbers (FWN), signed distance fields (SDF), and ray casting.
 */

#include "mesh_bvh.h"
#include "../primitive/triangle.h"
#include "../primitive/edge.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace mesh_bvh
{
/**
 * @brief Narrow-phase exact triangle-triangle intersection over candidate pairs.
 * @details One thread per candidate pair produced by the BVH broad phase. Pairs sharing a
 * vertex or an edge are discarded first -- adjacent triangles always touch and are not
 * self-intersections -- and the remainder go through the Moller triangle-triangle test.
 * Survivors are appended to a compacted output through an atomic counter.
 * @param[in] num_pairs Number of candidate pairs.
 * @param[in] query_ids Device array of first triangle indices.
 * @param[in] object_ids Device array of second triangle indices.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[out] out_query_ids Device array receiving confirmed first indices.
 * @param[out] out_object_ids Device array receiving confirmed second indices.
 * @param[in,out] valid_counter Device counter, atomically incremented per confirmed pair.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Output order is nondeterministic because slots are claimed by atomics.
 */
__global__ void filter_self_intersections_kernel(
        const int num_pairs,
        const int64_t *query_ids,
        const int64_t *object_ids,
        const float3 *vertices,
        const int3 *triangles,
        int64_t *out_query_ids,
        int64_t *out_object_ids,
        int64_t *valid_counter)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_pairs)
            return;

        int64_t idA = query_ids[idx];
        int64_t idB = object_ids[idx];

        int3 triA = triangles[idA];
        int3 triB = triangles[idB];

        // Filter shared vertices
        if (triA.x == triB.x || triA.x == triB.y || triA.x == triB.z ||
            triA.y == triB.x || triA.y == triB.y || triA.y == triB.z ||
            triA.z == triB.x || triA.z == triB.y || triA.z == triB.z)
        {
            return;
        }

        Triangle T1(vertices[triA.x], vertices[triA.y], vertices[triA.z]);
        Triangle T2(vertices[triB.x], vertices[triB.y], vertices[triB.z]);

        if (triangle::test_intersection(T1, T2))
        {
            uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int *)valid_counter, 1ULL);
            out_query_ids[write_idx] = idA;
            out_object_ids[write_idx] = idB;
        }
    }

    /**
     * @brief Launches exact triangle-triangle filtering over BVH candidate pairs.
     * @details Host wrapper; discards adjacent pairs and applies the Moller test to the rest.
     */
    void filter_self_intersections(
        const int num_pairs,
        const int64_t *query_ids,
        const int64_t *object_ids,
        const float3 *vertices,
        const int3 *triangles,
        int64_t *out_query_ids,
        int64_t *out_object_ids,
        int64_t *valid_counter)
    {
        int threads = NTHREADS;
        int blocks = (num_pairs + threads - 1) / threads;

        filter_self_intersections_kernel<<<blocks, threads>>>(
            num_pairs,
            query_ids,
            object_ids,
            vertices,
            triangles,
            out_query_ids,
            out_object_ids,
            valid_counter);
    }

/**
 * @brief Narrow-phase exact ray-triangle intersection over candidate pairs.
 * @details One thread per candidate pair from the broad phase, applying the
 * Moller-Trumbore test. Confirmed hits are compacted atomically, optionally carrying the
 * intersection point and ray parameter.
 * @param[in] num_pairs Number of candidate pairs.
 * @param[in] query_ids Device array of ray indices.
 * @param[in] object_ids Device array of triangle indices.
 * @param[in] ray_origins Device array of ray origins.
 * @param[in] ray_dirs Device array of ray directions.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[out] out_query_ids Device array receiving confirmed ray indices.
 * @param[out] out_object_ids Device array receiving confirmed triangle indices.
 * @param[out] out_intersect_pts Device array of intersection points.
 * @param[out] out_distances Device array of ray parameters, when requested.
 * @param[in] return_distance Whether distances are written.
 * @param[in,out] valid_counter Device counter, atomically incremented per hit.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Hits are emitted unordered, so they are not sorted along the ray. Sort by
 * distance if the nearest hit is required.
 */
__global__ void filter_ray_triangle_intersections_kernel(
        const int num_pairs,
        const int64_t *query_ids,
        const int64_t *object_ids,
        const float3 *ray_origins,
        const float3 *ray_dirs,
        const float3 *vertices,
        const int3 *triangles,
        int64_t *out_query_ids,
        int64_t *out_object_ids,
        float3 *out_intersect_pts,
        float *out_distances,
        bool return_distance,
        int64_t *valid_counter)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_pairs)
            return;

        int64_t ray_id = query_ids[idx];
        int64_t tri_id = object_ids[idx];

        Ray ray(ray_origins[ray_id], ray_dirs[ray_id]);

        int3 tri = triangles[tri_id];
        Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);

        float t_hit, u, v;
        if (T.is_intersect_ray(ray, t_hit, u, v))
        {
            uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int *)valid_counter, 1ULL);
            out_query_ids[write_idx] = ray_id;
            out_object_ids[write_idx] = tri_id;
            out_intersect_pts[write_idx] = ray.at(t_hit);
            if (return_distance)
            {
                out_distances[write_idx] = t_hit;
            }
        }
    }

    /**
     * @brief Launches exact ray-triangle filtering over BVH candidate pairs.
     * @details Host wrapper applying the Moller-Trumbore test to broad-phase candidates.
     */
    void filter_ray_triangle_intersections(
        const int num_pairs,
        const int64_t *query_ids,
        const int64_t *object_ids,
        const float3 *ray_origins,
        const float3 *ray_dirs,
        const float3 *vertices,
        const int3 *triangles,
        int64_t *out_query_ids,
        int64_t *out_object_ids,
        float3 *out_intersect_pts,
        float *out_distances,
        bool return_distance,
        int64_t *valid_counter)
    {
        int threads = NTHREADS;
        int blocks = (num_pairs + threads - 1) / threads;

        filter_ray_triangle_intersections_kernel<<<blocks, threads>>>(
            num_pairs,
            query_ids,
            object_ids,
            ray_origins,
            ray_dirs,
            vertices,
            triangles,
            out_query_ids,
            out_object_ids,
            out_intersect_pts,
            return_distance ? out_distances : nullptr,
            return_distance,
            valid_counter);
    }

    /**
     * @brief Determines inside/outside by counting ray crossings.
     * @details Casts a ray from the query point and counts how many triangles it crosses: an
     * odd count means inside. Robust to inconsistent winding, since only the number of
     * crossings matters, but sensitive to rays that graze an edge or vertex and are counted
     * ambiguously.
     * @param[in] p Query point.
     * @param[in] best_tri_id Nearest triangle, used to pick a well-conditioned ray direction.
     * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
     * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
     * @param[in] bvh_children Device array of BVH child index pairs.
     * @param[in] object_ids Device array mapping leaves to triangle indices.
     * @param[in] vertices Device array of mesh vertex coordinates.
     * @param[in] triangles Device array of triangle vertex indices.
     * @param[in] num_objects Number of triangles.
     * @return $-1$ inside, $+1$ outside.
     * @warning Uses a per-thread stack of `BVH_STACK_SIZE` entries in local memory.
     * @warning Requires a watertight mesh; a hole lets the ray escape and flips the parity.
     */
    __device__ __forceinline__ float compute_sign_ray_parity(
        const float3 &p,
        const int best_tri_id,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        const float3 *vertices,
        const int3 *triangles,
        const uint32_t num_objects)
    {
        int3 tri = triangles[best_tri_id];
        float3 v0 = vertices[tri.x];
        float3 v1 = vertices[tri.y];
        float3 v2 = vertices[tri.z];
        float3 centroid = (v0 + v1 + v2) / 3.0f;

        float3 ray_dir = maths::normalize_safe(
            centroid - p, make_float3(1.0f, 0.0f, 0.0f), 1e-6f);

        Ray ray2(p, ray_dir, 0.0f);

        int hit_count = 0;
        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0;

        while (stack_ptr >= 0)
        {
            int node_idx = stack[stack_ptr--];

            float t_hit_aabb;
            if (!ray2.is_intersect_aabb(bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx], t_hit_aabb))
            {
                continue;
            }

            if (node_idx >= num_objects - 1)
            {
                int tri_id = object_ids[node_idx - (num_objects - 1)];
                int3 tri = triangles[tri_id];
                Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);

                float t_hit, u, v;
                if (T.is_intersect_ray(ray2, t_hit, u, v))
                {
                    hit_count++;
                }
            }
            else
            {
                if (stack_ptr + 2 < BVH_STACK_SIZE)
                {
                    int2 children = bvh_children[node_idx];
                    stack[++stack_ptr] = children.x;
                    stack[++stack_ptr] = children.y;
                }
            }
        }

        return (hit_count % 2 == 1) ? -1.0f : 1.0f;
    }

    /**
     * @brief Signed solid angle a triangle subtends at a point.
     * @details Evaluated with the Van Oosterom-Strackee formula, whose `atan2` form stays
     * accurate for the small angles that dominate a distant summation. This is the per-triangle
     * term of the generalised winding number.
     * @param[in] p Viewpoint.
     * @param[in] a First triangle vertex.
     * @param[in] b Second triangle vertex.
     * @param[in] c Third triangle vertex.
     * @return Signed solid angle in steradians; the sign follows the winding.
     */
    __device__ __forceinline__ float compute_solid_angle_tri(float3 p, float3 a, float3 b, float3 c) {
        float3 a_v = a - p;
        float3 b_v = b - p;
        float3 c_v = c - p;
        float a_len = sqrtf(maths::dot2(a_v));
        float b_len = sqrtf(maths::dot2(b_v));
        float c_len = sqrtf(maths::dot2(c_v));
        
        float num = maths::dot(a_v, maths::cross(b_v, c_v));
        float den = a_len * b_len * c_len + maths::dot(a_v, b_v) * c_len + maths::dot(b_v, c_v) * a_len + maths::dot(c_v, a_v) * b_len;
        
        return 2.0f * atan2f(num, den);
    }

    /**
     * @brief Generalised winding number, evaluated hierarchically.
     * @details Sums the solid angles subtended by the mesh at the query point. For a closed
     * surface the total is $4\pi$ inside and $0$ outside, and it degrades gracefully rather
     * than failing on meshes with holes or self-intersections -- which is what makes it the
     * robust choice where ray parity breaks down. Subtrees far enough away relative to their
     * size are approximated by their precomputed ::WindingData aggregate instead of being
     * descended, controlled by an accuracy scale.
     * @param[in] p Query point.
     * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
     * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
     * @param[in] bvh_children Device array of BVH child index pairs.
     * @param[in] object_ids Device array mapping leaves to triangle indices.
     * @param[in] vertices Device array of mesh vertex coordinates.
     * @param[in] triangles Device array of triangle vertex indices.
     * @param[in] winding_data Device array of per-node winding aggregates.
     * @param[in] num_objects Number of triangles.
     * @return Winding number; near 1 inside a closed surface, near 0 outside.
     * @warning Uses a per-thread stack of `BVH_STACK_SIZE` entries in local memory.
     * @warning Cost grows as the accuracy scale tightens, since fewer subtrees can be
     * approximated. This is the most expensive sign mode.
     */
    __device__ __forceinline__ float compute_fast_winding_number(
        const float3 &p,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        const float3 *vertices,
        const int3 *triangles,
        const WindingData *winding_data,
        const uint32_t num_objects,
        const float accuracy_scale)
    {
        float total_omega = 0.0f;
        float accuracy_scale2 = accuracy_scale * accuracy_scale;

        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0;

        while (stack_ptr >= 0)
        {
            int node_idx = stack[stack_ptr--];

            WindingData data = winding_data[node_idx];
            float3 q = p - data.average_p;
            float qlength2 = maths::dot2(q);

            if (qlength2 > accuracy_scale2 * data.max_p_dist2)
            {
                if (qlength2 > 1e-6f) {
                    float qlength = sqrtf(qlength2);
                    float qlength3 = qlength2 * qlength;
                    float Omega_approx = -maths::dot(q, data.n) / qlength3;
                    total_omega += Omega_approx;
                }
            }
            else
            {
                if (node_idx >= num_objects - 1)
                {
                    int tri_id = object_ids[node_idx - (num_objects - 1)];
                    int3 tri = triangles[tri_id];
                    float3 v0 = vertices[tri.x];
                    float3 v1 = vertices[tri.y];
                    float3 v2 = vertices[tri.z];
                    
                    total_omega += compute_solid_angle_tri(p, v0, v1, v2);
                }
                else
                {
                    if (stack_ptr + 2 < BVH_STACK_SIZE)
                    {
                        int2 children = bvh_children[node_idx];
                        stack[++stack_ptr] = children.x;
                        stack[++stack_ptr] = children.y;
                    }
                }
            }
        }

        return total_omega * 0.07957747154f; // 1.0f / (4.0f * PI)
    }

    /**
     * @brief Determines inside/outside from the angle-weighted pseudonormal.
     * @details Compares the direction from the closest surface point to the query against the
     * pseudonormal of whichever feature -- face, edge, or vertex -- the closest point landed
     * on. Using angle-weighted normals at edges and vertices is what makes the test exact
     * rather than merely approximate near creases, and it needs no ray casting or summation,
     * making it the cheapest sign mode by a wide margin.
     * @param[in] p Query point.
     * @param[in] best_pt Closest point on the surface.
     * @param[in] best_tri_id Triangle containing @p best_pt.
     * @param[in] vertices Device array of mesh vertex coordinates.
     * @param[in] triangles Device array of triangle vertex indices.
     * @param[in] pseudonormal_vertices Device array of angle-weighted vertex pseudonormals.
     * @param[in] pseudonormal_edges Device array of edge pseudonormals.
     * @param[in] pseudonormal_faces Device array of face normals.
     * @return $-1$ inside, $+1$ outside.
     * @warning Valid only for a watertight, consistently wound mesh. On an open or
     * inconsistently oriented surface the sign flips unpredictably.
     */
    __device__ __forceinline__ float compute_pseudonormal_sign(
        const float3 &p,
        const float3 &best_pt,
        const int best_tri_id,
        const float3 *vertices,
        const int3 *triangles,
        const float3 *pseudonormal_vertices,
        const float3 *pseudonormal_edges,
        const float3 *pseudonormal_faces)
    {
        int3 tri = triangles[best_tri_id];
        float3 v0 = vertices[tri.x];
        float3 v1 = vertices[tri.y];
        float3 v2 = vertices[tri.z];
        
        float3 N = pseudonormal_faces[best_tri_id];
        bool found = false;
        const float eps_sq = 1e-10f;
        
        // 1. Check vertices
        if (maths::dot2(best_pt - v0) <= eps_sq) {
            N = pseudonormal_vertices[tri.x];
            found = true;
        } else if (maths::dot2(best_pt - v1) <= eps_sq) {
            N = pseudonormal_vertices[tri.y];
            found = true;
        } else if (maths::dot2(best_pt - v2) <= eps_sq) {
            N = pseudonormal_vertices[tri.z];
            found = true;
        }
        
        // 2. Check edges
        if (!found) {
            float3 u0 = v1 - v0;
            float len2_0 = maths::dot2(u0);
            if (len2_0 > 1e-10f) {
                float t0 = maths::saturate(maths::dot(best_pt - v0, u0) / len2_0);
                if (maths::dot2(v0 + t0 * u0 - best_pt) <= eps_sq) {
                    N = pseudonormal_edges[3 * best_tri_id + 0];
                    found = true;
                }
            }
        }
        if (!found) {
            float3 u1 = v2 - v1;
            float len2_1 = maths::dot2(u1);
            if (len2_1 > 1e-10f) {
                float t1 = maths::saturate(maths::dot(best_pt - v1, u1) / len2_1);
                if (maths::dot2(v1 + t1 * u1 - best_pt) <= eps_sq) {
                    N = pseudonormal_edges[3 * best_tri_id + 1];
                    found = true;
                }
            }
        }
        if (!found) {
            float3 u2 = v0 - v2;
            float len2_2 = maths::dot2(u2);
            if (len2_2 > 1e-10f) {
                float t2 = maths::saturate(maths::dot(best_pt - v2, u2) / len2_2);
                if (maths::dot2(v2 + t2 * u2 - best_pt) <= eps_sq) {
                    N = pseudonormal_edges[3 * best_tri_id + 2];
                    found = true;
                }
            }
        }
        
        return (maths::dot(p - best_pt, N) >= 0.0f) ? 1.0f : -1.0f;
    }

/**
 * @brief Finds the closest point on the mesh to each query, with an optional sign.
 * @details One thread per query. The BVH is descended with a running nearest distance that
 * prunes any subtree whose bound already exceeds it, and the exact closest point on each
 * surviving triangle is computed by barycentric projection.
 *
 * Sign determination is selectable because no single method suits every mesh: angle-weighted
 * pseudonormals are exact for watertight input and nearly free, generalised winding numbers
 * tolerate holes and self-intersections at the cost of the hierarchical ::WindingData
 * traversal, and flood-fill lookup handles the rest. The choice is the caller's
 * `sign_mode`.
 *
 * @param[in] num_queries Number of query points.
 * @param[in] num_objects Number of triangles.
 * @param[in] query_points Device array of query coordinates.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to triangle indices.
 * @param[in] winding_data Device array of per-node winding aggregates, or `nullptr`.
 * @param[in] pseudonormal_vertices Device array of angle-weighted vertex pseudonormals.
 * @param[in] pseudonormal_edges Device array of edge pseudonormals.
 * @param[in] pseudonormal_faces Device array of face normals.
 * @param[out] out_query_ids Device array of query indices.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Distances are unsigned unless a sign mode is selected; the sign convention is
 * negative inside.
 * @warning Traversal uses a per-thread stack of `BVH_STACK_SIZE` entries in local memory;
 * a pathologically unbalanced hierarchy can overflow it.
 * @warning Pseudonormal signing assumes a watertight, consistently oriented mesh. On open
 * or inconsistently wound surfaces the sign flips unpredictably -- prefer winding numbers
 * or flood fill there.
 */
__global__ void query_point_mesh_bvh_kernel(
        const int num_queries,
        const int num_objects,
        const float3 *__restrict__ query_points,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const WindingData *__restrict__ winding_data,
        const float3 *__restrict__ pseudonormal_vertices,
        const float3 *__restrict__ pseudonormal_edges,
        const float3 *__restrict__ pseudonormal_faces,
        int64_t *__restrict__ out_query_ids,
        int64_t *__restrict__ out_object_ids,
        float3 *__restrict__ out_projected_pts,
        float *__restrict__ out_distances,
        bool return_sdf,
        bool return_prj_pts,
        int sign_mode,
        const int8_t *__restrict__ flood_mask,
        float3 flood_min,
        float3 flood_spacing,
        int3 flood_dims,
        const int8_t *__restrict__ cf_coarse_mask,
        const int32_t *__restrict__ cf_boundary_lookup,
        const int8_t *__restrict__ cf_fine_masks,
        int3 cf_block_size,
        int3 cf_coarse_dims)
    {
        int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (q_idx >= num_queries)
            return;

        float3 p = query_points[q_idx];

        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0;

        float best_dist_sq = FLT_MAX;
        int best_tri_id = -1;
        float3 best_pt = make_float3(0, 0, 0);

        while (stack_ptr >= 0)
        {
            int node_idx = stack[stack_ptr--];

            float dist_sq = aabb::compute_squared_distance(p, bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx]);

            if (dist_sq > best_dist_sq)
                continue; // Prune branch

            if (node_idx >= num_objects - 1)
            {
                int tri_id = object_ids[node_idx - (num_objects - 1)];
                int3 tri = triangles[tri_id];
                Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);

                float3 closest_pt = T.compute_closest_point(p);
                float3 diff = p - closest_pt;
                float pt_dist_sq = maths::dot(diff, diff);

                if (pt_dist_sq < best_dist_sq)
                {
                    best_dist_sq = pt_dist_sq;
                    best_tri_id = tri_id;
                    best_pt = closest_pt;
                }
            }
            else
            {
                if (stack_ptr + 2 < BVH_STACK_SIZE)
                {
                    int2 children = bvh_children[node_idx];

                    float dist_l = aabb::compute_squared_distance(p, bvh_aabb_mins[children.x], bvh_aabb_maxs[children.x]);
                    float dist_r = aabb::compute_squared_distance(p, bvh_aabb_mins[children.y], bvh_aabb_maxs[children.y]);

                    if (dist_l > dist_r)
                    {
                        if (dist_l <= best_dist_sq) stack[++stack_ptr] = children.x;
                        if (dist_r <= best_dist_sq) stack[++stack_ptr] = children.y;
                    }
                    else
                    {
                        if (dist_r <= best_dist_sq) stack[++stack_ptr] = children.y;
                        if (dist_l <= best_dist_sq) stack[++stack_ptr] = children.x;
                    }
                }
            }
        }

        float dist = sqrtf(best_dist_sq);

        if (return_sdf && best_tri_id >= 0)
        {
            if (sign_mode == 0) {
                dist *= compute_sign_ray_parity(p, best_tri_id, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects);
            } else if (sign_mode == 1) {
                float wn = compute_fast_winding_number(p, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, winding_data, num_objects, 2.0f);
                dist *= (wn >= 0.5f) ? -1.0f : 1.0f;
            } else if (sign_mode == 2) {
                dist *= compute_pseudonormal_sign(p, best_pt, best_tri_id, vertices, triangles, pseudonormal_vertices, pseudonormal_edges, pseudonormal_faces);
            } else if (sign_mode == 3) {
                int i = roundf((p.x - flood_min.x) / flood_spacing.x);
                int j = roundf((p.y - flood_min.y) / flood_spacing.y);
                int k = roundf((p.z - flood_min.z) / flood_spacing.z);
                if (i >= 0 && i < flood_dims.x && j >= 0 && j < flood_dims.y && k >= 0 && k < flood_dims.z && flood_mask != nullptr) {
                    int idx = i * (flood_dims.y * flood_dims.z) + j * flood_dims.z + k;
                    int val = flood_mask[idx];
                    if (val != 2) {
                        // Dry (-2 or unreached interior vertex) -> strictly negative
                        dist = -dist;
                    }
                } else {
                    // Out of bounds of flood volume -> exterior (+1)
                }
            } else if (sign_mode == 4) {
                float wn = compute_fast_winding_number(p, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, winding_data, num_objects, 2.0f);
                if (wn <= 0.25f) {
                    // Strictly exterior (far outside) -> dist stays positive
                } else if (wn >= 0.75f) {
                    // Strictly interior (inside solid volume or overlapping components) -> strictly negative
                    dist = -dist;
                } else {
                    // Transition surface interface (0.25 < wn < 0.75) -> sub-voxel precision pseudonormal consensus
                    dist *= compute_pseudonormal_sign(p, best_pt, best_tri_id, vertices, triangles, pseudonormal_vertices, pseudonormal_edges, pseudonormal_faces);
                }
            } else if (sign_mode == 5) {
                int gi = roundf((p.x - flood_min.x) / flood_spacing.x);
                int gj = roundf((p.y - flood_min.y) / flood_spacing.y);
                int gk = roundf((p.z - flood_min.z) / flood_spacing.z);

                int ci = gi / cf_block_size.x;
                int cj = gj / cf_block_size.y;
                int ck = gk / cf_block_size.z;

                if (ci >= 0 && ci < cf_coarse_dims.x && cj >= 0 && cj < cf_coarse_dims.y && ck >= 0 && ck < cf_coarse_dims.z && cf_coarse_mask != nullptr) {
                    int c_idx = ci * (cf_coarse_dims.y * cf_coarse_dims.z) + cj * cf_coarse_dims.z + ck;
                    int c_val = cf_coarse_mask[c_idx];
                    if (c_val == 2) {
                        // Guaranteed Exterior Water -> positive sign
                    } else if (c_val == -1 || c_val == -2) {
                        // Guaranteed Interior -> strictly negative sign
                        dist = -dist;
                    } else if (c_val == 1 && cf_boundary_lookup != nullptr && cf_fine_masks != nullptr) {
                        int b_idx = cf_boundary_lookup[c_idx];
                        if (b_idx >= 0) {
                            int fi = gi % cf_block_size.x;
                            int fj = gj % cf_block_size.y;
                            int fk = gk % cf_block_size.z;
                            int fine_idx = b_idx * (cf_block_size.x * cf_block_size.y * cf_block_size.z) + fi * (cf_block_size.y * cf_block_size.z) + fj * cf_block_size.z + fk;
                            int fine_val = cf_fine_masks[fine_idx];
                            if (fine_val != 2) {
                                dist = -dist;
                            }
                        } else {
                            dist = -dist;
                        }
                    }
                }
            }
        }

        out_query_ids[q_idx] = q_idx;
        out_object_ids[q_idx] = best_tri_id;
        if (return_prj_pts && out_projected_pts != nullptr)
        {
            out_projected_pts[q_idx] = best_pt;
        }
        out_distances[q_idx] = dist;
    }

    /**
     * @brief Launches closest-point queries against the mesh, with an optional sign.
     * @details Host wrapper. The sign mode selects between pseudonormals, generalised winding
     * numbers, and flood-fill lookup, trading cost against tolerance of imperfect meshes.
     */
    void query_point_mesh_bvh(const int num_queries,
        const int num_objects,
        const float3 *query_points,
        const float3 *vertices,
        const int3 *triangles,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        const WindingData *winding_data,
        const float3 *pseudonormal_vertices,
        const float3 *pseudonormal_edges,
        const float3 *pseudonormal_faces,
        int64_t *out_query_ids,
        int64_t *out_object_ids,
        float3 *out_projected_pts,
        float *out_distances,
        bool return_sdf,
        bool return_prj_pts,
        int sign_mode,
        const int8_t *flood_mask,
        float3 flood_min,
        float3 flood_spacing,
        int3 flood_dims,
        const int8_t *cf_coarse_mask,
        const int32_t *cf_boundary_lookup,
        const int8_t *cf_fine_masks,
        int3 cf_block_size,
        int3 cf_coarse_dims)
    {
        int threads = NTHREADS;
        int blocks = (num_queries + threads - 1) / threads;

        query_point_mesh_bvh_kernel<<<blocks, threads>>>(
            num_queries,
            num_objects,
            query_points,
            vertices,
            triangles,
            bvh_aabb_mins,
            bvh_aabb_maxs,
            bvh_children,
            object_ids,
            winding_data,
            pseudonormal_vertices,
            pseudonormal_edges,
            pseudonormal_faces,
            out_query_ids,
            out_object_ids,
            out_projected_pts,
            out_distances,
            return_sdf,
            return_prj_pts,
            sign_mode,
            flood_mask,
            flood_min,
            flood_spacing,
            flood_dims,
            cf_coarse_mask,
            cf_boundary_lookup,
            cf_fine_masks,
            cf_block_size,
            cf_coarse_dims);
    }

/**
 * @brief Aggregates generalised winding number data from leaves to the root.
 * @details One thread per leaf, seeding its triangle's area, area-weighted centroid and
 * normal, then merging upwards. As in the AABB refit, an `atomicCAS` flag at each internal
 * node lets only the second arriving thread continue, so exactly one reaches the root and
 * the whole aggregation completes in a single launch.
 *
 * The stored aggregates are what make winding number evaluation hierarchical: a distant
 * subtree can be approximated by its summary instead of being descended triangle by
 * triangle.
 *
 * @param[in] num_objects Number of triangles.
 * @param[in] object_ids Device array mapping sorted leaf order to triangle indices.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] bvh_parents Device array of parent indices.
 * @param[in] bvh_children Device array of child index pairs.
 * @param[out] winding_data Device array of per-node ::WindingData aggregates.
 * @param[in,out] atomic_flags Device array of visit flags; must be zeroed before launch.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning @p atomic_flags must be cleared between builds, or a node may be merged before
 * both children are final.
 */
__global__ void bottom_up_winding_data_kernel(
        const int num_objects,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const int *__restrict__ bvh_parents,
        const int2 *__restrict__ bvh_children,
        WindingData *__restrict__ winding_data,
        int *__restrict__ atomic_flags)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_objects)
            return;

        int leaf_idx = num_objects - 1 + idx;
        int original_obj_id = object_ids[idx];

        int3 tri = triangles[original_obj_id];
        float3 a = vertices[tri.x];
        float3 b = vertices[tri.y];
        float3 c = vertices[tri.z];

        float3 ab = b - a;
        float3 ac = c - a;

        float3 N = 0.5f * maths::cross(ab, ac);
        float area = maths::norm(N);
        float3 P = (a + b + c) / 3.0f;

        winding_data[leaf_idx].n = N;
        winding_data[leaf_idx].area = area;
        winding_data[leaf_idx].area_p = P * area;
        winding_data[leaf_idx].average_p = P;

        float d1 = maths::dot2(a - P);
        float d2 = maths::dot2(b - P);
        float d3 = maths::dot2(c - P);
        winding_data[leaf_idx].max_p_dist2 = fmaxf(d1, fmaxf(d2, d3));

        int current_node = bvh_parents[leaf_idx];

        while (current_node != -1)
        {
            __threadfence();

            int old = atomicAdd(&atomic_flags[current_node], 1);

            if (old == 0)
            {
                return;
            }

            int left_child = bvh_children[current_node].x;
            int right_child = bvh_children[current_node].y;

            WindingData left = winding_data[left_child];
            WindingData right = winding_data[right_child];

            float3 parent_N = left.n + right.n;
            float3 parent_area_p = left.area_p + right.area_p;
            float parent_area = left.area + right.area;

            float3 parent_average_p = make_float3(0.0f, 0.0f, 0.0f);
            if (parent_area > 0.0f)
            {
                parent_average_p = parent_area_p / parent_area;
            }

            winding_data[current_node].n = parent_N;
            winding_data[current_node].area_p = parent_area_p;
            winding_data[current_node].area = parent_area;
            winding_data[current_node].average_p = parent_average_p;

            float dist_to_left = maths::norm(parent_average_p - left.average_p);
            float dist_to_right = maths::norm(parent_average_p - right.average_p);

            float max_dist_left = dist_to_left + sqrtf(left.max_p_dist2);
            float max_dist_right = dist_to_right + sqrtf(right.max_p_dist2);

            float max_dist = fmaxf(max_dist_left, max_dist_right);
            winding_data[current_node].max_p_dist2 = max_dist * max_dist;

            current_node = bvh_parents[current_node];
        }
    }

    /**
     * @brief Launches the hierarchical winding-number aggregation.
     * @details Host wrapper; must run before winding-number sign determination can be used.
     */
    void bottom_up_winding_data(
        const int num_objects,
        const int *object_ids,
        const float3 *vertices,
        const int3 *triangles,
        const int *bvh_parents,
        const int2 *bvh_children,
        WindingData *winding_data)
    {
        int threads = NTHREADS;
        int blocks = (num_objects + threads - 1) / threads;

        thrust::device_vector<int> atomic_flags(num_objects - 1, 0);

        bottom_up_winding_data_kernel<<<blocks, threads>>>(
            num_objects,
            object_ids,
            vertices,
            triangles,
            bvh_parents,
            bvh_children,
            winding_data,
            thrust::raw_pointer_cast(atomic_flags.data()));
    }
/**
 * @brief Tests whether each query box is intersected by any triangle.
 * @details One thread per box. The BVH prunes candidates and surviving triangles go through
 * the Akenine-Moller separating-axis test. Traversal stops at the first confirmed hit --
 * the answer is a boolean, so there is nothing to gain from continuing.
 * @param[in] num_queries Number of query boxes.
 * @param[in] num_objects Number of triangles.
 * @param[in] query_mins Device array of box lower bounds.
 * @param[in] query_maxs Device array of box upper bounds.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to triangle indices.
 * @param[out] out_intersect Device array of per-box boolean results.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Traversal uses a per-thread stack of `BVH_STACK_SIZE` entries in local memory;
 * a pathologically unbalanced hierarchy can overflow it.
 */
__global__ void query_voxel_mesh_bvh_kernel(
        const int num_queries,
        const int num_objects,
        const float3 *__restrict__ query_mins,
        const float3 *__restrict__ query_maxs,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        bool *__restrict__ out_intersect)
    {
        int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (q_idx >= num_queries)
            return;

        float3 q_min = query_mins[q_idx];
        float3 q_max = query_maxs[q_idx];

        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0;

        bool is_intersect = false;

        while (stack_ptr >= 0)
        {
            int node_idx = stack[stack_ptr--];

            if (!aabb::test_aabb_overlap(q_min, q_max, bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx]))
                continue;

            if (node_idx >= num_objects - 1)
            {
                int tri_id = object_ids[node_idx - (num_objects - 1)];
                int3 tri = triangles[tri_id];
                Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);

                if (T.is_voxel_intersect(q_min, q_max))
                {
                    is_intersect = true;
                    break;
                }
            }
            else
            {
                if (stack_ptr + 2 < BVH_STACK_SIZE)
                {
                    int2 children = bvh_children[node_idx];
                    stack[++stack_ptr] = children.x;
                    stack[++stack_ptr] = children.y;
                }
            }
        }

        out_intersect[q_idx] = is_intersect;
    }

    /**
     * @brief Launches box-versus-mesh overlap tests.
     * @details Host wrapper over the separating-axis test.
     */
    void query_voxel_mesh_bvh(
        const int num_queries,
        const int num_objects,
        const float3 *query_mins,
        const float3 *query_maxs,
        const float3 *vertices,
        const int3 *triangles,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        bool *out_intersect)
    {
        int threads = NTHREADS;
        int blocks = (num_queries + threads - 1) / threads;

        query_voxel_mesh_bvh_kernel<<<blocks, threads>>>(
            num_queries,
            num_objects,
            query_mins,
            query_maxs,
            vertices,
            triangles,
            bvh_aabb_mins,
            bvh_aabb_maxs,
            bvh_children,
            object_ids,
            out_intersect);
    }

/**
 * @brief Counts the grid voxels the mesh surface passes through.
 * @details Sizing pass for narrow-band voxelisation. One thread per voxel of the virtual
 * grid, testing its box against the mesh and atomically incrementing a global count. Only
 * the count is produced here, so the host can allocate exactly the right buffer before the
 * collection pass -- the dense volume is never materialised.
 * @param[in] res Per-axis voxel resolution.
 * @param[in] grid_min World coordinate of the grid's lower corner.
 * @param[in] voxel_size Per-axis voxel dimensions.
 * @param[in] num_objects Number of triangles.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to triangle indices.
 * @param[in,out] active_counter Device counter of active voxels.
 * @note Launched over one thread per grid cell; indices are `int64_t` because a $1024^3$
 * grid overflows 32-bit addressing.
 * @warning Traversal uses a per-thread stack of `BVH_STACK_SIZE` entries in local memory;
 * a pathologically unbalanced hierarchy can overflow it.
 */
__global__ void count_active_voxels_mesh_bvh_kernel(
        const int3 res,
        const float3 grid_min,
        const float3 voxel_size,
        const int num_objects,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        unsigned long long *__restrict__ active_counter)
    {
        int64_t rx = res.x;
        int64_t ry = res.y;
        int64_t rz = res.z;
        int64_t num_voxels = rx * ry * rz;

        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_voxels)
            return;

        int64_t vi = idx / (ry * rz);
        int64_t rem = idx % (ry * rz);
        int64_t vj = rem / rz;
        int64_t vk = rem % rz;

        float3 q_min = make_float3(
            grid_min.x + vi * voxel_size.x,
            grid_min.y + vj * voxel_size.y,
            grid_min.z + vk * voxel_size.z);

        float3 q_max = make_float3(
            q_min.x + voxel_size.x,
            q_min.y + voxel_size.y,
            q_min.z + voxel_size.z);

        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0;
        bool is_intersect = false;

        while (stack_ptr >= 0)
        {
            int node_idx = stack[stack_ptr--];

            if (!aabb::test_aabb_overlap(q_min, q_max, bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx]))
                continue;

            if (node_idx >= num_objects - 1)
            {
                int tri_id = object_ids[node_idx - (num_objects - 1)];
                int3 tri = triangles[tri_id];
                Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);

                if (T.is_voxel_intersect(q_min, q_max))
                {
                    is_intersect = true;
                    break;
                }
            }
            else
            {
                if (stack_ptr + 2 < BVH_STACK_SIZE)
                {
                    int2 children = bvh_children[node_idx];
                    stack[++stack_ptr] = children.x;
                    stack[++stack_ptr] = children.y;
                }
            }
        }

        if (is_intersect) {
            atomicAdd(active_counter, 1ULL);
        }
    }

/**
 * @brief Writes the linear indices of every voxel the mesh surface passes through.
 * @details The collection pass matching the counting kernel, repeating the same
 * intersection test and appending surviving indices through an atomic counter.
 * @param[in] res Per-axis voxel resolution.
 * @param[in] grid_min World coordinate of the grid's lower corner.
 * @param[in] voxel_size Per-axis voxel dimensions.
 * @param[in] num_objects Number of triangles.
 * @param[in] vertices Device array of mesh vertex coordinates.
 * @param[in] triangles Device array of triangle vertex indices.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to triangle indices.
 * @param[in,out] active_counter Device counter, atomically incremented per emission.
 * @param[out] out_active_ids Device array receiving linear voxel indices.
 * @note Launched over one thread per grid cell, with `int64_t` indexing.
 * @warning Emission order is nondeterministic. Sort the result if a canonical ordering is
 * needed -- downstream extraction generally expects sorted voxel ids.
 * @warning Traversal uses a per-thread stack of `BVH_STACK_SIZE` entries in local memory;
 * a pathologically unbalanced hierarchy can overflow it.
 */
__global__ void collect_active_voxels_mesh_bvh_kernel(
        const int3 res,
        const float3 grid_min,
        const float3 voxel_size,
        const int num_objects,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        unsigned long long *__restrict__ active_counter,
        int64_t *__restrict__ out_active_ids)
    {
        int64_t rx = res.x;
        int64_t ry = res.y;
        int64_t rz = res.z;
        int64_t num_voxels = rx * ry * rz;

        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_voxels)
            return;

        int64_t vi = idx / (ry * rz);
        int64_t rem = idx % (ry * rz);
        int64_t vj = rem / rz;
        int64_t vk = rem % rz;

        float3 q_min = make_float3(
            grid_min.x + vi * voxel_size.x,
            grid_min.y + vj * voxel_size.y,
            grid_min.z + vk * voxel_size.z);

        float3 q_max = make_float3(
            q_min.x + voxel_size.x,
            q_min.y + voxel_size.y,
            q_min.z + voxel_size.z);

        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0;
        bool is_intersect = false;

        while (stack_ptr >= 0)
        {
            int node_idx = stack[stack_ptr--];

            if (!aabb::test_aabb_overlap(q_min, q_max, bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx]))
                continue;

            if (node_idx >= num_objects - 1)
            {
                int tri_id = object_ids[node_idx - (num_objects - 1)];
                int3 tri = triangles[tri_id];
                Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);

                if (T.is_voxel_intersect(q_min, q_max))
                {
                    is_intersect = true;
                    break;
                }
            }
            else
            {
                if (stack_ptr + 2 < BVH_STACK_SIZE)
                {
                    int2 children = bvh_children[node_idx];
                    stack[++stack_ptr] = children.x;
                    stack[++stack_ptr] = children.y;
                }
            }
        }

        if (is_intersect) {
            unsigned long long write_idx = atomicAdd(active_counter, 1ULL);
            out_active_ids[write_idx] = idx;
        }
    }

    /**
     * @brief Launches the counting pass of narrow-band voxelisation.
     * @details Host wrapper producing only a count, so the caller can size its buffer exactly
     * without ever materialising the dense volume.
     */
    void count_active_voxels_mesh_bvh(
        const int3 res,
        const float3 grid_min,
        const float3 voxel_size,
        const int num_objects,
        const float3 *vertices,
        const int3 *triangles,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        int64_t *active_counter)
    {
        int64_t num_queries = (int64_t)res.x * res.y * res.z;
        int threads = NTHREADS;
        int blocks = (num_queries + threads - 1) / threads;

        count_active_voxels_mesh_bvh_kernel<<<blocks, threads>>>(
            res,
            grid_min,
            voxel_size,
            num_objects,
            vertices,
            triangles,
            bvh_aabb_mins,
            bvh_aabb_maxs,
            bvh_children,
            object_ids,
            (unsigned long long *)active_counter);
    }

    /**
     * @brief Launches the collection pass of narrow-band voxelisation.
     * @details Host wrapper writing the active voxel indices.
     * @warning Emission order is nondeterministic; sort if a canonical order is needed.
     */
    void collect_active_voxels_mesh_bvh(
        const int3 res,
        const float3 grid_min,
        const float3 voxel_size,
        const int num_objects,
        const float3 *vertices,
        const int3 *triangles,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        int64_t *active_counter,
        int64_t *out_active_ids)
    {
        int64_t num_queries = (int64_t)res.x * res.y * res.z;
        int threads = NTHREADS;
        int blocks = (num_queries + threads - 1) / threads;

        collect_active_voxels_mesh_bvh_kernel<<<blocks, threads>>>(
            res,
            grid_min,
            voxel_size,
            num_objects,
            vertices,
            triangles,
            bvh_aabb_mins,
            bvh_aabb_maxs,
            bvh_children,
            object_ids,
            (unsigned long long *)active_counter,
            out_active_ids);
    }
}
