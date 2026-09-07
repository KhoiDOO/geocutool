/**
 * @file gs_aabb.cu
 * @brief CUDA kernel implementations for analytical Axis-Aligned Bounding Box (AABB) bounds of 3D Gaussians.
 */

#include "gs.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <math_constants.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

namespace gs_aabb
{
    /**
     * @brief Closed-form AABB of one Gaussian's isosurface.
     * @details Bounds the ellipsoid $\{x : (x - \mu)^\top \Sigma^{-1} (x - \mu) = \text{iso}\}$
     * exactly, rather than falling back to a scale-derived sphere, which would be far looser
     * and would put many more false candidates through later broad-phase tests. Scales are
     * clamped to a fraction of the voxel size so a degenerate splat cannot produce an
     * ill-conditioned bound.
     * @param[in] mean Gaussian centre.
     * @param[in] scale Per-axis scale.
     * @param[in] covi Six upper-triangular entries of the inverse covariance.
     * @param[in] iso Isovalue defining the surface.
     * @param[in] tol Minimum scale as a fraction of the voxel size.
     * @param[in] level Octree level setting the reference voxel size.
     * @param[out] out_min AABB lower bound.
     * @param[out] out_max AABB upper bound.
     * @param[out] contact_points Representative surface points, or `nullptr`.
     */
    __device__ __forceinline__ void compute_gs_single_aabb(
        const float3 &mean,
        const float3 &scale,
        const float *__restrict__ covi,
        const float &iso,
        const float &tol,
        const uint32_t level,
        float3 &out_min,
        float3 &out_max,
        float3 *contact_points)
    {
        float two_level = (float)(1U << level);
        float voxelSize = 2.0f / two_level;
        float min_scale = tol * voxelSize;
        float3 modified_scale = scale;

        modified_scale.x = fmaxf(scale.x, min_scale);
        modified_scale.y = fmaxf(scale.y, min_scale);
        modified_scale.z = fmaxf(scale.z, min_scale);

        double detS = ((double)modified_scale.x) * ((double)modified_scale.y) * ((double)modified_scale.z);

        double c0 = covi[0];
        double c1 = covi[1];
        double c2 = covi[2];
        double c3 = covi[3];
        double c4 = covi[4];
        double c5 = covi[5];

        double h0 = c3 * c5 - c4 * c4;
        double h1 = c2 * c4 - c1 * c5;
        double h2 = c1 * c4 - c2 * c3;
        double h3 = c0 * c5 - c2 * c2;
        double h4 = c1 * c2 - c0 * c4;
        double h5 = c0 * c3 - c1 * c1;

        double w[3];
        w[0] = detS * sqrt(iso / h0);
        w[1] = detS * sqrt(iso / h3);
        w[2] = detS * sqrt(iso / h5);

        double3 Q[3];
        Q[0] = make_double3(h0, h1, h2);
        Q[1] = make_double3(h1, h3, h4);
        Q[2] = make_double3(h2, h4, h5);

        float3 P[6];
        #pragma unroll
        for (int i = 0; i < 3; i++)
        {
            P[2 * i] = make_float3((float)(w[i] * Q[i].x), (float)(w[i] * Q[i].y), (float)(w[i] * Q[i].z));
            P[2 * i + 1] = -1.0f * P[2 * i];
        }

        contact_points[0] = P[0];
        contact_points[1] = P[2];
        contact_points[2] = P[4];

        float3 Pmin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 Pmax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        #pragma unroll
        for (int i = 0; i < 6; i++)
        {
            Pmin.x = fminf(Pmin.x, P[i].x);
            Pmin.y = fminf(Pmin.y, P[i].y);
            Pmin.z = fminf(Pmin.z, P[i].z);
            Pmax.x = fmaxf(Pmax.x, P[i].x);
            Pmax.y = fmaxf(Pmax.y, P[i].y);
            Pmax.z = fmaxf(Pmax.z, P[i].z);
        }

        out_min = mean + Pmin;
        out_max = mean + Pmax;
    }

    template <bool multiple_isos>
/**
 * @brief Computes a tight AABB and contact point for each Gaussian at its isovalue.
 * @details One thread per Gaussian. The bounding box of an ellipsoidal isosurface is
 * obtained in closed form from the inverse covariance, giving a far tighter fit than a
 * scale-derived sphere and so far fewer false positives in later broad-phase queries.
 * Isovalues may be uniform or per-Gaussian, selected at compile time by the
 * `multiple_isos` template parameter to keep the branch out of the inner loop.
 * @param[in] num_gaussians Number of Gaussians $N$.
 * @param[in] means Device array of $N$ Gaussian centres.
 * @param[in] scales Device array of $N$ per-axis scales.
 * @param[in] covis Device array of $6N$ inverse covariance entries.
 * @param[in] isos Device array of $N$ per-Gaussian isovalues, used when `multiple_isos`.
 * @param[in] iso Uniform isovalue, used otherwise.
 * @param[in] tol Minimum scale as a fraction of the voxel size.
 * @param[in] level Octree level setting the reference voxel size.
 * @param[out] aabb_min Device array of $N$ AABB lower bounds.
 * @param[out] aabb_max Device array of $N$ AABB upper bounds.
 * @param[out] contact_points Device array of $N$ representative surface points.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Templated on `multiple_isos`; the host instantiates the variant it needs.
 */
__global__ void compute_gs_aabb_kernel(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float3 *__restrict__ scales,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        const float tol,
        const uint32_t level,
        float3 *__restrict__ aabb_min,
        float3 *__restrict__ aabb_max,
        float3 *__restrict__ contact_points)
    {
        uint32_t g_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (g_idx >= num_gaussians)
            return;

        float __iso = multiple_isos ? isos[g_idx] : iso;

        compute_gs_single_aabb(
            means[g_idx],
            scales[g_idx],
            covis + (g_idx * 6),
            __iso,
            tol,
            level,
            aabb_min[g_idx],
            aabb_max[g_idx],
            contact_points + (g_idx * 3));
    }

    void compute_gs_aabb(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float3 *__restrict__ scales,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        const float tol,
        const uint32_t level,
        float3 *__restrict__ aabb_min,
        float3 *__restrict__ aabb_max,
        float3 *__restrict__ contact_points)
    {

        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_gaussians + threads - 1) / threads;

        if (isos != nullptr)
        {
            compute_gs_aabb_kernel<true><<<blocks, threads>>>(
                num_gaussians,
                means,
                scales,
                covis,
                isos,
                iso,
                tol,
                level,
                aabb_min,
                aabb_max,
                contact_points);
        }
        else
        {
            compute_gs_aabb_kernel<false><<<blocks, threads>>>(
                num_gaussians,
                means,
                scales,
                covis,
                isos,
                iso,
                tol,
                level,
                aabb_min,
                aabb_max,
                contact_points);
        }
    }

    /**
     * @brief Tests whether a Gaussian's isosurface crosses any edge of a voxel.
     * @details Works in the Gaussian's local frame, where the ellipsoid becomes a unit sphere
     * and the test reduces to a quadratic along each edge. One of the three cases making up the
     * full voxel test.
     * @param[in] mean Gaussian centre.
     * @param[in] covi Six upper-triangular entries of the inverse covariance.
     * @param[in] vx_ab_min Voxel lower bound.
     * @param[in] vx_ab_max Voxel upper bound.
     * @param[in] iso Isovalue defining the surface.
     * @return True if the surface crosses an edge.
     * @note Assumes a cubic voxel.
     */
    __device__ __forceinline__ bool test_gs_intersect_voxel_edge(
        const float3 &mean,
        const float *covi,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max,
        const float iso)
    {

        float3 p = vx_ab_min - mean;             // move to local space of Gaussian
        float vsize = vx_ab_max.x - vx_ab_min.x; // assume cubic voxel

        // if edge crossings, true true
        for (int i = 0; i < 4; i++)
        {
            float dummy_t0, dummy_t1;

            if (gs::test_gs_segment(
                    covi[0], covi[1], covi[2], covi[3], covi[4], covi[5], iso,
                    make_float3(p.x, p.y + (i / 2 ? vsize : 0.0), p.z + (i % 2 ? vsize : 0.0)),
                    make_float3(p.x + vsize, p.y + (i / 2 ? vsize : 0.0), p.z + (i % 2 ? vsize : 0.0)),
                    false, dummy_t0, dummy_t1))
            {
                return true;
            }

            if (gs::test_gs_segment(
                    covi[0], covi[1], covi[2], covi[3], covi[4], covi[5], iso,
                    make_float3(p.x + (i / 2 ? vsize : 0.0), p.y, p.z + (i % 2 ? vsize : 0.0)),
                    make_float3(p.x + (i / 2 ? vsize : 0.0), p.y + vsize, p.z + (i % 2 ? vsize : 0.0)),
                    false, dummy_t0, dummy_t1))
            {
                return true;
            }

            if (gs::test_gs_segment(
                    covi[0], covi[1], covi[2], covi[3], covi[4], covi[5], iso,
                    make_float3(p.x + (i / 2 ? vsize : 0.0), p.y + (i % 2 ? vsize : 0.0), p.z),
                    make_float3(p.x + (i / 2 ? vsize : 0.0), p.y + (i % 2 ? vsize : 0.0), p.z + vsize),
                    false, dummy_t0, dummy_t1))
            {
                return true;
            }
        }

        return false;
    }

    /**
     * @brief Tests one face of a voxel against a Gaussian in its local frame.
     * @details Helper of test_gs_intersect_voxel_face(), solving for the parameter at which
     * the face plane meets the transformed surface and checking the hit lies within the face.
     * @param[in] p Voxel corner in the Gaussian's local frame.
     * @param[in,out] q Working vector for the face solve.
     * @param[in] vsize Voxel edge length.
     * @param[in] i Axis index of the face normal.
     * @param[in] j First in-plane axis index.
     * @param[in] k Second in-plane axis index.
     * @param[in] vs Face offset along the normal axis.
     * @return True if the face intersects the surface.
     */
    __device__ __forceinline__ bool test_gs_vx_face(
        const float *p,
        float *q,
        const float vsize,
        const int i,
        const int j,
        const int k,
        const float vs)
    {
        float t = (q[i] - (p[i] + vs)) / (2.0 * q[i]);

        if (0.0f <= t && t <= 1.0f)
        {
            float h[3];
            for (int l = 0; l < 3; l++)
                h[l] = (1.0 - 2.0 * t) * q[l];

            return p[j] <= h[j] && h[j] <= (p[j] + vsize) && p[k] <= h[k] && h[k] <= (p[k] + vsize);
        }

        return false;
    }

    /**
     * @brief Tests whether a Gaussian's isosurface crosses any face of a voxel.
     * @details Covers the case where the surface passes through a face interior without
     * touching an edge, which the edge test alone would miss.
     * @param[in] mean Gaussian centre.
     * @param[in] cp0 First contact point.
     * @param[in] cp1 Second contact point.
     * @param[in] cp2 Third contact point.
     * @param[in] vx_ab_min Voxel lower bound.
     * @param[in] vx_ab_max Voxel upper bound.
     * @return True if the surface crosses a face.
     * @note Assumes a cubic voxel.
     */
    __device__ __forceinline__ bool test_gs_intersect_voxel_face(
        const float3 &mean,
        float3 cp0,
        float3 cp1,
        float3 cp2,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max)
    {
        float3 p = vx_ab_min - mean;
        float vsize = vx_ab_max.x - vx_ab_min.x; // assume cubic voxel
        bool b[6];

        // need to index float3 components, so cast to float[]
        b[0] = test_gs_vx_face((float *)(&p), (float *)(&cp0), vsize, 0, 1, 2, 0.0);
        b[1] = test_gs_vx_face((float *)(&p), (float *)(&cp0), vsize, 0, 1, 2, vsize);

        b[2] = test_gs_vx_face((float *)(&p), (float *)(&cp1), vsize, 1, 0, 2, 0.0);
        b[3] = test_gs_vx_face((float *)(&p), (float *)(&cp1), vsize, 1, 0, 2, vsize);

        b[4] = test_gs_vx_face((float *)(&p), (float *)(&cp2), vsize, 2, 0, 1, 0.0);
        b[5] = test_gs_vx_face((float *)(&p), (float *)(&cp2), vsize, 2, 0, 1, vsize);

        return (b[0] || b[1] || b[2] || b[3] || b[4] || b[5]);
    }

    /**
     * @brief Full test of a Gaussian's isosurface against a voxel.
     * @details Ordered cheapest first: containment either way, then AABB overlap, then the
     * exact edge and face tests. Most candidate pairs are settled by the first two steps, so
     * the expensive geometry runs only where it can change the answer.
     * @param[in] gaus_idx Gaussian index, for diagnostics.
     * @param[in] mean Gaussian centre.
     * @param[in] covi Six upper-triangular entries of the inverse covariance.
     * @param[in] gs_ab_min Gaussian AABB lower bound.
     * @param[in] gs_ab_max Gaussian AABB upper bound.
     * @param[in] cp0 First contact point.
     * @param[in] cp1 Second contact point.
     * @param[in] cp2 Third contact point.
     * @param[in] vx_ab_min Voxel lower bound.
     * @param[in] vx_ab_max Voxel upper bound.
     * @return True if the Gaussian's isosurface meets the voxel.
     */
    __device__ __forceinline__ bool test_gs_intersect_voxel(
        const uint64_t gaus_idx,
        const float3 &mean,
        const float *covi,
        const float3 &gs_ab_min,
        const float3 &gs_ab_max,
        const float3 &cp0,
        const float3 &cp1,
        const float3 &cp2,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max,
        const float iso)
    {

        if (aabb::test_aabb_inside(gs_ab_min, gs_ab_max, vx_ab_min, vx_ab_max))
            return true;

        if (!aabb::test_aabb_overlap(gs_ab_min, gs_ab_max, vx_ab_min, vx_ab_max))
            return false;

        if (test_gs_intersect_voxel_face(mean, cp0, cp1, cp2, vx_ab_min, vx_ab_max))
            return true;

        return test_gs_intersect_voxel_edge(mean, covi, vx_ab_min, vx_ab_max, iso);
    }

    /**
     * @brief Computes the centroid and density of a Gaussian-voxel overlap region.
     * @details Intersects the two boxes and evaluates the Gaussian's density over the shared
     * volume, giving the weight used to decide whether the pair contributes meaningfully or
     * should be discarded.
     * @param[in] gs_ab_min Gaussian AABB lower bound.
     * @param[in] gs_ab_max Gaussian AABB upper bound.
     * @param[in] vx_ab_min Voxel lower bound.
     * @param[in] vx_ab_max Voxel upper bound.
     * @param[in] mean Gaussian centre.
     * @param[in] covi Six upper-triangular entries of the inverse covariance.
     * @param[in] opacity Gaussian opacity.
     * @param[in] return_centroids Whether the centroid is written.
     * @param[out] out_centroid Centroid of the overlap region.
     * @param[out] out_density Density integrated over the overlap.
     */
    __device__ __forceinline__ void compute_overlap_metrics(
        const float3 &gs_ab_min,
        const float3 &gs_ab_max,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max,
        const float3 &mean,
        const float *covi,
        const float opacity,
        const bool return_centroids,
        float3 &out_centroid,
        float &out_density,
        float &out_aspect_ratio,
        float &out_penetration)
    {
        // 1. Calculate Overlap Box boundaries
        float3 overlap_min;
        float3 overlap_max;
        aabb::compute_aabb_overlap(gs_ab_min, gs_ab_max, vx_ab_min, vx_ab_max, overlap_min, overlap_max);

        if (return_centroids)
        {
            aabb::compute_aabb_centroid(overlap_min, overlap_max, out_centroid);
            gs::compute_density(out_centroid, mean, covi, opacity, out_density);
        }

        // 3. Get the physical dimensions of the overlap box
        float3 dims;
        aabb::compute_aabb_dim_size(overlap_min, overlap_max, dims);

        // 4. Find the smallest and largest dimensions
        float min_dim = fminf(dims.x, fminf(dims.y, dims.z));
        float max_dim = fmaxf(dims.x, fmaxf(dims.y, dims.z));

        // Assume cubic voxels: size is max - min on any axis
        float voxel_size = vx_ab_max.x - vx_ab_min.x;

        // 5. Calculate Metrics (with safeguards against divide-by-zero)
        out_aspect_ratio = (max_dim > 1e-6f) ? (min_dim / max_dim) : 0.0f;
        out_penetration = (voxel_size > 1e-6f) ? (min_dim / voxel_size) : 0.0f;
    }

    template <bool multiple_isos>
/**
 * @brief Reports every (voxel, Gaussian) pair whose supports overlap.
 * @details One thread per voxel, descending the Gaussian BVH with a private stack. Leaves
 * surviving the AABB test are re-examined exactly: the Gaussian's density is evaluated
 * inside the voxel and the pair is kept only if it clears both the aspect-ratio and
 * density thresholds, which discards the many boxes that touch a voxel without
 * contributing meaningful mass.
 * @param[in] num_voxels Number of voxels $V$.
 * @param[in] num_gaussians Number of Gaussians $N$.
 * @param[in] vx_aabb_mins Device array of $V$ voxel lower bounds.
 * @param[in] vx_aabb_maxs Device array of $V$ voxel upper bounds.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to Gaussian indices.
 * @param[in] means Device array of $N$ Gaussian centres.
 * @param[in] covis Device array of $6N$ inverse covariance entries.
 * @param[in] opacities Device array of $N$ opacities.
 * @param[in] gs_aabb_mins Device array of $N$ Gaussian AABB lower bounds.
 * @param[in] gs_aabb_maxs Device array of $N$ Gaussian AABB upper bounds.
 * @param[in] contact_points Device array of $N$ representative surface points.
 * @param[in] isos Device array of $N$ per-Gaussian isovalues.
 * @param[in] iso Uniform isovalue fallback.
 * @param[in] ar_threshold Aspect-ratio rejection threshold.
 * @param[in] p_threshold Density rejection threshold.
 * @param[in] return_centroids Whether to emit centroids and densities.
 * @param[out] hit_mask Device array of $V$ flags marking voxels with any hit.
 * @param[out] out_voxel_ids Device array receiving the voxel index of each pair.
 * @param[out] out_gaus_ids Device array receiving the Gaussian index of each pair.
 * @param[out] centroids Device array of per-pair centroids, when requested.
 * @param[out] densities Device array of per-pair densities, when requested.
 * @param[in,out] global_counter Device counter, atomically incremented per pair.
 * @param[in] max_capacity Capacity of the output arrays.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Traversal uses a per-thread stack of `BVH_STACK_SIZE` entries in local memory;
 * a pathologically unbalanced hierarchy can overflow it.
 * @warning Output slots are claimed atomically, so pair ordering varies between runs.
 * Emission stops at @p max_capacity; compare the final counter against it to detect
 * truncation.
 */
__global__ void query_gs_voxel_pair_intersection_bvh_kernel(
        const uint32_t num_voxels,
        const uint32_t num_gaussians,
        const float3 *__restrict__ vx_aabb_mins,
        const float3 *__restrict__ vx_aabb_maxs,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float *__restrict__ covis,
        const float *__restrict__ opacities,
        const float3 *__restrict__ gs_aabb_mins,
        const float3 *__restrict__ gs_aabb_maxs,
        const float3 *__restrict__ contact_points,
        const float *__restrict__ isos,
        const float iso,
        const float ar_threshold,
        const float p_threshold,
        const bool return_centroids,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_voxel_ids,
        int64_t *__restrict__ out_gaus_ids,
        float3 *__restrict__ centroids,
        float *__restrict__ densities,
        int64_t *__restrict__ global_counter,
        const int64_t max_capacity)
    {
        uint32_t v_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (v_idx >= num_voxels)
            return;

        float3 vx_ab_min = vx_aabb_mins[v_idx];
        float3 vx_ab_max = vx_aabb_maxs[v_idx];

        bool any_hit = false;

        // --- BVH LOCAL STACK ---
        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0; // Push the Root Node

        while (stack_ptr >= 0)
        {
            int node_idx = stack[stack_ptr--];

            // 1. BROAD PHASE: Voxel AABB vs BVH Node AABB
            if (aabb::test_aabb_overlap(vx_ab_min, vx_ab_max, bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx]))
            {
                if (node_idx >= num_gaussians - 1)
                {
                    // Recover the original Gaussian index
                    int leaf_idx = node_idx - (num_gaussians - 1);
                    uint32_t g_idx = object_ids[leaf_idx];

                    // Fetch Gaussian properties
                    float3 mean = means[g_idx];
                    const float *covi = covis + (g_idx * 6);
                    float opacity = opacities[g_idx];
                    float3 gs_ab_min = gs_aabb_mins[g_idx];
                    float3 gs_ab_max = gs_aabb_maxs[g_idx];
                    float3 cp0 = contact_points[g_idx * 3 + 0];
                    float3 cp1 = contact_points[g_idx * 3 + 1];
                    float3 cp2 = contact_points[g_idx * 3 + 2];
                    float __iso = multiple_isos ? isos[g_idx] : iso;

                    bool hit = test_gs_intersect_voxel(
                        g_idx,
                        mean,
                        covi,
                        gs_ab_min,
                        gs_ab_max,
                        cp0,
                        cp1,
                        cp2,
                        vx_ab_min,
                        vx_ab_max,
                        __iso);

                    if (hit)
                    {
                        float3 centroid;
                        float density;
                        float aspect_ratio;
                        float penetration;

                        compute_overlap_metrics(
                            gs_ab_min, gs_ab_max, vx_ab_min, vx_ab_max,
                            mean, covi, opacity, return_centroids,
                            centroid, density, aspect_ratio, penetration);

                        // Threshold check
                        if (aspect_ratio >= ar_threshold && penetration >= p_threshold)
                        {
                            any_hit = true;
                            uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int *)global_counter, 1ULL);

                            if (write_idx < max_capacity)
                            {
                                out_voxel_ids[write_idx] = v_idx;
                                out_gaus_ids[write_idx] = g_idx;

                                if (return_centroids)
                                {
                                    centroids[write_idx] = centroid;
                                    densities[write_idx] = density;
                                }
                            }
                        }
                    }
                }
                else
                {
                    // INTERNAL NODE: Push children to stack
                    if (stack_ptr + 2 < BVH_STACK_SIZE)
                    {
                        int2 children = bvh_children[node_idx];
                        stack[++stack_ptr] = children.x;
                        stack[++stack_ptr] = children.y;
                    }
                }
            }
        }

        hit_mask[v_idx] = any_hit;
    }

    void query_gs_voxel_pair_intersection_bvh(
        const uint32_t num_voxels,
        const uint32_t num_gaussians,
        const float3 *__restrict__ vx_aabb_mins,
        const float3 *__restrict__ vx_aabb_maxs,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float *__restrict__ covis,
        const float *__restrict__ opacities,
        const float3 *__restrict__ gs_aabb_mins,
        const float3 *__restrict__ gs_aabb_maxs,
        const float3 *__restrict__ contact_points,
        const float *__restrict__ isos,
        const float iso,
        const float ar_threshold,
        const float p_threshold,
        const bool return_centroids,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_voxel_ids,
        int64_t *__restrict__ out_gaus_ids,
        float3 *__restrict__ centroids,
        float *__restrict__ densities,
        int64_t *__restrict__ global_counter,
        const int64_t max_capacity)
    {
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_voxels + threads - 1) / threads;

        if (isos != nullptr)
        {
            query_gs_voxel_pair_intersection_bvh_kernel<true><<<blocks, threads>>>(
                num_voxels,
                num_gaussians,
                vx_aabb_mins,
                vx_aabb_maxs,
                bvh_aabb_mins,
                bvh_aabb_maxs,
                bvh_children,
                object_ids,
                means,
                covis,
                opacities,
                gs_aabb_mins,
                gs_aabb_maxs,
                contact_points,
                isos,
                iso,
                ar_threshold,
                p_threshold,
                return_centroids,
                hit_mask,
                out_voxel_ids,
                out_gaus_ids,
                centroids,
                densities,
                global_counter,
                max_capacity);
        }
        else
        {
            query_gs_voxel_pair_intersection_bvh_kernel<false><<<blocks, threads>>>(
                num_voxels,
                num_gaussians,
                vx_aabb_mins,
                vx_aabb_maxs,
                bvh_aabb_mins,
                bvh_aabb_maxs,
                bvh_children,
                object_ids,
                means,
                covis,
                opacities,
                gs_aabb_mins,
                gs_aabb_maxs,
                contact_points,
                isos,
                iso,
                ar_threshold,
                p_threshold,
                return_centroids,
                hit_mask,
                out_voxel_ids,
                out_gaus_ids,
                centroids,
                densities,
                global_counter,
                max_capacity);
        }
    }

    template <bool multiple_isos>
/**
 * @brief Reports every (edge, Gaussian) pair whose supports intersect.
 * @details One thread per edge. A broad-phase AABB around the segment prunes the BVH,
 * then surviving Gaussians receive an exact segment-versus-ellipsoid test. Used to decide
 * which splats a grid edge passes through when building sparse structures over a splat
 * cloud.
 * @param[in] num_edges Number of edges $E$.
 * @param[in] num_gaussians Number of Gaussians $N$.
 * @param[in] edge_starts Device array of $E$ segment start points.
 * @param[in] edge_ends Device array of $E$ segment end points.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to Gaussian indices.
 * @param[in] means Device array of $N$ Gaussian centres.
 * @param[in] covis Device array of $6N$ inverse covariance entries.
 * @param[in] isos Device array of $N$ per-Gaussian isovalues.
 * @param[in] iso Uniform isovalue fallback.
 * @param[out] hit_mask Device array of $E$ flags marking edges with any hit.
 * @param[out] out_edge_ids Device array receiving the edge index of each pair.
 * @param[out] out_gaus_ids Device array receiving the Gaussian index of each pair.
 * @param[in,out] global_counter Device counter, atomically incremented per pair.
 * @param[in] max_capacity Capacity of the output arrays.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Traversal uses a per-thread stack of `BVH_STACK_SIZE` entries in local memory;
 * a pathologically unbalanced hierarchy can overflow it.
 * @warning Output slots are claimed atomically, so pair ordering varies between runs.
 * Emission stops at @p max_capacity; compare the final counter against it to detect
 * truncation.
 */
__global__ void query_gs_edge_pair_intersection_bvh_kernel(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_edge_ids,
        int64_t *__restrict__ out_gaus_ids,
        int64_t *__restrict__ global_counter,
        const int64_t max_capacity)
    {
        uint32_t e_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (e_idx >= num_edges)
            return;

        float3 edge_start = edge_starts[e_idx];
        float3 edge_end = edge_ends[e_idx];

        // Create a fast Broad-Phase AABB for the edge
        float3 e_ab_min = make_float3(fminf(edge_start.x, edge_end.x), fminf(edge_start.y, edge_end.y), fminf(edge_start.z, edge_end.z));
        float3 e_ab_max = make_float3(fmaxf(edge_start.x, edge_end.x), fmaxf(edge_start.y, edge_end.y), fmaxf(edge_start.z, edge_end.z));

        bool any_hit = false;

        // --- BVH LOCAL STACK ---
        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0; // Push the Root Node

        while (stack_ptr >= 0)
        {
            int node_idx = stack[stack_ptr--];

            // 1. BROAD PHASE
            if (aabb::test_aabb_overlap(e_ab_min, e_ab_max, bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx]))
            {
                if (node_idx >= num_gaussians - 1)
                {
                    // 2. NARROW PHASE
                    int leaf_idx = node_idx - (num_gaussians - 1);
                    uint32_t g_idx = object_ids[leaf_idx];

                    float3 mean = means[g_idx];
                    const float *covi = covis + (g_idx * 6);
                    float __iso = multiple_isos ? isos[g_idx] : iso;

                    float3 local_start = edge_start - mean;
                    float3 local_end = edge_end - mean;
                    float dummy_t_entry, dummy_t_exit;

                    bool hit = gs::test_gs_segment(
                        covi[0], covi[1], covi[2], covi[3], covi[4], covi[5],
                        __iso, local_start, local_end, false, dummy_t_entry, dummy_t_exit);

                    if (hit)
                    {
                        any_hit = true;
                        uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int *)global_counter, 1ULL);

                        if (write_idx < max_capacity)
                        {
                            out_edge_ids[write_idx] = e_idx;
                            out_gaus_ids[write_idx] = g_idx;
                        }
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
        }
        hit_mask[e_idx] = any_hit;
    }

    void query_gs_edge_pair_intersection_bvh(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_edge_ids,
        int64_t *__restrict__ out_gaus_ids,
        int64_t *__restrict__ global_counter,
        const int64_t max_capacity)
    {
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_edges + threads - 1) / threads;

        if (isos != nullptr)
        {
            query_gs_edge_pair_intersection_bvh_kernel<true><<<blocks, threads>>>(
                num_edges,
                num_gaussians,
                edge_starts,
                edge_ends,
                bvh_aabb_mins,
                bvh_aabb_maxs,
                bvh_children,
                object_ids,
                means, covis,
                isos,
                iso,
                hit_mask,
                out_edge_ids,
                out_gaus_ids,
                global_counter,
                max_capacity);
        }
        else
        {
            query_gs_edge_pair_intersection_bvh_kernel<false><<<blocks, threads>>>(
                num_edges,
                num_gaussians,
                edge_starts,
                edge_ends,
                bvh_aabb_mins,
                bvh_aabb_maxs,
                bvh_children,
                object_ids,
                means,
                covis,
                isos,
                iso,
                hit_mask,
                out_edge_ids,
                out_gaus_ids,
                global_counter,
                max_capacity);
        }
    }

    template <bool multiple_isos>
/**
 * @brief Finds the first Gaussian each edge intersects.
 * @details One thread per edge. Unlike the pair-emitting variant this kernel records a
 * single Gaussian per edge at a fixed output slot, so it needs no atomic counter and its
 * result is deterministic. Suited to occlusion and visibility tests where only the
 * existence of a blocker matters.
 * @param[in] num_edges Number of edges $E$.
 * @param[in] num_gaussians Number of Gaussians $N$.
 * @param[in] edge_starts Device array of $E$ segment start points.
 * @param[in] edge_ends Device array of $E$ segment end points.
 * @param[in] bvh_aabb_mins Device array of BVH node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of BVH node upper bounds.
 * @param[in] bvh_children Device array of BVH child index pairs.
 * @param[in] object_ids Device array mapping leaves to Gaussian indices.
 * @param[in] means Device array of $N$ Gaussian centres.
 * @param[in] opacities Device array of $N$ opacities.
 * @param[in] covis Device array of $6N$ inverse covariance entries.
 * @param[in] isos Device array of $N$ per-Gaussian isovalues.
 * @param[in] iso Uniform isovalue fallback.
 * @param[out] hit_mask Device array of $E$ flags marking intersected edges.
 * @param[out] out_gaus_ids Device array of $E$ Gaussian indices, $-1$ where none.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Reports the first qualifying Gaussian encountered, which is not necessarily the
 * nearest along the segment.
 * @warning Traversal uses a per-thread stack of `BVH_STACK_SIZE` entries in local memory;
 * a pathologically unbalanced hierarchy can overflow it.
 */
__global__ void query_gs_edge_intersection_bvh_kernel(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float *__restrict__ opacities,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_gaus_ids)
    {
        uint32_t e_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (e_idx >= num_edges)
            return;

        float3 edge_start = edge_starts[e_idx];
        float3 edge_end = edge_ends[e_idx];

        float3 e_ab_min = make_float3(fminf(edge_start.x, edge_end.x), fminf(edge_start.y, edge_end.y), fminf(edge_start.z, edge_end.z));
        float3 e_ab_max = make_float3(fmaxf(edge_start.x, edge_end.x), fmaxf(edge_start.y, edge_end.y), fmaxf(edge_start.z, edge_end.z));

        bool any_hit = false;
        float max_density = -1.0f;
        int best_gs = -1;

        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0;

        while (stack_ptr >= 0)
        {
            int node_idx = stack[stack_ptr--];

            if (aabb::test_aabb_overlap(e_ab_min, e_ab_max, bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx]))
            {
                if (node_idx >= num_gaussians - 1)
                {
                    int leaf_idx = node_idx - (num_gaussians - 1);
                    uint32_t g_idx = object_ids[leaf_idx];

                    float3 mean = means[g_idx];
                    const float *covi = covis + (g_idx * 6);
                    float __iso = multiple_isos ? isos[g_idx] : iso;

                    float3 local_start = edge_start - mean;
                    float3 local_end = edge_end - mean;
                    float t_entry, t_exit;

                    bool hit = gs::test_gs_segment(
                        covi[0], covi[1], covi[2], covi[3], covi[4], covi[5],
                        __iso, local_start, local_end, true, t_entry, t_exit);

                    if (hit)
                    {
                        any_hit = true;
                        float t_mid = (fmaxf(t_entry, 0.0f) + fminf(t_exit, 1.0f)) * 0.5f;
                        float3 p_mid = local_start + t_mid * (local_end - local_start);

                        float density;
                        gs::compute_density_local(p_mid, covi, opacities[g_idx], density);

                        if (density > max_density)
                        {
                            max_density = density;
                            best_gs = g_idx;
                        }
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
        }

        // Write the single best result exactly once per thread
        out_gaus_ids[e_idx] = best_gs;
        hit_mask[e_idx] = any_hit;
    }

    void query_gs_edge_intersection_bvh(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float *__restrict__ opacities,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_gaus_ids)
    {
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_edges + threads - 1) / threads;

        if (isos != nullptr)
        {
            query_gs_edge_intersection_bvh_kernel<true><<<blocks, threads>>>(
                num_edges,
                num_gaussians,
                edge_starts,
                edge_ends,
                bvh_aabb_mins,
                bvh_aabb_maxs,
                bvh_children,
                object_ids,
                means,
                opacities,
                covis,
                isos,
                iso,
                hit_mask,
                out_gaus_ids);
        }
        else
        {
            query_gs_edge_intersection_bvh_kernel<false><<<blocks, threads>>>(
                num_edges,
                num_gaussians,
                edge_starts,
                edge_ends,
                bvh_aabb_mins,
                bvh_aabb_maxs,
                bvh_children,
                object_ids,
                means,
                opacities,
                covis,
                isos,
                iso,
                hit_mask,
                out_gaus_ids);
        }
    }
}