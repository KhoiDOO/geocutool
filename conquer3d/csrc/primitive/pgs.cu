/**
 * @file pgs.cu
 * @brief CUDA kernel implementations for Periodic Gaussian Splatting (PGS) cluster tangency radius solving.
 */

#include "pgs.h"
#include "pgs_math.cuh"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <math_constants.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

namespace pgs
{
/**
 * @brief Solves the tangency radius of each periodic Gaussian against its neighbours.
 * @details One thread per Gaussian. Periodic Gaussians carry an orientation as well as a
 * covariance, so the support radius is the point at which a splat becomes tangent to its
 * neighbours along the normal direction rather than a plain Mahalanobis cutoff. Gaussians
 * whose configuration admits no valid tangency -- degenerate or fully enclosed by a
 * neighbour -- are reported through @p invalid_mask instead of receiving a fabricated
 * radius.
 * @param[in] num_gaussians Number of Gaussians $N$.
 * @param[in] means Device array of $N$ Gaussian centres.
 * @param[in] normals Device array of $N$ orientation vectors.
 * @param[in] covis Device array of $6N$ inverse covariance entries.
 * @param[in] tree_points Device array of centres in KD-tree order.
 * @param[in] tree_inds Device array of permutation indices back to original order.
 * @param[in] k Number of neighbours to consider; must not exceed `MAX_K`.
 * @param[out] isos Device array of $N$ tangency radii.
 * @param[out] invalid_mask Device array of $N$ flags marking unsolvable Gaussians.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Entries flagged in @p invalid_mask leave the matching @p isos value
 * unspecified; filter on the mask before use.
 */
__global__ void solve_pgs_cluster_tangency_radius_kernel(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ covis,
        const float3 *__restrict__ tree_points,
        const int64_t *__restrict__ tree_inds,
        const int k,
        float *__restrict__ isos,
        bool *__restrict__ invalid_mask)
    {
        uint32_t g_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (g_idx >= num_gaussians) return;

        float3 mean = means[g_idx];
        float3 normal = normals[g_idx];
        const float *covi = covis + (g_idx * 6);

        float best_dists[MAX_K];
        int64_t best_inds[MAX_K];

        #pragma unroll
        for (int i = 0; i < MAX_K; i++) {
            best_dists[i] = FLT_MAX;
            best_inds[i] = -1;
        }

        kdtree::query_kdtree_loop(mean, num_gaussians, tree_points, tree_inds, k, best_dists, best_inds);

        float min_iso = FLT_MAX;
        bool any_success = false;

        for (int i = 0; i < k; i++) {
            int64_t neighbor_idx = best_inds[i];

            if (neighbor_idx == -1) continue;

            if (neighbor_idx == g_idx) continue;

            float3 neighbor_mean = means[neighbor_idx];
            float3 neighbor_normal = normals[neighbor_idx];

            float iso;
            bool success = pgs::solve_pgs_pair_tangency_radius(mean, normal, covi, neighbor_mean, neighbor_normal, iso);
            if (success) {
                min_iso = fminf(min_iso, iso);
                any_success = true;
            }
        }

        isos[g_idx] = min_iso;
        invalid_mask[g_idx] = !any_success;
    }

    void solve_pgs_cluster_tangency_radius(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ covis,
        const int k,
        float *__restrict__ isos,
        bool *__restrict__ invalid_mask
    )
    {
        thrust::device_vector<float3> cloned_means(means, means + num_gaussians);
        thrust::device_vector<int64_t> oinds(num_gaussians);
        thrust::sequence(oinds.begin(), oinds.end());

        kdtree::build(
            num_gaussians,
            thrust::raw_pointer_cast(cloned_means.data()),
            thrust::raw_pointer_cast(oinds.data())
        );

        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_gaussians + threads - 1) / threads;

        solve_pgs_cluster_tangency_radius_kernel<<<blocks, threads>>>(
            num_gaussians,
            means,
            normals,
            covis,
            thrust::raw_pointer_cast(cloned_means.data()),
            thrust::raw_pointer_cast(oinds.data()),
            k,
            isos,
            invalid_mask);
    }
}