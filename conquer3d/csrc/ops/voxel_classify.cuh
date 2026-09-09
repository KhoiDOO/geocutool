/**
 * @file voxel_classify.cuh
 * @brief Corner-sign classification shared by the cube-based extractors.
 *
 * @details Marching Cubes and grid Marching Tetrahedra begin identically: pack each voxel's
 * eight corner signs into an 8-bit case index, which decides that cell's topology outright.
 * Only what happens *after* classification differs -- one indexes cube tables, the other
 * splits the cell into tetrahedra. The classification stage was duplicated verbatim in both
 * translation units, so it lives here once.
 *
 * @note The kernel and its launcher carry internal linkage: a `__global__` defined in a
 * header would otherwise be emitted by every translation unit that includes it and collide
 * at link time. Each unit gets its own copy of the code, which is what the duplicated
 * source produced anyway.
 */

#ifndef VOXEL_CLASSIFY_CUH
#define VOXEL_CLASSIFY_CUH

#include "../constants.h"
#include <cuda_runtime.h>
#include <stdint.h>

namespace voxel_classify {

/**
     * @brief Packs eight corner signs into a Marching Cubes case index.
     * @details Sets bit $i$ when corner $i$ lies below the isolevel, producing the 8-bit code
     * that indexes every topology table. Branch-free enough to compile to predicated
     * instructions, so it costs nothing in warp divergence.
     * @param[in] sv0 Value at corner 0.
     * @param[in] sv1 Value at corner 1.
     * @param[in] sv2 Value at corner 2.
     * @param[in] sv3 Value at corner 3.
     * @param[in] sv4 Value at corner 4.
     * @param[in] sv5 Value at corner 5.
     * @param[in] sv6 Value at corner 6.
     * @param[in] sv7 Value at corner 7.
     * @param[in] iso Isolevel separating inside from outside.
     * @param[out] voxel_code The resulting 8-bit case index.
     * @note Codes 0 and 255 mean the cell is entirely outside or inside and emits nothing.
     */
    __device__ __forceinline__ void compute_voxel_code(
        float sv0, float sv1, float sv2, float sv3,
        float sv4, float sv5, float sv6, float sv7,
        float iso, uint8_t &voxel_code)
    {
        voxel_code = 0;
        if (sv0 < iso)
            voxel_code |= 1;
        if (sv1 < iso)
            voxel_code |= 2;
        if (sv2 < iso)
            voxel_code |= 4;
        if (sv3 < iso)
            voxel_code |= 8;
        if (sv4 < iso)
            voxel_code |= 16;
        if (sv5 < iso)
            voxel_code |= 32;
        if (sv6 < iso)
            voxel_code |= 64;
        if (sv7 < iso)
            voxel_code |= 128;
    }

/**
 * @brief Classifies every voxel by the sign pattern of its eight corners.
 * @details Stage 1 of extraction. One thread per voxel; each compares its corner values
 * against the isolevel and packs the results into a 256-case code. The code alone
 * determines the voxel's surface topology, so all later stages are table lookups rather
 * than searches. Inactive voxels -- entirely inside or outside -- receive code 0 and are
 * compacted away by the host before stage 2, which is what makes cost scale with surface
 * area rather than volume.
 * @param[in] num_voxels Number of voxels.
 * @param[in] voxels Device array of corner indices, eight per voxel.
 * @param[in] sdf Device array of scalar field values at the grid vertices.
 * @param[in] iso Isolevel separating inside from outside.
 * @param[out] voxel_codes Device array of one sign code per voxel.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 */
static __global__ void compute_active_voxels_kernel(
        const uint32_t num_voxels,
        const uint32_t *__restrict__ voxels,
        const float *__restrict__ sdf,
        const float iso,
        uint8_t *__restrict__ voxel_codes)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_voxels)
            return;

        uint32_t v0 = voxels[idx * 8 + 0];
        uint32_t v1 = voxels[idx * 8 + 1];
        uint32_t v2 = voxels[idx * 8 + 2];
        uint32_t v3 = voxels[idx * 8 + 3];
        uint32_t v4 = voxels[idx * 8 + 4];
        uint32_t v5 = voxels[idx * 8 + 5];
        uint32_t v6 = voxels[idx * 8 + 6];
        uint32_t v7 = voxels[idx * 8 + 7];

        uint8_t voxel_code = 0;
        compute_voxel_code(
            sdf[v0], sdf[v1], sdf[v2], sdf[v3],
            sdf[v4], sdf[v5], sdf[v6], sdf[v7],
            iso, voxel_code);

        voxel_codes[idx] = voxel_code;
    }

/**
     * @brief Launches stage 1: classify every voxel by its corner signs.
     * @details Host wrapper sizing a 1D grid over all voxels and launching the classification
     * kernel. See the kernel for the parallel decomposition.
     * @param[in] num_voxels Number of voxels.
     * @param[in] voxels Device array of corner indices.
     * @param[in] sdf Device array of scalar field values.
     * @param[in] iso Isolevel separating inside from outside.
     * @param[out] voxel_codes Device array of per-voxel sign codes.
     */
    static inline void compute_active_voxels(
        const uint32_t num_voxels,
        const uint32_t *voxels,
        const float *sdf,
        const float iso,
        uint8_t *voxel_codes)
    {
        int block_size = NTHREADS;
        int grid_size = (num_voxels + block_size - 1) / block_size;
        compute_active_voxels_kernel<<<grid_size, block_size>>>(
            num_voxels, voxels, sdf, iso, voxel_codes);
    }

}

#endif // VOXEL_CLASSIFY_CUH
