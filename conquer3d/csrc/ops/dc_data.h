#ifndef CONQUER3D_OPS_DC_DATA_H
#define CONQUER3D_OPS_DC_DATA_H

/**
 * @file dc_data.h
 * @brief Constant topology tables for Dual Contouring.
 *
 * @details Dual Contouring inverts the Marching Cubes arrangement: one vertex is placed
 * inside each active cell, positioned by minimising a quadratic error function over the
 * surface normals, and faces are formed by connecting the cells around each bipolar edge.
 * That is what lets it reproduce sharp creases and corners that vertex-on-edge methods round
 * away. These tables supply the cell-local topology the QEF kernels index into.
 */

#include <cuda_runtime.h>

namespace conquer3d {
namespace ops {

// 12 Edges in conquer3d CCW convention:
// 0:(0,1), 1:(1,2), 2:(3,2), 3:(0,3) [Bottom -Z]
// 4:(4,5), 5:(5,6), 6:(7,6), 7:(4,7) [Top +Z]
// 8:(0,4), 9:(1,5), 10:(2,6), 11:(3,7) [Vertical +Z]
/**
 * @brief Local corner index pair spanned by each of the 12 cube edges.
 *
 * @details Uses the Conquer3D counter-clockwise convention: edges 0-3 bound the $-Z$ face,
 * 4-7 the $+Z$ face, and 8-11 run vertically along $+Z$.
 */
static __constant__ int dc_edge_corners[12][2] = {
    {0, 1}, {1, 2}, {3, 2}, {0, 3},
    {4, 5}, {5, 6}, {7, 6}, {4, 7},
    {0, 4}, {1, 5}, {2, 6}, {3, 7}
};

// Relative coordinates (u, v, w) in [0, 1]^3 for 8 corners:
/**
 * @brief Unit-cube $(u, v, w)$ coordinates of each of the 8 voxel corners.
 *
 * @details Local parametric coordinates in $[0, 1]^3$, used to evaluate the trilinear field
 * and its gradient inside a cell without reconstructing world positions.
 */
static __constant__ float dc_corner_uvw[8][3] = {
    {0.0f, 0.0f, 0.0f}, // 0
    {1.0f, 0.0f, 0.0f}, // 1
    {1.0f, 1.0f, 0.0f}, // 2
    {0.0f, 1.0f, 0.0f}, // 3
    {0.0f, 0.0f, 1.0f}, // 4
    {1.0f, 0.0f, 1.0f}, // 5
    {1.0f, 1.0f, 1.0f}, // 6
    {0.0f, 1.0f, 1.0f}  // 7
};

// Canonical topological Cartesian quadrant slot for each local edge ID e in [0, 11]
/**
 * @brief Cartesian quadrant slot assigned to each of the 12 local edges.
 *
 * @details Dual methods place vertices per edge rather than per cell, so each edge must map
 * deterministically to one of four quadrant slots. Because adjacent cells agree on this
 * mapping, the dual quad around a shared edge is assembled consistently from all four
 * incident cells.
 */
static __constant__ int dc_edge_quadrant[12] = {
    0, // e=0 (+X) -> slot 0
    3, // e=1 (+Y) -> slot 3
    3, // e=2 (+X) -> slot 3
    0, // e=3 (+Y) -> slot 0
    1, // e=4 (+X) -> slot 1
    2, // e=5 (+Y) -> slot 2
    2, // e=6 (+X) -> slot 2
    1, // e=7 (+Y) -> slot 1
    0, // e=8 (+Z) -> slot 0
    1, // e=9 (+Z) -> slot 1
    2, // e=10 (+Z) -> slot 2
    3  // e=11 (+Z) -> slot 3
};

} // namespace ops
} // namespace conquer3d

#endif // CONQUER3D_OPS_DC_DATA_H
