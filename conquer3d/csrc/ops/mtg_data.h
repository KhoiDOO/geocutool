#ifndef MTG_DATA_H
#define MTG_DATA_H

/**
 * @file mtg_data.h
 * @brief Constant lookup tables for Marching Tetrahedra over a regular cubic grid.
 *
 * @details Each voxel is split into six tetrahedra sharing the cube's main diagonal, which
 * inherits Marching Tetrahedra's freedom from topological ambiguity while still operating on
 * a structured grid. The decomposition is consistent across neighbouring cells, so extracted
 * surfaces meet without cracks.
 */

#include <cuda_runtime.h>
#include <stdint.h>

namespace mtg {

/**
 * @brief Decomposition of one voxel into 6 tetrahedra, as cube corner indices.
 *
 * @details All six tetrahedra share the $0 \rightarrow 6$ main diagonal. Neighbouring
 * voxels use the same split, which is what keeps the extracted surface watertight across
 * cell boundaries.
 */
__constant__ int mtg_tets[6][4] = {
    {0, 2, 3, 7},
    {0, 2, 7, 6},
    {0, 4, 6, 7},
    {0, 6, 1, 2},
    {0, 6, 4, 1},
    {5, 6, 1, 4}
};

// The 6 local edges of a tetrahedron, mapped to its 4 local vertices (0-3).
/**
 * @brief Local vertex pair spanned by each of the 6 tetrahedron edges.
 *
 * @details Indices are local to a tetrahedron (0-3), not to the voxel; map them through
 * ::mtg_tets to reach cube corners.
 */
__constant__ int mtg_edgeConnection[6][2] = {
    {0, 1}, // Edge 0
    {0, 2}, // Edge 1
    {0, 3}, // Edge 2
    {1, 2}, // Edge 3
    {1, 3}, // Edge 4
    {2, 3}  // Edge 5
};

// The triangles formed for each of the 16 cases of a tetrahedron.
// Contains up to 2 triangles (6 edges), terminated by -1.
// Winding is consistent so that normals point outward (towards >= iso).
/**
 * @brief Edge triples forming the triangles of each of the 16 tetrahedron sign cases.
 *
 * @details Up to two triangles per tetrahedron, `-1` terminated.
 */
__constant__ int mtg_triTable[16][7] = {
    {-1, -1, -1, -1, -1, -1, -1}, // 0x00
    { 0,  1,  2, -1, -1, -1, -1}, // 0x01
    { 0,  4,  3, -1, -1, -1, -1}, // 0x02
    { 1,  2,  4,  1,  4,  3, -1}, // 0x03
    { 1,  3,  5, -1, -1, -1, -1}, // 0x04
    { 0,  5,  2,  0,  3,  5, -1}, // 0x05
    { 0,  4,  5,  0,  5,  1, -1}, // 0x06
    { 2,  4,  5, -1, -1, -1, -1}, // 0x07
    { 2,  5,  4, -1, -1, -1, -1}, // 0x08
    { 0,  1,  5,  0,  5,  4, -1}, // 0x09
    { 0,  5,  3,  0,  2,  5, -1}, // 0x0A
    { 1,  5,  3, -1, -1, -1, -1}, // 0x0B
    { 1,  3,  4,  1,  4,  2, -1}, // 0x0C
    { 0,  3,  4, -1, -1, -1, -1}, // 0x0D
    { 0,  2,  1, -1, -1, -1, -1}, // 0x0E
    {-1, -1, -1, -1, -1, -1, -1}  // 0x0F
};

// Number of triangles for each of the 16 cases.
/**
 * @brief Triangle count emitted by each of the 16 tetrahedron sign cases (0, 1 or 2).
 */
__constant__ int mtg_num_tris[16] = {
    0, 1, 1, 2, 1, 2, 2, 1,
    1, 2, 2, 1, 2, 1, 1, 0
};

} // namespace mtg

#endif // MTG_DATA_H