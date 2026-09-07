#ifndef MT_DATA_H
#define MT_DATA_H

/**
 * @file mt_data.h
 * @brief Constant lookup tables for Marching Tetrahedra on unstructured tetrahedral meshes.
 *
 * @details A tetrahedron has four corners and therefore only $2^4 = 16$ sign cases, none of
 * them topologically ambiguous -- the reason Marching Tetrahedra yields consistent, crack-free
 * surfaces where cube-based methods need an explicit disambiguation rule. Each case emits at
 * most two triangles.
 */

#include <cuda_runtime.h>
#include <stdint.h>

namespace mt {

/**
 * @brief Local vertex pair spanned by each of the 6 tetrahedron edges.
 *
 * @details Edges are ordered $(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)$; every table below
 * indexes edges in this order.
 */
__constant__ int tetEdgeConnection[6][2] = {
    {0, 1}, // Edge 0
    {0, 2}, // Edge 1
    {0, 3}, // Edge 2
    {1, 2}, // Edge 3
    {1, 3}, // Edge 4
    {2, 3}  // Edge 5
};

/**
 * @brief 6-bit mask of bipolar edges for each of the 16 corner-sign cases.
 *
 * @details Bit $e$ is set when edge $e$ crosses the isolevel and so carries an interpolated
 * vertex. The table is symmetric about its midpoint because inverting every corner sign
 * preserves the crossing set while reversing orientation.
 */
__constant__ int tetEdgeTable[16] = {
    0x00, // 0: None
    0x07, // 1 (0001: v0): edges 0, 1, 2
    0x19, // 2 (0010: v1): edges 0, 3, 4
    0x1E, // 3 (0011: v0, v1): edges 1, 2, 3, 4
    0x2A, // 4 (0100: v2): edges 1, 3, 5
    0x2D, // 5 (0101: v0, v2): edges 0, 2, 3, 5
    0x33, // 6 (0110: v1, v2): edges 0, 1, 4, 5
    0x34, // 7 (0111: v0, v1, v2): edges 2, 4, 5
    0x34, // 8 (1000: v3): edges 2, 4, 5
    0x33, // 9 (1001: v0, v3): edges 0, 1, 4, 5
    0x2D, // 10 (1010: v1, v3): edges 0, 2, 3, 5
    0x2A, // 11 (1011: v0, v1, v3): edges 1, 3, 5
    0x1E, // 12 (1100: v2, v3): edges 1, 2, 3, 4
    0x19, // 13 (1101: v0, v2, v3): edges 0, 3, 4
    0x07, // 14 (1110: v1, v2, v3): edges 0, 1, 2
    0x00  // 15: None
};

/**
 * @brief Triangle count emitted by each of the 16 corner-sign cases (0, 1 or 2).
 */
__constant__ int tetNumTris[16] = {
    0, 1, 1, 2, 1, 2, 2, 1,
    1, 2, 2, 1, 2, 1, 1, 0
};

/**
 * @brief Exclusive prefix scan of emitted triangle vertices across the 16 cases.
 *
 * @details 17 entries, so case $c$ emits `tetTriNumTable[c + 1] - tetTriNumTable[c]`
 * vertices. Used to size output buffers without a counting pass.
 */
__constant__ int tetTriNumTable[17] = {
    0, 0, 3, 6, 12, 15, 21, 27, 30, 33, 39, 45, 48, 54, 57, 60, 60
};

/**
 * @brief Edge triples forming the triangles of each corner-sign case.
 *
 * @details Up to two triangles (six edge indices) per case, `-1` terminated. Indices refer
 * to the edge order defined by ::tetEdgeConnection.
 */
__constant__ int tetTriTable[16][7] = {
    {-1, -1, -1, -1, -1, -1, -1}, // 0x00
    { 0,  1,  2, -1, -1, -1, -1}, // 0x01
    { 0,  4,  3, -1, -1, -1, -1}, // 0x02
    { 2,  1,  4,  4,  3,  1, -1}, // 0x03
    { 1,  3,  5, -1, -1, -1, -1}, // 0x04
    { 0,  5,  2,  0,  3,  5, -1}, // 0x05
    { 0,  4,  5,  0,  1,  5, -1}, // 0x06
    { 2,  5,  4, -1, -1, -1, -1}, // 0x07
    { 2,  4,  5, -1, -1, -1, -1}, // 0x08
    { 0,  5,  4,  0,  1,  5, -1}, // 0x09
    { 0,  2,  5,  0,  5,  3, -1}, // 0x0A
    { 1,  5,  3, -1, -1, -1, -1}, // 0x0B
    { 2,  4,  1,  4,  1,  3, -1}, // 0x0C
    { 0,  3,  4, -1, -1, -1, -1}, // 0x0D
    { 0,  2,  1, -1, -1, -1, -1}, // 0x0E
    {-1, -1, -1, -1, -1, -1, -1}  // 0x0F
};

} // namespace mt

#endif // MT_DATA_H
