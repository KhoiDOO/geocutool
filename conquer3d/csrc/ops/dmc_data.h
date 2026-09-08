#ifndef CONQUER3D_OPS_DMC_DATA_H
#define CONQUER3D_OPS_DMC_DATA_H

/**
 * @file dmc_data.h
 * @brief Constant topology tables for Dual Marching Cubes.
 *
 * @details Dual Marching Cubes decomposes each cell into independent contours and emits one
 * dual vertex per contour rather than one per cell. Cells that Dual Contouring would collapse
 * to a single vertex -- and thereby pinch -- instead receive several, which is what
 * guarantees strictly 2-manifold output. The tables here encode both the cell-local topology
 * and the full 256-case contour patterns.
 */

#include <cuda_runtime.h>

namespace conquer3d {
namespace ops {

/**
 * @brief Sentinel marking a Dual Marching Cubes case that needs runtime disambiguation.
 *
 * @details Stored in ::dmc_ambig_table in place of an equivalence-class representative. A
 * cell carrying this value must have its face and interior connectivity resolved by the
 * asymptotic decider before contours can be emitted.
 */
#define DMC_AMBIGUOUS 254

/**
 * @brief Sentinel marking an ambiguous classical Marching Cubes case.
 *
 * @details Shares the value of ::DMC_AMBIGUOUS so the two extraction paths can consult the
 * same table without translating between sentinels.
 */
#define MC_AMBIGUOUS 254

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
static __constant__ int dmc_edge_corners[12][2] = {
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
static __constant__ float dmc_corner_uvw[8][3] = {
    {0.0f, 0.0f, 0.0f}, // 0
    {1.0f, 0.0f, 0.0f}, // 1
    {1.0f, 1.0f, 0.0f}, // 2
    {0.0f, 1.0f, 0.0f}, // 3
    {0.0f, 0.0f, 1.0f}, // 4
    {1.0f, 0.0f, 1.0f}, // 5
    {1.0f, 1.0f, 1.0f}, // 6
    {0.0f, 1.0f, 1.0f}  // 7
};

/**
 * @brief Cartesian quadrant slot assigned to each of the 12 local edges.
 *
 * @details Dual methods place vertices per edge rather than per cell, so each edge must map
 * deterministically to one of four quadrant slots. Because adjacent cells agree on this
 * mapping, the dual quad around a shared edge is assembled consistently from all four
 * incident cells.
 */
static __constant__ int dmc_edge_quadrant[12] = {
    0, // e=0 (+X) -> slot 0
    3, // e=1 (+Y) -> slot 3
    1, // e=2 (+X) -> slot 1
    0, // e=3 (+Y) -> slot 0
    3, // e=4 (+X) -> slot 3
    2, // e=5 (+Y) -> slot 2
    2, // e=6 (+X) -> slot 2
    1, // e=7 (+Y) -> slot 1
    0, // e=8 (+Z) -> slot 0
    1, // e=9 (+Z) -> slot 1
    2, // e=10 (+Z) -> slot 2
    3  // e=11 (+Z) -> slot 3
};

/**
 * @brief Maps each corner-sign case to its topological equivalence class representative.
 *
 * @details Entries carrying the sentinel ::DMC_AMBIGUOUS (254) mark cases whose face or
 * interior connectivity cannot be settled from corner signs alone and must be resolved at
 * runtime by the asymptotic decider. All other cases index a canonical representative,
 * collapsing the 256 configurations onto the far smaller set of distinct topologies.
 */
static __constant__ unsigned char dmc_ambig_table[256] = {
    0, // quitte: 0 <-> mc: 0, class representative: 0
    1, // quitte: 1 <-> mc: 1, class representative: 1
    2, // quitte: 2 <-> mc: 2, class representative: 1
    3, // quitte: 3 <-> mc: 3, class representative: 3
    4, // quitte: 4 <-> mc: 8, class representative: 1
    5, // quitte: 5 <-> mc: 9, class representative: 3
    MC_AMBIGUOUS, // quitte: 6 <-> mc: 10, class representative: 6
    7, // quitte: 7 <-> mc: 11, class representative: 7
    8, // quitte: 8 <-> mc: 4, class representative: 1
    MC_AMBIGUOUS, // quitte: 9 <-> mc: 5, class representative: 6
    10, // quitte: 10 <-> mc: 6, class representative: 3
    11, // quitte: 11 <-> mc: 7, class representative: 7
    12, // quitte: 12 <-> mc: 12, class representative: 3
    13, // quitte: 13 <-> mc: 13, class representative: 7
    14, // quitte: 14 <-> mc: 14, class representative: 7
    15, // quitte: 15 <-> mc: 15, class representative: 15
    16, // quitte: 16 <-> mc: 16, class representative: 1
    17, // quitte: 17 <-> mc: 17, class representative: 3
    MC_AMBIGUOUS, // quitte: 18 <-> mc: 18, class representative: 6
    19, // quitte: 19 <-> mc: 19, class representative: 7
    MC_AMBIGUOUS, // quitte: 20 <-> mc: 24, class representative: 6
    21, // quitte: 21 <-> mc: 25, class representative: 7
    MC_AMBIGUOUS, // quitte: 22 <-> mc: 26, class representative: 22
    23, // quitte: 23 <-> mc: 27, class representative: 23
    MC_AMBIGUOUS, // quitte: 24 <-> mc: 20, class representative: 24
    MC_AMBIGUOUS, // quitte: 25 <-> mc: 21, class representative: 25
    MC_AMBIGUOUS, // quitte: 26 <-> mc: 22, class representative: 25
    27, // quitte: 27 <-> mc: 23, class representative: 27
    MC_AMBIGUOUS, // quitte: 28 <-> mc: 28, class representative: 25
    29, // quitte: 29 <-> mc: 29, class representative: 29
    MC_AMBIGUOUS, // quitte: 30 <-> mc: 30, class representative: 30
    31, // quitte: 31 <-> mc: 31, class representative: 7
    32, // quitte: 32 <-> mc: 32, class representative: 1
    MC_AMBIGUOUS, // quitte: 33 <-> mc: 33, class representative: 6
    34, // quitte: 34 <-> mc: 34, class representative: 3
    35, // quitte: 35 <-> mc: 35, class representative: 7
    MC_AMBIGUOUS, // quitte: 36 <-> mc: 40, class representative: 24
    MC_AMBIGUOUS, // quitte: 37 <-> mc: 41, class representative: 25
    MC_AMBIGUOUS, // quitte: 38 <-> mc: 42, class representative: 25
    39, // quitte: 39 <-> mc: 43, class representative: 29
    MC_AMBIGUOUS, // quitte: 40 <-> mc: 36, class representative: 6
    MC_AMBIGUOUS, // quitte: 41 <-> mc: 37, class representative: 22
    42, // quitte: 42 <-> mc: 38, class representative: 7
    43, // quitte: 43 <-> mc: 39, class representative: 23
    MC_AMBIGUOUS, // quitte: 44 <-> mc: 44, class representative: 25
    MC_AMBIGUOUS, // quitte: 45 <-> mc: 45, class representative: 30
    46, // quitte: 46 <-> mc: 46, class representative: 27
    47, // quitte: 47 <-> mc: 47, class representative: 7
    48, // quitte: 48 <-> mc: 48, class representative: 3
    49, // quitte: 49 <-> mc: 49, class representative: 7
    50, // quitte: 50 <-> mc: 50, class representative: 7
    51, // quitte: 51 <-> mc: 51, class representative: 15
    MC_AMBIGUOUS, // quitte: 52 <-> mc: 56, class representative: 25
    53, // quitte: 53 <-> mc: 57, class representative: 27
    MC_AMBIGUOUS, // quitte: 54 <-> mc: 58, class representative: 30
    55, // quitte: 55 <-> mc: 59, class representative: 7
    MC_AMBIGUOUS, // quitte: 56 <-> mc: 52, class representative: 25
    MC_AMBIGUOUS, // quitte: 57 <-> mc: 53, class representative: 30
    58, // quitte: 58 <-> mc: 54, class representative: 29
    59, // quitte: 59 <-> mc: 55, class representative: 7
    MC_AMBIGUOUS, // quitte: 60 <-> mc: 60, class representative: 60
    MC_AMBIGUOUS, // quitte: 61 <-> mc: 61, class representative: 25
    MC_AMBIGUOUS, // quitte: 62 <-> mc: 62, class representative: 25
    63, // quitte: 63 <-> mc: 63, class representative: 3
    64, // quitte: 64 <-> mc: 128, class representative: 1
    MC_AMBIGUOUS, // quitte: 65 <-> mc: 129, class representative: 6
    MC_AMBIGUOUS, // quitte: 66 <-> mc: 130, class representative: 24
    MC_AMBIGUOUS, // quitte: 67 <-> mc: 131, class representative: 25
    68, // quitte: 68 <-> mc: 136, class representative: 3
    69, // quitte: 69 <-> mc: 137, class representative: 7
    MC_AMBIGUOUS, // quitte: 70 <-> mc: 138, class representative: 25
    71, // quitte: 71 <-> mc: 139, class representative: 27
    MC_AMBIGUOUS, // quitte: 72 <-> mc: 132, class representative: 6
    MC_AMBIGUOUS, // quitte: 73 <-> mc: 133, class representative: 22
    MC_AMBIGUOUS, // quitte: 74 <-> mc: 134, class representative: 25
    MC_AMBIGUOUS, // quitte: 75 <-> mc: 135, class representative: 30
    76, // quitte: 76 <-> mc: 140, class representative: 7
    77, // quitte: 77 <-> mc: 141, class representative: 23
    78, // quitte: 78 <-> mc: 142, class representative: 29
    79, // quitte: 79 <-> mc: 143, class representative: 7
    80, // quitte: 80 <-> mc: 144, class representative: 3
    81, // quitte: 81 <-> mc: 145, class representative: 7
    MC_AMBIGUOUS, // quitte: 82 <-> mc: 146, class representative: 25
    83, // quitte: 83 <-> mc: 147, class representative: 29
    84, // quitte: 84 <-> mc: 152, class representative: 7
    85, // quitte: 85 <-> mc: 153, class representative: 15
    MC_AMBIGUOUS, // quitte: 86 <-> mc: 154, class representative: 30
    87, // quitte: 87 <-> mc: 155, class representative: 7
    MC_AMBIGUOUS, // quitte: 88 <-> mc: 148, class representative: 25
    MC_AMBIGUOUS, // quitte: 89 <-> mc: 149, class representative: 30
    MC_AMBIGUOUS, // quitte: 90 <-> mc: 150, class representative: 60
    MC_AMBIGUOUS, // quitte: 91 <-> mc: 151, class representative: 25
    92, // quitte: 92 <-> mc: 156, class representative: 27
    93, // quitte: 93 <-> mc: 157, class representative: 7
    MC_AMBIGUOUS, // quitte: 94 <-> mc: 158, class representative: 25
    95, // quitte: 95 <-> mc: 159, class representative: 3
    MC_AMBIGUOUS, // quitte: 96 <-> mc: 160, class representative: 6
    MC_AMBIGUOUS, // quitte: 97 <-> mc: 161, class representative: 22
    MC_AMBIGUOUS, // quitte: 98 <-> mc: 162, class representative: 25
    MC_AMBIGUOUS, // quitte: 99 <-> mc: 163, class representative: 30
    MC_AMBIGUOUS, // quitte: 100 <-> mc: 168, class representative: 25
    MC_AMBIGUOUS, // quitte: 101 <-> mc: 169, class representative: 30
    MC_AMBIGUOUS, // quitte: 102 <-> mc: 170, class representative: 60
    MC_AMBIGUOUS, // quitte: 103 <-> mc: 171, class representative: 25
    MC_AMBIGUOUS, // quitte: 104 <-> mc: 164, class representative: 22
    MC_AMBIGUOUS, // quitte: 105 <-> mc: 165, class representative: 105
    MC_AMBIGUOUS, // quitte: 106 <-> mc: 166, class representative: 30
    MC_AMBIGUOUS, // quitte: 107 <-> mc: 167, class representative: 22
    MC_AMBIGUOUS, // quitte: 108 <-> mc: 172, class representative: 30
    MC_AMBIGUOUS, // quitte: 109 <-> mc: 173, class representative: 22
    MC_AMBIGUOUS, // quitte: 110 <-> mc: 174, class representative: 25
    MC_AMBIGUOUS, // quitte: 111 <-> mc: 175, class representative: 6
    112, // quitte: 112 <-> mc: 176, class representative: 7
    113, // quitte: 113 <-> mc: 177, class representative: 23
    114, // quitte: 114 <-> mc: 178, class representative: 27
    115, // quitte: 115 <-> mc: 179, class representative: 7
    116, // quitte: 116 <-> mc: 184, class representative: 29
    117, // quitte: 117 <-> mc: 185, class representative: 7
    MC_AMBIGUOUS, // quitte: 118 <-> mc: 186, class representative: 25
    119, // quitte: 119 <-> mc: 187, class representative: 3
    MC_AMBIGUOUS, // quitte: 120 <-> mc: 180, class representative: 30
    MC_AMBIGUOUS, // quitte: 121 <-> mc: 181, class representative: 22
    MC_AMBIGUOUS, // quitte: 122 <-> mc: 182, class representative: 25
    MC_AMBIGUOUS, // quitte: 123 <-> mc: 183, class representative: 6
    MC_AMBIGUOUS, // quitte: 124 <-> mc: 188, class representative: 25
    MC_AMBIGUOUS, // quitte: 125 <-> mc: 189, class representative: 6
    MC_AMBIGUOUS, // quitte: 126 <-> mc: 190, class representative: 24
    127, // quitte: 127 <-> mc: 191, class representative: 1
    128, // quitte: 128 <-> mc: 64, class representative: 1
    MC_AMBIGUOUS, // quitte: 129 <-> mc: 65, class representative: 24
    MC_AMBIGUOUS, // quitte: 130 <-> mc: 66, class representative: 6
    MC_AMBIGUOUS, // quitte: 131 <-> mc: 67, class representative: 25
    MC_AMBIGUOUS, // quitte: 132 <-> mc: 72, class representative: 6
    MC_AMBIGUOUS, // quitte: 133 <-> mc: 73, class representative: 25
    MC_AMBIGUOUS, // quitte: 134 <-> mc: 74, class representative: 22
    MC_AMBIGUOUS, // quitte: 135 <-> mc: 75, class representative: 30
    136, // quitte: 136 <-> mc: 68, class representative: 3
    MC_AMBIGUOUS, // quitte: 137 <-> mc: 69, class representative: 25
    138, // quitte: 138 <-> mc: 70, class representative: 7
    139, // quitte: 139 <-> mc: 71, class representative: 29
    140, // quitte: 140 <-> mc: 76, class representative: 7
    141, // quitte: 141 <-> mc: 77, class representative: 27
    142, // quitte: 142 <-> mc: 78, class representative: 23
    143, // quitte: 143 <-> mc: 79, class representative: 7
    MC_AMBIGUOUS, // quitte: 144 <-> mc: 80, class representative: 6
    MC_AMBIGUOUS, // quitte: 145 <-> mc: 81, class representative: 25
    MC_AMBIGUOUS, // quitte: 146 <-> mc: 82, class representative: 22
    MC_AMBIGUOUS, // quitte: 147 <-> mc: 83, class representative: 30
    MC_AMBIGUOUS, // quitte: 148 <-> mc: 88, class representative: 22
    MC_AMBIGUOUS, // quitte: 149 <-> mc: 89, class representative: 30
    MC_AMBIGUOUS, // quitte: 150 <-> mc: 90, class representative: 105
    MC_AMBIGUOUS, // quitte: 151 <-> mc: 91, class representative: 22
    MC_AMBIGUOUS, // quitte: 152 <-> mc: 84, class representative: 25
    MC_AMBIGUOUS, // quitte: 153 <-> mc: 85, class representative: 60
    MC_AMBIGUOUS, // quitte: 154 <-> mc: 86, class representative: 30
    MC_AMBIGUOUS, // quitte: 155 <-> mc: 87, class representative: 25
    MC_AMBIGUOUS, // quitte: 156 <-> mc: 92, class representative: 30
    MC_AMBIGUOUS, // quitte: 157 <-> mc: 93, class representative: 25
    MC_AMBIGUOUS, // quitte: 158 <-> mc: 94, class representative: 22
    MC_AMBIGUOUS, // quitte: 159 <-> mc: 95, class representative: 6
    160, // quitte: 160 <-> mc: 96, class representative: 3
    MC_AMBIGUOUS, // quitte: 161 <-> mc: 97, class representative: 25
    162, // quitte: 162 <-> mc: 98, class representative: 7
    163, // quitte: 163 <-> mc: 99, class representative: 27
    MC_AMBIGUOUS, // quitte: 164 <-> mc: 104, class representative: 25
    MC_AMBIGUOUS, // quitte: 165 <-> mc: 105, class representative: 60
    MC_AMBIGUOUS, // quitte: 166 <-> mc: 106, class representative: 30
    MC_AMBIGUOUS, // quitte: 167 <-> mc: 107, class representative: 25
    168, // quitte: 168 <-> mc: 100, class representative: 7
    MC_AMBIGUOUS, // quitte: 169 <-> mc: 101, class representative: 30
    170, // quitte: 170 <-> mc: 102, class representative: 15
    171, // quitte: 171 <-> mc: 103, class representative: 7
    172, // quitte: 172 <-> mc: 108, class representative: 29
    MC_AMBIGUOUS, // quitte: 173 <-> mc: 109, class representative: 25
    174, // quitte: 174 <-> mc: 110, class representative: 7
    175, // quitte: 175 <-> mc: 111, class representative: 3
    176, // quitte: 176 <-> mc: 112, class representative: 7
    177, // quitte: 177 <-> mc: 113, class representative: 29
    178, // quitte: 178 <-> mc: 114, class representative: 23
    179, // quitte: 179 <-> mc: 115, class representative: 7
    MC_AMBIGUOUS, // quitte: 180 <-> mc: 120, class representative: 30
    MC_AMBIGUOUS, // quitte: 181 <-> mc: 121, class representative: 25
    MC_AMBIGUOUS, // quitte: 182 <-> mc: 122, class representative: 22
    MC_AMBIGUOUS, // quitte: 183 <-> mc: 123, class representative: 6
    184, // quitte: 184 <-> mc: 116, class representative: 27
    MC_AMBIGUOUS, // quitte: 185 <-> mc: 117, class representative: 25
    186, // quitte: 186 <-> mc: 118, class representative: 7
    187, // quitte: 187 <-> mc: 119, class representative: 3
    MC_AMBIGUOUS, // quitte: 188 <-> mc: 124, class representative: 25
    MC_AMBIGUOUS, // quitte: 189 <-> mc: 125, class representative: 24
    MC_AMBIGUOUS, // quitte: 190 <-> mc: 126, class representative: 6
    191, // quitte: 191 <-> mc: 127, class representative: 1
    192, // quitte: 192 <-> mc: 192, class representative: 3
    MC_AMBIGUOUS, // quitte: 193 <-> mc: 193, class representative: 25
    MC_AMBIGUOUS, // quitte: 194 <-> mc: 194, class representative: 25
    MC_AMBIGUOUS, // quitte: 195 <-> mc: 195, class representative: 60
    196, // quitte: 196 <-> mc: 200, class representative: 7
    197, // quitte: 197 <-> mc: 201, class representative: 29
    MC_AMBIGUOUS, // quitte: 198 <-> mc: 202, class representative: 30
    MC_AMBIGUOUS, // quitte: 199 <-> mc: 203, class representative: 25
    200, // quitte: 200 <-> mc: 196, class representative: 7
    MC_AMBIGUOUS, // quitte: 201 <-> mc: 197, class representative: 30
    202, // quitte: 202 <-> mc: 198, class representative: 27
    MC_AMBIGUOUS, // quitte: 203 <-> mc: 199, class representative: 25
    204, // quitte: 204 <-> mc: 204, class representative: 15
    205, // quitte: 205 <-> mc: 205, class representative: 7
    206, // quitte: 206 <-> mc: 206, class representative: 7
    207, // quitte: 207 <-> mc: 207, class representative: 3
    208, // quitte: 208 <-> mc: 208, class representative: 7
    209, // quitte: 209 <-> mc: 209, class representative: 27
    MC_AMBIGUOUS, // quitte: 210 <-> mc: 210, class representative: 30
    MC_AMBIGUOUS, // quitte: 211 <-> mc: 211, class representative: 25
    212, // quitte: 212 <-> mc: 216, class representative: 23
    213, // quitte: 213 <-> mc: 217, class representative: 7
    MC_AMBIGUOUS, // quitte: 214 <-> mc: 218, class representative: 22
    MC_AMBIGUOUS, // quitte: 215 <-> mc: 219, class representative: 6
    216, // quitte: 216 <-> mc: 212, class representative: 29
    MC_AMBIGUOUS, // quitte: 217 <-> mc: 213, class representative: 25
    MC_AMBIGUOUS, // quitte: 218 <-> mc: 214, class representative: 25
    MC_AMBIGUOUS, // quitte: 219 <-> mc: 215, class representative: 24
    220, // quitte: 220 <-> mc: 220, class representative: 7
    221, // quitte: 221 <-> mc: 221, class representative: 3
    MC_AMBIGUOUS, // quitte: 222 <-> mc: 222, class representative: 6
    223, // quitte: 223 <-> mc: 223, class representative: 1
    224, // quitte: 224 <-> mc: 224, class representative: 7
    MC_AMBIGUOUS, // quitte: 225 <-> mc: 225, class representative: 30
    226, // quitte: 226 <-> mc: 226, class representative: 29
    MC_AMBIGUOUS, // quitte: 227 <-> mc: 227, class representative: 25
    228, // quitte: 228 <-> mc: 232, class representative: 27
    MC_AMBIGUOUS, // quitte: 229 <-> mc: 233, class representative: 25
    MC_AMBIGUOUS, // quitte: 230 <-> mc: 234, class representative: 25
    MC_AMBIGUOUS, // quitte: 231 <-> mc: 235, class representative: 24
    232, // quitte: 232 <-> mc: 228, class representative: 23
    MC_AMBIGUOUS, // quitte: 233 <-> mc: 229, class representative: 22
    234, // quitte: 234 <-> mc: 230, class representative: 7
    MC_AMBIGUOUS, // quitte: 235 <-> mc: 231, class representative: 6
    236, // quitte: 236 <-> mc: 236, class representative: 7
    MC_AMBIGUOUS, // quitte: 237 <-> mc: 237, class representative: 6
    238, // quitte: 238 <-> mc: 238, class representative: 3
    239, // quitte: 239 <-> mc: 239, class representative: 1
    240, // quitte: 240 <-> mc: 240, class representative: 15
    241, // quitte: 241 <-> mc: 241, class representative: 7
    242, // quitte: 242 <-> mc: 242, class representative: 7
    243, // quitte: 243 <-> mc: 243, class representative: 3
    244, // quitte: 244 <-> mc: 248, class representative: 7
    245, // quitte: 245 <-> mc: 249, class representative: 3
    MC_AMBIGUOUS, // quitte: 246 <-> mc: 250, class representative: 6
    247, // quitte: 247 <-> mc: 251, class representative: 1
    248, // quitte: 248 <-> mc: 244, class representative: 7
    MC_AMBIGUOUS, // quitte: 249 <-> mc: 245, class representative: 6
    250, // quitte: 250 <-> mc: 246, class representative: 3
    251, // quitte: 251 <-> mc: 247, class representative: 1
    252, // quitte: 252 <-> mc: 252, class representative: 3
    253, // quitte: 253 <-> mc: 253, class representative: 1
    254, // quitte: 254 <-> mc: 254, class representative: 1
    255 // quitte: 255 <-> mc: 255, class representative: 0
};

/**
 * @brief Packed contour patterns for all 256 corner-sign cases.
 *
 * @details A flat $256 \times 17$ byte array. Row $c$ begins at offset $17c$; its first
 * entry is the number of contours in the cell, after which each contour is stored as a
 * length followed by that many edge indices, with unused entries set to `-1`. Flattening
 * keeps the whole table at 4352 bytes so it stays resident in constant memory and is
 * broadcast to every thread in a warp at full speed.
 */
static __constant__ char dmc_r_pattern[4352] = {
    0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 8, 3, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 11, 2, 0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 11, 2, 1, 9, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 2, 10, 9, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 8, 3, 2, 10, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 10, 1, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 10, 1, 0, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 9, 0, 3, 11, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 9, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 3, 0, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 1, 9, 4, 7, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 4, 7, 11, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    3, 3, 3, 3, 9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1,
    1, 6, 4, 7, 11, 2, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 4, 7, 3, 0, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 2, 10, 9, 0, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 2, 10, 9, 4, 7, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 10, 1, 3, 11, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 11, 10, 1, 0, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 4, 7, 8, 9, 0, 3, 11, 10, -1, -1, -1, -1, -1, -1,
    1, 5, 4, 7, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 5, 4, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 5, 4, 8, 3, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 11, 2, 0, 8, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 5, 4, 0, 1, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 2, 1, 5, 4, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1,
    3, 3, 3, 3, 3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1,
    1, 5, 2, 10, 5, 4, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 2, 10, 5, 4, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 3, 11, 10, 1, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 4, 9, 5, 0, 8, 11, 10, 1, -1, -1, -1, -1, -1, -1,
    1, 6, 5, 4, 0, 3, 11, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 5, 4, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 7, 8, 9, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 3, 0, 9, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 7, 8, 0, 1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 1, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 9, 5, 7, 8, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 9, 5, 7, 11, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 2, 3, 11, 0, 1, 5, 7, 8, -1, -1, -1, -1, -1, -1,
    1, 5, 11, 2, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 7, 8, 9, 5, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 10, 1, 2, 9, 5, 7, 3, 0, -1, -1, -1, -1, -1, -1,
    1, 6, 8, 0, 2, 10, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 2, 10, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 4, 9, 5, 7, 8, 10, 1, 3, 11, -1, -1, -1, -1, -1, -1,
    1, 7, 5, 7, 11, 10, 1, 0, 9, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 7, 11, 10, 5, 7, 8, 0, 3, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 11, 10, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 1, 9, 8, 3, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 2, 3, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 0, 8, 7, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 7, 6, 2, 3, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 6, 2, 1, 9, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1,
    3, 3, 3, 3, 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1,
    2, 4, 3, 9, 0, 2, 10, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 6, 11, 7, 2, 10, 9, 8, 3, -1, -1, -1, -1, -1, -1,
    1, 5, 7, 6, 10, 1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 7, 6, 10, 1, 0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 0, 3, 7, 6, 10, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 7, 6, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 8, 4, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 6, 11, 3, 0, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 6, 11, 8, 4, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 9, 4, 6, 11, 3, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 2, 3, 8, 4, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 0, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 1, 9, 0, 2, 3, 8, 4, 6, -1, -1, -1, -1, -1, -1,
    1, 5, 1, 9, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 8, 4, 6, 11, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 1, 2, 10, 3, 0, 4, 6, 11, -1, -1, -1, -1, -1, -1,
    2, 4, 4, 11, 8, 4, 6, 0, 2, 10, 9, -1, -1, -1, -1, -1, -1,
    1, 7, 10, 9, 4, 6, 11, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 1, 3, 8, 4, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 10, 1, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 7, 4, 6, 10, 9, 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 10, 9, 4, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1,
    3, 3, 3, 3, 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1,
    2, 4, 3, 0, 1, 5, 4, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 11, 7, 6, 8, 3, 1, 5, 4, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 2, 3, 7, 6, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 9, 5, 4, 0, 8, 7, 6, 2, -1, -1, -1, -1, -1, -1,
    2, 4, 4, 6, 2, 3, 7, 1, 5, 4, 0, -1, -1, -1, -1, -1, -1,
    1, 7, 6, 2, 1, 5, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1,
    3, 3, 3, 3, 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1,
    4, 3, 3, 3, 3, 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5,
    2, 3, 5, 7, 6, 11, 5, 4, 0, 2, 10, -1, -1, -1, -1, -1, -1,
    2, 6, 3, 4, 8, 3, 2, 10, 5, 11, 7, 6, -1, -1, -1, -1, -1,
    2, 3, 5, 9, 5, 4, 10, 1, 3, 7, 6, -1, -1, -1, -1, -1, -1,
    2, 6, 3, 6, 10, 1, 0, 8, 7, 9, 5, 4, -1, -1, -1, -1, -1,
    1, 7, 4, 0, 3, 7, 6, 10, 5, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 7, 6, 10, 5, 4, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 9, 5, 6, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 6, 11, 3, 0, 9, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 11, 8, 0, 1, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 6, 11, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 8, 9, 5, 6, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 9, 5, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 7, 1, 5, 6, 2, 3, 8, 0, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 1, 5, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 1, 2, 10, 9, 5, 6, 11, 8, -1, -1, -1, -1, -1, -1,
    2, 6, 3, 11, 3, 0, 9, 5, 6, 1, 2, 10, -1, -1, -1, -1, -1,
    1, 7, 11, 8, 0, 2, 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 6, 11, 3, 2, 10, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 7, 1, 3, 8, 9, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 10, 1, 0, 9, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 8, 3, 1, 9, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 0, 8, 11, 2, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1,
    3, 3, 3, 3, 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1,
    2, 3, 5, 5, 10, 6, 1, 9, 8, 11, 2, -1, -1, -1, -1, -1, -1,
    1, 4, 6, 5, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 6, 5, 1, 2, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 6, 5, 9, 0, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 5, 9, 8, 3, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 3, 11, 6, 5, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 0, 8, 11, 6, 5, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 3, 11, 6, 5, 9, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 6, 5, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 3, 0, 4, 7, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1,
    3, 3, 3, 3, 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1,
    2, 3, 5, 10, 6, 5, 1, 9, 4, 7, 3, -1, -1, -1, -1, -1, -1,
    3, 3, 3, 3, 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1,
    2, 3, 5, 5, 10, 6, 4, 7, 11, 2, 0, -1, -1, -1, -1, -1, -1,
    4, 3, 3, 3, 3, 0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6,
    2, 6, 3, 2, 1, 9, 4, 7, 11, 5, 10, 6, -1, -1, -1, -1, -1,
    2, 4, 3, 1, 2, 6, 5, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 4, 1, 2, 6, 5, 3, 0, 4, 7, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 8, 4, 7, 9, 0, 2, 6, 5, -1, -1, -1, -1, -1, -1,
    1, 7, 7, 3, 2, 6, 5, 9, 4, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 8, 4, 7, 3, 11, 6, 5, 1, -1, -1, -1, -1, -1, -1,
    1, 7, 5, 1, 0, 4, 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 6, 3, 5, 9, 0, 3, 11, 6, 8, 4, 7, -1, -1, -1, -1, -1,
    1, 6, 6, 5, 9, 4, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 4, 9, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 10, 6, 4, 9, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 0, 1, 10, 6, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 8, 3, 1, 10, 6, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 4, 9, 10, 6, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 4, 0, 8, 11, 2, 4, 9, 10, 6, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 3, 11, 2, 0, 1, 10, 6, 4, -1, -1, -1, -1, -1, -1,
    1, 7, 6, 4, 8, 11, 2, 1, 10, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 4, 9, 1, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 3, 0, 8, 1, 2, 6, 4, 9, -1, -1, -1, -1, -1, -1,
    1, 4, 0, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 8, 3, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 6, 4, 9, 1, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 7, 8, 11, 6, 4, 9, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 3, 11, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 6, 4, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 10, 6, 7, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 7, 3, 0, 9, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 10, 6, 7, 8, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 10, 6, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 2, 3, 11, 10, 6, 7, 8, 9, -1, -1, -1, -1, -1, -1,
    1, 7, 2, 0, 9, 10, 6, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 6, 3, 8, 0, 1, 10, 6, 7, 2, 3, 11, -1, -1, -1, -1, -1,
    1, 6, 11, 2, 1, 10, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 1, 2, 6, 7, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 7, 2, 6, 7, 3, 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 7, 8, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 7, 3, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 7, 8, 9, 1, 3, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 7, 8, 0, 3, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 5, 10, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 5, 10, 11, 7, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 3, 11, 7, 5, 10, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1,
    2, 4, 4, 7, 5, 10, 11, 9, 8, 3, 1, -1, -1, -1, -1, -1, -1,
    1, 5, 5, 10, 2, 3, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 2, 0, 8, 7, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 9, 0, 1, 5, 10, 2, 3, 7, -1, -1, -1, -1, -1, -1,
    1, 7, 9, 8, 7, 5, 10, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 1, 2, 11, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 0, 8, 3, 1, 2, 11, 7, 5, -1, -1, -1, -1, -1, -1,
    1, 6, 7, 5, 9, 0, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 7, 7, 5, 9, 8, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 1, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 0, 8, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 9, 0, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 9, 8, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 8, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 0, 4, 5, 10, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 0, 1, 9, 8, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1,
    1, 7, 10, 11, 3, 1, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 5, 10, 2, 3, 8, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 5, 10, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 6, 3, 10, 2, 3, 8, 4, 5, 0, 1, 9, -1, -1, -1, -1, -1,
    1, 6, 5, 10, 2, 1, 9, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 5, 1, 2, 11, 8, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 7, 0, 4, 5, 1, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 7, 0, 2, 11, 8, 4, 5, 9, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 8, 4, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 0, 4, 5, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 8, 4, 5, 9, 0, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 11, 7, 4, 9, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 5, 0, 8, 3, 4, 9, 10, 11, 7, -1, -1, -1, -1, -1, -1,
    1, 6, 1, 10, 11, 7, 4, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 7, 3, 1, 10, 11, 7, 4, 8, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 9, 10, 2, 3, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 7, 9, 10, 2, 0, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 7, 3, 7, 4, 0, 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 3, 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 11, 7, 4, 9, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    2, 6, 3, 7, 4, 9, 1, 2, 11, 0, 8, 3, -1, -1, -1, -1, -1,
    1, 5, 11, 7, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 11, 7, 4, 8, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 4, 9, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 4, 9, 1, 0, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 4, 0, 3, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 9, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 3, 0, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 0, 1, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 3, 1, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 2, 3, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 9, 10, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 2, 3, 8, 0, 1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 5, 1, 2, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 6, 3, 0, 9, 1, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 0, 2, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 4, 1, 3, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 3, 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
};

/**
 * @brief Bit-packed edge indices of each cube face, for the on-the-fly asymptotic decider.
 *
 * @details Face order is $-Z, +Z, -Y, +Y, -X, +X$. Each 16-bit word packs the face's four
 * edge indices into 4-bit fields, so the decider unpacks a face with shifts and masks rather
 * than an extra constant-memory read.
 */
static __constant__ unsigned short dmc_e_face[6] = {
    (unsigned short)291, (unsigned short)18277, (unsigned short)18696, (unsigned short)10859, (unsigned short)33719, (unsigned short)38305
};

/**
 * @brief Bit-packed corner indices of each cube face, matching ::dmc_e_face ordering.
 *
 * @details Four 4-bit fields per word, giving the corners whose values the bilinear saddle
 * evaluation needs when disambiguating a face.
 */
static __constant__ unsigned short dmc_v_face[6] = {
    (unsigned short)12576, (unsigned short)25717, (unsigned short)5380, (unsigned short)29538, (unsigned short)8292, (unsigned short)30001
};

} // namespace ops
} // namespace conquer3d

#endif // CONQUER3D_OPS_DMC_DATA_H
