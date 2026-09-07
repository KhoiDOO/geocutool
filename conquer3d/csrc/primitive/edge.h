/**
 * @file edge.h
 * @brief Undirected edge primitive structure with canonical index sorting for topological hashing.
 */

#ifndef EDGE_H
#define EDGE_H

#include <cuda_runtime.h>
#include <cstdint>
#include "../maths/f3x1.h"

/**
 * @brief Canonical undirected edge between vertex indices $(v_0 \le v_1)$.
 */
struct Edge {
    uint32_t v0; ///< Minimum vertex index.
    uint32_t v1; ///< Maximum vertex index.

    /** @brief Default constructor. */
    __host__ __device__ Edge() : v0(0), v1(0) {}

    /**
     * @brief Constructs a canonically sorted edge where $v_0 = \min(a, b)$ and $v_1 = \max(a, b)$.
     */
    __host__ __device__ Edge(uint32_t a, uint32_t b) {
        v0 = a < b ? a : b;
        v1 = a > b ? a : b;
    }

    /**
     * @brief Equality comparison.
     * @details Edge keys are stored with endpoints sorted, so two edges compare equal
     * regardless of the traversal direction that produced them -- which is what lets a shared
     * edge deduplicate across the triangles meeting at it.
     * @param[in] other Edge to compare against.
     * @return True if both endpoints match.
     */
    __host__ __device__ bool operator==(const Edge& other) const {
        return v0 == other.v0 && v1 == other.v1;
    }

    /**
     * @brief Inequality comparison.
     * @param[in] other Edge to compare against.
     * @return True if the endpoints differ.
     */
    __host__ __device__ bool operator!=(const Edge& other) const {
        return v0 != other.v0 || v1 != other.v1;
    }

    /**
     * @brief Lexicographic ordering by endpoint indices.
     * @details Provides the strict weak ordering the device sort requires, so equal keys form
     * contiguous runs and can be uniqued in a single pass.
     * @param[in] other Edge to compare against.
     * @return True if this edge orders before @p other.
     */
    __host__ __device__ bool operator<(const Edge& other) const {
        if (v0 != other.v0) return v0 < other.v0;
        return v1 < other.v1;
    }

    /**
     * @brief Computes 3D Euclidean midpoint of this edge.
     */
    __host__ __device__ float3 compute_midpoint(const float3* vertices) const {
        float3 p0 = vertices[v0];
        float3 p1 = vertices[v1];
        return (p0 + p1) * 0.5f;
    }
};

#endif // EDGE_H
