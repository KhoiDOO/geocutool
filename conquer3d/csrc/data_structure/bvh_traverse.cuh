/**
 * @file bvh_traverse.cuh
 * @brief Shared stack-based descent over a Linear BVH.
 *
 * @details Every query in the library walks the hierarchy the same way: pop a node, ask
 * whether it is worth entering, and either visit it as a leaf or push its two children.
 * Only the node test and the leaf action differ. Writing that loop once and passing those
 * two steps as functors keeps the traversal -- including the stack bound, the leaf index
 * arithmetic and the overflow guard -- in a single place, so a fix lands everywhere rather
 * than in one of eighteen copies.
 *
 * The functors are device lambdas and are fully inlined, so the generated code matches the
 * hand-written loop it replaces.
 *
 * @note Two query shapes are deliberately *not* expressed here and remain hand-written:
 * nearest-point search, which pushes the closer child last and prunes each child against
 * the running best distance, and the Fast Winding Number descent, which does work on the
 * branch where the node test *fails* rather than skipping it. Forcing either into this
 * interface would obscure it for no gain.
 */

#ifndef BVH_TRAVERSE_CUH
#define BVH_TRAVERSE_CUH

#include "../constants.h"
#include <cuda_runtime.h>

namespace bvh
{
    /**
     * @brief Walks the hierarchy, visiting every leaf the node test admits.
     *
     * @tparam NodeTest  Callable `bool(int node_idx)`; return false to prune the subtree.
     * @tparam LeafVisit Callable `bool(int leaf_idx)` receiving the leaf's slot in
     *         `object_ids`; return false to abandon the traversal, which is how
     *         any-hit queries stop at their first result.
     *
     * @param[in] num_objects Number of primitives $N$; internal nodes occupy $[0, N-1)$ and
     *     leaves $[N-1, 2N-1)$.
     * @param[in] children Device array of $N - 1$ child index pairs.
     * @param[in] node_test Predicate deciding whether to enter a node.
     * @param[in] leaf_visit Action performed at an admitted leaf.
     *
     * @warning The stack holds `BVH_STACK_SIZE` entries in local memory. A hierarchy deeper
     * than that silently drops the remaining children rather than failing, so a
     * pathologically unbalanced tree can under-report. Morton ordering keeps real trees
     * far shallower than the bound.
     */
    template <class NodeTest, class LeafVisit>
    __device__ __forceinline__ void traverse(
        const int num_objects,
        const int2 *__restrict__ children,
        NodeTest node_test,
        LeafVisit leaf_visit)
    {
        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0;

        while (stack_ptr >= 0)
        {
            const int node_idx = stack[stack_ptr--];

            if (!node_test(node_idx))
                continue;

            if (node_idx >= num_objects - 1)
            {
                if (!leaf_visit(node_idx - (num_objects - 1)))
                    return;
            }
            else if (stack_ptr + 2 < BVH_STACK_SIZE)
            {
                const int2 c = children[node_idx];
                if (c.x >= 0) stack[++stack_ptr] = c.x;
                if (c.y >= 0) stack[++stack_ptr] = c.y;
            }
        }
    }
}

#endif // BVH_TRAVERSE_CUH
