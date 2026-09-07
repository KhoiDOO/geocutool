/**
 * @file bvh.cu
 * @brief CUDA kernel implementations for parallel Radix Linear Bounding Volume Hierarchy (LBVH) (Karras 2012).
 */

#include "bvh.h"
#include "zcurve.h"
#include "../primitive/ray.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>

namespace bvh
{
    /**
     * @brief An axis-aligned box, as carried through the Thrust bounds reduction.
     */
    struct Node {
        float3 min_pt; ///< Lower corner.
        float3 max_pt; ///< Upper corner.
    };

    /**
     * @brief Thrust binary functor merging two boxes into their union.
     * @details Associative and commutative, as a parallel reduction requires, so the scene
     * bounds can be computed in one `thrust::reduce` regardless of how the work is split.
     */
    struct Reduce {
        /**
         * @brief Merges two boxes into their union.
         * @param[in] a First box.
         * @param[in] b Second box.
         * @return The smallest box containing both.
         */
        __device__ __forceinline__ Node operator()(const Node& a, const Node& b) const {
            Node res;
            aabb::compute_aabb_union(a.min_pt, a.max_pt, b.min_pt, b.max_pt, res.min_pt, res.max_pt);
            return res;
        }
    };

    /**
     * @brief Thrust unary functor turning a primitive index into its ::bvh::Node box.
     * @details Fused with the reduction so the per-primitive boxes are never materialised as
     * an intermediate array.
     */
    struct Transform {
        const float3* mins; ///< Device array of primitive lower bounds.
        const float3* maxs; ///< Device array of primitive upper bounds.
        
        /**
         * @brief Binds the functor to the primitive bounds arrays.
         * @param[in] _mins Device array of primitive lower bounds.
         * @param[in] _maxs Device array of primitive upper bounds.
         */
        Transform(const float3* _mins, const float3* _maxs) : mins(_mins), maxs(_maxs) {}

        /**
         * @brief Loads one primitive's bounds as a ::bvh::Node.
         * @param[in] idx Primitive index.
         * @return The primitive's box.
         */
        __device__ __forceinline__ Node operator()(const int& idx) const {
            Node node;
            node.min_pt = mins[idx];
            node.max_pt = maxs[idx];
            return node;
        }
    };

    /**
     * @brief Length of the common high-order bit prefix of two Morton codes.
     * @details The similarity measure Karras's hierarchy construction is built on: the longer
     * the shared prefix, the closer two primitives lie in space. Equal codes fall back to
     * comparing indices, which breaks ties deterministically and keeps the emitted tree
     * well-formed when several primitives share a cell.
     * @param[in] num_objects Number of primitives, used for the bounds check.
     * @param[in] morton_codes Device array of sorted Morton codes.
     * @param[in] i First index.
     * @param[in] j Second index; out-of-range values return $-1$ to terminate a range scan.
     * @return Shared prefix length in bits, or $-1$ when @p j is out of range.
     */
    __device__ __forceinline__ int common_prefix(
        const int num_objects,
        const uint32_t* morton_codes,
        const int i, 
        const int j)
    {
        // Out of bounds check: if j is outside the array, they share no prefix (-1)
        if (j < 0 || j >= num_objects) return -1;

        uint32_t key_i = morton_codes[i];
        uint32_t key_j = morton_codes[j];

        if (key_i == key_j) {
            // Duplicate coordinate fallback: Add 32 and use the array index as a unique tie-breaker
            return 32 + __clz(i ^ j);
        }

        // Standard hardware-accelerated matching prefix count
        return __clz(key_i ^ key_j);
    }

/**
 * @brief Assigns a Morton code to each primitive from its AABB centroid.
 * @details One thread per primitive. The centroid is normalised into the scene bounds and
 * interleaved into a 30-bit Morton code. Sorting primitives by this code places spatial
 * neighbours adjacently, which is the precondition for Karras's hierarchy emission --
 * the tree structure is read directly off the sorted code sequence.
 * @param[in] num_objects Number of primitives $N$.
 * @param[in] aabb_mins Device array of $N$ AABB lower bounds.
 * @param[in] aabb_maxs Device array of $N$ AABB upper bounds.
 * @param[in] scene_min Lower bound of the whole scene, used to normalise centroids.
 * @param[in] scene_max Upper bound of the whole scene.
 * @param[out] morton_codes Device array of $N$ Morton codes.
 * @param[out] object_ids Device array of $N$ identity indices, permuted by the sort that
 *     follows so results can be mapped back to the caller's ordering.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Primitives whose centroids fall within $2^{-10}$ of the scene extent share a code.
 * Duplicates are handled during hierarchy emission by falling back to index comparison.
 */
__global__ void compute_morton_codes_kernel(
        const uint32_t num_objects,
        const float3* __restrict__ aabb_mins,
        const float3* __restrict__ aabb_maxs,
        const float3 scene_min,
        const float3 scene_max,
        uint32_t* __restrict__ morton_codes,
        int* __restrict__ object_ids)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_objects) return;

        float3 min_pt = aabb_mins[idx];
        float3 max_pt = aabb_maxs[idx];

        float3 centroid;
        aabb::compute_aabb_centroid(min_pt, max_pt, centroid);

        float3 extents;
        aabb::compute_aabb_extent(scene_min, scene_max, extents);

        // Normalize the centroid to a [0.0, 1.0] range
        float3 normalized = make_float3(
            (centroid.x - scene_min.x) / extents.x,
            (centroid.y - scene_min.y) / extents.y,
            (centroid.z - scene_min.z) / extents.z
        );

        morton_codes[idx] = zcurve::morton3D(normalized.x, normalized.y, normalized.z);
        
        object_ids[idx] = idx; 
    }

/**
 * @brief Builds the internal-node hierarchy from sorted Morton codes (Karras, 2012).
 * @details One thread per internal node, of which there are exactly $N - 1$. Each thread
 * determines its node's range by comparing common prefix lengths with its neighbours,
 * locates the split by binary search, and writes its two children -- with no
 * synchronisation between threads at all. That independence is the whole point of the
 * Karras construction: the entire hierarchy is emitted in a single parallel pass.
 * @param[in] num_objects Number of primitives $N$.
 * @param[in] morton_codes Device array of $N$ Morton codes in ascending order.
 * @param[out] bvh_children Device array of $N - 1$ child index pairs.
 * @param[out] bvh_parents Device array of $2N - 1$ parent indices for the bottom-up pass.
 * @note Launched over $N - 1$ threads; internal nodes occupy indices $[0, N-1)$ and leaves
 * $[N-1, 2N-1)$.
 * @warning Requires @p morton_codes to be sorted ascending. Unsorted input produces a
 * structurally valid but spatially meaningless tree, which degrades queries to a linear
 * scan rather than failing outright.
 */
__global__ void karras_emit_hierarchy_kernel(
        const int num_objects,
        const uint32_t* __restrict__ morton_codes,
        int2* __restrict__ bvh_children,
        int* __restrict__ bvh_parents)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        // We only launch N-1 threads, one for each internal node
        if (idx >= num_objects - 1) return;

        // Determine direction of the range (+1 or -1)
        int delta_right = common_prefix(num_objects, morton_codes, idx, idx + 1);
        int delta_left  = common_prefix(num_objects, morton_codes, idx, idx - 1);
        int d = (delta_right > delta_left) ? 1 : -1;

        // Compute upper bound for the length of the range
        int delta_min = min(delta_right, delta_left);
        int l_max = 2;
        while (common_prefix(num_objects, morton_codes, idx, idx + l_max * d) > delta_min) {
            l_max *= 2; // Jump by powers of 2
        }

        // Find the other end using binary search
        int l = 0;
        for (int t = l_max / 2; t >= 1; t /= 2) {
            if (common_prefix(num_objects, morton_codes, idx, idx + (l + t) * d) > delta_min) {
                l += t;
            }
        }
        int j = idx + l * d;

        // Find the split position using binary search
        int delta_node = common_prefix(num_objects, morton_codes, idx, j);
        int s = 0;
        int t = l;
        
        do {
            t = (t + 1) >> 1;
            if (common_prefix(num_objects, morton_codes, idx, idx + (s + t) * d) > delta_node) {
                s += t;
            }
        } while (t > 1);
        
        int gamma = idx + s * d + min(d, 0);

        // Output child and parent pointers
        int min_idx = min(idx, j);
        int max_idx = max(idx, j);
        
        int left  = (min_idx == gamma) ? (num_objects - 1 + gamma) : gamma;
        int right = (max_idx == gamma + 1) ? (num_objects - 1 + gamma + 1) : (gamma + 1);

        bvh_children[idx] = make_int2(left, right);
        bvh_parents[left] = idx;
        bvh_parents[right] = idx;
    }

/**
 * @brief Propagates bounding boxes from leaves to the root.
 * @details One thread per leaf. Each seeds its own AABB, then walks towards the root
 * merging bounds. At every internal node an `atomicCAS` flag decides which of the two
 * arriving threads continues: the first exits, the second -- knowing both children are now
 * final -- merges and proceeds. Exactly one thread therefore reaches the root, and the
 * whole refit completes in one launch with no global barrier.
 * @param[in] num_objects Number of primitives $N$.
 * @param[in] object_ids Device array mapping sorted leaf order to original indices.
 * @param[in] in_aabb_mins Device array of $N$ primitive AABB lower bounds.
 * @param[in] in_aabb_maxs Device array of $N$ primitive AABB upper bounds.
 * @param[in] bvh_parents Device array of $2N - 1$ parent indices.
 * @param[in] bvh_children Device array of $N - 1$ child index pairs.
 * @param[out] bvh_aabb_mins Device array of $2N - 1$ node lower bounds.
 * @param[out] bvh_aabb_maxs Device array of $2N - 1$ node upper bounds.
 * @param[in,out] atomic_flags Device array of $N - 1$ visit flags; must be zeroed before
 *     launch.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning @p atomic_flags must be cleared between builds. Stale flags let the first
 * thread through at a node whose sibling has not finished, producing bounds that silently
 * omit part of the subtree.
 */
__global__ void bottom_up_aabb_kernel(
        const int num_objects,
        const int* __restrict__ object_ids,
        const float3* __restrict__ in_aabb_mins,
        const float3* __restrict__ in_aabb_maxs,
        const int* __restrict__ bvh_parents,
        const int2* __restrict__ bvh_children,
        float3* __restrict__ bvh_aabb_mins,
        float3* __restrict__ bvh_aabb_maxs,
        int* __restrict__ atomic_flags)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx >= num_objects) return;

        int leaf_idx = num_objects - 1 + idx;
        int original_obj_id = object_ids[idx];
        
        bvh_aabb_mins[leaf_idx] = in_aabb_mins[original_obj_id];
        bvh_aabb_maxs[leaf_idx] = in_aabb_maxs[original_obj_id];

        int current_node = bvh_parents[leaf_idx];

        while (current_node != -1) 
        {
            __threadfence();

            // Atomically flag that a child has arrived
            int old = atomicAdd(&atomic_flags[current_node], 1);

            if (old == 0) {
                return;
            }

            // 3. Parent AABB Computation
            int left_child = bvh_children[current_node].x;
            int right_child = bvh_children[current_node].y;

            aabb::compute_aabb_union(
                bvh_aabb_mins[left_child], bvh_aabb_maxs[left_child],
                bvh_aabb_mins[right_child], bvh_aabb_maxs[right_child],
                bvh_aabb_mins[current_node], bvh_aabb_maxs[current_node]
            );

            // Move up
            current_node = bvh_parents[current_node];
        }
    }

    void build(
        const uint32_t num_objects,
        const uint32_t num_nodes,
        const float3 *__restrict__ in_aabb_mins, 
        const float3 *__restrict__ in_aabb_maxs, 
        float3 *__restrict__ bvh_aabb_mins,      
        float3 *__restrict__ bvh_aabb_maxs,      
        int2 *__restrict__ bvh_children,         
        int *__restrict__ bvh_parents,           
        int *__restrict__ object_ids)
    {
        at::cuda::ThrustAllocator allocator;
        auto policy = thrust::cuda::par(allocator).on(at::cuda::getCurrentCUDAStream());

        thrust::counting_iterator<int> iter(0);
        Node init_node;
        init_node.min_pt = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        init_node.max_pt = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        Node scene_bounds = thrust::transform_reduce(
            policy,
            iter, iter + num_objects,
            Transform(in_aabb_mins, in_aabb_maxs),
            init_node,
            Reduce()
        );

        thrust::device_vector<uint32_t> morton_codes(num_objects);
        
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_objects + threads - 1) / threads;

        compute_morton_codes_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_objects,
            in_aabb_mins,
            in_aabb_maxs,
            scene_bounds.min_pt,
            scene_bounds.max_pt,
            thrust::raw_pointer_cast(morton_codes.data()),
            object_ids
        );

        thrust::sort_by_key(
            policy,
            morton_codes.begin(),
            morton_codes.end(),
            thrust::device_pointer_cast(object_ids)
        );

        cudaMemsetAsync(bvh_parents, -1, sizeof(int) * num_nodes, at::cuda::getCurrentCUDAStream());

        uint32_t internal_threads = NTHREADS;
        uint32_t internal_blocks = (num_objects - 1 + internal_threads - 1) / internal_threads;

        karras_emit_hierarchy_kernel<<<internal_blocks, internal_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_objects,
            thrust::raw_pointer_cast(morton_codes.data()),
            bvh_children,
            bvh_parents
        );

        uint32_t leaf_threads = NTHREADS;
        uint32_t leaf_blocks = (num_objects + leaf_threads - 1) / leaf_threads;

        thrust::device_vector<int> atomic_flags(num_objects - 1, 0);

        bottom_up_aabb_kernel<<<leaf_blocks, leaf_threads>>>(
            num_objects,
            object_ids,
            in_aabb_mins,
            in_aabb_maxs,
            bvh_parents,
            bvh_children,
            bvh_aabb_mins,
            bvh_aabb_maxs,
            thrust::raw_pointer_cast(atomic_flags.data())
        );

        cudaDeviceSynchronize();
    }

/**
 * @brief Reports every BVH primitive whose AABB overlaps each query box.
 * @details One thread per query box, descending the hierarchy with a private stack and
 * emitting a (query, object) pair for each overlapping leaf. Pairs are appended to a shared
 * buffer through an atomic counter.
 * @param[in] num_queries Number of query boxes $Q$.
 * @param[in] num_objects Number of primitives in the tree $N$.
 * @param[in] query_mins Device array of $Q$ query AABB lower bounds.
 * @param[in] query_maxs Device array of $Q$ query AABB upper bounds.
 * @param[in] bvh_aabb_mins Device array of $2N - 1$ node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of $2N - 1$ node upper bounds.
 * @param[in] bvh_children Device array of $N - 1$ child index pairs.
 * @param[in] object_ids Device array mapping leaves to original primitive indices.
 * @param[out] out_query_ids Device array receiving the query index of each pair.
 * @param[out] out_object_ids Device array receiving the primitive index of each pair.
 * @param[in,out] hit_counter Device counter, atomically incremented per pair.
 * @param[in] max_capacity Capacity of the output arrays.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Traversal uses a per-thread stack of `BVH_STACK_SIZE` entries held in local
 * memory. A deeply unbalanced hierarchy can overflow it; Morton ordering keeps the tree
 * shallow enough in practice, but pathological input remains a risk.
 * @warning Output slots are claimed with a single atomic on @p hit_counter, so pair
 * ordering is nondeterministic between runs. Writes stop once @p max_capacity is
 * reached; compare the final counter against it to detect truncation.
 */
__global__ void query_bvh_kernel(
        const uint32_t num_queries,
        const uint32_t num_objects,
        const float3* __restrict__ query_mins,
        const float3* __restrict__ query_maxs,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        int64_t* __restrict__ out_query_ids,
        int64_t* __restrict__ out_object_ids,
        int64_t* __restrict__ hit_counter,
        const int64_t max_capacity)
    {
        uint32_t q_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (q_idx >= num_queries) return;

        float3 q_min = query_mins[q_idx];
        float3 q_max = query_maxs[q_idx];

        // The local thread stack
        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        
        // Push the Root Node (Index 0) to start
        stack[0] = 0; 

        while (stack_ptr >= 0) 
        {
            int node_idx = stack[stack_ptr--]; 

            if (aabb::test_aabb_overlap(q_min, q_max, bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx])) 
            {
                if (node_idx >= num_objects - 1) 
                {
                    int leaf_idx = node_idx - (num_objects - 1);
                    int original_obj_id = object_ids[leaf_idx];

                    uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int*)hit_counter, 1ULL);

                    if (write_idx < max_capacity) {
                        out_query_ids[write_idx] = q_idx;
                        out_object_ids[write_idx] = original_obj_id;
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
    }

    void query(
        const uint32_t num_queries,
        const uint32_t num_objects,
        const float3* __restrict__ query_mins,
        const float3* __restrict__ query_maxs,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        int64_t* __restrict__ out_query_ids,
        int64_t* __restrict__ out_object_ids,
        int64_t* __restrict__ hit_counter,
        const int64_t max_capacity)
    {
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_queries + threads - 1) / threads;

        query_bvh_kernel<<<blocks, threads>>>(
            num_queries,
            num_objects,
            query_mins,
            query_maxs,
            bvh_aabb_mins,
            bvh_aabb_maxs,
            bvh_children,
            object_ids,
            out_query_ids,
            out_object_ids,
            hit_counter,
            max_capacity
        );
    }

/**
 * @brief Reports every overlapping pair of primitives within one BVH.
 * @details One thread per leaf, querying the tree that contains it. Self-pairs are
 * skipped, and a pair is emitted only when the querying index is the lower of the two, so
 * each unordered pair is reported exactly once rather than twice.
 * @param[in] num_objects Number of primitives $N$.
 * @param[in] bvh_aabb_mins Device array of $2N - 1$ node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of $2N - 1$ node upper bounds.
 * @param[in] bvh_children Device array of $N - 1$ child index pairs.
 * @param[in] object_ids Device array mapping leaves to original primitive indices.
 * @param[out] out_query_ids Device array receiving the first index of each pair.
 * @param[out] out_object_ids Device array receiving the second index of each pair.
 * @param[in,out] hit_counter Device counter, atomically incremented per pair.
 * @param[in] max_capacity Capacity of the output arrays.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Traversal uses a per-thread stack of `BVH_STACK_SIZE` entries held in local
 * memory. A deeply unbalanced hierarchy can overflow it; Morton ordering keeps the tree
 * shallow enough in practice, but pathological input remains a risk.
 * @warning Output slots are claimed with a single atomic on @p hit_counter, so pair
 * ordering is nondeterministic between runs. Writes stop once @p max_capacity is
 * reached; compare the final counter against it to detect truncation.
 */
__global__ void query_self_bvh_kernel(
        const uint32_t num_objects,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        int64_t* __restrict__ out_query_ids,
        int64_t* __restrict__ out_object_ids,
        int64_t* __restrict__ hit_counter,
        const int64_t max_capacity)
    {
        uint32_t leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (leaf_idx >= num_objects) return;

        int node_offset = num_objects - 1;
        float3 q_min = bvh_aabb_mins[node_offset + leaf_idx];
        float3 q_max = bvh_aabb_maxs[node_offset + leaf_idx];
        int original_q_id = object_ids[leaf_idx];

        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0; 

        while (stack_ptr >= 0) 
        {
            int node_idx = stack[stack_ptr--]; 

            if (aabb::test_aabb_overlap(q_min, q_max, bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx])) 
            {
                if (node_idx >= num_objects - 1) 
                {
                    int other_leaf_idx = node_idx - (num_objects - 1);
                    int original_obj_id = object_ids[other_leaf_idx];

                    if (original_q_id < original_obj_id) 
                    {
                        uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int*)hit_counter, 1ULL);

                        if (write_idx < max_capacity) {
                            out_query_ids[write_idx] = original_q_id;
                            out_object_ids[write_idx] = original_obj_id;
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
    }

    void query_self(
        const uint32_t num_objects,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        int64_t* __restrict__ out_query_ids,
        int64_t* __restrict__ out_object_ids,
        int64_t* __restrict__ hit_counter,
        const int64_t max_capacity)
    {
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_objects + threads - 1) / threads;

        query_self_bvh_kernel<<<blocks, threads>>>(
            num_objects,
            bvh_aabb_mins,
            bvh_aabb_maxs,
            bvh_children,
            object_ids,
            out_query_ids,
            out_object_ids,
            hit_counter,
            max_capacity
        );
    }

/**
 * @brief Reports every primitive whose AABB a ray enters.
 * @details One thread per ray, using the Kay-Kajiya slab test at each node and descending
 * only into children the ray actually meets. This is a broad-phase pass: it returns
 * candidates whose bounding boxes are hit, not confirmed surface intersections, so an
 * exact test against the primitives is still required downstream.
 * @param[in] num_queries Number of rays $Q$.
 * @param[in] num_objects Number of primitives $N$.
 * @param[in] ray_origins Device array of $Q$ ray origins.
 * @param[in] ray_dirs Device array of $Q$ ray directions.
 * @param[in] bvh_aabb_mins Device array of $2N - 1$ node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of $2N - 1$ node upper bounds.
 * @param[in] bvh_children Device array of $N - 1$ child index pairs.
 * @param[in] object_ids Device array mapping leaves to original primitive indices.
 * @param[out] out_query_ids Device array receiving the ray index of each hit.
 * @param[out] out_object_ids Device array receiving the primitive index of each hit.
 * @param[in,out] hit_counter Device counter, atomically incremented per hit.
 * @param[in] max_capacity Capacity of the output arrays.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @warning Traversal uses a per-thread stack of `BVH_STACK_SIZE` entries held in local
 * memory. A deeply unbalanced hierarchy can overflow it; Morton ordering keeps the tree
 * shallow enough in practice, but pathological input remains a risk.
 * @warning Output slots are claimed with a single atomic on @p hit_counter, so pair
 * ordering is nondeterministic between runs. Writes stop once @p max_capacity is
 * reached; compare the final counter against it to detect truncation.
 */
__global__ void query_ray_bvh_kernel(
        const uint32_t num_queries,
        const uint32_t num_objects,
        const float3* __restrict__ ray_origins,
        const float3* __restrict__ ray_dirs,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        int64_t* __restrict__ out_query_ids,
        int64_t* __restrict__ out_object_ids,
        int64_t* __restrict__ hit_counter,
        const int64_t max_capacity)
    {
        uint32_t q_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (q_idx >= num_queries) return;

        Ray ray(ray_origins[q_idx], ray_dirs[q_idx]);

        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0; 

        while (stack_ptr >= 0) 
        {
            int node_idx = stack[stack_ptr--]; 
            float t_hit;

            if (ray.is_intersect_aabb(bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx], t_hit)) 
            {
                if (node_idx >= num_objects - 1) 
                {
                    int leaf_idx = node_idx - (num_objects - 1);
                    int original_obj_id = object_ids[leaf_idx];

                    uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int*)hit_counter, 1ULL);

                    if (write_idx < max_capacity) {
                        out_query_ids[write_idx] = q_idx;
                        out_object_ids[write_idx] = original_obj_id;
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
    }

    void query_ray(
        const uint32_t num_queries,
        const uint32_t num_objects,
        const float3* __restrict__ ray_origins,
        const float3* __restrict__ ray_dirs,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        int64_t* __restrict__ out_query_ids,
        int64_t* __restrict__ out_object_ids,
        int64_t* __restrict__ hit_counter,
        const int64_t max_capacity)
    {
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_queries + threads - 1) / threads;

        query_ray_bvh_kernel<<<blocks, threads>>>(
            num_queries,
            num_objects,
            ray_origins,
            ray_dirs,
            bvh_aabb_mins,
            bvh_aabb_maxs,
            bvh_children,
            object_ids,
            out_query_ids,
            out_object_ids,
            hit_counter,
            max_capacity
        );
    }

/**
 * @brief Finds the primitive whose AABB is nearest to each query point.
 * @details One thread per query, tracking the best squared distance found so far and
 * pruning any subtree whose bound already exceeds it. Unlike the other query kernels this
 * one writes a single result per query at a fixed index, so it needs no atomics and its
 * output is deterministic.
 * @param[in] num_queries Number of query points $Q$.
 * @param[in] num_objects Number of primitives $N$.
 * @param[in] query_points Device array of $Q$ coordinates.
 * @param[in] bvh_aabb_mins Device array of $2N - 1$ node lower bounds.
 * @param[in] bvh_aabb_maxs Device array of $2N - 1$ node upper bounds.
 * @param[in] bvh_children Device array of $N - 1$ child index pairs.
 * @param[in] object_ids Device array mapping leaves to original primitive indices.
 * @param[out] out_query_ids Device array of $Q$ query indices.
 * @param[out] out_object_ids Device array of $Q$ nearest primitive indices, $-1$ if none.
 * @param[out] out_distances Device array of $Q$ squared distances to the nearest AABB.
 * @note Launched with `NTHREADS` threads per block over a 1D grid.
 * @note Distances are to the bounding box, not the primitive surface; a box is only a
 * lower bound on the true distance.
 * @warning Traversal uses a per-thread stack of `BVH_STACK_SIZE` entries held in local
 * memory. A deeply unbalanced hierarchy can overflow it; Morton ordering keeps the tree
 * shallow enough in practice, but pathological input remains a risk.
 */
__global__ void query_point_bvh_kernel(
        const uint32_t num_queries,
        const uint32_t num_objects,
        const float3* __restrict__ query_points,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        int64_t* __restrict__ out_query_ids,
        int64_t* __restrict__ out_object_ids,
        float* __restrict__ out_distances)
    {
        uint32_t q_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (q_idx >= num_queries) return;

        float3 p = query_points[q_idx];

        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0; 
        
        float best_dist_sq = FLT_MAX;
        int best_leaf_original_id = -1;

        while (stack_ptr >= 0) 
        {
            int node_idx = stack[stack_ptr--]; 

            float dist_sq = aabb::compute_squared_distance(p, bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx]);
            
            if (dist_sq > best_dist_sq) continue; // Prune branch

            if (node_idx >= num_objects - 1) 
            {
                if (dist_sq < best_dist_sq) {
                    best_dist_sq = dist_sq;
                    best_leaf_original_id = object_ids[node_idx - (num_objects - 1)];
                }
            } 
            else 
            {
                if (stack_ptr + 2 < BVH_STACK_SIZE)
                {
                    int2 children = bvh_children[node_idx];
                    
                    float dist_l = aabb::compute_squared_distance(p, bvh_aabb_mins[children.x], bvh_aabb_maxs[children.x]);
                    float dist_r = aabb::compute_squared_distance(p, bvh_aabb_mins[children.y], bvh_aabb_maxs[children.y]);
                    
                    if (dist_l > dist_r) {
                        stack[++stack_ptr] = children.x; 
                        stack[++stack_ptr] = children.y;
                    } else {
                        stack[++stack_ptr] = children.y;
                        stack[++stack_ptr] = children.x;
                    }
                }
            }
        }
        
        out_query_ids[q_idx] = q_idx;
        out_object_ids[q_idx] = best_leaf_original_id;
        out_distances[q_idx] = sqrtf(best_dist_sq);
    }

    void query_point(
        const uint32_t num_queries,
        const uint32_t num_objects,
        const float3* __restrict__ query_points,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        int64_t* __restrict__ out_query_ids,
        int64_t* __restrict__ out_object_ids,
        float* __restrict__ out_distances)
    {
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_queries + threads - 1) / threads;

        query_point_bvh_kernel<<<blocks, threads>>>(
            num_queries,
            num_objects,
            query_points,
            bvh_aabb_mins,
            bvh_aabb_maxs,
            bvh_children,
            object_ids,
            out_query_ids,
            out_object_ids,
            out_distances
        );
    }
}