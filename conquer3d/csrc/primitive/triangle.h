/**
 * @file triangle.h
 * @brief CUDA device primitives for 3D Triangle geometric tests (Ray-Triangle, Closest Point, Triangle-Triangle, Triangle-Box).
 */

#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "../maths/maths.h"

#include "ray.h"
#include "aabb.h"

/**
 * @brief 3D Triangle geometric primitive structure with hardware-optimized device methods.
 */
struct Triangle {
    float3 v0, ///< First corner.
           v1, ///< Second corner.
           v2; ///< Third corner.

    /**
     * @brief Default constructor; all three corners are set to the origin.
     */
    __host__ __device__ __forceinline__ Triangle() 
        : v0(make_float3(0.0f, 0.0f, 0.0f)), v1(make_float3(0.0f, 0.0f, 0.0f)), v2(make_float3(0.0f, 0.0f, 0.0f)) {}

    /**
     * @brief Constructs a triangle from three corners.
     * @details Winding determines the normal's direction: counter-clockwise as seen from
     * outside yields an outward normal.
     * @param[in] a First corner.
     * @param[in] b Second corner.
     * @param[in] c Third corner.
     */
    __host__ __device__ __forceinline__ Triangle(const float3& a, const float3& b, const float3& c)
        : v0(a), v1(b), v2(c) {}

    /**
     * @brief Ray-triangle intersection via the Moller-Trumbore algorithm.
     * @details Solves for the ray parameter and barycentric coordinates directly, never
     * forming the triangle's plane equation -- cheaper and better conditioned than the
     * plane-then-containment approach.
     * @param[in] ray The ray to test.
     * @param[out] t_hit Ray parameter at the intersection.
     * @param[out] u First barycentric coordinate.
     * @param[out] v Second barycentric coordinate.
     * @return True if the ray meets the triangle within its valid range.
     * @note The third barycentric coordinate is $1 - u - v$.
     * @warning Rays near-parallel to the plane are rejected by an epsilon test on the
     * determinant and report no hit, even when exactly coplanar.
     */
    __host__ __device__ __forceinline__ bool is_intersect_ray(const Ray& ray, float& t_hit, float& u, float& v) const {
        float3 edge1 = v1 - v0;
        float3 edge2 = v2 - v0;
        float3 h = maths::cross(ray.direction, edge2);
        float a = maths::dot(edge1, h);

        // If a is near zero, ray is parallel to triangle
        if (a > -1e-6f && a < 1e-6f) return false; 

        float f = 1.0f / a;
        float3 s = ray.origin - v0;
        u = f * maths::dot(s, h);

        if (u < 0.0f || u > 1.0f) return false;

        float3 q = maths::cross(s, edge1);
        v = f * maths::dot(ray.direction, q);

        if (v < 0.0f || u + v > 1.0f) return false;

        float t = f * maths::dot(edge2, q);
        
        // Ensure hit is within ray's valid range
        if (t >= ray.t_min && t <= ray.t_max) { 
            t_hit = t;
            return true;
        }
        return false;
    }

    /**
     * @brief Closest point on the triangle to a query point.
     * @details Classifies the query against the triangle's seven Voronoi regions -- three
     * vertices, three edges, and the face interior -- and projects accordingly. Handling the
     * regions explicitly is what keeps the result on the triangle when the plane projection
     * falls outside it.
     * @param[in] p Query point.
     * @return The closest point, which may lie on a vertex or an edge.
     */
    __host__ __device__ __forceinline__ float3 compute_closest_point(const float3& p) const {
        float3 ab = v1 - v0;
        float3 ac = v2 - v0;
        float3 ap = p - v0;

        float d1 = maths::dot(ab, ap);
        float d2 = maths::dot(ac, ap);
        if (d1 <= 0.0f && d2 <= 0.0f) return v0;

        float3 bp = p - v1;
        float d3 = maths::dot(ab, bp);
        float d4 = maths::dot(ac, bp);
        if (d3 >= 0.0f && d4 <= d3) return v1;

        float vc = d1*d4 - d3*d2;
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
            float v = d1 / (d1 - d3);
            return v0 + v * ab;
        }

        float3 cp = p - v2;
        float d5 = maths::dot(ab, cp);
        float d6 = maths::dot(ac, cp);
        if (d6 >= 0.0f && d5 <= d6) return v2;

        float vb = d5*d2 - d1*d6;
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
            float w = d2 / (d2 - d6);
            return v0 + w * ac;
        }

        float va = d3*d6 - d5*d4;
        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return v1 + w * (v2 - v1);
        }

        float denom = 1.0f / (va + vb + vc);
        float v = vb * denom;
        float w = vc * denom;
        return v0 + ab * v + ac * w;
    }

    /**
     * @brief Unit geometric normal.
     * @details Normalised cross product of two edge vectors; direction follows the winding.
     * @return The unit normal.
     * @warning Degenerate triangles yield NaNs.
     */
    __host__ __device__ __forceinline__ float3 compute_normal() const {
        float3 edge1 = v1 - v0;
        float3 edge2 = v2 - v0;
        return maths::normalize(maths::cross(edge1, edge2));
    }

    /**
     * @brief Triangle area.
     * @details Half the magnitude of the edge cross product.
     * @return The area; zero for a degenerate triangle.
     */
    __host__ __device__ __forceinline__ float compute_area() const {
        float3 edge1 = v1 - v0;
        float3 edge2 = v2 - v0;
        float3 cross = maths::cross(edge1, edge2);
        return 0.5f * sqrtf(maths::dot(cross, cross));
    }

    /**
     * @brief Whether any interior angle exceeds 90 degrees.
     * @details Detected from the sign of the edge dot products, with no inverse trigonometry.
     * Obtuse triangles place their circumcentre outside themselves, which is why mixed Voronoi
     * area computation needs a barycentric fallback for them.
     * @return True if the triangle is obtuse.
     */
    __host__ __device__ __forceinline__ bool is_obtuse() const {
        float3 e01 = v1 - v0;
        float3 e02 = v2 - v0;
        float3 e10 = v0 - v1;
        float3 e12 = v2 - v1;
        float3 e20 = v0 - v2;
        float3 e21 = v1 - v2;

        float d0 = maths::dot(e01, e02);
        float d1 = maths::dot(e10, e12);
        float d2 = maths::dot(e20, e21);

        return (d0 < 0.0f) || (d1 < 0.0f) || (d2 < 0.0f);
    }

    /**
     * @brief Cotangents of the three interior angles.
     * @details Evaluated as $\cot\theta = \cos\theta / \sin\theta$ from dot and cross
     * products without computing the angle. These are the weights of the discrete
     * Laplace-Beltrami operator.
     * @param[out] cot0 Cotangent at the first corner.
     * @param[out] cot1 Cotangent at the second corner.
     * @param[out] cot2 Cotangent at the third corner.
     * @warning Cotangents diverge as an angle approaches 0 or $\pi$, so nearly degenerate
     * triangles yield very large weights that can destabilise the assembled operator.
     */
    __host__ __device__ __forceinline__ void compute_cotangents(float& cot0, float& cot1, float& cot2) const {
        float3 e01 = v1 - v0;
        float3 e12 = v2 - v1;
        float3 e20 = v0 - v2;

        float d0 = maths::dot(e01, -e20);
        float d1 = maths::dot(e12, -e01);
        float d2 = maths::dot(e20, -e12);

        cot0 = d0 / fmaxf(maths::norm(maths::cross(e01, -e20)), 1e-8f);
        cot1 = d1 / fmaxf(maths::norm(maths::cross(e12, -e01)), 1e-8f);
        cot2 = d2 / fmaxf(maths::norm(maths::cross(e20, -e12)), 1e-8f);
    }

    /**
     * @brief The three interior angles, in radians.
     * @param[out] a0 Angle at the first corner.
     * @param[out] a1 Angle at the second corner.
     * @param[out] a2 Angle at the third corner.
     * @note The three sum to $\pi$ up to floating-point error.
     */
    __host__ __device__ __forceinline__ void compute_angles(float& a0, float& a1, float& a2) const {
        float3 e01 = v1 - v0;
        float3 e02 = v2 - v0;
        float3 e10 = v0 - v1;
        float3 e12 = v2 - v1;
        float3 e20 = v0 - v2;
        float3 e21 = v1 - v2;

        float n01 = maths::norm(e01);
        float n02 = maths::norm(e02);
        float n12 = maths::norm(e12);

        float d0 = maths::dot(e01, e02) / fmaxf(n01 * n02, 1e-8f);
        float d1 = maths::dot(e10, e12) / fmaxf(n01 * n12, 1e-8f);
        float d2 = maths::dot(e20, e21) / fmaxf(n02 * n12, 1e-8f);

        a0 = acosf(maths::clamp(d0, -1.0f, 1.0f));
        a1 = acosf(maths::clamp(d1, -1.0f, 1.0f));
        a2 = acosf(maths::clamp(d2, -1.0f, 1.0f));
    }

    /**
     * @brief Centroid of the triangle.
     * @return The arithmetic mean of the three corners.
     */
    __host__ __device__ __forceinline__ float3 compute_centroid() const {
        return (v0 + v1 + v2) * (1.0f / 3.0f);
    }

    /**
     * @brief Samples a point uniformly by area.
     * @details Warps two uniform variates through $(1 - \sqrt{r_1})$, $\sqrt{r_1}(1 - r_2)$,
     * $\sqrt{r_1} r_2$. The square root is what makes samples uniform over the triangle
     * rather than clustered towards a corner.
     * @param[in] r1 First uniform variate in $[0, 1]$.
     * @param[in] r2 Second uniform variate in $[0, 1]$.
     * @return The sampled point.
     */
    __host__ __device__ __forceinline__ float3 sample_point(float r1, float r2) const {
        float sqrt_r1 = sqrtf(r1);
        float u = 1.0f - sqrt_r1;
        float v = r2 * sqrt_r1;
        float w = 1.0f - u - v;
        return v0 * u + v1 * v + v2 * w;
    }

    /**
     * @brief Axis-aligned bounding box of the triangle.
     * @param[out] aabb_min Lower bound.
     * @param[out] aabb_max Upper bound.
     */
    __host__ __device__ __forceinline__ void compute_aabb(float3 &aabb_min, float3 &aabb_max) const {
        aabb_min = make_float3(fminf(v0.x, fminf(v1.x, v2.x)), fminf(v0.y, fminf(v1.y, v2.y)), fminf(v0.z, fminf(v1.z, v2.z)));
        aabb_max = make_float3(fmaxf(v0.x, fmaxf(v1.x, v2.x)), fmaxf(v0.y, fmaxf(v1.y, v2.y)), fmaxf(v0.z, fmaxf(v1.z, v2.z)));
    }

    /**
     * @brief Centre of the circle through all three corners.
     * @details With @p strict_inside set, a circumcentre falling outside the triangle -- as
     * happens for every obtuse triangle -- is replaced by the midpoint of the longest edge.
     * That clamp keeps mixed Voronoi area computation from producing negative contributions.
     * @param[in] strict_inside Whether to clamp the result into the triangle.
     * @return The circumcentre.
     * @warning Undefined for degenerate triangles, whose circumradius is unbounded.
     */
    __host__ __device__ __forceinline__ float3 compute_circumcenter(bool strict_inside = false) const {
        float3 a = v0 - v2;
        float3 b = v1 - v2;
        float3 cross_ab = maths::cross(a, b);
        float denom = 2.0f * maths::dot(cross_ab, cross_ab);
        
        if (fabsf(denom) < 1e-8f) {
            return compute_centroid();
        }
        
        float3 circumcenter = v2 + (maths::cross(cross_ab, a) * maths::dot(b, b) + maths::cross(b, cross_ab) * maths::dot(a, a)) * (1.0f / denom);
        
        if (strict_inside) {
            if (!test_point_inside_on_tria_plane(circumcenter)) {
                float d01 = maths::dot(v0 - v1, v0 - v1);
                float d12 = maths::dot(v1 - v2, v1 - v2);
                float d20 = maths::dot(v2 - v0, v2 - v0);
                
                if (d01 >= d12 && d01 >= d20) {
                    circumcenter = (v0 + v1) * 0.5f;
                } else if (d12 >= d01 && d12 >= d20) {
                    circumcenter = (v1 + v2) * 0.5f;
                } else {
                    circumcenter = (v2 + v0) * 0.5f;
                }
            }
        }
        
        return circumcenter;
    }

    /**
     * @brief Normalised shape quality score.
     * @details Relates area to edge lengths, so the score is scale invariant: 1 for an
     * equilateral triangle, falling to 0 as it degenerates.
     * @return Quality in $[0, 1]$.
     */
    __host__ __device__ __forceinline__ float compute_quality() const {
        float area = compute_area();
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v1;
        float3 e3 = v0 - v2;
        float l1 = maths::norm(e1);
        float l2 = maths::norm(e2);
        float l3 = maths::norm(e3);
        float l = fmaxf(fmaxf(l1, l2), l3);
        float s = (l1 + l2 + l3) * 0.5f;
        float q = (6 / sqrtf(3.0f)) * (area / (s * l));
        return q;
    }

    /**
     * @brief How close the triangle is to equilateral.
     * @details Compares edge lengths against their mean, scoring 1 when all three agree.
     * @return Regularity in $[0, 1]$.
     */
    __host__ __device__ __forceinline__ float compute_triangle_regularity() const {
        float area = compute_area();
        if (area <= 1e-8f) return 0.0f;
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v1;
        float3 e3 = v0 - v2;
        float l1 = maths::norm(e1);
        float l2 = maths::norm(e2);
        float l3 = maths::norm(e3);
        float s = (l1 + l2 + l3) * 0.5f;
        float r = area / s;
        float l = fmaxf(fmaxf(l1, l2), l3);
        return (2.0f * sqrtf(3.0f) * r) / l;
    }

    /**
     * @brief Circumradius divided by shortest edge length.
     * @details The quantity Delaunay refinement bounds, and so the natural quality measure for
     * meshes produced by such algorithms.
     * @return The ratio; lower is better, with $1/\sqrt{3}$ the equilateral minimum.
     */
    __host__ __device__ __forceinline__ float compute_radius_edge_ratio() const {
        float area = compute_area();
        if (area <= 1e-8f) return 0.0f;
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v1;
        float3 e3 = v0 - v2;
        float l1 = maths::norm(e1);
        float l2 = maths::norm(e2);
        float l3 = maths::norm(e3);
        float R = (l1 * l2 * l3) / (4.0f * area);
        float e = fminf(fminf(l1, l2), l3);
        if (e <= 1e-8f) return 0.0f;
        return R / e;
    }

    /**
     * @brief Largest deviation of an interior angle from 60 degrees.
     * @details Catches badly shaped triangles that area-based scores miss because they are
     * small rather than thin.
     * @return The maximum deviation in radians.
     */
    __host__ __device__ __forceinline__ float compute_angle_deviation() const {
        float a0, a1, a2;
        compute_angles(a0, a1, a2);
        float opt = 1.0471975511965977f; // 60 degrees in radians (PI / 3)
        return (fabsf(a0 - opt) + fabsf(a1 - opt) + fabsf(a2 - opt)) / 3.0f;
    }

    /**
     * @brief Circumradius divided by inradius.
     * @details Attains its minimum of 2 for an equilateral triangle and grows without bound as
     * the triangle degenerates.
     * @return The ratio, at least 2 for any valid triangle.
     */
    __host__ __device__ __forceinline__ float compute_radii_ratio() const {
        float area = compute_area();
        if (area <= 1e-8f) return 0.0f;
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v1;
        float3 e3 = v0 - v2;
        float l1 = maths::norm(e1);
        float l2 = maths::norm(e2);
        float l3 = maths::norm(e3);
        float s = (l1 + l2 + l3) * 0.5f;
        float r = area / s;
        float R = (l1 * l2 * l3) / (4.0f * area);
        return r / R;
    }

    /**
     * @brief Aspect ratio under a selectable definition.
     * @details Meshing literature defines aspect ratio inconsistently, so @p mode selects
     * between the longest-to-shortest edge ratio and the longest-edge-to-inradius form rather
     * than committing to one.
     * @param[in] mode Definition selector.
     * @return The aspect ratio; larger values indicate worse conditioning.
     */
    __host__ __device__ __forceinline__ float compute_ar(int mode) const {
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v1;
        float3 e3 = v0 - v2;
        float l1 = maths::norm(e1);
        float l2 = maths::norm(e2);
        float l3 = maths::norm(e3);

        if (mode == 0){
            float area = compute_area();
            float l = fmaxf(fmaxf(l1, l2), l3);
            return (sqrtf(3.0f) * l * l) / (4.0f * area);
        } else if (mode == 1){
            float s = (l1 + l2 + l3) * 0.5f;
            return (l1 * l2 * l3) / (8.0f * (s - l1) * (s - l2) * (s - l3));
        } else if (mode == 2){
            float l_min = fminf(fminf(l1, l2), l3);
            float l_max = fmaxf(fmaxf(l1, l2), l3);
            return l_min / l_max;
        } else {
            return 0.0f;
        }
    }

    /**
     * @brief Whether a point lies on the triangle's supporting plane.
     * @details Compares $|\mathbf{n} \cdot (\mathbf{p} - \mathbf{v}_0)|$ against a tolerance.
     * A coplanarity test only; it says nothing about lying within the triangle's bounds.
     * @param[in] p Query point.
     * @param[in] eps Distance tolerance. Defaults to 1e-5.
     * @return True if within @p eps of the plane.
     */
    __host__ __device__ __forceinline__ bool test_point_on_tria_plane(const float3& p, float eps = 1e-5f) const {
        float3 n = compute_normal();
        float dist = fabsf(maths::dot(n, p - v0));
        return dist <= eps;
    }

    /**
     * @brief Whether a coplanar point lies within the triangle.
     * @details Solves for barycentric coordinates and checks all three are non-negative.
     * @param[in] p Query point, assumed coplanar.
     * @return True if inside, edges and vertices included; false for degenerate triangles.
     * @warning A point off the plane projects onto it and may still report true; pair with
     * test_point_on_tria_plane(), or use test_point_inside().
     */
    __host__ __device__ __forceinline__ bool test_point_inside_on_tria_plane(const float3& p) const {
        float3 edge0 = v1 - v0;
        float3 edge1 = v2 - v0;
        float3 v2_p = p - v0;

        float d00 = maths::dot(edge0, edge0);
        float d01 = maths::dot(edge0, edge1);
        float d11 = maths::dot(edge1, edge1);
        float d20 = maths::dot(v2_p, edge0);
        float d21 = maths::dot(v2_p, edge1);

        float denom = d00 * d11 - d01 * d01;
        
        if (fabsf(denom) < 1e-8f) return false;

        float invDenom = 1.0f / denom;
        float v = (d11 * d20 - d01 * d21) * invDenom;
        float w = (d00 * d21 - d01 * d20) * invDenom;
        float u = 1.0f - v - w;

        return (u >= 0.0f) && (v >= 0.0f) && (w >= 0.0f);
    }

    /**
     * @brief Whether a point lies on the triangle's surface.
     * @details The full containment predicate: coplanar within @p eps *and* inside the
     * barycentric bounds, short-circuiting on the cheaper plane test first.
     * @param[in] p Query point.
     * @param[in] eps Coplanarity tolerance. Defaults to 1e-5.
     * @return True if the point lies on the triangle.
     */
    __host__ __device__ __forceinline__ bool test_point_inside(const float3& p, float eps = 1e-5f) const {
        if (!test_point_on_tria_plane(p, eps)) return false;
        return test_point_inside_on_tria_plane(p);
    }

    /**
     * @brief Exact intersection test against another triangle.
     * @details Implements the Moller interval-overlap test: each triangle is classified
     * against the other's supporting plane, and where both straddle it the overlap of
     * their intersection intervals along the line of plane intersection decides the
     * result. Coplanar pairs fall back to a 2D edge-overlap test.
     * @param[in] T2 The triangle to test against.
     * @return True if the two triangles intersect.
     * @note Triangles sharing a vertex or an edge report an intersection; filter
     * adjacency beforehand when testing a mesh against itself.
     */
    __host__ __device__ __inline__ bool test_intersection(const Triangle& T2) const {
        const float3& V0 = v0;
        const float3& V1 = v1;
        const float3& V2 = v2;
        const float3& U0 = T2.v0;
        const float3& U1 = T2.v1;
        const float3& U2 = T2.v2;
        float3 E1 = V1 - V0;
        float3 E2 = V2 - V0;
        float3 N1 = maths::cross(E1, E2);
        
        float dU0 = maths::dot(N1, U0 - V0);
        float dU1 = maths::dot(N1, U1 - V0);
        float dU2 = maths::dot(N1, U2 - V0);
        
        if (dU0 * dU1 > 0.0f && dU0 * dU2 > 0.0f) return false;
        
        float3 D1 = U1 - U0;
        float3 D2 = U2 - U0;
        float3 N2 = maths::cross(D1, D2);
        
        float dV0 = maths::dot(N2, V0 - U0);
        float dV1 = maths::dot(N2, V1 - U0);
        float dV2 = maths::dot(N2, V2 - U0);
        
        if (dV0 * dV1 > 0.0f && dV0 * dV2 > 0.0f) return false;
        
        float3 D = maths::cross(N1, N2);
        
        float pV0 = maths::dot(D, V0);
        float pV1 = maths::dot(D, V1);
        float pV2 = maths::dot(D, V2);
        
        float pU0 = maths::dot(D, U0);
        float pU1 = maths::dot(D, U1);
        float pU2 = maths::dot(D, U2);
        
        float isect1[2], isect2[2];
        
        if (dV0 * dV1 > 0.0f) {
            isect1[0] = pV2 + (pV0 - pV2) * dV2 / (dV2 - dV0);
            isect1[1] = pV2 + (pV1 - pV2) * dV2 / (dV2 - dV1);
        } else if (dV0 * dV2 > 0.0f) {
            isect1[0] = pV1 + (pV0 - pV1) * dV1 / (dV1 - dV0);
            isect1[1] = pV1 + (pV2 - pV1) * dV1 / (dV1 - dV2);
        } else {
            isect1[0] = pV0 + (pV1 - pV0) * dV0 / (dV0 - dV1);
            isect1[1] = pV0 + (pV2 - pV0) * dV0 / (dV0 - dV2);
        }
        
        if (dU0 * dU1 > 0.0f) {
            isect2[0] = pU2 + (pU0 - pU2) * dU2 / (dU2 - dU0);
            isect2[1] = pU2 + (pU1 - pU2) * dU2 / (dU2 - dU1);
        } else if (dU0 * dU2 > 0.0f) {
            isect2[0] = pU1 + (pU0 - pU1) * dU1 / (dU1 - dU0);
            isect2[1] = pU1 + (pU2 - pU1) * dU1 / (dU1 - dU2);
        } else {
            isect2[0] = pU0 + (pU1 - pU0) * dU0 / (dU0 - dU1);
            isect2[1] = pU0 + (pU2 - pU0) * dU0 / (dU0 - dU2);
        }
        
        if (isect1[0] > isect1[1]) { float tmp = isect1[0]; isect1[0] = isect1[1]; isect1[1] = tmp; }
        if (isect2[0] > isect2[1]) { float tmp = isect2[0]; isect2[0] = isect2[1]; isect2[1] = tmp; }
        
        if (isect1[1] < isect2[0] || isect2[1] < isect1[0]) return false;
        
        return true;
    }

    /**
     * @brief Triangle versus axis-aligned box overlap test.
     * @details The Akenine-Moller separating-axis test over 13 candidate axes: the box's three
     * face normals, the triangle's normal, and the nine cross products of their edge
     * directions. Finding one separating axis proves disjointness; exhausting all 13 proves
     * overlap. This is the predicate behind narrow-band voxelisation.
     * @param[in] voxel_min Box lower bound.
     * @param[in] voxel_max Box upper bound.
     * @return True if the triangle meets the box.
     * @note Touching counts as intersecting.
     */
    __host__ __device__ __forceinline__ bool is_voxel_intersect(const float3& voxel_min, const float3& voxel_max) const {
        float3 tri_min, tri_max;
        compute_aabb(tri_min, tri_max);
        
        if (!aabb::test_aabb_overlap(voxel_min, voxel_max, tri_min, tri_max)) return false;

        float3 extents = (voxel_max - voxel_min) * 0.5f;
        float3 voxel_center = voxel_min + extents;

        float3 p0 = v0 - voxel_center;
        float3 p1 = v1 - voxel_center;
        float3 p2 = v2 - voxel_center;

        float3 f0 = p1 - p0;
        float3 f1 = p2 - p1;
        float3 f2 = p0 - p2;

/**
 * @brief Evaluates one separating-axis candidate of the triangle-box overlap test.
 * @details Projects the triangle's corners and the box's half-extents onto the candidate
 * axis and returns early when the intervals are disjoint. Written as a macro so all 13
 * axes expand inline, keeping the test free of call overhead inside the kernel.
 */
#define TEST_AXIS(axis) \
        do { \
            float p0_proj = maths::dot(p0, axis); \
            float p1_proj = maths::dot(p1, axis); \
            float p2_proj = maths::dot(p2, axis); \
            float r = maths::dot(extents, maths::abs(axis)); \
            if (fminf(fminf(p0_proj, p1_proj), p2_proj) > r || fmaxf(fmaxf(p0_proj, p1_proj), p2_proj) < -r) return false; \
        } while(0)

        TEST_AXIS(make_float3(0.0f, -f0.z, f0.y));
        TEST_AXIS(make_float3(0.0f, -f1.z, f1.y));
        TEST_AXIS(make_float3(0.0f, -f2.z, f2.y));

        TEST_AXIS(make_float3(f0.z, 0.0f, -f0.x));
        TEST_AXIS(make_float3(f1.z, 0.0f, -f1.x));
        TEST_AXIS(make_float3(f2.z, 0.0f, -f2.x));

        TEST_AXIS(make_float3(-f0.y, f0.x, 0.0f));
        TEST_AXIS(make_float3(-f1.y, f1.x, 0.0f));
        TEST_AXIS(make_float3(-f2.y, f2.x, 0.0f));

#undef TEST_AXIS

        float3 normal = maths::cross(f0, f1);
        float radius = maths::dot(extents, maths::abs(normal));
        float d = maths::dot(normal, p0);
        if (fabsf(d) > radius) return false;

        return true;
    }
};

/**
 * @brief Free-function overloads operating on raw `float3` corners.
 *
 * @details Every function here forwards to the matching ::Triangle method after
 * constructing a temporary. Kernels usually hold three loose `float3` corners rather than
 * a Triangle object, and the compiler inlines the temporary away entirely, so these
 * wrappers cost nothing at runtime while keeping call sites free of boilerplate.
 */
namespace triangle
{
    /**
     * @brief Unit geometric normal of a triangle.
     * @details Forwards to Triangle::compute_normal(); orientation follows the winding.
     * @param[in] v0 First corner.
     * @param[in] v1 Second corner.
     * @param[in] v2 Third corner.
     * @return The unit normal.
     * @warning Degenerate triangles yield NaNs.
     */
    __device__ __inline__ float3 compute_normal(const float3 &v0, const float3 &v1, const float3 &v2)
    {
        return Triangle(v0, v1, v2).compute_normal();
    }

    /**
     * @brief Area of a triangle.
     * @details Forwards to Triangle::compute_area().
     * @param[in] v0 First corner.
     * @param[in] v1 Second corner.
     * @param[in] v2 Third corner.
     * @return The area.
     */
    __device__ __inline__ float compute_area(const float3 &v0, const float3 &v1, const float3 &v2)
    {
        return Triangle(v0, v1, v2).compute_area();
    }

    /**
     * @brief Whether a triangle has an obtuse interior angle.
     * @details Forwards to Triangle::is_obtuse(). Obtuse triangles need the barycentric
     * fallback in mixed Voronoi area computation, since their circumcentre lies outside.
     * @param[in] v0 First corner.
     * @param[in] v1 Second corner.
     * @param[in] v2 Third corner.
     * @return True if any interior angle exceeds 90 degrees.
     */
    __host__ __device__ __inline__ bool is_obtuse(const float3 &v0, const float3 &v1, const float3 &v2)
    {
        return Triangle(v0, v1, v2).is_obtuse();
    }

    /**
     * @brief Centroid of a triangle.
     * @details Forwards to Triangle::compute_centroid().
     * @param[in] v0 First corner.
     * @param[in] v1 Second corner.
     * @param[in] v2 Third corner.
     * @return The arithmetic mean of the three corners.
     */
    __device__ __inline__ float3 compute_centroid(const float3 &v0, const float3 &v1, const float3 &v2)
    {
        return Triangle(v0, v1, v2).compute_centroid();
    }

    /**
     * @brief Circumcentre of a triangle.
     * @details Forwards to Triangle::compute_circumcenter(). With @p strict_inside set, a
     * circumcentre falling outside the triangle is replaced by the midpoint of the longest
     * edge, which keeps Voronoi area computations well defined on obtuse triangles.
     * @param[in] v0 First corner.
     * @param[in] v1 Second corner.
     * @param[in] v2 Third corner.
     * @param[in] strict_inside Whether to clamp the result into the triangle.
     * @return The circumcentre.
     */
    __host__ __device__ __inline__ float3 compute_circumcenter(const float3 &v0, const float3 &v1, const float3 &v2, bool strict_inside = false)
    {
        return Triangle(v0, v1, v2).compute_circumcenter(strict_inside);
    }

    /**
     * @brief Axis-aligned bounding box of a triangle.
     * @details Forwards to Triangle::compute_aabb().
     * @param[in] v0 First corner.
     * @param[in] v1 Second corner.
     * @param[in] v2 Third corner.
     * @param[out] aabb_min Lower bound.
     * @param[out] aabb_max Upper bound.
     */
    __device__ __inline__ void compute_aabb(const float3 &v0, const float3 &v1, const float3 &v2, float3 &aabb_min, float3 &aabb_max)
    {
        Triangle(v0, v1, v2).compute_aabb(aabb_min, aabb_max);
    }

    /**
     * @brief Exact triangle-triangle intersection test.
     * @details Forwards to Triangle::test_intersection().
     * @param[in] T1 First triangle.
     * @param[in] T2 Second triangle.
     * @return True if the triangles intersect.
     * @note Triangles sharing a vertex or edge report an intersection; filter adjacency
     * beforehand when testing a mesh against itself.
     */
    __host__ __device__ __inline__ bool test_intersection(const Triangle& T1, const Triangle& T2)
    {
        return T1.test_intersection(T2);
    }

    /**
     * @brief Triangle versus axis-aligned box overlap test.
     * @details Forwards to Triangle::is_voxel_intersect(), which applies the Akenine-Moller
     * separating-axis test. This is the predicate behind narrow-band voxelisation.
     * @param[in] v0 First corner.
     * @param[in] v1 Second corner.
     * @param[in] v2 Third corner.
     * @param[in] voxel_min Box lower bound.
     * @param[in] voxel_max Box upper bound.
     * @return True if the triangle meets the box.
     */
    __host__ __device__ __inline__ bool is_voxel_intersect(const float3 &v0, const float3 &v1, const float3 &v2, const float3 &voxel_min, const float3 &voxel_max)
    {
        return Triangle(v0, v1, v2).is_voxel_intersect(voxel_min, voxel_max);
    }

    /**
     * @brief Whether a point lies on a triangle's supporting plane.
     * @details Forwards to Triangle::test_point_on_tria_plane(). A coplanarity test only; it
     * says nothing about lying within the triangle's bounds.
     * @param[in] v0 First corner.
     * @param[in] v1 Second corner.
     * @param[in] v2 Third corner.
     * @param[in] p Query point.
     * @param[in] eps Distance tolerance. Defaults to 1e-5.
     * @return True if within @p eps of the plane.
     */
    __host__ __device__ __inline__ bool test_point_on_tria_plane(const float3 &v0, const float3 &v1, const float3 &v2, const float3 &p, float eps = 1e-5f)
    {
        return Triangle(v0, v1, v2).test_point_on_tria_plane(p, eps);
    }

    /**
     * @brief Whether a coplanar point lies within a triangle.
     * @details Forwards to Triangle::test_point_inside_on_tria_plane(), a barycentric
     * containment test that assumes coplanarity.
     * @param[in] v0 First corner.
     * @param[in] v1 Second corner.
     * @param[in] v2 Third corner.
     * @param[in] p Query point, assumed coplanar.
     * @return True if inside, edges and vertices included.
     * @warning A point off the plane projects onto it and may still report True; pair with
     * test_point_on_tria_plane() for full containment.
     */
    __host__ __device__ __inline__ bool test_point_inside_on_tria_plane(const float3 &v0, const float3 &v1, const float3 &v2, const float3 &p)
    {
        return Triangle(v0, v1, v2).test_point_inside_on_tria_plane(p);
    }

    /**
     * @brief Whether a point lies on the triangle's surface.
     * @details Forwards to Triangle::test_point_inside(), combining the coplanarity and
     * barycentric tests.
     * @param[in] v0 First corner.
     * @param[in] v1 Second corner.
     * @param[in] v2 Third corner.
     * @param[in] p Query point.
     * @param[in] eps Coplanarity tolerance. Defaults to 1e-5.
     * @return True if the point lies on the triangle.
     */
    __host__ __device__ __inline__ bool test_point_inside(const float3 &v0, const float3 &v1, const float3 &v2, const float3 &p, float eps = 1e-5f)
    {
        return Triangle(v0, v1, v2).test_point_inside(p, eps);
    }

    /**
     * @brief Samples a point uniformly inside a triangle.
     * @details Forwards to Triangle::sample_point(), which warps two uniform variates through
     * the square-root transform so samples are distributed uniformly by area rather than
     * clustering towards one corner.
     * @param[in] v0 First corner.
     * @param[in] v1 Second corner.
     * @param[in] v2 Third corner.
     * @param[in] r1 First uniform variate in $[0, 1]$.
     * @param[in] r2 Second uniform variate in $[0, 1]$.
     * @return The sampled point.
     */
    __device__ __inline__ float3 sample_point(const float3 &v0, const float3 &v1, const float3 &v2, float r1, float r2)
    {
        return Triangle(v0, v1, v2).sample_point(r1, r2);
    }
} // namespace triangle

#endif // TRIANGLE_H