#ifndef BASE_H
#define BASE_H

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>

#ifndef __CUDACC__
static inline __host__ __device__ double rsqrt(double a) {
    return 1. / sqrt(a);
}

static inline __host__ __device__ float rsqrtf(float a) {
    return 1. / sqrtf(a);
}
#endif

typedef struct
{
    float m[3][3];
} float3x3;

typedef struct
{
    float m[3][4];
} float3x4;

typedef struct
{
    float m[4][4];
} float4x4;

static __inline__ __host__ __device__ float3 mul3x3(float3 a, float3x3 m) {
    return make_float3(
        a.x * m.m[0][0] + a.y * m.m[1][0] + a.z * m.m[2][0],
        a.x * m.m[0][1] + a.y * m.m[1][1] + a.z * m.m[2][1],
        a.x * m.m[0][2] + a.y * m.m[1][2] + a.z * m.m[2][2]
    );
}

static __inline__ __host__ __device__ float3 mul3x4(float3 a, float4x4 m) {
    return make_float3(
        a.x * m.m[0][0] + a.y * m.m[1][0] + a.z * m.m[2][0] + m.m[3][0],
        a.x * m.m[0][1] + a.y * m.m[1][1] + a.z * m.m[2][1] + m.m[3][1],
        a.x * m.m[0][2] + a.y * m.m[1][2] + a.z * m.m[2][2] + m.m[3][2]
    );
}

static __inline__ __host__ __device__ float3 mul3x4(float3x4 m, float4 a) {
    return make_float3(
        a.x * m.m[0][0] + a.y * m.m[0][1] + a.z * m.m[0][2] + a.w * m.m[0][3],
        a.x * m.m[1][0] + a.y * m.m[1][1] + a.z * m.m[1][2] + a.w * m.m[1][3],
        a.x * m.m[2][0] + a.y * m.m[2][1] + a.z * m.m[2][2] + a.w * m.m[2][3]
    );
}

static __inline__ __host__ __device__ float4 mul4x4(float4 a, float4x4 m) {
    return make_float4(
        a.x * m.m[0][0] + a.y * m.m[1][0] + a.z * m.m[2][0] + a.w * m.m[3][0],
        a.x * m.m[0][1] + a.y * m.m[1][1] + a.z * m.m[2][1] + a.w * m.m[3][1],
        a.x * m.m[0][2] + a.y * m.m[1][2] + a.z * m.m[2][2] + a.w * m.m[3][2],
        a.x * m.m[0][3] + a.y * m.m[1][3] + a.z * m.m[2][3] + a.w * m.m[3][3]
    );
}

static __inline__ __host__ __device__ float4 mul4x4(float4x4 m, float4 a) {
    return make_float4(
        a.x * m.m[0][0] + a.y * m.m[0][1] + a.z * m.m[0][2] + a.w * m.m[0][3],
        a.x * m.m[1][0] + a.y * m.m[1][1] + a.z * m.m[1][2] + a.w * m.m[1][3],
        a.x * m.m[2][0] + a.y * m.m[2][1] + a.z * m.m[2][2] + a.w * m.m[2][3],
        a.x * m.m[3][0] + a.y * m.m[3][1] + a.z * m.m[3][2] + a.w * m.m[3][3]
    );
}

static __inline__ __host__ __device__  float3 crs3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - b.y * a.z,
        a.z * b.x - b.z * a.x,
        a.x * b.y - b.x * a.y
    );
}

static __inline__ __host__ __device__ float3x3 make_float3x3(
    float a00, float a01, float a02,
    float a10, float a11, float a12,
    float a20, float a21, float a22) {
    float3x3 a;
    a.m[0][0] = a00; a.m[0][1] = a01; a.m[0][2] = a02;
    a.m[1][0] = a10; a.m[1][1] = a11; a.m[1][2] = a12;
    a.m[2][0] = a20; a.m[2][1] = a21; a.m[2][2] = a22;
    return a;
}

static __inline__ __host__ __device__ float4x4 make_float4x4(
    float a00, float a01, float a02, float a03,
    float a10, float a11, float a12, float a13,
    float a20, float a21, float a22, float a23,
    float a30, float a31, float a32, float a33) {
    float4x4 a;
    a.m[0][0] = a00; a.m[0][1] = a01; a.m[0][2] = a02; a.m[0][3] = a03;
    a.m[1][0] = a10; a.m[1][1] = a11; a.m[1][2] = a12; a.m[1][3] = a13;
    a.m[2][0] = a20; a.m[2][1] = a21; a.m[2][2] = a22; a.m[2][3] = a23;
    a.m[3][0] = a30; a.m[3][1] = a31; a.m[3][2] = a32; a.m[3][3] = a33;
    return a;
}

static __inline__ __host__ __device__ float3x3 operator* (const float3x3& a, const float3x3& b)
{
    float3x3 c;

    c.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0] + a.m[0][2] * b.m[2][0];
    c.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1] + a.m[0][2] * b.m[2][1];
    c.m[0][2] = a.m[0][0] * b.m[0][2] + a.m[0][1] * b.m[1][2] + a.m[0][2] * b.m[2][2];

    c.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0] + a.m[1][2] * b.m[2][0];
    c.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1] + a.m[1][2] * b.m[2][1];
    c.m[1][2] = a.m[1][0] * b.m[0][2] + a.m[1][1] * b.m[1][2] + a.m[1][2] * b.m[2][2];

    c.m[2][0] = a.m[2][0] * b.m[0][0] + a.m[2][1] * b.m[1][0] + a.m[2][2] * b.m[2][0];
    c.m[2][1] = a.m[2][0] * b.m[0][1] + a.m[2][1] * b.m[1][1] + a.m[2][2] * b.m[2][1];
    c.m[2][2] = a.m[2][0] * b.m[0][2] + a.m[2][1] * b.m[1][2] + a.m[2][2] * b.m[2][2];

    return c;
}


static __inline__ __host__ __device__ float3x4 operator* (const float3x4& a, const float4x4& b)
{
    float3x4 c;

    c.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0] + a.m[0][2] * b.m[2][0] + a.m[0][3] * b.m[3][0];
    c.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1] + a.m[0][2] * b.m[2][1] + a.m[0][3] * b.m[3][1];
    c.m[0][2] = a.m[0][0] * b.m[0][2] + a.m[0][1] * b.m[1][2] + a.m[0][2] * b.m[2][2] + a.m[0][3] * b.m[3][2];
    c.m[0][3] = a.m[0][0] * b.m[0][3] + a.m[0][1] * b.m[1][3] + a.m[0][2] * b.m[2][3] + a.m[0][3] * b.m[3][3];

    c.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0] + a.m[1][2] * b.m[2][0] + a.m[1][3] * b.m[3][0];
    c.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1] + a.m[1][2] * b.m[2][1] + a.m[1][3] * b.m[3][1];
    c.m[1][2] = a.m[1][0] * b.m[0][2] + a.m[1][1] * b.m[1][2] + a.m[1][2] * b.m[2][2] + a.m[1][3] * b.m[3][2];
    c.m[1][3] = a.m[1][0] * b.m[0][3] + a.m[1][1] * b.m[1][3] + a.m[1][2] * b.m[2][3] + a.m[1][3] * b.m[3][3];

    c.m[2][0] = a.m[2][0] * b.m[0][0] + a.m[2][1] * b.m[1][0] + a.m[2][2] * b.m[2][0] + a.m[2][3] * b.m[3][0];
    c.m[2][1] = a.m[2][0] * b.m[0][1] + a.m[2][1] * b.m[1][1] + a.m[2][2] * b.m[2][1] + a.m[2][3] * b.m[3][1];
    c.m[2][2] = a.m[2][0] * b.m[0][2] + a.m[2][1] * b.m[1][2] + a.m[2][2] * b.m[2][2] + a.m[2][3] * b.m[3][2];
    c.m[2][3] = a.m[2][0] * b.m[0][3] + a.m[2][1] * b.m[1][3] + a.m[2][2] * b.m[2][3] + a.m[2][3] * b.m[3][3];

    return c;
}

static __inline__ __host__ __device__ float4x4  operator*(const float4x4& a, const float4x4& b)
{
    float4x4 c;

    c.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0] + a.m[0][2] * b.m[2][0] + a.m[0][3] * b.m[3][0];
    c.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1] + a.m[0][2] * b.m[2][1] + a.m[0][3] * b.m[3][1];
    c.m[0][2] = a.m[0][0] * b.m[0][2] + a.m[0][1] * b.m[1][2] + a.m[0][2] * b.m[2][2] + a.m[0][3] * b.m[3][2];
    c.m[0][3] = a.m[0][0] * b.m[0][3] + a.m[0][1] * b.m[1][3] + a.m[0][2] * b.m[2][3] + a.m[0][3] * b.m[3][3];

    c.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0] + a.m[1][2] * b.m[2][0] + a.m[1][3] * b.m[3][0];
    c.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1] + a.m[1][2] * b.m[2][1] + a.m[1][3] * b.m[3][1];
    c.m[1][2] = a.m[1][0] * b.m[0][2] + a.m[1][1] * b.m[1][2] + a.m[1][2] * b.m[2][2] + a.m[1][3] * b.m[3][2];
    c.m[1][3] = a.m[1][0] * b.m[0][3] + a.m[1][1] * b.m[1][3] + a.m[1][2] * b.m[2][3] + a.m[1][3] * b.m[3][3];

    c.m[2][0] = a.m[2][0] * b.m[0][0] + a.m[2][1] * b.m[1][0] + a.m[2][2] * b.m[2][0] + a.m[2][3] * b.m[3][0];
    c.m[2][1] = a.m[2][0] * b.m[0][1] + a.m[2][1] * b.m[1][1] + a.m[2][2] * b.m[2][1] + a.m[2][3] * b.m[3][1];
    c.m[2][2] = a.m[2][0] * b.m[0][2] + a.m[2][1] * b.m[1][2] + a.m[2][2] * b.m[2][2] + a.m[2][3] * b.m[3][2];
    c.m[2][3] = a.m[2][0] * b.m[0][3] + a.m[2][1] * b.m[1][3] + a.m[2][2] * b.m[2][3] + a.m[2][3] * b.m[3][3];

    c.m[3][0] = a.m[3][0] * b.m[0][0] + a.m[3][1] * b.m[1][0] + a.m[3][2] * b.m[2][0] + a.m[3][3] * b.m[3][0];
    c.m[3][1] = a.m[3][0] * b.m[0][1] + a.m[3][1] * b.m[1][1] + a.m[3][2] * b.m[2][1] + a.m[3][3] * b.m[3][1];
    c.m[3][2] = a.m[3][0] * b.m[0][2] + a.m[3][1] * b.m[1][2] + a.m[3][2] * b.m[2][2] + a.m[3][3] * b.m[3][2];
    c.m[3][3] = a.m[3][0] * b.m[0][3] + a.m[3][1] * b.m[1][3] + a.m[3][2] * b.m[2][3] + a.m[3][3] * b.m[3][3];

    return c;
}

static __inline__ __host__ __device__ float4x4 transpose(const float4x4& a) {
    float4x4 b;
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
        b.m[i][j] = a.m[j][i];
    return b;
}

static __inline__ __host__ __device__ float3x3 transpose(const float3x3& a) {
    float3x3 b;
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
        b.m[i][j] = a.m[j][i];
    return b;
}

static inline __host__ __device__ void copy(float3x3 &a, float3x3 b) {
    a.m[0][0] = b.m[0][0]; a.m[0][1] = b.m[0][1]; a.m[0][2] = b.m[0][2];
    a.m[1][0] = b.m[1][0]; a.m[1][1] = b.m[1][1]; a.m[1][2] = b.m[1][2];
    a.m[2][0] = b.m[2][0]; a.m[2][1] = b.m[2][1]; a.m[2][2] = b.m[2][2];
}

static inline __host__ __device__ double dot(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline __host__ __device__ double dot2(double3 a) {
    return dot(a, a);
}

static inline __host__ __device__ double3 cross(double3 a, double3 b) {
    return make_double3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

static inline __host__ __device__ double3 normalize(double3 v) {
    double invLen = rsqrt(dot2(v));
    return make_double3(invLen * v.x, invLen * v.y, invLen * v.z);
}

static inline __host__ __device__ double3 operator+(double3 a, double3 b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static inline __host__ __device__ double3 operator-(double3 a, double3 b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static inline __host__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static inline __host__ __device__ void operator+=(float3 &a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

static inline __host__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static inline __host__ __device__ void operator-=(float3 &a, float3 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

static inline __host__ __device__ float3 operator*(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

static inline __host__ __device__ float3 operator*(float b, float3 a) {
  return make_float3(b * a.x, b * a.y, b * a.z);
}

static inline __host__ __device__ void operator*=(float3 &a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

static inline __host__ __device__ float3 operator/(float3 a, const float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
}

static inline __host__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline __host__ __device__ float dot2(float3 a) {
    return dot(a, a);
}

static inline __host__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

static inline __host__ __device__ float3 normalize(float3 v) {
    float invLen = rsqrtf(dot2(v));
    return v * invLen;
}


static inline __host__ __device__ bool equals(float3 a, float3 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

static __inline__ __host__ __device__  double4 crs4(float3 aa, float3 bb, float3 cc) {
    double3 a = make_double3(aa.x, aa.y, aa.z);
    double3 b = make_double3(bb.x, bb.y, bb.z);
    double3 c = make_double3(cc.x, cc.y, cc.z);
    double4 P;
    P.x = a.z*b.y - a.y*b.z - a.z*c.y + b.z*c.y + a.y*c.z - b.y*c.z;
    P.y = a.x*b.z + a.z*c.x - b.z*c.x - a.x*c.z + b.x*c.z - a.z*b.x;
    P.z = a.y*b.x - a.x*b.y - a.y*c.x + b.y*c.x + a.x*c.y - b.x*c.y;
    P.w = a.y*b.z*c.x + a.z*b.x*c.y - a.x*b.z*c.y - a.y*b.x*c.z + a.x*b.y*c.z -a.z*b.y*c.x;
    return P;
}

static __host__ __device__ __forceinline__ float project_edge(
    const float3& vertex, const float3& edge, const float3& point) {
    const float3 point_vec = point - vertex;
    const float length = dot2(edge);
    return dot(point_vec, edge) / length;
}

static __host__ __device__ __forceinline__ float3 project_plane(
    const float3& vertex, const float3& normal, const float3& point) {
    const float3 unit_normal = normalize(normal);
    const float dist = (point.x - vertex.x) * unit_normal.x + \
                        (point.y - vertex.y) * unit_normal.y + \
                        (point.z - vertex.z) * unit_normal.z;
    return point - (unit_normal * dist);
}

static __host__ __device__ __forceinline__ bool is_not_above(
    const float3& vertex, const float3& edge, const float3& normal, const float3& point) {
    const float3 edge_normal = cross(normal, edge);
    return dot(edge_normal, point - vertex) <= 0;
}

static __host__ __device__ __forceinline__ float3 point_at(
    const float3& vertex, const float3& edge, const float& t) {
    return vertex + (edge * t);
}

static inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static __inline__ __host__ __device__ float3 operator* (const float3x4 m, const float4 a)
{
    return mul3x4(m, a);
}

static __inline__ __host__ __device__ float4 operator* (const float4 a, const float4x4 M)
{
    return mul4x4(a, M);
}

static __inline__ __host__ __device__ float4 operator* (const float4x4 m, const float4 a)
{
    // return mul4x4(m, a);
    return make_float4(
        a.x * m.m[0][0] + a.y * m.m[0][1] + a.z * m.m[0][2] + a.w * m.m[0][3],
        a.x * m.m[1][0] + a.y * m.m[1][1] + a.z * m.m[1][2] + a.w * m.m[1][3],
        a.x * m.m[2][0] + a.y * m.m[2][1] + a.z * m.m[2][2] + a.w * m.m[2][3],
        a.x * m.m[3][0] + a.y * m.m[3][1] + a.z * m.m[3][2] + a.w * m.m[3][3]
    );
}

#endif // BASE_H