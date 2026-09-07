#include <torch/extension.h>
#include "../../primitive/triangle.h"
#include "../../check.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Converts a 3-element tensor into a `float3`.
 * @details Moves the tensor to host memory as float32 before reading it, so a caller may
 * pass a CUDA tensor without a separate transfer. Intended for scalar-sized arguments
 * only.
 * @param[in] t A 1D tensor of exactly three elements.
 * @return The equivalent `float3`.
 * @warning Synchronises on a device-to-host copy; never call this per element in a loop.
 */
inline float3 tensor_to_float3(const torch::Tensor& t) {
    TORCH_CHECK(t.dim() == 1 && t.size(0) == 3, "Tensor must be 1D with 3 elements");
    auto t_contig = t.contiguous().cpu().to(torch::kFloat32);
    float* ptr = t_contig.data_ptr<float>();
    return make_float3(ptr[0], ptr[1], ptr[2]);
}

/**
 * @brief Converts a `float3` into a 3-element CPU tensor.
 * @param[in] f The vector to convert.
 * @return A `(3,)` float32 tensor on the host.
 */
inline torch::Tensor float3_to_tensor(const float3& f) {
    return torch::tensor({f.x, f.y, f.z}, torch::dtype(torch::kFloat32));
}

/**
 * @brief Registers the ::Triangle geometric primitive on the extension module.
 * @details Called once from `pybind.cpp` with the root module, so every symbol
 * defined here lands directly on `conquer3d._C`.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_primitive_triangle(py::module_& m) {
    py::class_<Triangle>(m, "Triangle", R"pbdoc(
        3D Triangle geometric primitive structure with hardware-optimized device methods.

        Example:
            >>> import torch
            >>> from conquer3d._C import Triangle
            >>> tri = Triangle(torch.tensor([0.,0.,0.]), torch.tensor([1.,0.,0.]), torch.tensor([0.,1.,0.]))
            >>> area = tri.compute_area()
        )pbdoc")
        .def(py::init([](const torch::Tensor& v0, const torch::Tensor& v1, const torch::Tensor& v2) {
                 return Triangle(tensor_to_float3(v0), tensor_to_float3(v1), tensor_to_float3(v2));
             }),
             py::arg("v0"), py::arg("v1"), py::arg("v2"),
             R"pbdoc(
             Constructs a 3D Triangle from 3 corner vertex coordinates.

             Args:
                 v0 (torch.Tensor): (3,) float32 corner vertex.
                 v1 (torch.Tensor): (3,) float32 corner vertex.
                 v2 (torch.Tensor): (3,) float32 corner vertex.

             Example:
                 >>> tri = Triangle(v0, v1, v2)
             )pbdoc")
        .def("is_intersect_ray", [](const Triangle& self, const Ray& ray) {
                 float t_hit, u, v;
                 bool hit = self.is_intersect_ray(ray, t_hit, u, v);
                 return py::make_tuple(hit, t_hit, u, v);
             },
             py::arg("ray"),
             R"pbdoc(
             Tests ray-triangle intersection using Möller-Trumbore algorithm.

             Args:
                 ray (Ray): 3D ray primitive.

             Returns:
                 Tuple[bool, float, float, float]: (hit, t_hit, u, v) where u, v are barycentric coordinates.

             Example:
                 >>> hit, t, u, v = tri.is_intersect_ray(ray)
             )pbdoc")
        .def("compute_closest_point", [](const Triangle& self, const torch::Tensor& p) {
                 return float3_to_tensor(self.compute_closest_point(tensor_to_float3(p)));
             },
             py::arg("p"),
             R"pbdoc(
             Computes closest projected point on triangle surface (Ericson real-time collision).

             Args:
                 p (torch.Tensor): (3,) float32 3D query point.

             Returns:
                 torch.Tensor: (3,) float32 closest point coordinates on triangle.

             Example:
                 >>> proj = tri.compute_closest_point(p)
             )pbdoc")
        .def("compute_normal", [](const Triangle& self) {
                 return float3_to_tensor(self.compute_normal());
             },
             R"pbdoc(
             Computes normalized outward unit face normal.

             Returns:
                 torch.Tensor: (3,) float32 unit normal vector.

             Example:
                 >>> n = tri.compute_normal()
             )pbdoc")
        .def("compute_area", &Triangle::compute_area,
             R"pbdoc(
             Computes triangle surface area.

             Returns:
                 float: Surface area of the triangle.

             Example:
                 >>> a = tri.compute_area()
             )pbdoc")
        .def("sample_point", [](const Triangle& self, float r1, float r2) {
                 return float3_to_tensor(self.sample_point(r1, r2));
             },
             py::arg("r1"), py::arg("r2"),
             R"pbdoc(
             Samples uniform area-weighted point on triangle surface using square root mapping.

             Args:
                 r1 (float): Random float in [0, 1].
                 r2 (float): Random float in [0, 1].

             Returns:
                 torch.Tensor: (3,) float32 sampled 3D point.

             Example:
                 >>> p = tri.sample_point(0.2, 0.7)
             )pbdoc")
        .def("compute_aabb", [](const Triangle& self) {
                 float3 aabb_min, aabb_max;
                 self.compute_aabb(aabb_min, aabb_max);
                 return py::make_tuple(float3_to_tensor(aabb_min), float3_to_tensor(aabb_max));
             },
             R"pbdoc(
             Computes tight axis-aligned bounding box.

             Returns:
                 Tuple[torch.Tensor, torch.Tensor]: (aabb_min, aabb_max) as (3,) float32 tensors.

             Example:
                 >>> aabb_min, aabb_max = tri.compute_aabb()
             )pbdoc")
        .def("test_intersection", [](const Triangle& self, const Triangle& other) {
                 return self.test_intersection(other);
             },
             py::arg("other"),
             R"pbdoc(
             Moeller (1997) triangle-triangle collision test.

             Args:
                 other (Triangle): Other triangle primitive.

             Returns:
                 bool: True if triangles intersect, False otherwise.

             Example:
                 >>> collides = tri1.test_intersection(tri2)
             )pbdoc")
        .def("is_obtuse", [](const Triangle& self) {
                 return self.is_obtuse();
             },
             R"pbdoc(
             Checks if triangle has any internal angle greater than 90 degrees.

             Returns:
                 bool: True if obtuse, False otherwise.

             Example:
                 >>> is_ob = tri.is_obtuse()
             )pbdoc")
        .def("compute_centroid", [](const Triangle& self) {
                 return float3_to_tensor(self.compute_centroid());
             },
             R"pbdoc(
             Computes barycentric centroid: $(v_0 + v_1 + v_2) / 3$.

             Returns:
                 torch.Tensor: (3,) float32 centroid coordinates.

             Example:
                 >>> c = tri.compute_centroid()
             )pbdoc")
        .def("compute_circumcenter", [](const Triangle& self, bool strict_inside) {
                 return float3_to_tensor(self.compute_circumcenter(strict_inside));
             },
             py::arg("strict_inside") = false,
             R"pbdoc(
             Computes circumcenter of the triangle.

             Args:
                 strict_inside (bool, optional): If True and triangle is obtuse, clamps to edge midpoint. Defaults to False.

             Returns:
                 torch.Tensor: (3,) float32 circumcenter coordinates.

             Example:
                 >>> cc = tri.compute_circumcenter()
             )pbdoc")
        .def("test_point_on_tria_plane", [](const Triangle& self, const torch::Tensor& p, float eps) {
                 return self.test_point_on_tria_plane(tensor_to_float3(p), eps);
             },
             py::arg("p"), py::arg("eps") = 1e-5f,
             R"pbdoc(
             Tests whether a point lies on the triangle's supporting plane.

             Evaluates the absolute point-to-plane distance
             $|\mathbf{n} \cdot (\mathbf{p} - \mathbf{v}_0)|$ against a tolerance, where
             $\mathbf{n}$ is the unit face normal. This is a coplanarity test only; it
             says nothing about whether the point falls within the triangle's bounds.

             Args:
                 p (torch.Tensor): (3,) float32 query point.
                 eps (float, optional): Maximum absolute distance from the plane still
                     treated as coplanar. Defaults to 1e-5.

             Returns:
                 bool: True if the point is within `eps` of the supporting plane.

             Example:
                 >>> tri.test_point_on_tria_plane(torch.tensor([0.2, 0.3, 0.0]))
             )pbdoc")
        .def("test_point_inside_on_tria_plane", [](const Triangle& self, const torch::Tensor& p) {
                 return self.test_point_inside_on_tria_plane(tensor_to_float3(p));
             },
             py::arg("p"),
             R"pbdoc(
             Tests whether a coplanar point falls inside the triangle.

             Solves for the barycentric coordinates $(u, v, w)$ of the point against the
             edge vectors and reports whether all three are non-negative. The point is
             assumed to already lie on the supporting plane -- a point far off the plane
             projects onto it and may still report True, so pair this with
             `test_point_on_tria_plane` (or use `test_point_inside`) for a full
             containment test.

             Args:
                 p (torch.Tensor): (3,) float32 query point, assumed coplanar.

             Returns:
                 bool: True if the point lies within the triangle, edges and vertices
                 included. False for degenerate triangles, whose barycentric denominator
                 collapses below 1e-8.

             Example:
                 >>> tri.test_point_inside_on_tria_plane(tri.compute_centroid())
             )pbdoc")
        .def("test_point_inside", [](const Triangle& self, const torch::Tensor& p, float eps) {
                 return self.test_point_inside(tensor_to_float3(p), eps);
             },
             py::arg("p"), py::arg("eps") = 1e-5f,
             R"pbdoc(
             Tests whether a point lies on the triangle itself.

             The full containment predicate: the point must be coplanar within `eps`
             **and** fall inside the triangle's barycentric bounds. Equivalent to
             `test_point_on_tria_plane` followed by `test_point_inside_on_tria_plane`,
             short-circuiting on the cheaper plane test first.

             Args:
                 p (torch.Tensor): (3,) float32 query point.
                 eps (float, optional): Coplanarity tolerance passed to the plane test.
                     Defaults to 1e-5.

             Returns:
                 bool: True if the point lies on the triangle's surface.

             Example:
                 >>> tri.test_point_inside(tri.compute_centroid())
                 True
             )pbdoc")
        .def("is_voxel_intersect", [](const Triangle& self, const torch::Tensor& voxel_min, const torch::Tensor& voxel_max) {
                 return self.is_voxel_intersect(tensor_to_float3(voxel_min), tensor_to_float3(voxel_max));
             },
             py::arg("voxel_min"), py::arg("voxel_max"),
             R"pbdoc(
             Akenine-Moeller (2001) separating axis theorem (SAT) triangle-box overlap test.

             Args:
                 voxel_min (torch.Tensor): (3,) float32 lower box corner.
                 voxel_max (torch.Tensor): (3,) float32 upper box corner.

             Returns:
                 bool: True if triangle intersects or overlaps box, False otherwise.

             Example:
                 >>> is_overlap = tri.is_voxel_intersect(box_min, box_max)
             )pbdoc");
}
