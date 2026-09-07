#include <torch/extension.h>
#include "../../primitive/ray.h"
#include "../../check.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Converts a 3-element tensor into a `float3`.
 * @details The ::Ray binding's copy of the shared conversion helper, kept local to avoid a
 * cross-translation-unit dependency between binding files.
 * @param[in] t A 1D tensor of exactly three elements.
 * @return The equivalent `float3`.
 * @warning Synchronises on a device-to-host copy.
 */
inline float3 tensor_to_float3_ray(const torch::Tensor& t) {
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
inline torch::Tensor float3_to_tensor_ray(const float3& f) {
    return torch::tensor({f.x, f.y, f.z}, torch::dtype(torch::kFloat32));
}

/**
 * @brief Registers the ::Ray geometric primitive on the extension module.
 * @details Called once from `pybind.cpp` with the root module, so every symbol
 * defined here lands directly on `conquer3d._C`.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_primitive_ray(py::module_& m) {
    py::class_<Ray>(m, "Ray", R"pbdoc(
        Parametric 3D Ray geometric primitive with hardware fast AABB slab intersection methods.

        Example:
            >>> import torch
            >>> from conquer3d._C import Ray
            >>> ray = Ray(torch.tensor([0., 0., 0.]), torch.tensor([0., 0., 1.]))
            >>> pt = ray.at(2.5)
        )pbdoc")
        .def(py::init([](const torch::Tensor& origin, const torch::Tensor& dir) {
                 return Ray(tensor_to_float3_ray(origin), tensor_to_float3_ray(dir));
             }),
             py::arg("origin"), py::arg("direction"),
             R"pbdoc(
             Constructs a 3D Ray from origin and direction vectors.

             Args:
                 origin (torch.Tensor): (3,) float32 origin coordinates.
                 direction (torch.Tensor): (3,) float32 direction vector.

             Example:
                 >>> ray = Ray(origin, direction)
             )pbdoc")
        .def_property_readonly("origin", [](const Ray& self) { return float3_to_tensor_ray(self.origin); }, "Ray 3D origin position (3,) float32.")
        .def_property_readonly("direction", [](const Ray& self) { return float3_to_tensor_ray(self.direction); }, "Ray 3D direction vector (3,) float32.")
        .def_property_readonly("inv_direction", [](const Ray& self) { return float3_to_tensor_ray(self.inv_direction); }, "Precomputed reciprocal direction (3,) float32.")
        .def_property_readonly("t_min", [](const Ray& self) { return self.t_min; }, "Near clipping limit.")
        .def_property_readonly("t_max", [](const Ray& self) { return self.t_max; }, "Far clipping limit.")
        .def("at", [](const Ray& self, float t) {
                 return float3_to_tensor_ray(self.at(t));
             },
             py::arg("t"),
             R"pbdoc(
             Evaluates 3D position along ray: $p(t) = \text{origin} + t \cdot \text{direction}$.

             Args:
                 t (float): Parametric distance along ray.

             Returns:
                 torch.Tensor: (3,) float32 evaluated 3D point.

             Example:
                 >>> p = ray.at(1.5)
             )pbdoc")
        .def("is_intersect_aabb", [](const Ray& self, const torch::Tensor& aabb_min, const torch::Tensor& aabb_max) {
                 float t_hit;
                 bool hit = self.is_intersect_aabb(tensor_to_float3_ray(aabb_min), tensor_to_float3_ray(aabb_max), t_hit);
                 return py::make_tuple(hit, t_hit);
             },
             py::arg("aabb_min"), py::arg("aabb_max"),
             R"pbdoc(
             Tests Ray-AABB intersection via Kay-Kajiya slab method.

             Args:
                 aabb_min (torch.Tensor): (3,) float32 lower box bounds.
                 aabb_max (torch.Tensor): (3,) float32 upper box bounds.

             Returns:
                 Tuple[bool, float]: (hit, t_hit) where hit is True if ray intersects box and t_hit is entry distance.

             Example:
                 >>> hit, t = ray.is_intersect_aabb(box_min, box_max)
             )pbdoc");
}
