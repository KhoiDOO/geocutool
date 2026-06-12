#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_primitive_gs(py::module_& m);
void bind_primitive_pgs(py::module_& m);

void bind_ds_kdtree(py::module_& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Geocutool Python bindings";

    bind_primitive_gs(m);
    bind_primitive_pgs(m);

    bind_ds_kdtree(m);
}