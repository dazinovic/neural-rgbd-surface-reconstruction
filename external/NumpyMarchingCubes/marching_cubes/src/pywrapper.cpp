
#include "pywrapper.h"

#include "marching_cubes.h"

#include <stdexcept>
#include <array>


PyObject* marching_cubes(PyArrayObject* arr, double isovalue, double truncation)
{
    if(PyArray_NDIM(arr) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported.");
    
    // Prepare data.
    npy_intp* shape = PyArray_DIMS(arr);
    std::array<long, 3> lower{0, 0, 0};
    std::array<long, 3> upper{shape[0]-1, shape[1]-1, shape[2]-1};
    long numx = upper[0] - lower[0] + 1;
    long numy = upper[1] - lower[1] + 1;
    long numz = upper[2] - lower[2] + 1;
    std::vector<double> vertices;
    std::vector<size_t> polygons;
    
//    auto pyarray_to_cfunc = [&](long x, long y, long z) -> double {
//        const npy_intp c[3] = {x, y, z};
//        return PyArray_SafeGet<double>(arr, c);
//    };

    npy_accessor tsdf_accessor(arr, {numx, numy, numz});

    // Marching cubes.
    marching_cubes(tsdf_accessor, isovalue, truncation, vertices, polygons);
    
    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));
    
    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;
    
    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);
    
    return res;
}

