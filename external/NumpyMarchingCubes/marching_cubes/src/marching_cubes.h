#ifndef _MARCHING_CUBES_H
#define _MARCHING_CUBES_H

#include "pyarraymodule.h"
#include <array>
#include <vector>

struct npy_accessor {
    npy_accessor(PyArrayObject* arr, const std::array<long, 3> size) : m_arr(arr), m_size(size) {}
    const std::array<long, 3>& size() const {
        return m_size;
    }
    double operator()(long x, long y, long z) const {
        const npy_intp c[3] = {x, y, z};
        return PyArray_SafeGet<double>(m_arr, c);
    }

    PyArrayObject* m_arr;
    const std::array<long, 3> m_size;
};

void marching_cubes(const npy_accessor& tsdf_accessor, double isovalue, double truncation,
    std::vector<double>& vertices, std::vector<unsigned long>& polygons);

#endif // _MARCHING_CUBES_H