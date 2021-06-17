#pragma once

#include <numpy/arrayobject.h>
#include <Eigen/Core>

#include "fnnls.h"

namespace eigen_wrapper {

template<typename T>
inline int determine_dtype()
{
    int typenum_dtype;
    if (std::is_same<T, float>::value) {
        typenum_dtype = NPY_FLOAT;
    }
    else if (std::is_same<T, double>::value) {
        typenum_dtype = NPY_DOUBLE;
    }
    else {
        throw std::invalid_argument("Only 32 bit float and 64 bit double are supported!");
    }

    return typenum_dtype;
}

template<typename T>
inline PyArrayObject *_1darray_copy_to_numpy(const T *data_ptr, const ssize_t length)
{
    import_array();
    PyObject *p_object = PyArray_SimpleNew(1, &length, determine_dtype<T>());
    if (!p_object) {
       //Something went wrong, return and let python handle the exception.
       return nullptr;
    }
    PyArrayObject* p_array = reinterpret_cast<PyArrayObject*>(p_object);
    std::memcpy(PyArray_DATA(p_array), data_ptr, length * sizeof(T));
    return p_array;
}

template<typename T>
inline PyArrayObject *onedarray(Eigen::PlainObjectBase<fnnls::VectorX_<T>> &&m)
{
    return _1darray_copy_to_numpy(m.data(), m.size());
}

template<typename T>
inline PyArrayObject *onedarray(const Eigen::PlainObjectBase<fnnls::VectorX_<T>> &m)
{
    return _1darray_copy_to_numpy(m.data(), m.size());
}

template <typename MatrixType>
class Map : public Eigen::Map<MatrixType> {
public:
    typedef Eigen::Map<MatrixType> Base;

    Map(PyArrayObject *object) : Base(reinterpret_cast<typename MatrixType::Scalar*>(PyArray_DATA(object)),
                                      (PyArray_NDIM(object) == 1) ? PyArray_DIM(object, 0) : PyArray_DIM(object, 1),
                                      (PyArray_NDIM(object) == 1) ? 1 : PyArray_DIM(object, 0)),
                                      object_(object) {

        if (!PyArray_ISONESEGMENT(object)) {
            throw std::invalid_argument("Numpy arrays must be in one contiguous segment to be able to be transferred to an Eigen Map.");
        }
        if (!PyArray_IS_C_CONTIGUOUS(object) && PyArray_NDIM(object) > 1) {
            throw std::invalid_argument("Only C-contiguous numpy arrays are supported if dimension is > 1");
        }
        Py_INCREF(object_);
    }

    virtual ~Map() {
        Py_DECREF(object_);
    }

private:
    PyArrayObject* const object_;
};

}
