#pragma once

#include "eigen_wrapper.h"
#include "fnnls.h"

namespace fnnls_python_wrapper {

template<typename T>
inline PyArrayObject *call_fnnls_solver(PyArrayObject *array_Z,
                                        PyArrayObject *array_x,
                                        const int max_iterations,
                                        const T tolerance)
{
    return eigen_wrapper::onedarray<T>(fnnls::fnnls_solver<T>(eigen_wrapper::Map<fnnls::MatrixX_<T>>(array_Z),
                                                              eigen_wrapper::Map<fnnls::VectorX_<T>>(array_x),
                                                              max_iterations,
                                                              tolerance));
}

template<typename T>
class StorePreCompute
{
    private:
        fnnls::MatrixX_<T> ZT_old;
        fnnls::MatrixX_<T> ZTZ;

    public:
        StorePreCompute() {};
        PyArrayObject *call_fnnls_solver_precompute(PyArrayObject *array_Z,
                                                    PyArrayObject *array_x,
                                                    int max_iterations,
                                                    T tolerance)
        {
            const auto ZT_new = eigen_wrapper::Map<fnnls::MatrixX_<T>>(array_Z);
            const int n_rows = ZT_new.rows();
            const int n_cols = ZT_new.cols();
            constexpr T tone = T{1.0};
            if ((ZT_old.rows() != n_rows) || (ZT_old.cols() != n_cols) || !(ZT_old - ZT_new).isMuchSmallerThan(tone)) {
                ZTZ.resize(n_rows, n_rows);
                ZTZ = fnnls::construct_ZTZ<T>(ZT_new.sparseView(), n_rows);
                ZT_old.resize(n_rows, n_cols);
                ZT_old = ZT_new;
            }

            return eigen_wrapper::onedarray<T>(fnnls::fnnls_solver<T>(ZT_new,
                                                                      eigen_wrapper::Map<fnnls::VectorX_<T>>(array_x),
                                                                      max_iterations,
                                                                      tolerance,
                                                                      ZTZ.data()));
        }
};

}
