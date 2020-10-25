//
// Created by ben on 2020/10/20.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include <mkl.h>
#include <omp.h>
#include "StopWatch.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <functional>

struct Matrix {
    ~Matrix() { reset_buffer(0, 0); }

    Matrix(size_t nrow, size_t ncol) : m_nrow(nrow), m_ncol(ncol) {
        reset_buffer(nrow, ncol);
    }

    Matrix(size_t nrow, size_t ncol, std::vector<double> const &vec)
            : m_nrow(nrow), m_ncol(ncol) {
        reset_buffer(nrow, ncol);
        (*this) = vec;
    }

    Matrix(Matrix const &other) : m_nrow(other.m_nrow), m_ncol(other.m_ncol) {
        reset_buffer(other.m_nrow, other.m_ncol);
        for (size_t i = 0; i < m_nrow; ++i) {
            for (size_t j = 0; j < m_ncol; ++j) {
                (*this)(i, j) = other(i, j);
            }
        }
    }

    Matrix &operator=(std::vector<double> const &vec) {
        if (size() != vec.size()) {
            throw std::out_of_range("number of elements mismatch");
        }

        size_t k = 0;
        for (size_t i = 0; i < m_nrow; ++i) {
            for (size_t j = 0; j < m_ncol; ++j) {
                (*this)(i, j) = vec[k];
                ++k;
            }
        }

        return *this;
    }

    Matrix &operator=(Matrix const &other) {
        if (this == &other) {
            return *this;
        }
        if (m_nrow != other.m_nrow || m_ncol != other.m_ncol) {
            reset_buffer(other.m_nrow, other.m_ncol);
        }
        for (size_t i = 0; i < m_nrow; ++i) {
            for (size_t j = 0; j < m_ncol; ++j) {
                (*this)(i, j) = other(i, j);
            }
        }
        return *this;
    }

    Matrix &operator=(Matrix &&other) {
        if (this == &other) {
            return *this;
        }
        reset_buffer(0, 0);
        std::swap(m_nrow, other.m_nrow);
        std::swap(m_ncol, other.m_ncol);
        std::swap(m_buffer, other.m_buffer);
        return *this;
    }

    Matrix &operator=(double val) {
        for (size_t i = 0, end = m_nrow * m_ncol; i < end; ++i) {
            m_buffer[i] = val;
        }
        return *this;
    }

    Matrix &operator+=(const Matrix &other) {
        if (m_nrow != other.m_nrow || m_ncol != other.m_ncol) {
            throw std::out_of_range("the shape of first matrix differs from"
                                    "that of second matrix");
        }

        for (size_t i = 0, end = m_nrow * m_ncol; i < end; ++i) {
            m_buffer[i] += other.m_buffer[i];
        }
        return *this;
    }

    double operator()(size_t row, size_t col) const {
        return m_buffer[index(row, col)];
    }
    double &operator()(size_t row, size_t col) {
        return m_buffer[index(row, col)];
    }

    double operator[](size_t idx) const { return m_buffer[idx]; }
    double &operator[](size_t idx) { return m_buffer[idx]; }

    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }

    size_t size() const { return m_nrow * m_ncol; }
    double buffer(size_t i) const { return m_buffer[i]; }
    std::vector<double> buffer_vector() const {
        return std::vector<double>(m_buffer, m_buffer + size());
    }

    Matrix transpose() const;

    size_t index(size_t row, size_t col) const { return row * m_ncol + col; }

    void save(Matrix &mat, size_t it, size_t jt);
    void zero() {
        for (size_t i = 0; i < m_nrow; ++i) {
            for (size_t j = 0; j < m_ncol; ++j) {
                (*this)(i, j) = 0;
            }
        }
    }
    void reset_buffer(size_t nrow, size_t ncol) {
        if (m_buffer) {
            delete[] m_buffer;
        }
        const size_t nelement = nrow * ncol;
        if (nelement) {
            m_buffer = new double[nelement];
        } else {
            m_buffer = nullptr;
        }
        m_nrow = nrow;
        m_ncol = ncol;
    }

    friend Matrix multiply_naive(Matrix const &mat1, Matrix const &mat2);
    friend Matrix multiply_tile(const Matrix &mat1, const Matrix &mat2, const size_t tile_size);
    friend Matrix multiply_mkl(Matrix const &mat1, Matrix const &mat2);
    friend bool operator==(Matrix const &mat1, Matrix const &mat2);

    size_t m_nrow = 0;
    size_t m_ncol = 0;
    double *m_buffer = nullptr;
};

Matrix Matrix::transpose() const {
    Matrix ret(nrow(), ncol());

    for (size_t i = 0; i < ret.nrow(); ++i) {
        for (size_t j = 0; j < ret.ncol(); ++j) {
            ret(j, i) = (*this)(i, j);
        }
    }

    return ret;
}

bool operator==(Matrix const &mat1, Matrix const &mat2) {
    if ((mat1.ncol() != mat2.ncol()) && (mat1.nrow() != mat2.ncol())) {
        return false;
    }

    for (size_t i = 0; i < mat1.nrow(); ++i) {
        for (size_t j = 0; j < mat1.ncol(); ++j) {
            if (mat1(i, j) != mat2(i, j)) {
                return false;
            }
        }
    }

    return true;
}

bool operator!=(Matrix const &mat1, Matrix const &mat2) {
    return !(mat1 == mat2);
}

/*
 * Throw an exception if the shapes of the two matrices don't support
 * multiplication.
 */
void validate_multiplication(Matrix const & mat1, Matrix const & mat2)
{
    if (mat1.ncol() != mat2.nrow())
    {
        throw std::out_of_range(
                "the number of first matrix column "
                "differs from that of second matrix row");
    }
}

/*
 * Get the number of floating-point operations.
 */
size_t calc_nflo(Matrix const & mat1, Matrix const & mat2)
{
    return mat1.nrow() * mat1.ncol() * mat2.ncol();
}


Matrix multiply_naive(Matrix const & mat1, Matrix const & mat2){
    validate_multiplication(mat1,mat2);
    Matrix ret(mat1.nrow(), mat2.ncol());
    const size_t nrow1 = mat1.nrow();
    const size_t ncol1 = mat1.ncol();
    const size_t nrow2 = mat2.nrow();
    const size_t ncol2 = mat2.ncol();


    for (size_t i=0; i<nrow1; ++i)
    {
        const size_t base1 = i * ncol1;
        for (size_t k=0; k<ncol2; ++k)
        {
            double v = 0;
            for (size_t j=0; j<ncol1; ++j)
            {
                v += mat1.m_buffer[base1 + j] * mat2.m_buffer[j*ncol2 + k];
            }
            ret.m_buffer[base1 + k] = v;
        }
    }



    return ret;

}


Matrix multiply_tile(Matrix const & mat1, Matrix const & mat2,size_t lsize){
    validate_multiplication(mat1,mat2);
    Matrix ret(mat1.nrow(), mat2.ncol());
    ret.zero();
    const size_t nrow1 = mat1.nrow();
    const size_t ncol1 = mat1.ncol();
    const size_t nrow2 = mat2.nrow();
    const size_t ncol2 = mat2.ncol();
    //const size_t m1rt = mat2.ncol();
    //const size_t m2ct = mat2.ncol();
    //const size_t m1ct = mat2.ncol();
    const size_t m1rt = ceil(nrow1 / lsize);
    const size_t m1ct = ceil(ncol1 / lsize);
    const size_t m2ct = ceil(ncol2 / lsize);
    double v1,v2;

    for (size_t z=0;z<lsize;++z){
        for (size_t x=0;x<lsize;++x){
            for (size_t i=0;i<m1rt;++i){
                for (size_t k=0;k<m2ct;++k){
                    for(size_t p=0;p<lsize;++p){
                        for(size_t j=0;j<m1ct;++j){
                            v1 = i + m1rt * x < nrow1 && j + m1ct * p < ncol1 ?mat1(i + m1rt * x,j + m1ct * p): 0;
                            v2 = j + m1ct * p < nrow2 && k + z * m2ct < ncol2 ?mat2(j + m1ct * p,k + z * m2ct): 0;
                            if (i + m1rt * x<nrow1 && k + z * m2ct<ncol2){
                                ret(i + m1rt * x,k + z * m2ct) =ret(i + m1rt * x,k + z * m2ct) + v1 * v2;
                            }
                        }
                    }
                }
            }
        }
    }

    return ret;
}
Matrix multiply_mkl(Matrix const & mat1, Matrix const & mat2){
    mkl_set_num_threads(1);
    Matrix ret(mat1.nrow(), mat2.ncol());

    cblas_dgemm(
        CblasRowMajor /* const CBLAS_LAYOUT Layout */
      , CblasNoTrans /* const CBLAS_TRANSPOSE transa */
      , CblasNoTrans /* const CBLAS_TRANSPOSE transb */
      , mat1.nrow() /* const MKL_INT m */
      , mat2.ncol() /* const MKL_INT n */
      , mat1.ncol() /* const MKL_INT k */
      , 1.0 /* const double alpha */
      , mat1.m_buffer /* const double *a */
      , mat1.ncol() /* const MKL_INT lda */
      , mat2.m_buffer /* const double *b */
      , mat2.ncol() /* const MKL_INT ldb */
      , 0.0 /* const double beta */
      , ret.m_buffer /* double * c */
      , ret.ncol() /* const MKL_INT ldc */
    );


    return ret;
}
PYBIND11_MODULE(_matrix, m)
{
m.doc() = "pybind11 matrix multiplication test";
m.def("multiply_naive", & multiply_naive, "naive method");
m.def("multiply_tile", & multiply_tile, "Tiling solution");
m.def("multiply_mkl", & multiply_mkl, "use mkl");
py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
.def(py::init<size_t, size_t>())
.def(py::init<size_t, size_t, const std::vector<double> &>())
.def(py::init<const Matrix &>())
.def_property("nrow", &Matrix::nrow, nullptr)
.def_property("ncol", &Matrix::ncol, nullptr)
.def("__eq__", &operator==)
.def("buffer_vector", &Matrix::buffer_vector)
.def("__setitem__", [](Matrix &mat, std::pair<size_t, size_t> i,
double val) { mat(i.first, i.second) = val; })
.def("__getitem__", [](Matrix &mat, std::pair<size_t, size_t> i) {
return mat(i.first, i.second);
});
}