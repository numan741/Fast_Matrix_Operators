#include "Mat.h"
#include <stdexcept>
#include <exception>
#include "LinearAlgebra.h"
#include <tuple>
extern "C" {
#include <Python.h>
#include <numpy/arrayobject.h>
}

Mat::Mat(int rows, int cols) : rows_(rows), cols_(cols) {
    allocSpace();
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            p[i][j] = 0;
        }
    }
}
Mat::Mat(npy_intp rows, npy_intp cols,double* Mp ) : rows_(rows), cols_(cols)
{
    allocSpace();
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            p[i][j] = Mp[i * cols_ + j];
        }
    }
}

Mat::Mat() : rows_(1), cols_(1)
{
    allocSpace();
    p[0][0] = 0;
}

Mat::~Mat()
{
    for (int i = 0; i < rows_; ++i) {
        delete[] p[i];
    }
    delete[] p;
}

Mat::Mat(const Mat& m) : rows_(m.rows()), cols_(m.cols())
{
    allocSpace();
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            p[i][j] = m.p[i][j];
        }
    }
}

Mat Mat::inv() {

    Mat I = createIdentity(rows_);
    Mat AI = augment(*this, I);
    Mat U = gaussianEliminate(AI);
    Mat IAInverse = rowReduceFromGaussian(U);
    Mat AInverse(rows_, cols_);
    
    for (int i = 0; i < AInverse.rows_; ++i) {
        
        for (int j = 0; j < AInverse.cols_; ++j) {
            AInverse(i, j) = IAInverse(i, j + cols_);
        }
    }
    return AInverse;
}
double  Mat::det() {
    double det = 1;
    //Mat u = std::get<1>(LU_Decompose(*this));
    Mat U = gaussianEliminate(*this);
    for (int i = 0; i <cols_; i++) {
        det = det * U(i, i);
    }
    return det;
    

}
 PyObject* Mat::toNumpy() {
     npy_intp ndim = 2;
     npy_intp nRows = static_cast<npy_intp>(rows_);  // NOTE: This narrows the integer
     npy_intp nCols = static_cast<npy_intp>(cols_);
     npy_intp dims[2] = { nRows, nCols };

     PyObject* result = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
     if (result == NULL) throw std::bad_alloc();

     double* resultDataPtr = static_cast<double*>(PyArray_DATA((PyArrayObject*)result));
     for (int i = 0; i < nRows; i++) {
         for (int j = 0; j < nCols; j++) {
             resultDataPtr[i * nCols + j] = p[i][j];
         }
     }
     return result;
}

void Mat::allocSpace()
{
    p = new float* [rows_];
    for (int i = 0; i < rows_; ++i) {
        p[i] = new float[cols_];
    }
}

Mat Mat::operator+(const Mat& m2)
{
    Mat temp(*this);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            temp.p[i][j]=p[i][j] + m2.p[i][j];
        }
    }
    return temp;
}

Mat Mat::operator-(const Mat& m2)
{
    Mat temp(*this);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            temp.p[i][j]=p[i][j] - m2.p[i][j];
        }
    }
    return temp;
}

Mat Mat::operator*( const Mat& m)
{
    Mat temp(rows_, m.cols_);
    for (int i = 0; i < temp.rows_; ++i) {
        for (int j = 0; j < temp.cols_; ++j) {
            for (int k = 0; k < cols_; ++k) {
                temp.p[i][j] += (p[i][k] * m.p[k][j]);
            }
        }
    }
    return temp;
}

Vect Mat::operator*(const Vect& v)
{
    Vect temp(rows_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            for (int k = 0; k < cols_; ++k) {
                temp(i) += (p[i][k] * temp(k));
            }
        }
    }
    return temp;
}

Mat Mat::operator*(double num)
{
    Mat temp(*this);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            temp.p[i][j] *= num;
        }
    }
    return temp;
}

Mat Mat::operator/(double num)
{
    Mat temp(*this);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            temp.p[i][j] /= num;
        }
    }
    return temp;
}


void Mat::swapRows(int r1, int r2)
{
    float* temp = p[r1];
    p[r1] = p[r2];
    p[r2] = temp;
}

Mat& Mat::operator=(const Mat& m)
{
    if (this == &m) {
        return *this;
    }

    if (rows_ != m.rows_ || cols_ != m.cols_) {
        for (int i = 0; i < rows_; ++i) {
            delete[]  p[i];
        }
        delete[] p;

        rows_ = m.rows_;
        cols_ = m.cols_;
        allocSpace();
    }

    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            p[i][j] = m.p[i][j];
        }
    }
    return *this;
}


