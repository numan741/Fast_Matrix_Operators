#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <iostream>
#include <Python.h>
#include "Vect.h"
#include <numpy/arrayobject.h>
#include <vector>

class Mat
{
public:
    Mat(int rows, int cols);
    Mat(npy_intp rows, npy_intp cols,double* Mp);
    Mat();
    ~Mat();
    Mat(const Mat&);
    inline float& operator()(int x, int y) { return p[x][y]; }

    Mat& operator=(const Mat&);

    Mat operator+(const Mat&);
    Mat operator-(const Mat&);
    Mat operator*(const Mat&);
    Vect operator*(const Vect&);
    Mat operator*(double);
    Mat operator/(double);
    
    void swapRows(int, int);

    Mat inv();
    Mat pinv();
    double det();
    PyObject* toNumpy();


    float** data() { return p; }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

private:
    
    int rows_, cols_;
    float** p;
    void allocSpace();
    
};
#endif