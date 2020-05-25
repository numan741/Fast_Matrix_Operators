#include "Vect.h"

Vect::Vect(int ele) : ele_(ele) {
    allocSpace();
    for (int i = 0; i < ele_; ++i) {
        p[i] = 0;
    }
}

Vect::Vect(int ele, double* Mp) : ele_(ele)
{
    allocSpace();
    for (int i = 0; i < ele_; ++i) {
            p[i] = Mp[i];
    }
}

Vect::Vect() : ele_(1)
{
    allocSpace();
    p[0] = 0;
}
Vect::Vect(const Vect& v):ele_(v.ele_) {
    allocSpace();
    for (int i = 0; i < ele_; ++i) {
            p[i] = v.p[i];
    }

}

Vect::~Vect()
{
    delete[] p;
}

void Vect::allocSpace()
{
    p = new double[ele_];
   
}

Vect Vect::operator+(const Vect& v) {
    Vect temp(*this);
    for (int i = 0; i < ele_; ++i) {
            temp.p[i]=p[i] + v.p[i];
    }
    return temp;
}
Vect Vect::operator-(const Vect& v) {
    Vect temp(*this);
    for (int i = 0; i < ele_; ++i) {
        temp.p[i] = p[i] - v.p[i];
    }
    return temp;
}
Vect Vect::operator*(double num) {
    Vect temp(*this);
    for (int i = 0; i < ele_; ++i) {
        temp.p[i]=p[i] * num;
    }
    return temp;
}
Vect Vect::operator/(double num) {
    Vect temp(*this);
    for (int i = 0; i < ele_; ++i) {
        temp.p[i] =p[i]/ num;
    }
    return temp;
}
Vect& Vect::operator=(const Vect& m) {
    if (this == &m) {
        return *this;
    }

    if (ele_ != m.ele_) {
        delete[] p;
        ele_ = m.ele_;
        allocSpace();
    }

    for (int i = 0; i < ele_; ++i) {
            p[i] = m.p[i];
    }
    return *this;      
}

PyObject* Vect::toNumpy() {
    npy_intp ndim = 2;
    npy_intp nele = static_cast<npy_intp>(ele_);  // NOTE: This narrows the integer
    npy_intp dims[2] = { 1, nele };

    PyObject* result = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    if (result == NULL) throw std::bad_alloc();

    double* resultDataPtr = static_cast<double*>(PyArray_DATA((PyArrayObject*)result));
    
        for (int j = 0; j < nele; j++) {
            resultDataPtr[j] = p[j];
        }
    return result;
}

