#include <iostream>

extern "C" {
#include <Python.h>
#include <numpy/arrayobject.h>
}
class Vect
{
public:
    Vect(int ele);
    Vect(int ele, double* Mp);
    Vect(const Vect& v);
    Vect();
    ~Vect();
    
    inline double& operator()(int x) { return p[x]; }
    Vect operator+(const Vect&);
    Vect operator-(const Vect&);
    Vect operator*(double);
    Vect operator/(double);
    Vect& operator=(const Vect&);
    PyObject* toNumpy();

    size_t eles() const { return ele_; }
    double* data() { return p; }
private:
    int ele_;
    double* p;
    void allocSpace();
};

