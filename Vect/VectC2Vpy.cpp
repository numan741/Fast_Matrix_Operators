#include <Python.h>
#include <iostream>
#include "Vect.h"
#include <vector>
#include <numpy/arrayobject.h>
class WrongDimensions : public std::exception
{
public:
    WrongDimensions() {}
    const char* what() const noexcept { return msg.c_str(); }

private:
    std::string msg = "The dimensions were incorrect";
};

static const std::vector<npy_intp> getPyArrayDimensions(PyArrayObject* pyarr)
{
    npy_intp ndims = PyArray_NDIM(pyarr);
    npy_intp* dims = PyArray_SHAPE(pyarr);
    std::vector<npy_intp> result;
    for (int i = 0; i < ndims; i++) {
        result.push_back(dims[i]);
    }
    return result;
}

static bool checkPyArrayDimensions(PyArrayObject* pyarr)
{
    const auto dims = getPyArrayDimensions(pyarr);
    assert(dims.size() <= 2 && dims.size() > 0);
    if (dims[0] != 1) {
        return false;
    }
    else {
        return true;
    }
    
}

Vect* PyArrayToVect(PyArrayObject* pyarr)
{
   // if (!checkPyArrayDimensions(pyarr)) throw WrongDimensions();
    int arrTypeCode;
    arrTypeCode = NPY_DOUBLE;
    //npy_intp ndims = PyArray_NDIM(pyarr);
    npy_intp* dims = PyArray_SHAPE(pyarr);
    PyArray_Descr* reqDescr = PyArray_DescrFromType(arrTypeCode);
    PyArrayObject* cleanArr = (PyArrayObject*)PyArray_FromArray(pyarr, reqDescr, NPY_ARRAY_C_CONTIGUOUS);

    double* dataPtr = static_cast<double*>(PyArray_DATA(cleanArr));
    Vect* result=new Vect(dims[0], dataPtr);  
    Py_DECREF(cleanArr);
    return result;
}

PyObject* toNumpy(Vect vec)
{
    npy_intp ndim = 1;
    npy_intp nele = static_cast<npy_intp>(vec.eles());
    npy_intp dims[2] = {nele, };

    PyObject* result = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    double* resultDataPtr = static_cast<double*>(PyArray_DATA((PyArrayObject*)result));
    
        for (int i = 0; i < nele; i++) {
            resultDataPtr[i] = vec(i);
        }
    return result;
}


PyObject* construct(PyObject* self, PyObject* args)
{
    PyArrayObject* myArray = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &myArray)) {
        return NULL;
    }
    Vect* vect = PyArrayToVect(myArray);
    PyObject* MatCapsule = PyCapsule_New((void*)vect, "VectPtr", NULL);
    PyCapsule_SetPointer(MatCapsule, (void*)vect);
    return Py_BuildValue("O", MatCapsule);
}



PyObject* AddOperator(PyObject* self, PyObject* args)
{
    PyObject* VectCapsule1_;   // Capsule with the pointer to Vector object
    PyObject* VectCapsule2_;         // Second Vector

    PyArg_ParseTuple(args, "OO",
        &VectCapsule1_,
        &VectCapsule2_);
    Vect* v1 = (Vect*)PyCapsule_GetPointer(VectCapsule1_, "VectPtr");
    Vect* v2 = (Vect*)PyCapsule_GetPointer(VectCapsule2_, "VectPtr");
    
    Vect* vf = new Vect(v1->operator+(*v2));
    PyObject* MatCapsule = PyCapsule_New((void*)vf, "VectPtr", NULL);
    PyCapsule_SetPointer(MatCapsule, (void*)vf);
    return Py_BuildValue("O", MatCapsule);
}

PyObject* SubOperator(PyObject* self, PyObject* args)
{
    
    PyObject* VectCapsule1_;  
    PyObject* VectCapsule2_;        

    PyArg_ParseTuple(args, "OO",
        &VectCapsule1_,
        &VectCapsule2_);
    Vect* v1 = (Vect*)PyCapsule_GetPointer(VectCapsule1_, "VectPtr");
    Vect* v2 = (Vect*)PyCapsule_GetPointer(VectCapsule2_, "VectPtr");

    Vect* vf = new Vect(v1->operator-(*v2));

    PyObject* MatCapsule = PyCapsule_New((void*)vf, "VectPtr", NULL);
    PyCapsule_SetPointer(MatCapsule, (void*)vf);
    
    return Py_BuildValue("O", MatCapsule);
}

PyObject* ScalarMul(PyObject* self, PyObject* args)
{
    
    PyObject* VectCapsule1_;   
    double Scalar;         

    PyArg_ParseTuple(args, "Od",
        &VectCapsule1_,
        &Scalar);
    Vect* v1 = (Vect*)PyCapsule_GetPointer(VectCapsule1_, "VectPtr");
    
    Vect* vf = new Vect(v1->operator*(Scalar));
    PyObject* MatCapsule = PyCapsule_New((void*)vf, "VectPtr", NULL);
    PyCapsule_SetPointer(MatCapsule, (void*)vf);
    return Py_BuildValue("O", MatCapsule);
}

PyObject* ScalarDiv(PyObject* self, PyObject* args)
{

    PyObject* VectCapsule1_;   
    double Scalar;         

    PyArg_ParseTuple(args, "Od",
        &VectCapsule1_,
        &Scalar);
    Vect* v1 = (Vect*)PyCapsule_GetPointer(VectCapsule1_, "VectPtr");

    Vect* vf = new Vect(v1->operator/(Scalar));
    PyObject* MatCapsule = PyCapsule_New((void*)vf, "VectPtr", NULL);
    PyCapsule_SetPointer(MatCapsule, (void*)vf);
    return Py_BuildValue("O", MatCapsule);
}

PyObject* toNum(PyObject* self, PyObject* args)
{

    PyObject* VectCapsule1_;  

    PyArg_ParseTuple(args, "O",
        &VectCapsule1_);
    Vect* v1 = (Vect*)PyCapsule_GetPointer(VectCapsule1_, "VectPtr");
    return Py_BuildValue("O", toNumpy(*v1));
}

PyObject* delete_object(PyObject* self, PyObject* args)
{
    
    PyObject* vecCapsule_;   
    PyArg_ParseTuple(args, "O",
        &vecCapsule_);
    Vect* cab = (Vect*)PyCapsule_GetPointer(vecCapsule_, "VectPtr");
    delete cab;
    return Py_BuildValue("");
}


PyMethodDef cVect_Functions[] =
{
    /*
     *  Structures which define functions ("methods") provided by the module.
     */
        {"construct",                   // C++/Py Constructor
          construct, METH_VARARGS,
         "Create `Vect` object"},

        {"AddOperator",                     // C++/Py wrapper 
          AddOperator, METH_VARARGS,
         "Plus Operator Overloading"},

        {"SubOperator",                       // C++/Py wrapper 
          SubOperator, METH_VARARGS,
         "Subtract Operator Overloading"},

         {"ScalarMul",                       // C++/Py wrapper 
          ScalarMul, METH_VARARGS,
         "Scalar Multiplication"},

         {"ScalarDiv",                       // C++/Py wrapper 
          ScalarDiv, METH_VARARGS,
         "Scalar Division"},

        {"toNum",               // C++/Py wrapper 
          toNum, METH_VARARGS,
         "Numpy Conversion"},

        {"delete_object",               // C++/Py Destructor
          delete_object, METH_VARARGS,
         "Delete Vector object"},

        {NULL, NULL, 0, NULL}      // Last function description must be empty.
                                   // Otherwise, it will create seg fault while
                                   // importing the module.
};


struct PyModuleDef cVect_Module =
{
       PyModuleDef_HEAD_INIT,
       "cVect",               // Name of the module.
       NULL,                 // Docstring for the module - in this case empty.
       -1,  
       cVect_Functions         // Structures of type `PyMethodDef` with functions
                             // (or "methods") provided by the module.
};


PyMODINIT_FUNC 
    PyInit_cVect(void)
{
    import_array();
    PyObject* m= PyModule_Create(&cVect_Module);
    if (m == NULL) return NULL;
    return m;
}