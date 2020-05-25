#include <Python.h>
#include <iostream>
#include "Mat.h"
#include <vector>
#include "Vect.h"
#include <numpy/arrayobject.h>

static PyObject* DimensionCheckError = NULL;
static PyObject* InvConditionError = NULL;
static PyObject* MulDimensionError = NULL;

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

static bool checkPyArrayDimensions(PyArrayObject* pyarr, const npy_intp dim0, const npy_intp dim1)
{
    const auto dims = getPyArrayDimensions(pyarr);
    assert(dims.size() <= 2 && dims.size() > 0);
    if (dims.size() == 1) {
        return (dims[0] == dim0 || dim0 == -1) && (dim1 == 1 || dim1 == -1);
    }
    else {
        return (dims[0] == dim0 || dim0 == -1) && (dims[1] == dim1 || dim1 == -1);
    }
}

Mat* convertPyArrayToMat(PyArrayObject* pyarr, int nrows, int ncols)
{
    if (!checkPyArrayDimensions(pyarr, nrows, ncols)) throw WrongDimensions();
    int arrTypeCode;
    arrTypeCode = NPY_DOUBLE;
    const auto dims = getPyArrayDimensions(pyarr);
    PyArray_Descr* reqDescr = PyArray_DescrFromType(arrTypeCode);
    if (reqDescr == NULL) throw std::bad_alloc();
    PyArrayObject* cleanArr = (PyArrayObject*)PyArray_FromArray(pyarr, reqDescr, NPY_ARRAY_CARRAY);
    if (cleanArr == NULL) throw std::bad_alloc();
    reqDescr = NULL;  
    double* dataPtr = static_cast<double*>(PyArray_DATA(cleanArr));
    Mat* result= new Mat(dims[0], dims[1], dataPtr);  
    Py_DECREF(cleanArr);

    return result;
}

PyObject* toNumpy(Mat& matrix)
{
    npy_intp ndim = 2;
    npy_intp nRows = static_cast<npy_intp>(matrix.rows());
    npy_intp nCols = static_cast<npy_intp>(matrix.cols());
    npy_intp dims[2] = { nRows, nCols };

    PyObject* result = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    if (result == NULL) throw std::bad_alloc();

    double* resultDataPtr = static_cast<double*>(PyArray_DATA((PyArrayObject*)result));
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            resultDataPtr[i * nCols + j] = matrix(i, j);
        }
    }
    return result;
}


PyObject* construct(PyObject* self, PyObject* args)
{
    PyArrayObject* myArray = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &myArray)) {
        return NULL;
    }
    Mat* M1 = convertPyArrayToMat(myArray,-1,-1);
    PyObject* MatCapsule = PyCapsule_New((void*)M1, "MatPtr", NULL);
    PyCapsule_SetPointer(MatCapsule, (void*)M1);
    return Py_BuildValue("O", MatCapsule);
}

PyObject* AddOperator(PyObject* self, PyObject* args)
{
    PyObject* MatCapsule1_;   // Capsule with the pointer to Matrix object
    PyObject* MatCapsule2_;         

    PyArg_ParseTuple(args, "OO",
        &MatCapsule1_,
        &MatCapsule2_);
    Mat* m1 = (Mat*)PyCapsule_GetPointer(MatCapsule1_, "MatPtr");
    Mat* m2 = (Mat*)PyCapsule_GetPointer(MatCapsule2_, "MatPtr");

    Mat* mf = new Mat(m1->operator+(*m2));
    PyObject* MatCapsule = PyCapsule_New((void*)mf, "MatPtr", NULL);
    PyCapsule_SetPointer(MatCapsule, (void*)mf);
    return Py_BuildValue("O", MatCapsule);
}

PyObject* SubOperator(PyObject* self, PyObject* args)
{

    PyObject* MatCapsule1_;   // Capsule with the pointer to Matrix object
    PyObject* MatCapsule2_;         // Second Arguments

    PyArg_ParseTuple(args, "OO",
        &MatCapsule1_,
        &MatCapsule2_);
    Mat* m1 = (Mat*)PyCapsule_GetPointer(MatCapsule1_, "MatPtr");
    Mat* m2 = (Mat*)PyCapsule_GetPointer(MatCapsule2_, "MatPtr");

    Mat* vf = new Mat(m1->operator-(*m2));

    PyObject* MatCapsule = PyCapsule_New((void*)vf, "MatPtr", NULL);
    PyCapsule_SetPointer(MatCapsule, (void*)vf);
    return Py_BuildValue("O", MatCapsule);
}

PyObject* MulVectOperator(PyObject* self, PyObject* args)
{

    PyObject* MatCapsule1_;   // Capsule with the pointer to Matrix object
    PyObject* VectCapsule1_;         // Second Arguments

    PyArg_ParseTuple(args, "OO",
        &MatCapsule1_,
        &VectCapsule1_);
    Mat* m1 = (Mat*)PyCapsule_GetPointer(MatCapsule1_, "MatPtr");
    Vect* v1 = (Vect*)PyCapsule_GetPointer(VectCapsule1_, "MatPtr");

    Vect* vf = new Vect(m1->operator*(*v1));

    PyObject* MatCapsule = PyCapsule_New((void*)vf, "MatPtr", NULL);
    PyCapsule_SetPointer(MatCapsule, (void*)vf);
    return Py_BuildValue("O", MatCapsule);
}

PyObject* MulOperator(PyObject* self, PyObject* args)
{
    PyObject* MatCapsule1_;   // Capsule with the pointer to Matrix object
    PyObject* MatCapsule2_;         // Second Arguments

    PyArg_ParseTuple(args, "OO",
        &MatCapsule1_,
        &MatCapsule2_);
    Mat* m1 = (Mat*)PyCapsule_GetPointer(MatCapsule1_, "MatPtr");
    Mat* m2 = (Mat*)PyCapsule_GetPointer(MatCapsule2_, "MatPtr");
    if (m1->cols() != m2->rows()) {
        PyErr_SetString(MulDimensionError, " Matrix Dimension not matched ==> Columns of 1st matrix != Rows of 2nd matrix");
        return NULL;
    }
    Mat* vf = new Mat(m1->operator*(*m2));

    PyObject* MatCapsule = PyCapsule_New((void*)vf, "MatPtr", NULL);
    PyCapsule_SetPointer(MatCapsule, (void*)vf);
    return Py_BuildValue("O", MatCapsule);
}

PyObject* Inverse(PyObject* self, PyObject* args)
{

    PyObject* MatCapsule1_;   // Capsule with the pointer to Matrix object

    PyArg_ParseTuple(args, "O",
        &MatCapsule1_);
    Mat* m1 = (Mat*)PyCapsule_GetPointer(MatCapsule1_, "MatPtr");
    if (m1->cols()!=m1->rows()) {
        PyErr_SetString(DimensionCheckError, "This is not square Matrix");
        return NULL;
    }else if(m1->det() == 0.0 && m1->cols() != m1->rows()) {
        PyErr_SetString(InvConditionError, " Matrix is singular so it does not have Inverse ");
        return NULL;
    }

    Mat* mf = new Mat(m1->inv());

    PyObject* MatCapsule = PyCapsule_New((void*)mf, "MatPtr", NULL);
    PyCapsule_SetPointer(MatCapsule, (void*)mf);
    return Py_BuildValue("O", MatCapsule);
}

PyObject* ScalarMul(PyObject* self, PyObject* args)
{

    PyObject* MatCapsule1_;   // Capsule with the pointer to Matrix object
    double Scalar;         // Scalar Value to mul with Matrix

    PyArg_ParseTuple(args, "Od",
        &MatCapsule1_,
        &Scalar);
    Mat* v1 = (Mat*)PyCapsule_GetPointer(MatCapsule1_, "MatPtr");

    Mat* vf = new Mat(v1->operator*(Scalar));
    PyObject* MatCapsule = PyCapsule_New((void*)vf, "MatPtr", NULL);
    PyCapsule_SetPointer(MatCapsule, (void*)vf);
    return Py_BuildValue("O", MatCapsule);
}

PyObject* ScalarDiv(PyObject* self, PyObject* args)
{

    PyObject* MatCapsule1_;   // Capsule with the pointer to `Mator` object
    double Scalar;
    PyArg_ParseTuple(args, "Od",&MatCapsule1_,&Scalar);
    Mat* v1 = (Mat*)PyCapsule_GetPointer(MatCapsule1_, "MatPtr");
    Mat* vf = new Mat(v1->operator/(Scalar));
    PyObject* MatCapsule = PyCapsule_New((void*)vf, "MatPtr", NULL);
    PyCapsule_SetPointer(MatCapsule, (void*)vf);
    return Py_BuildValue("O", MatCapsule);
}

PyObject* toNum(PyObject* self, PyObject* args)
{
    PyObject* MatCapsule1_;
    PyArg_ParseTuple(args, "O",&MatCapsule1_);
    Mat* v1 = (Mat*)PyCapsule_GetPointer(MatCapsule1_, "MatPtr");
    return Py_BuildValue("O", toNumpy(*v1));
}

PyObject* delete_object(PyObject* self, PyObject* args)
{
    PyObject* vecCapsule_;
    PyArg_ParseTuple(args, "O",
        &vecCapsule_);
    Mat* cab = (Mat*)PyCapsule_GetPointer(vecCapsule_, "MatPtr");
    delete cab;
    return Py_BuildValue("");
}


PyMethodDef cMat_Functions[] =
{
    /*
     *  Structures which define functions ("methods") provided by the module.
     */
        {"construct",                   // C++/Py Constructor
          construct, METH_VARARGS,
         "Create `Mat` object"},

        {"AddOperator",                     // C++/Py wrapper 
          AddOperator, METH_VARARGS,
         "Plus Operator Overloading"},

        {"SubOperator",                       // C++/Py wrapper 
          SubOperator, METH_VARARGS,
         "Subtract Operator Overloading"},

         {"MulVectOperator",                       // C++/Py wrapper 
          MulVectOperator, METH_VARARGS,
         "Multiply Vector Overloading"},

         {"MulOperator",                       // C++/Py wrapper 
          MulOperator, METH_VARARGS,
         "Multiply Operator Overloading"},

         {"Inverse",                       // C++/Py wrapper 
          Inverse, METH_VARARGS,
         "Inverse func Overloading"},

         {"ScalarMul",                       // C++/Py wrapper 
          ScalarMul, METH_VARARGS,
         "Scalar Multiplication"},

         {"ScalarDiv",                       // C++/Py wrapper 
          ScalarDiv, METH_VARARGS,
         "Scalar Division"},

        {"toNum",               // C++/Py wrapper 
          toNum, METH_VARARGS,
         "Convert to Numpy"},

        {"delete_object",               // C++/Py Destructor
          delete_object, METH_VARARGS,
         "Delete Matrix object"},

        {NULL, NULL, 0, NULL}      // Last function description must be empty.
                                   // Otherwise, it will create seg fault while
                                   // importing the module.
};


struct PyModuleDef cMat_Module =
{
       PyModuleDef_HEAD_INIT,
       "cMat",               // Name of the module.
       NULL,                 // Docstring for the module - in this case empty.
       -1,
       cMat_Functions         // Structures of type `PyMethodDef` with functions
                             // (or "methods") provided by the module.
};


PyMODINIT_FUNC
    PyInit_cMat(void)
{
    import_array();
    PyObject* m = PyModule_Create(&cMat_Module);
    if (m == NULL) return NULL;
    DimensionCheckError = PyErr_NewException("cMat.DimensionCheckMulError", NULL, NULL);
    PyModule_AddObject(m, "DimensionCheckMul", DimensionCheckError);

    InvConditionError = PyErr_NewException("cMat.InvConditionError", NULL, NULL);
    PyModule_AddObject(m, "InvCondition", InvConditionError);

    MulDimensionError = PyErr_NewException("cMat.MulDimensionError", NULL, NULL);
    PyModule_AddObject(m, "MulDimension", MulDimensionError);  
    return m;
}