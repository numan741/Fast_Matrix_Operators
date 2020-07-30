# Fast_Vector/Matrix_Operators
Implementation of Fast Matrix Functionality. All the functionality is developed in C++ with Object-Oriented Programming(OOP) techniques. 
# Discription
The library have a python frontend, meaning the user should be able to use these classes in Python code, upon importing your python script.
Morover, library have C++ backend, that the python frontend should call upon a DLL, written in C++, to do the calculations.
# Implementation
The python library compiles the C++ backend upon importing (User must have a Visual Studio and MSBuild installation on the computer, AND MSBuild is in the PATH)
# Tools 
Cpython
C++
Python
# Details
The lib has 2 classes, Matrix and Vector. After importing the lib, the user should be able to create a matrix (arbitrary size, not necessarily a square matrix) or a vector (arbitrary size) by initializing it with a numpy array. All data should be stored in float32, all calulations should be done in float32.
If S is a scalar, V is a vector, M is a matrix, implement the following operators (with operator overloading):
o (elementwise): V-V, V+V
o (elementwise): M-M, M+M
o (elementwise): V*S, M*S, V/S, M/S
o (NOT elementwise, regular matrix and vector multiplications): M*V, M*M Throw exception if dimensions are not matching

