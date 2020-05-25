#pragma once

#include "Mat.h";
//#include "Vect.h";
#include <tuple>
#include <vector>

#define EPS 1e-10
#define rcond 1e-15


Mat createIdentity(int);

Mat augment(Mat, Mat);

Mat gaussianEliminate(Mat);

Mat rowReduceFromGaussian(Mat);

std::tuple<Mat,Mat,Mat> SVD_Decompose(Mat);
std::tuple<Mat, Mat> LU_Decompose(Mat);

Mat transpose(Mat);

Mat solve(Mat, Mat);

Vect transpose(Vect);

