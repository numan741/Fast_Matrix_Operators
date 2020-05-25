#include "LinearAlgebra.h"
#include <algorithm>
#include <xmmintrin.h>
#include <vector>

using std::domain_error;
Mat augment(Mat A, Mat B)
{
    Mat AB(A.rows(), A.cols() + B.cols());
    
    for (int i = 0; i < AB.rows(); ++i) {
        for (int j = 0; j < AB.cols(); ++j) {
            if (j < A.cols())
                AB(i, j) = A(i, j);
            else
                AB(i, j) = B(i, j - B.cols());
        }
    }
    return AB;
}

Mat gaussianEliminate(Mat A)
{
    Mat Ab=A;
    int rows = Ab.rows();
    int cols = Ab.cols();
    int Acols = cols - 1;

    int i = 0; // row tracker
    int j = 0; // column tracker

    // iterate through the rows
    while (i < rows)
    {
        // find a pivot for the row
        bool pivot_found = false;
        while (j < Acols && !pivot_found)
        {
            if (Ab(i, j) != 0) { // pivot not equal to 0
                pivot_found = true;
            }
            else { // check for a possible swap
                int max_row = i;
                double max_val = 0;
                for (int k = i + 1; k < rows; ++k)
                {
                    double cur_abs = Ab(k, j) >= 0 ? Ab(k, j) : -1 * Ab(k, j);
                    if (cur_abs > max_val)
                    {
                        max_row = k;
                        max_val = cur_abs;
                    }
                }
                if (max_row != i) {
                    Ab.swapRows(max_row, i);
                    pivot_found = true;
                }
                else {
                    j++;
                }
            }
        }

        // perform elimination as normal if pivot was found
        if (pivot_found)
        {
            
            for (int t = i + 1; t < rows; ++t) {
                for (int s = j + 1; s < cols; ++s) {
                    Ab(t, s) = Ab(t, s) - Ab(i, s) * (Ab(t, j) / Ab(i, j));
                    if (Ab(t, s) < EPS && Ab(t, s) > -1 * EPS)
                        Ab(t, s) = 0;
                }
                Ab(t, j) = 0;
            }
        }

        i++;
        j++;
    }

    return Ab;
}

Mat rowReduceFromGaussian(Mat Rin)
{
    Mat R=Rin;
    int rows = R.rows();
    int cols = R.cols();

    int i = rows - 1; // row tracker
    int j = cols - 2; // column tracker

    // iterate through every row
    while (i >= 0)
    {
        // find the pivot column
        int k = j - 1;
        while (k >= 0) {
            if (R(i, k) != 0)
                j = k;
            k--;
        }

        // zero out elements above pivots if pivot not 0
        if (R(i, j) != 0) {
           
            for (int t = i - 1; t >= 0; --t) {
                for (int s = 0; s < cols; ++s) {
                    if (s != j) {
                        R(t, s) = R(t, s) - R(i, s) * (R(t, j) / R(i, j));
                        if (R(t, s) < EPS && R(t, s) > -1 * EPS)
                            R(t, s) = 0;
                    }
                }
                R(t, j) = 0;
            }

            // divide row by pivot
            
            for (int k = j + 1; k < cols; ++k) {
                R(i, k) = R(i, k) / R(i, j);
                if (R(i, k) < EPS && R(i, k) > -1 * EPS)
                    R(i, k) = 0;
            }
            R(i, j) = 1;

        }

        i--;
        j--;
    }

    return R;
}
Mat createIdentity(int size)
{
    Mat temp(size, size);
    
    for (int i = 0; i < temp.rows(); ++i) {
        for (int j = 0; j < temp.cols(); ++j) {
            if (i == j) {
                temp(i,j) = 1;
            }
            else {
                temp(i,j) = 0;
            }
        }
    }
    return temp;
}

Mat transpose(Mat A)
{
    Mat ret(A.cols(), A.rows());
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            ret(j,i) = A(i,j);
        }
    }
    return ret;
}

Mat solve(Mat A, Mat b)
{
    // Gaussian elimination
    for (int i = 0; i < A.rows(); ++i) {
        if (A(i,i) == 0) {
            // pivot 0 - throw error
            throw domain_error("Error: the coefficient matrix has 0 as a pivot. Please fix the input and try again.");
        }
        for (int j = i + 1; j < A.rows(); ++j) {
            for (int k = i + 1; k < A.cols(); ++k) {
                A(j,k) -= A(i,k) * (A(j,i) / A(i,i));
                if (A(j, k) < EPS && A(j, k) > -1 * EPS)
                    A(j, k) = 0;
            }
            b(j,0) -= b(i,0) * (A(j,i) / A(i,i));
            if (A(j,0) < EPS && A(j,0) > -1 * EPS)
                A(j,0) = 0;
            A(j,i) = 0;
        }
    }

    // Back substitution
    Mat x(b.rows(), 1);
    x((x.rows() - 1),0) = b((x.rows() - 1),0) / A((x.rows() - 1),(x.rows() - 1));
    if (x((x.rows() - 1),0) < EPS && x((x.rows() - 1),0) > -1 * EPS)
        x((x.rows() - 1),0) = 0;
    
    for (int i = x.rows() - 2; i >= 0; --i) {
        int sum = 0;
        for (int j = i + 1; j < x.rows(); ++j) {
            sum += A(i,j) * x(j,0);
        }
        x(i,0) = (b(i,0) - sum) / A(i,i);
        if (x(i,0) < EPS && x(i,0) > -1 * EPS)
            x(i,0) = 0;
    }

    return x;
}

std::tuple<Mat, Mat, Mat> SVD_Decompose(Mat) {

}

std::tuple<Mat, Mat> LU_Decompose(Mat A) {
    int n = A.cols();
    Mat lower(n, n);
    Mat upper(n, n);
    if (A(0,0) == 0) {
        A.swapRows(2, 1);
    }
    for (int i = 0; i < n; i++) {
        for (int k = i; k < n; k++) { 
            float sum = 0;
            for (int j = 0; j < i; j++)
                sum += (lower(i, j) * upper(j, k));
            upper(i,k) = A(i,k) - sum;
            
               
        }

        for (int k = i; k < n; k++) {
            if (i == k) {
                lower(i, i) = 1; 
            }
            else {
                float sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (lower(k,j) * upper(j,i));
                lower(k,i) = (A(k,i) - sum) / upper(i,i);
            }
        }
    }
    return std::make_tuple(lower,upper);

}

double det(Mat A) {

}





