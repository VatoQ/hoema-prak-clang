#include "matrix.h"
#include "vector.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

double f1(const Vector* x);
double f2(const Vector* x);

int main(int argc, char* argv[])
{
    // symmetric matrix
    double m_vals[] = {
        2.0, -0.5, 0.1, -0.5, 1.0, -0.1, 0.1, -0.1, 2.0,
    };
    Matrix M = Matrix_new_vals(3, 3, m_vals);
    printf("M:\n");
    Matrix_print(&M);

    Matrix M_inv = Matrix_new(0, 0, 0.0);
    Matrix_inverse(&M_inv, &M);

    printf("\nM_inv: \n");
    Matrix_print(&M_inv);

    Matrix res = Matrix_new(0, 0, 0.0);

    Matrix_Matrix_dot(&res, &M, &M_inv);

    printf("\nM @ M_inv: \n");
    Matrix_print(&res);

    Matrix_free(&M_inv);
    Matrix_free(&M);

    return EXIT_SUCCESS;
}

double f1(const Vector* x)
{
    const double a = x->values[0];
    const double b = x->values[1];

    return sin(a * b) + sin(a) + cos(b);
}

double f2(const Vector* x)
{
    const double a = x->values[0];
    const double b = x->values[1];
    const double c = x->values[2];

    // -(2x_1^2 - 2x_1x_2 + x_2^2 + x_3^2 - 2x_1 - 4x_3)
    return -2 * (a * a - a * b - a - 2 * c) - b * b - c * c;
}
