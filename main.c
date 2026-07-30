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
    double m_init_vals[] = { 2.0, 1.0, 1.0, 2.0 };
    const size_t m       = 2;
    const size_t n       = 2;
    Matrix M             = Matrix_new_vals(m, n, m_init_vals);

    double v_init_vals[] = { 2.0, 3.0 };
    Vector v             = Vector_new_vals(n, v_init_vals);
    Vector target        = Vector_new(0, 0.0);

    int status = Matrix_Vector_dot(&target, &M, &v);

    if (status == MATRIX_MATH_SUCCESS)
    {
        Vector_print(&target);
        printf("\n");
    }
    else if (status == MATRIX_BASIC_ERROR)
    {
        perror("There has been an error during calculation\n");
    }
    else if (status == MATRIX_DIMENSION_ERROR)
    {
        perror("There was a dimension mismatch\n");
    }

    Matrix M_inv = Matrix_new(0, 0, 0.0);

    status = Matrix_inverse(&M_inv, &M);

    if (status == MATRIX_MATH_SUCCESS)
    {
        Matrix_print(&M_inv);
    }
    else if (status == MATRIX_BASIC_ERROR)
    {
        perror("There has been an error during calculation\n");
    }
    else if (status == MATRIX_DIMENSION_ERROR)
    {
        perror("There was a dimension mismatch\n");
    }

    Vector_free(&target);
    Matrix_free(&M);
    Matrix_free(&M_inv);
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
