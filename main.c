#include "matrix.h"
#include "vector.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

double f1(const Vector* x);
double f2(const Vector* x);
Vector f3(const Vector* x);

int main(int argc, char* argv[])
{
    double x_vals[] = { 1.0, 2.0, 0.0, 3.0 };
    Vector x        = Vector_new_vals(4, x_vals);

    Matrix jacobi = Matrix_new(0, 0, 0.0);

    int status = Matrix_jacobi(&jacobi, &x, f3);

    printf("Jacobi Matrix: \n");
    Matrix_print(&jacobi);
    Matrix_free(&jacobi);
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

Vector f3(const Vector* x)
{
    if (x->dim != 4)
    {
        perror("Dimension error in f3\n");
    }
    Vector f_x    = Vector_new(3, 0.0);
    f_x.values[0] = x->values[0] * x->values[1] * exp(x->values[2]);
    f_x.values[1] = x->values[1] * x->values[2] * x->values[3];
    f_x.values[2] = x->values[3];

    return f_x;
}
