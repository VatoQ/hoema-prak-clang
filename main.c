#include "logging.h"
#include "matrix.h"
#include "vector.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
    Log_set_log_mode(MATH_LOG_STDERR);
    double vals1[] = { 1.0,  -0.25, 0.01, -0.2, -0.25, 1.2,  0.05, 0.12,
                       0.01, 0.05,  0.45, 0.23, -0.2,  0.12, 0.23, 1.0 };
    double vals2[] = { -0.23, 0.5,   -0.22, -0.07, 0.5,   0.921, 0.111, 0.236,
                       -0.22, 0.111, 3.1,   0.512, -0.07, 0.236, 0.512, 4.5 };

    Matrix M1 = Matrix_new_vals(4, 4, vals1);
    Matrix M2 = Matrix_new_vals(4, 4, vals2);

    Matrix M3 = Matrix_new(0, 0, 0);

    int status = Matrix_Matrix_dot_jordan(&M3, &M1, &M2);

    Matrix_print(&M1);
    printf("@\n");
    Matrix_print(&M2);
    printf("=\n");
    Matrix_print(&M3);
    return 0;
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
