#include "../include/logging.h"
#include "../include/matrix.h"
#include "../include/prng.h"
#include "../include/vector.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
    const size_t N = 5;
    const size_t n = N;
    Matrix M       = Matrix_new(N, N, 0);
    Matrix M_tmp   = Matrix_zeros_like(&M);
    Vector eigs    = Vector_zeros(N);
    Vector v       = Vector_zeros(N);
    for (size_t i = 0; i < n; i++)
    {
        Vector tmp    = Vector_new_random_normal(N, 0.0, 1);
        double lambda = 1.0 / Vector_norm(&tmp);

        Vector_scale(&tmp, lambda);
        Matrix_Vector_outer(&M_tmp, &tmp, &tmp);
        Matrix_add(&M, &M_tmp);
        double norm_M_inv = 1.0 / Matrix_norm(&M, FROBENIUS);
        Matrix_scale(&M, norm_M_inv);

        printf("vector %zu\n", i);
        Vector_print(&tmp);
        printf("\n");
        printf("Matrix %zu\n", i);
        Matrix_print(&M);
        Vector_free(&tmp);
    }

    Matrix_eigvals(&eigs, &M, false);
    double norm_M = Matrix_norm(&M, FROBENIUS);

    printf("\nEigenvalues:\n");
    Vector_print(&eigs);
    printf("\n");
    printf("||M||_2: %f\n", norm_M);
    Vector_free(&v);
    Vector_free(&eigs);
    Matrix_free(&M);
    Matrix_free(&M_tmp);

    return EXIT_SUCCESS;
}
