#include <stdbool.h>
#define PARALLEL_THRESHOLD 1000
#include "../include/matrix.h"
#include "performance_tracker.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct
{
    Matrix* out;
    const Matrix* in;
} inverse_args;

void inverse_callback(void* ctx)
{
    inverse_args* a = ctx;
    Matrix_inverse(a->out, a->in);
}

typedef struct
{
    Vector* eigen;
    const Matrix* M;
    bool sort;

} eigen_args;

void eigen_callback(void* ctx)
{
    eigen_args* a = ctx;
    Matrix_eigvals(a->eigen, a->M, a->sort);
}

int main(int argc, char** argv)
{
    const size_t runs   = 6;
    const size_t ignore = 2;

    const size_t m     = 100;
    const size_t n     = m;
    const size_t FLOPS = (size_t)(8.0 / 3.0 * n * n * n);
    Matrix M           = Matrix_new_random_symmetric(m);
    Matrix M_inv       = Matrix_zeros_like(&M);
    inverse_args ia    = { &M_inv, &M };
    callable_t c       = { inverse_callback, &ia };

    track_performance(runs, ignore, FLOPS, m, "Matrix_inverse()", c);

    Vector eigenvalues       = Vector_new(m, 0);
    eigen_args ea            = { &eigenvalues, &M, false };
    callable_t call          = { eigen_callback, &ea };
    const size_t eigen_flops = 300 * m * m * m;

    track_performance(runs, ignore, eigen_flops, m, "Matrix_eigvals()", call);

    Matrix_free(&M_inv);
    return EXIT_SUCCESS;
}
