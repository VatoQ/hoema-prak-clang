#include "../include/logging.h"
#include "../include/matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv)
{
    // Inverse:
    // M * N * ( 6n )  + M * (2N + 1)
    // + M * N * (6n)   + M
    // 2 (N^2 * M * 6) + M (2N + 2)
    // 12M*N^2 + 2MN+2M
    // 2M ( 6N^2 + N + 1)
    // M = N:
    // 2N(6N^2 + N + 1)
    const size_t m     = 800;
    const size_t n     = m;
    const size_t FLOPS = 2 * m * (6 * n * n + n + 1);
    Matrix M           = Matrix_new_random_symmetric(m);
    Matrix M_inv       = Matrix_zeros_like(&M);

    struct timespec start, end;
    printf("Enter Matrix_invers()...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    Matrix_inverse(&M_inv, &M);
    clock_gettime(CLOCK_MONOTONIC, &end);
    long seconds = end.tv_sec - start.tv_sec;
    long nanos   = end.tv_nsec - start.tv_nsec;

    double elapsed         = seconds + nanos * 1e-9;
    double flop_per_second = (double)FLOPS / elapsed;

    printf("GFLOP/s: %lf\n", flop_per_second / 1000000000);

    return EXIT_SUCCESS;
}
