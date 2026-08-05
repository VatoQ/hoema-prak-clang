#include "../include/logging.h"
#define PARALLEL_THRESHOLD 1000
#include "../include/matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define runs 5
#define ignore 1

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
    const size_t m                  = 800;
    const size_t n                  = m;
    const size_t FLOPS              = (size_t)(8.0 / 3.0 * n * n * n);
    Matrix M                        = Matrix_new_random_symmetric(m);
    Matrix M_inv                    = Matrix_zeros_like(&M);
    double runtimes[runs - ignore]  = { 0 };
    double flops_arr[runs - ignore] = { 0 };

    printf("Benchmark Matrix_inverse()...\n");

    double rt_sum = 0.0;
    double f_sum  = 0.0;
    for (int r = 0; r < runs; r++)
    {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        Matrix_inverse(&M_inv, &M);
        clock_gettime(CLOCK_MONOTONIC, &end);
        long seconds = end.tv_sec - start.tv_sec;
        long nanos   = end.tv_nsec - start.tv_nsec;

        double elapsed         = seconds + nanos * 1e-9;
        double flop_per_second = (double)FLOPS / elapsed;
        printf("Run %d: GFLOP/s = %lf", r, flop_per_second / 1000000000);
        printf(", Runtime = %lf", r, elapsed);
        if (r >= ignore)
        {
            runtimes[r - ignore]  = elapsed;
            flops_arr[r - ignore] = flop_per_second;
            rt_sum += elapsed;
            f_sum += flop_per_second;
            printf("\n");
        }
        else
        {
            printf(" (not in average)\n");
        }
    }
    rt_sum /= (runs - ignore);
    f_sum /= (runs - ignore);

    printf("GFLOP/s (average): %lf\n", f_sum / 1000000000);
    printf("Runtime (average): %lf\n", rt_sum);

    return EXIT_SUCCESS;
}
