#ifndef PERFORMANCE_TRACKER_H
#define PERFORMANCE_TRACKER_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct
{
    void (*fn)(void* ctx);
    void* ctx;
} callable_t;

void track_performance(const size_t runs,
                       const size_t ignore,
                       const size_t FLOPS,
                       const size_t problem_size,
                       const char* name,
                       callable_t call)
{

    double* runtimes  = calloc(runs - ignore, sizeof(double));
    double* flops_arr = calloc(runs - ignore, sizeof(double));

    printf("Benchmark %s, problem size %zu...\n", name, problem_size);
    double rt_sum = 0.0;
    double f_sum  = 0.0;
    for (int r = 0; r < runs; r++)
    {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        call.fn(call.ctx);
        clock_gettime(CLOCK_MONOTONIC, &end);
        long seconds           = end.tv_sec - start.tv_sec;
        long nanos             = end.tv_nsec - start.tv_nsec;
        double elapsed         = seconds + nanos * 1e-9;
        double flop_per_second = (double)FLOPS / elapsed;
        if (r < 10)
        {
            printf("Run  ");
        }
        else
        {
            printf("Run ");
        }
        printf("%d: GFLOP/s = %.3lf", r, flop_per_second / 1000000000);
        printf(", Runtime = %.7lf", elapsed);
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

    printf("GFLOP/s (average): %.3lf\n", f_sum / 1000000000);
    printf("Runtime (average): %.3lf\n", rt_sum);
    printf("\n");

    free(runtimes);
    free(flops_arr);
}

#endif // PERFORMANCE_TRACKER_H
