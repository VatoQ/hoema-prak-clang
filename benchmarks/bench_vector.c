#include "../include/prng.h"
#define PARALLEL_THRESHOLD 200000
#include "../include/vector.h"
#include "performance_tracker.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    double* real_target;
    Vector* target;
    const Vector *u, *v;
    double lambda;
} compute_args;

void add_callback(void* ctx)
{
    compute_args* a = ctx;
    Vector_add(a->target, a->v);
}
void sub_callback(void* ctx)
{
    compute_args* a = ctx;
    Vector_sub(a->target, a->v);
}

void scale_callback(void* ctx)
{
    compute_args* a = ctx;
    Vector_scale(a->target, a->lambda);
}

void dot_callback(void* ctx)
{
    compute_args* a = ctx;
    Vector_dot(a->real_target, a->u, a->v);
}

int main(int argc, char* argv[])
{
    const size_t runs   = 10;
    const size_t ignore = 3;
    PRNG_State prng     = PRNG_State_init(NO_SEED);
    const size_t N      = 25000000;
    const Vector u      = Vector_new_random_normal(N, 0, 1);
    const Vector v      = Vector_new_random_normal(N, 0, 1);
    Vector sink         = Vector_zeros(N);
    const double lambda = PRNG_State_random_double(&prng);
    double real_target  = 0.0;

    compute_args cargs = { 0 };
    cargs.target       = &sink;
    cargs.v            = &v;
    cargs.u            = &u;
    cargs.lambda       = lambda;
    cargs.real_target  = &real_target;

    callable_t add = { add_callback, &cargs };
    track_performance(runs, ignore, N, N, "Vector_add()", add);

    callable_t sub = { sub_callback, &cargs };
    track_performance(runs, ignore, N, N, "Vector_sub()", sub);

    callable_t scale = { scale_callback, &cargs };
    track_performance(runs, ignore, N, N, "Vector_scale()", scale);

    callable_t dot = { dot_callback, &cargs };
    track_performance(runs, ignore, 2 * N, N, "Vector_dot()", dot);

    return EXIT_SUCCESS;
}
