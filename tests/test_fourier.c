#include "../include/fourier.h"
#include "../include/vector.h"
#include "acutest.h"
#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

void test_data_points_new(void)
{
    const size_t N = 100;
    DataPoints dp  = DataPoints_new(N);
    Vector a       = Vector_new_random_normal(N, 0, 1, Real);
    double norm_a2 = Vector_norm(&a) * 2;
    Vector b       = Vector_new_random_normal(N, 0, 1, Real);
    double norm_b2 = Vector_norm(&b) * 2;
    norm_a2        = 1 / norm_a2;
    norm_b2        = 1 / norm_b2;

    Vector_scale(&a, &norm_a2);
    Vector_scale(&b, &norm_b2);
    ACCESS_VOID(real_t, a_values, a.values);
    ACCESS_VOID(real_t, b_values, b.values);
    for (size_t i = 0; i < N; i++)
    {
        dp.data[i] = a_values[i] + b_values[i] * I;
    }
    DataPoints freq = DataPoints_new(N);
    DataPoints tmp  = DataPoints_new(N);

    fourier_transform(&freq, &dp, TO);
    fourier_transform(&tmp, &freq, FROM);

    for (size_t i = 0; i < N; i++)
    {
        TEST_CHECK(fabs(creal(dp.data[i]) - creal(tmp.data[i])) < EPS);
        TEST_CHECK(fabs(cimag(dp.data[i]) - cimag(tmp.data[i])) < EPS);
    }
    DataPoints_free(&dp);
    DataPoints_free(&freq);
    DataPoints_free(&tmp);
}

TEST_LIST = { { "test_data_points_new", test_data_points_new },
              { NULL, NULL } };
