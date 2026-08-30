// #define DEBUG
#include "../include/fourier.h"
#include "../include/vector.h"
#include "acutest.h"
#include <complex.h>
#include <math.h>
#include <stddef.h>

void test_round_trip_random(void)
{
    const size_t N = 100;
    DataPoints dp  = DataPoints_new(N);
    Vector a       = Vector_new_random_normal(N, 0, 1, Real);
    double norm_a2 = 1 / Vector_norm(&a) / 2;
    Vector b       = Vector_new_random_normal(N, 0, 1, Real);
    double norm_b2 = 1 / Vector_norm(&b) / 2;

    Vector_scale(&a, &norm_a2);
    Vector_scale(&b, &norm_b2);
    ACCESS_VOID(real_t, a_values, a.values);
    ACCESS_VOID(real_t, b_values, b.values);
    for (size_t i = 0; i < N; i++)
    {
        dp.data[i] = a_values[i] + b_values[i] * I;
    }
    Vector_free(&a);
    Vector_free(&b);
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

void test_impulse_response(void)
{
    // An impulse in the time domain must produce a constant spectrum.
    const size_t N  = 64;
    DataPoints dp   = DataPoints_new(N);
    DataPoints freq = DataPoints_new(N);
    DataPoints tmp  = DataPoints_new(N);

    for (size_t i = 0; i < N; i++)
    {
        dp.data[i] = (i == 0) ? 1.0 + 0.0 * I : 0.0 + 0.0 * I;
    }

    fourier_transform(&freq, &dp, TO);
    fourier_transform(&tmp, &freq, FROM);

    for (size_t i = 0; i < N; i++)
    {
        DEBUG_PRINT("freq.data[%zu] = %f\n", i, creal(freq.data[i]));
        TEST_CHECK(fabs(creal(freq.data[i]) - 1 / sqrt(N)) < EPS);
        TEST_CHECK(fabs(cimag(freq.data[i]) - 0.0) < EPS);
        TEST_CHECK(fabs(creal(dp.data[i]) - creal(tmp.data[i])) < EPS);
        TEST_CHECK(fabs(cimag(dp.data[i]) - cimag(tmp.data[i])) < EPS);
    }
}

void test_single_frequency(void)
{
    const size_t N  = 64;
    const size_t k  = 5;
    DataPoints dp   = DataPoints_new(N);
    DataPoints freq = DataPoints_new(N);

    for (size_t n = 0; n < N; n++)
    {
        double phase = 2.0 * PI * k * n / N;
        dp.data[n]   = cos(phase) + sin(phase) * I;
    }

    fourier_transform(&freq, &dp, TO);

    for (size_t i = 0; i < N; i++)
    {
        DEBUG_PRINT("freq.data[%zu] = %f\n", i, cabs(freq.data[i]));
        if (i == N - k)
        {
            TEST_CHECK(cabs(freq.data[i]) > EPS);
        }
        else
        {
            TEST_CHECK(cabs(freq.data[i]) < EPS);
        }
    }

    DataPoints_free(&dp);
    DataPoints_free(&freq);
}

void test_real_signal_symmetry(void)
{
    const size_t N  = 64;
    DataPoints dp   = DataPoints_new(N);
    DataPoints freq = DataPoints_new(N);

    for (size_t i = 0; i < N; i++)
    {
        dp.data[i] = (double)i + 0.0 * I;
    }

    fourier_transform(&freq, &dp, TO);

    for (size_t i = 1; i < N; i++)
    {
        complex_t a = freq.data[i];
        complex_t b = freq.data[N - i];
        DEBUG_PRINT("a: ");
        DEBUG_CODE(PRINT_COMPLEX(a); puts(""));
        DEBUG_PRINT("b: ");
        DEBUG_CODE(PRINT_COMPLEX(b); puts(""));
        TEST_CHECK(fabs(creal(a) - creal(b)) < EPS);
        TEST_CHECK(fabs(cimag(a) + cimag(b)) < EPS);
    }
    DataPoints_free(&dp);
    DataPoints_free(&freq);
}

void test_energy_preservation(void)
{
    const size_t N  = 128;
    DataPoints dp   = DataPoints_new(N);
    DataPoints freq = DataPoints_new(N);

    Vector v = Vector_new_random_normal(N, 0, 1, Complex);
    ACCESS_VOID(complex_t, v_values, v.values);

    for (size_t i = 0; i < N; i++)
    {
        dp.data[i] = v_values[i];
    }
    Vector_free(&v);

    fourier_transform(&freq, &dp, TO);

    real_t E_time = 0.0;
    real_t E_freq = 0.0;

    for (size_t i = 0; i < N; i++)
    {
        E_time += creal(dp.data[i]) * creal(dp.data[i]) +
                  cimag(dp.data[i]) * cimag(dp.data[i]);
        E_freq += creal(freq.data[i]) * creal(freq.data[i]) +
                  cimag(freq.data[i]) * cimag(freq.data[i]);
    }
    TEST_CHECK(fabs(E_time - E_freq) < EPS);

    DataPoints_free(&dp);
    DataPoints_free(&freq);
}

TEST_LIST = { { "test_impulse_response", test_impulse_response },
              { "test_round_trip_random", test_round_trip_random },
              { "test_single_frequency", test_single_frequency },
              { "test_real_signal_symmetry", test_real_signal_symmetry },
              { "test_energy_preservation", test_energy_preservation },
              { NULL, NULL } };
