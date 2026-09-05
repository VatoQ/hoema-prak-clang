#define DEBUG
#include "../include/config.h"
#include "../include/real_functions.h"
#include "acutest.h"
#include <math.h>
#include <stddef.h>

void test_rfn_cos(void)
{
    enum
    {
        N = 50
    };
    real_t dx = -2 * 2 * PI / N;
    real_t x  = 0;

    for (size_t i = 0; i < N; i++)
    {
        real_t c1 = rfn_cos(x);
        real_t c2 = cos(x);
        DEBUG_PRINT("\ncos(%f) = (a) %f, (e)  %f", x, c1, c2);

        x += dx;

        TEST_CHECK(fabs(c1 - c2) < EPS);
    }
}

void test_rfn_cotan(void)
{
    enum
    {
        N = 50
    };
    real_t dx = 2 * PI / N;
    real_t x  = 0;

    for (size_t i = 0; i < N - 2; i++)
    {
        real_t c1 = rfn_cotan(x);
        real_t c2 = cos(x) / sin(x);
        if (c1 < 1e14 && c2 < 1e14)
        {
            DEBUG_PRINT("\ncotan(%f) = (e) %f, (a) %f", x, c1, c2);

            TEST_CHECK(fabs(c1 - c2) < EPS);
        }
        x += dx;
    }
}

void test_rfn_exp(void)
{
    enum
    {
        enum_bases_count = 3,
        enum_fvals_count = 20,
    };

    real_t bases[enum_bases_count] = { PI, 2, 4.2 };
    // TODO: widen range by better algorithm
    const real_t fval_min = -3, fval_max = 3,
                 dx = (fval_max - fval_min) / enum_fvals_count;

    real_t x = fval_min;
    for (size_t j = 0; j <= enum_fvals_count; j++)
    {
        real_t f_x_actual   = rfn_exp(x);
        real_t f_x_expected = exp(x);
        DEBUG_PRINT("\nexp(%f) = (a) %f, (e) %f", x, f_x_actual, f_x_expected);
        TEST_CHECK(fabs(f_x_actual - f_x_expected) < EPS);
        x += dx;
    }
}

void test_rfn_mod(void)
{
    /* a > 0, m > 0 */
    TEST_CHECK(fabs(rfn_mod(10.0, 3.0) - 1.0) < EPS);
    TEST_CHECK(fabs(rfn_mod(10.0, 5.0) - 0.0) < EPS);
    TEST_CHECK(fabs(rfn_mod(3.0, 10.0) - 3.0) < EPS);

    /* a < 0, m < 0 */
    TEST_CHECK(fabs(rfn_mod(-10.0, -3.0) - -1.0) < EPS);
    TEST_CHECK(fabs(rfn_mod(-10.0, -5.0) - 0.0) < EPS);
    TEST_CHECK(fabs(rfn_mod(-3.0, -10.0) - -3.0) < EPS);

    /* a < 0, m > 0 */
    TEST_CHECK(fabs(rfn_mod(-10.0, 3.0) - 1.0) < EPS);
    TEST_CHECK(fabs(rfn_mod(-10.0, 5.0) - 0.0) < EPS);
    TEST_CHECK(fabs(rfn_mod(-3.0, 10.0) - 3.0) < EPS);

    /* a > 0, m < 0 */
    TEST_CHECK(fabs(rfn_mod(10.0, -3.0) - -1.0) < EPS);
    TEST_CHECK(fabs(rfn_mod(10.0, -5.0) - 0.0) < EPS);
    TEST_CHECK(fabs(rfn_mod(3.0, -10.0) - -3.0) < EPS);

    /* zero / unsupported combinations */
    TEST_CHECK(fabs(rfn_mod(0.0, 3.0) - 0.0) < EPS);
    TEST_CHECK(fabs(rfn_mod(0.0, -3.0) - 0.0) < EPS);
    TEST_CHECK(fabs(rfn_mod(3.0, 0.0) - 0.0) < EPS);
    TEST_CHECK(fabs(rfn_mod(-3.0, 0.0) - 0.0) < EPS);
    TEST_CHECK(fabs(rfn_mod(0.0, 0.0) - 0.0) < EPS);
}

void test_rfn_pow_i(void)
{
    enum
    {
        enum_bases_count = 3,
        enum_xval_max    = 15,
        enum_xval_min    = -15,
    };

    real_t bases[enum_bases_count] = { PI, 2, 3.456789 };

    for (size_t i = 0; i < enum_bases_count; i++)
    {

        for (ssize_t j = enum_xval_min; j < enum_xval_max; j++)
        {
            real_t f_x_actual   = rfn_pow_i(bases[i], j);
            real_t f_x_expected = pow(bases[i], j);
            DEBUG_PRINT("\n%f**%ld = (a)%f, (e)%f",
                        bases[i],
                        j,
                        f_x_expected,
                        f_x_actual);
            TEST_CHECK(fabs(f_x_actual - f_x_expected) < EPS);
        }
    }
}

void test_rfn_sin(void)
{
    enum
    {
        N = 50
    };
    real_t dx = 2 * 2 * PI / N;
    real_t x  = 0;

    for (size_t i = 0; i < N; i++)
    {
        real_t c1 = rfn_sin(x);
        real_t c2 = sin(x);
        DEBUG_PRINT("\nsin(%f) = (a) %f, (e) = %f", x, c1, c2);

        x += dx;

        TEST_CHECK(fabs(c1 - c2) < EPS);
    }
}

void test_rfn_tan(void)
{
    enum
    {
        N = 50
    };
    real_t dx = 2 * PI / N;
    real_t x  = 0;

    for (size_t i = 0; i < N - 2; i++)
    {
        real_t c1 = rfn_tan(x);
        real_t c2 = tan(x);
        if (c1 < 1e14 && c2 < 1e14)
        {
            DEBUG_PRINT("\ntan(%f) = (a) %f, (e) = %f", x, c1, c2);

            TEST_CHECK(fabs(c1 - c2) < EPS);
        }
        x += dx;
    }
}

TEST_LIST = {
    { "test_rfn_cos", test_rfn_cos },     { "test_rfn_cotan", test_rfn_cotan },
    { "test_rfn_exp", test_rfn_exp },     { "test_rfn_mod", test_rfn_mod },
    { "test_rfn_pow_i", test_rfn_pow_i }, { "test_rfn_sin", test_rfn_sin },
    { "test_rfn_tan", test_rfn_tan },     { NULL, NULL },
};
