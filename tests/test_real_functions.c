// #define DEBUG
#include "../include/config.h"
#include "../include/real_functions.h"
#include "acutest.h"
#include <math.h>
#include <stddef.h>

#define N 50

void test_cos(void)
{
    real_t dx = -2 * 2 * PI / N;
    real_t x  = 0;

    for (size_t i = 0; i < N; i++)
    {
        real_t c1 = rfn_cos(x);
        real_t c2 = cos(x);
        DEBUG_PRINT("\nc1 = %f, c2 = %f", c1, c2);

        x += dx;

        TEST_CHECK(fabs(c1 - c2) < EPS);
    }
}

void test_sin(void)
{
    real_t dx = 2 * 2 * PI / N;
    real_t x  = 0;

    for (size_t i = 0; i < N; i++)
    {
        real_t c1 = rfn_sin(x);
        real_t c2 = sin(x);
        DEBUG_PRINT("\nc1 = %f, c2 = %f", c1, c2);

        x += dx;

        TEST_CHECK(fabs(c1 - c2) < EPS);
    }
}

TEST_LIST = {
    { "test_cos", test_cos },
    { "test_sin", test_sin },
    { NULL, NULL },
};
