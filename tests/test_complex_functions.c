#include <complex.h>
#include <math.h>
#define DEBUG
#include "../include/complex_functions.h"
#include "../include/config.h"
#include "acutest.h"
#include <stddef.h>

void test_exp(void)
{
    const size_t N  = 50;
    const real_t dx = 2 * PI / N;
    real_t phi      = 0;
    for (size_t i = 0; i < N; i++)
    {
        complex_t z_actual   = cfn_exp(phi);
        complex_t z_expected = cexp(phi * I);
        TEST_CHECK(fabs(creal(z_actual) - creal(z_expected)) < EPS);
        TEST_CHECK(fabs(cimag(z_actual) - cimag(z_expected)) < EPS);
    }
}

TEST_LIST = {
    {
      "test_exp",
      test_exp,
    },
    { NULL, NULL },
};
