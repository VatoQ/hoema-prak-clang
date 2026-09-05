#include "../include/complex_functions.h"
#include "../include/real_functions.h"

complex_t cfn_exp(real_t phi)
{
    return rfn_cos(phi) + rfn_sin(phi) * I;
}

complex_t cfn_pow_i(complex_t z, int_t n)
{
    if (n < 0 && cabs(z) < EPS) // prevent division by zero in the n < 0 case
    {
        return ieee_nan() + ieee_nan() * I;
    }
    if (n < 0)
    {
        z = 1 / z;
        n = -n;
    }
    if (n == 0)
    {
        return 1;
    }

    complex_t y = 1;
    while (n > 1)
    {
        if (n % 2 == 1)
        {
            y *= z;
            n--;
        }
        z *= z;
        n >>= 1;
    }
    return z * y;
}
