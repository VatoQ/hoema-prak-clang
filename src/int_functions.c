#include "../include/int_functions.h"
#include <stddef.h>

int_t ifn_factorial(int_t n)
{
    int_t f = 1;
    for (size_t i = 1; i <= n; i++)
    {
        int_t right_power = 1 << (i + 1);
        f *= i;
    }
    return f;
}

int_t ifn_pow(int_t b, int_t e)
{
    int_t a = 1;
    for (size_t i = 0; i < e; i++)
    {
        a *= b;
    }
    return a;
}

int_t ifn_pow_mod(int_t b, int_t e, int_t m)
{
    if (m == 1)
    {
        return 0;
    }
    int_t result = 1;
    b %= m;
    while (e > 0)
    {
        if (e % 2 == 1)
        {
            result = (result * b) % m;
        }
        e = e >> 1;
    }
    return result;
}
