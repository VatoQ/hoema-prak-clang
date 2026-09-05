// #define DEBUG
#include "../include/real_functions.h"
#include "../include/int_functions.h"
#include <stddef.h>
#include <stdlib.h>

static size_t MAX_STEP = 8;

real_t rfn_abs(real_t x)
{
    return (x >= 0) ? x : -x;
}

static real_t _rfn_cos_sin_worker(real_t x, int_t offset)
{
    real_t sign = 1;
    real_t sum  = 0;

    for (size_t k = 0; k < MAX_STEP; k++)
    {
        sum +=
          sign * rfn_pow_i(x, k * 2 + offset) / ifn_factorial(k * 2 + offset);
        sign *= -1;
    }

    return sum;
}

real_t rfn_cos(real_t x)
{
    x = rfn_mod(x, 2 * PI);

    if (x <= PI * 0.5)
    {
        return _rfn_cos_sin_worker(x, 0);
    }

    else if (x <= PI)
    {
        return -_rfn_cos_sin_worker(PI - x, 0);
    }

    else if (x < PI * 1.5)
    {
        return -_rfn_cos_sin_worker(x - PI, 0);
    }

    return _rfn_cos_sin_worker(2 * PI - x, 0);
}

real_t rfn_cotan(real_t x)
{
    return rfn_cos(x) / rfn_sin(x);
}

real_t rfn_exp(real_t x)
{
    real_t sum = 0.0;

    for (size_t k = 0; k < 20; k++)
    {
        sum += rfn_pow_i(x, k) / ifn_factorial(k);
    }
    return sum;
}

real_t rfn_mod(real_t a, real_t m)
{
    real_t k = 0;

    if (a > 0 && m > 0)
    {
        while (k + m <= a)
        {
            k += m;
        }
        return a - k;
    }
    if (a < 0 && m < 0)
    {
        while (k + m >= a)
        {
            k += m;
        }
        return a - k;
    }
    if (a < 0 && m > 0)
    {
        while (k - m >= a)
        {
            k -= m;
        }
        return k - a;
    }
    if (a > 0 && m < 0)
    {
        while (k - m <= a)
        {
            k -= m;
        }
        return k - a;
    }
    return 0;
}

real_t rfn_pow_i(real_t x, int_t n)
{
    if (n < 0 && rfn_abs(x) < EPS) // prevent division by zero in the n < 0 case
    {
        return ieee_nan();
    }
    if (n < 0)
    {
        x = 1 / x;
        n = -n;
    }
    if (n == 0)
    {
        return 1;
    }

    real_t y = 1;
    while (n > 1)
    {
        if (n % 2 == 1)
        {
            y *= x;
            n--;
        }
        x *= x;
        n >>= 1;
    }
    return x * y;
}

real_t rfn_sin(real_t x)
{
    x = rfn_mod(x, 2 * PI);

    if (rfn_abs(x) < PI * 0.5)
    {
        real_t sign = (x < 0) ? -1 : 1;
        return sign * _rfn_cos_sin_worker(rfn_abs(x), 1);
    }
    if (rfn_abs(x) < PI)
    {
        real_t sign = (x < 0) ? -1 : 1;
        return sign * _rfn_cos_sin_worker(PI - rfn_abs(x), 1);
    }
    if (rfn_abs(x) < PI * 1.5)
    {
        real_t sign = (x < 0) ? 1 : -1;
        return sign * _rfn_cos_sin_worker(rfn_abs(x) - PI, 1);
    }
    real_t sign = (x < 0) ? 1 : -1;
    return sign * _rfn_cos_sin_worker(2 * PI - rfn_abs(x), 1);
}

real_t rfn_tan(real_t x)
{
    return rfn_sin(x) / rfn_cos(x);
}
