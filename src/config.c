#include "../include/config.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>

size_t detect_L3_cache_size(void)
{
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");

    if (!f)
    {
        return 0;
    }
    char buf[64];
    if (!fgets(buf, sizeof(buf), f))
    {
        fclose(f);
        return 0;
    }

    fclose(f);

    size_t size = strtoull(buf, NULL, 10);

    if (strstr(buf, "K"))
    {
        size *= 1024;
    }

    if (strstr(buf, "M"))
    {
        size *= 1024 * 1024;
    }

    return size;
}

parallel_thresholds PARALLEL_THRESHOLDS = { 15000, 1000000 };

void config_init(void)
{
    const size_t L3 = detect_L3_cache_size();
    if (L3 != 0)
    {
        PARALLEL_THRESHOLDS.upper = L3 * 2;
    }
}

size_t elem_size(DataType dt)
{
    switch (dt)
    {
        case Int:
        {
            return sizeof(int_t);
        }
        case Real:
        {
            return sizeof(real_t);
        }
        case Complex:
        {
            return sizeof(complex_t);
        }
        default:
        {
            return 1;
        }
    }
}

static inline void set_int_limit(int_t* out, LimitType lt)
{
    switch (lt)
    {
        case Min:
        {
            if (sizeof(int_t) == sizeof(int))
                *out = INT_MIN;
            else if (sizeof(int_t) == sizeof(long))
                *out = LONG_MIN;
            else if (sizeof(int_t) == sizeof(long long))
                *out = LLONG_MIN;
            break;
        }
        case Max:
        {
            if (sizeof(int_t) == sizeof(int))
                *out = INT_MAX;
            else if (sizeof(int_t) == sizeof(long))
                *out = LONG_MAX;
            else if (sizeof(int_t) == sizeof(long long))
                *out = LLONG_MAX;
            break;
        }
        default:
        {
            *out = 0;
            break;
        }
    }
}

static inline void set_real_limit(real_t* out, LimitType lt)
{
    switch (lt)
    {
        case Min:
        {
            if (sizeof(real_t) == sizeof(float))
                *out = -FLT_MAX;
            else if (sizeof(real_t) == sizeof(double))
                *out = -DBL_MAX;
            else if (sizeof(real_t) == sizeof(long double))
                *out = -LDBL_MAX;
            break;
        }
        case Max:
        {
            if (sizeof(real_t) == sizeof(float))
                *out = FLT_MAX;
            else if (sizeof(real_t) == sizeof(double))
                *out = DBL_MAX;
            else if (sizeof(real_t) == sizeof(long double))
                *out = LDBL_MAX;
            break;
        }
        case Infty:
        {
            *out = INFINITY;
            break;
        }
        default:
        {
            *out = 0;
        }
    }
}

static inline void set_complex_limit(complex_t* out, LimitType lt)
{
    real_t rmin, rmax;

    // reuse real limit logic for the real part
    set_real_limit(&rmin, Min);
    set_real_limit(&rmax, Max);

    switch (lt)
    {
        case Min:
        {
            *out = rmin + 0.0 * I;
            break;
        }

        case Max:
        {
            *out = rmax + 0.0 * I;
            break;
        }

        case Infty:
        {
            *out = INFINITY + INFINITY * I;
            break;
        }
        default:
        {
            *out = 0;
        }
    }
}

int get_limit(void* target, DataType dt, LimitType lt)
{
    switch (dt)
    {
        case Int:
        {
            set_int_limit((int_t*)target, lt);
            break;
        }
        case Real:
        {
            set_real_limit((real_t*)target, lt);
            break;
        }
        case Complex:
        {
            set_complex_limit((complex_t*)target, lt);
            break;
        }
        default:
        {
            return CONFIG_ERROR;
        }
    }
    return CONFIG_SUCCESS;
}
