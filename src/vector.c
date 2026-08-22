#include <iso646.h>
#define _POSIX_C_SOURCE 200112L
#include "../include/logging.h"
#include "../include/prng.h"
#include "../include/vector.h"
#include <alloca.h>
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#ifndef PARALLEL_THRESHOLD
#define PARALLEL_THRESHOLD 1000
#endif // PARALLEL_THRESHOLD

static PRNG_State prng = { 0 };

static int _parallel_condition(size_t dim)
{
    const size_t lower = PARALLEL_THRESHOLDS.lower;
    const size_t upper = PARALLEL_THRESHOLDS.upper;

    return (dim >= lower) && (dim <= upper);
}

/**
 * @brief Check status of a given vector.
 *
 * @param[] v Vector to be inspected.
 * @return 1 if `v->values == NULL` or `v->dim == 0`, otherwise 0
 */
int Vector_is_empty(const Vector* v)
{
    return (v->values == NULL) || (v->dim == 0);
}

int Vector_dimension_match(const Vector* u, const Vector* v)
{
    return u->dim == v->dim;
}

void Vector_prepare_target(Vector* target, const size_t dim, const DataType dt)
{
    if (Vector_is_empty(target))
    {
        *target = Vector_zeros(dim, dt);
        return;
    }

    if (target->dim == dim)
    {
        return;
    }

    Log_log("Target vector shape mismatch, reallocating", LOG_RT_WARNING);

    Vector_free(target);
    *target = Vector_zeros(dim, dt);
}

static void _new_int(const size_t dim, void* v_values, const void* init_val)
{
    ACCESS_VOID(int_t, v_values_t, v_values);
#pragma omp simd
    for (size_t i = 0; i < dim; i++)
    {
        v_values_t[i] = *(int_t*)init_val;
    }
}

static void _new_real(const size_t dim, void* v_values, const void* init_val)
{
    ACCESS_VOID(real_t, v_values_t, v_values);
#pragma omp simd
    for (size_t i = 0; i < dim; i++)
    {
        v_values_t[i] = *(real_t*)init_val;
    }
}

static void _new_cmpl(const size_t dim, void* v_values, const void* init_val)
{
    ACCESS_VOID(complex_t, v_values_t, v_values);
#pragma omp simd
    for (size_t i = 0; i < dim; i++)
    {
        v_values_t[i] = *(complex_t*)init_val;
    }
}

static void (*_new_units[TYPE_COUNT])(const size_t dim,
                                      void* v_values,
                                      const void* init_val) = {
    [Int]     = _new_int,
    [Real]    = _new_real,
    [Complex] = _new_cmpl
};

Vector Vector_new(const size_t dim, const void* init_val, const DataType dt)
{
    Vector v = Vector_zeros(dim, dt);
    _new_units[dt](dim, v.values, init_val);
    return v;
}

static void _assign_int(void* restrict target,
                        const void* source,
                        const size_t dim)
{
    ACCESS_VOID(int_t, ptr_t, target);
    ACCESS_VOID(int_t, ptr_s, source);
    for (size_t i = 0; i < dim; i++)
    {
        ptr_t[i] = ptr_s[i];
    }
}

static void _assign_real(void* restrict target,
                         const void* source,
                         const size_t dim)
{
    ACCESS_VOID(real_t, ptr_t, target);
    ACCESS_VOID(real_t, ptr_s, source);
    for (size_t i = 0; i < dim; i++)
    {
        ptr_t[i] = ptr_s[i];
    }
}

static void _assign_cmpl(void* restrict target,
                         const void* source,
                         const size_t dim)
{
    ACCESS_VOID(complex_t, ptr_t, target);
    ACCESS_VOID(complex_t, ptr_s, source);
    for (size_t i = 0; i < dim; i++)
    {
        ptr_t[i] = ptr_s[i];
    }
}

static void (*_assign_units[TYPE_COUNT])(void* restrict target,
                                         const void* source,
                                         const size_t dim) = {
    [Int]     = _assign_int,
    [Real]    = _assign_real,
    [Complex] = _assign_cmpl,
};

Vector Vector_new_vals(const size_t dim,
                       const void* init_vals,
                       const DataType dt)
{
    Vector v = Vector_zeros(dim, dt);
    _assign_units[dt](v.values, init_vals, dim);
    return v;
}

static void _normal_int(const size_t dim,
                        const double mean,
                        const double variance,
                        void* v_values)
{
    ACCESS_VOID(int_t, vals_t, v_values);
    for (size_t n = 0; n < dim; n++)
    {
        double rnd = PRNG_State_normal(&prng, mean, variance);
        vals_t[n]  = rnd;
    }
}

static void _normal_real(const size_t dim,
                         const double mean,
                         const double variance,
                         void* v_values)
{
    ACCESS_VOID(real_t, vals_t, v_values);
    for (size_t n = 0; n < dim; n++)
    {
        double rnd = PRNG_State_normal(&prng, mean, variance);
        vals_t[n]  = rnd;
    }
}

static void _normal_cmpl(const size_t dim,
                         const double mean,
                         const double variance,
                         void* v_values)
{
    ACCESS_VOID(complex_t, vals_t, v_values);
    for (size_t n = 0; n < dim; n++)
    {
        double a    = PRNG_State_normal(&prng, mean, variance);
        double b    = PRNG_State_normal(&prng, mean, variance);
        complex_t z = a + b * I;
        vals_t[n]   = z;
    }
}

static void (*_normal_units[TYPE_COUNT])(const size_t dim,
                                         const double mean,
                                         const double variance,
                                         void* v_values) = {
    [Int]     = _normal_int,
    [Real]    = _normal_real,
    [Complex] = _normal_cmpl
};

Vector Vector_new_random_normal(const size_t dim,
                                const double mean,
                                const double variance,
                                const DataType dt)
{
    Vector_init_prng(NO_SEED);
    Vector v = Vector_zeros(dim, dt);
    _normal_units[dt](dim, mean, variance, v.values);
    return v;
}

static void _uniform_int(const size_t dim,
                         const double min,
                         const double max,
                         void* v_values)
{
    ACCESS_VOID(int_t, vals_t, v_values);
    for (size_t n = 0; n < dim; n++)
    {
        double rnd = PRNG_State_random_double_range(&prng, min, max);
        vals_t[n]  = rnd;
    }
}

static void _uniform_real(const size_t dim,
                          const double min,
                          const double max,
                          void* v_values)
{
    ACCESS_VOID(real_t, vals_t, v_values);
    for (size_t n = 0; n < dim; n++)
    {
        double rnd = PRNG_State_random_double_range(&prng, min, max);
        vals_t[n]  = rnd;
    }
}

static void _uniform_cmpl(const size_t dim,
                          const double min,
                          const double max,
                          void* v_values)
{
    ACCESS_VOID(complex_t, vals_t, v_values);
    for (size_t n = 0; n < dim; n++)
    {
        double a    = PRNG_State_random_double_range(&prng, min, max);
        double b    = PRNG_State_random_double_range(&prng, min, max);
        complex_t z = a + b * I;
        vals_t[n]   = z;
    }
}

static void (*_uniform_units[TYPE_COUNT])(const size_t dim,
                                          const double min,
                                          const double max,
                                          void* v_values) = {
    [Int]     = _uniform_int,
    [Real]    = _uniform_real,
    [Complex] = _uniform_cmpl
};

Vector Vector_new_random_uniform(const size_t dim,
                                 const double min,
                                 const double max,
                                 const DataType dt)
{
    Vector_init_prng(NO_SEED);
    Vector v = Vector_zeros(dim, dt);
    _uniform_units[dt](dim, min, max, v.values);
    return v;
}

Vector Vector_new_copy(const Vector* v)
{
    Vector res = Vector_zeros(v->dim, v->dt);
    memcpy(res.values, v->values, v->dim * elem_size(v->dt));
    res.dim = v->dim;

    return res;
}

static int _all_close_int_s(const size_t u_dim,
                            const void* u_values,
                            const void* v_values)
{
    ACCESS_VOID(int_t, ptr_u, u_values);
    ACCESS_VOID(int_t, ptr_v, v_values);
    for (size_t n = 0; n < u_dim; n++)
    {

        if (ptr_u[n] != ptr_v[n])
        {
            return 0;
        }
    }
    return 1;
}

static int _all_close_real_s(const size_t u_dim,
                             const void* u_values,
                             const void* v_values)
{

    ACCESS_VOID(real_t, ptr_u, u_values);
    ACCESS_VOID(real_t, ptr_v, v_values);

    for (size_t n = 0; n < u_dim; n++)
    {

        if (fabs(ptr_u[n] - ptr_v[n]) > EPS)
        {
            return 0;
        }
    }
    return 1;
}

static int _all_close_cmpl_s(const size_t u_dim,
                             const void* u_values,
                             const void* v_values)
{
    ACCESS_VOID(complex_t, ptr_u, u_values);
    ACCESS_VOID(complex_t, ptr_v, v_values);
    for (size_t n = 0; n < u_dim; n++)
    {

        if (fabs(creal(ptr_u[n]) - creal(ptr_v[n])) > EPS ||
            fabs(cimag(ptr_u[n]) - cimag(ptr_v[n])) > EPS)
        {
            return 0;
        }
    }
    return 1;
}

static int (*_all_close_units_s[TYPE_COUNT])(const size_t u_dim,
                                             const void* u_values,
                                             const void* v_values) = {
    [Int]     = _all_close_int_s,
    [Real]    = _all_close_real_s,
    [Complex] = _all_close_cmpl_s
};

static int _all_close_scalar(const Vector* u, const Vector* v)
{
    DataType dt = MAX(u->dt, v->dt);
    return _all_close_units_s[dt](u->dim, u->values, v->values);
}

static int _all_close_int_p(const size_t u_dim,
                            const void* u_values,
                            const void* v_values)
{
    int all_close = 1;
#pragma omp parallel for
    for (size_t n = 0; n < u_dim; n++)
    {
        ACCESS_VOID(int_t, ptr_u, u_values + n);
        ACCESS_VOID(int_t, ptr_v, v_values + n);

        if (*ptr_u != *ptr_v)
        {
            all_close = 0;
        }
    }
    return all_close;
}

static int _all_close_real_p(const size_t u_dim,
                             const void* u_values,
                             const void* v_values)
{
    int all_close = 1;
#pragma omp parallel for
    for (size_t n = 0; n < u_dim; n++)
    {
        ACCESS_VOID(real_t, ptr_u, u_values + n);
        ACCESS_VOID(real_t, ptr_v, v_values + n);

        if (fabs(*ptr_u - *ptr_v) > EPS)
        {
            all_close = 0;
        }
    }
    return all_close;
}

static int _all_close_cmpl_p(const size_t u_dim,
                             const void* u_values,
                             const void* v_values)
{
    int all_close = 1;
#pragma omp parallel for
    for (size_t n = 0; n < u_dim; n++)
    {
        ACCESS_VOID(complex_t, ptr_u, u_values + n);
        ACCESS_VOID(complex_t, ptr_v, v_values + n);

        if (fabs(creal(*ptr_u) - creal(*ptr_v)) > EPS ||
            fabs(cimag(*ptr_u) - cimag(*ptr_v)) > EPS)
        {
            all_close = 0;
        }
    }
    return all_close;
}

static int (*_all_close_units_p[TYPE_COUNT])(const size_t u_dim,
                                             const void* u_values,
                                             const void* v_values) = {
    [Int]     = _all_close_int_p,
    [Real]    = _all_close_real_p,
    [Complex] = _all_close_cmpl_p
};

static int _all_close_parallel(const Vector* u, const Vector* v)
{
    DataType dt = MAX(u->dt, v->dt);
    return _all_close_units_p[dt](u->dim, u->values, v->values);
}

int Vector_all_close(const Vector* u, const Vector* v)
{
    if (v->dim != u->dim)
    {
        return 0;
    }

    if (_parallel_condition(v->dim * elem_size(v->dt)))
    {
        return _all_close_parallel(u, v);
    }

    return _all_close_scalar(u, v);
}

void Vector_copy(Vector* target, const Vector* v)
{
    Vector_prepare_target(target, v->dim, v->dt);
    memcpy(target->values, v->values, v->dim * elem_size(v->dt));
    target->dim = v->dim;
}

Vector Vector_zeros_like(const Vector* v)
{
    Vector res = Vector_zeros(v->dim, v->dt);
    return res;
}

Vector Vector_zeros(const size_t dim, const DataType dt)
{
    Vector res = { 0 };
    if (dim > 0)
    {
        res.values = NULL;
        if (posix_memalign((void**)&res.values, 32, dim * elem_size(dt)) != 0)
        {
            Log_log("Error allocating data for Vector_zeros()", LOG_RT_ERROR);
            return (Vector){ 0 };
        }
        complex_t z = 0;
        _new_units[dt](dim, res.values, &z);
        res.dim = dim;
    }
    res.dt = dt;
    return res;
}

static Vector _ones_int(const size_t dim)
{
    int_t one = 1;
    return Vector_new(dim, &one, Int);
}

static Vector _ones_real(const size_t dim)
{
    real_t one = 1;
    return Vector_new(dim, &one, Real);
}

static Vector _ones_cmpl(const size_t dim)
{
    complex_t one = 1;
    return Vector_new(dim, &one, Complex);
}

static Vector (*_ones_units[TYPE_COUNT])(const size_t dim) = {
    [Int]     = _ones_int,
    [Real]    = _ones_real,
    [Complex] = _ones_cmpl,
};

Vector Vector_ones(const size_t dim, const DataType dt)
{
    return _ones_units[dt](dim);
}

static void _get_at_int(void* target, const void* v_values, const size_t index)
{
    ASSIGN_UNTYPED(int_t, target, v_values + index);
}

static void _get_at_real(void* target, const void* v_values, const size_t index)
{
    ASSIGN_UNTYPED(real_t, target, v_values + index);
}

static void _get_at_cmpl(void* target, const void* v_values, const size_t index)
{
    ASSIGN_UNTYPED(complex_t, target, v_values + index);
}

static void (*_get_at_units[TYPE_COUNT])(void* target,
                                         const void* v_values,
                                         const size_t index) = {
    [Int]     = _get_at_int,
    [Real]    = _get_at_real,
    [Complex] = _get_at_cmpl,
};

int Vector_at(void* target, const Vector* v, const size_t index)
{
    if (index >= v->dim)
    {
        Log_log("Dimension error in Vector_at()", LOG_RT_ERROR);
        return VECTOR_DIMENSION_ERROR;
    }

    _get_at_units[v->dt](target, v->values, index);

    return VECTOR_SUCCESS;
}

int Vector_get_at(double* target, const Vector* v, const size_t index)
{
    if (index >= v->dim)
    {
        Log_log("Dimension error in Vector_get_at()", LOG_RT_ERROR);
        return VECTOR_DIMENSION_ERROR;
    }
    _get_at_units[v->dt](target, v->values, index);

    return VECTOR_SUCCESS;
}

static void _set_at_int(void* target, const void* v_values, const size_t index)
{
    ACCESS_VOID(int_t, values_t, target);
    ASSIGN_TYPED(int_t, &values_t[index], v_values);
}

static void _set_at_real(void* target, const void* v_values, const size_t index)
{
    ACCESS_VOID(real_t, values_t, target);
    ASSIGN_TYPED(real_t, &values_t[index], v_values);
}

static void _set_at_cmpl(void* target, const void* v_values, const size_t index)
{
    ACCESS_VOID(complex_t, values_t, target);
    complex_t* val  = (complex_t*)v_values;
    values_t[index] = creal(*val) + cimag(*val) * I;
}

static void (*_set_at_units[TYPE_COUNT])(void* target,
                                         const void* v_values,
                                         const size_t index) = {
    [Int]     = _set_at_int,
    [Real]    = _set_at_real,
    [Complex] = _set_at_cmpl,
};

int Vector_set_item(Vector* v, const size_t index, const void* val)
{
    if (index >= v->dim)
    {
        Log_log("Dimension error in Vector_set_item()", LOG_RT_ERROR);
        return VECTOR_DIMENSION_ERROR;
    }
    _set_at_units[v->dt](v->values, val, index);
    return VECTOR_SUCCESS;
}

size_t Vector_get_dim(const Vector* v)
{
    return v->dim;
}

void Vector_free(Vector* v)
{
    free(v->values);
    v->values = NULL;
    v->dim    = 0;
}

static void _print_int(const size_t dim, const void* v_values)
{
    ACCESS_VOID(int_t, v_values_t, v_values);
    for (int i = 0; i < dim; i++)
    {
        int_t val = v_values_t[i];
        printf("%zu ", val);
    }
}

static void _print_real(const size_t dim, const void* v_values)
{
    ACCESS_VOID(real_t, v_values_t, v_values);
    for (int i = 0; i < dim; i++)
    {
        real_t val = v_values_t[i];
        printf("%f ", val);
    }
}

static void _print_cmpl(const size_t dim, const void* v_values)
{
    ACCESS_VOID(complex_t, v_values_t, v_values);
    for (int i = 0; i < dim; i++)
    {
        complex_t val = v_values_t[i];
        PRINT_COMPLEX(val);
        printf(" ");
    }
}

static void (*_print_units[TYPE_COUNT])(const size_t dim,
                                        const void* v_values) = {
    [Int]     = _print_int,
    [Real]    = _print_real,
    [Complex] = _print_cmpl,
};

void Vector_print(Vector* v)
{
    printf("( ");
    _print_units[v->dt](v->dim, v->values);
    printf(")");
}

static void _max_int_s(void* t, const size_t v_dim, const void* v_values)
{

    int_t max;
    get_limit(&max, Int, Min);
    ACCESS_VOID(int_t, v_values_t, v_values);

    for (size_t i = 0; i < v_dim; i++)
    {
        if (v_values_t[i] > max)
        {
            max = v_values_t[i];
        }
    }
    ASSIGN_UNTYPED(int_t, t, &max);
}

static void _max_int_p(void* t, const size_t v_dim, const void* v_values)
{

    int_t max;
    get_limit(&max, Int, Min);

    ACCESS_VOID(int_t, v_values_t, v_values);
#pragma omp parallel for
    for (size_t i = 0; i < v_dim; i++)
    {
        if (v_values_t[i] > max)
        {
            max = v_values_t[i];
        }
    }
    ASSIGN_UNTYPED(int_t, t, &max);
}

static void _max_real_s(void* t, const size_t v_dim, const void* v_values)
{

    real_t max;
    get_limit(&max, Real, Infty);
    max *= -1;
    ACCESS_VOID(real_t, v_values_t, v_values);

    for (size_t i = 0; i < v_dim; i++)
    {
        if (v_values_t[i] > max)
        {
            max = v_values_t[i];
        }
    }
    ASSIGN_UNTYPED(real_t, t, &max);
}

static void _max_real_p(void* t, const size_t v_dim, const void* v_values)
{

    real_t max;
    get_limit(&max, Real, Infty);
    max *= -1;
    ACCESS_VOID(real_t, v_values_t, v_values);

#pragma omp parallel for
    for (size_t i = 0; i < v_dim; i++)
    {
        if (v_values_t[i] > max)
        {
            max = v_values_t[i];
        }
    }
    ASSIGN_UNTYPED(real_t, t, &max);
}

static void _max_cmpl_s(void* t, const size_t v_dim, const void* v_values)
{

    real_t max;
    get_limit(&max, Real, Infty);
    max *= -1;
    ACCESS_VOID(complex_t, v_values_t, v_values);

    for (size_t i = 0; i < v_dim; i++)
    {
        real_t abs_z = cabs(v_values_t[i]);

        if (abs_z > max)
        {
            max = v_values_t[i];
        }
    }
    ASSIGN_UNTYPED(real_t, t, &max);
}

static void _max_cmpl_p(void* t, const size_t v_dim, const void* v_values)
{

    real_t max;
    get_limit(&max, Real, Min);
    ACCESS_VOID(complex_t, v_values_t, v_values);

#pragma omp parallel for
    for (size_t i = 0; i < v_dim; i++)
    {
        real_t abs_z = cabs(v_values_t[i]);

        if (abs_z > max)
        {
            max = v_values_t[i];
        }
    }
    ASSIGN_UNTYPED(real_t, t, &max);
}

static void (*_max_units_s[TYPE_COUNT])(void* t,
                                        const size_t v_dim,
                                        const void* v_values) = {
    [Int]     = _max_int_s,
    [Real]    = _max_real_s,
    [Complex] = _max_cmpl_s,

};

static void (*_max_units_p[TYPE_COUNT])(void* t,
                                        const size_t v_dim,
                                        const void* v_values) = {
    [Int]     = _max_int_p,
    [Real]    = _max_real_p,
    [Complex] = _max_cmpl_p,

};

static void _max_parallel(void* t, const Vector* v)
{
    _max_units_p[v->dt](t, v->dim, v->values);
}

static void _max_scalar(void* t, const Vector* v)
{
    _max_units_s[v->dt](t, v->dim, v->values);
}

void Vector_max(void* target, const Vector* v)
{
    if (_parallel_condition(v->dim * elem_size(v->dt)))
    {
        _max_parallel(target, v);
        return;
    }
    _max_scalar(target, v);
}

static void _min_int_s(void* t, const size_t v_dim, const void* v_values)
{

    int_t min;
    get_limit(&min, Int, Max);

    for (size_t i = 0; i < v_dim; i++)
    {
        ACCESS_VOID(int_t, ptr, v_values + i);
        if (*ptr < min)
        {
            min = *ptr;
        }
    }
    ASSIGN_UNTYPED(int_t, t, &min);
}

static void _min_int_p(void* t, const size_t v_dim, const void* v_values)
{

    int_t min;
    get_limit(&min, Int, Max);

#pragma omp parallel for
    for (size_t i = 0; i < v_dim; i++)
    {
        ACCESS_VOID(int_t, ptr, v_values + i);
        if (*ptr < min)
        {
            min = *ptr;
        }
    }
    ASSIGN_UNTYPED(int_t, t, &min);
}

static void _min_real_s(void* t, const size_t v_dim, const void* v_values)
{

    real_t min;
    get_limit(&min, Real, Infty);

    for (size_t i = 0; i < v_dim; i++)
    {
        ACCESS_VOID(real_t, ptr, v_values + i);
        if (*ptr < min)
        {
            min = *ptr;
        }
    }
    ASSIGN_UNTYPED(real_t, t, &min);
}

static void _min_real_p(void* t, const size_t v_dim, const void* v_values)
{

    real_t min;
    get_limit(&min, Real, Infty);

#pragma omp parallel for
    for (size_t i = 0; i < v_dim; i++)
    {
        ACCESS_VOID(real_t, ptr, v_values + i);
        if (*ptr < min)
        {
            min = *ptr;
        }
    }
    ASSIGN_UNTYPED(real_t, t, &min);
}

static void _min_cmpl_s(void* t, const size_t v_dim, const void* v_values)
{

    real_t min;
    get_limit(&min, Real, Infty);

    for (size_t i = 0; i < v_dim; i++)
    {
        ACCESS_VOID(complex_t, ptr, v_values + i);
        real_t abs_z = cabs(*ptr);

        if (abs_z < min)
        {
            min = *ptr;
        }
    }
    ASSIGN_UNTYPED(real_t, t, &min);
}

static void _min_cmpl_p(void* t, const size_t v_dim, const void* v_values)
{

    real_t min;
    get_limit(&min, Real, Min);

#pragma omp parallel for
    for (size_t i = 0; i < v_dim; i++)
    {
        ACCESS_VOID(complex_t, ptr, v_values + i);
        real_t abs_z = cabs(*ptr);
        if (abs_z < min)
        {
            min = *ptr;
        }
    }
    ASSIGN_UNTYPED(real_t, t, &min);
}

static void (*_min_units_s[TYPE_COUNT])(void* t,
                                        const size_t v_dim,
                                        const void* v_values) = {
    [Int]     = _min_int_s,
    [Real]    = _min_real_s,
    [Complex] = _min_cmpl_s,

};

static void (*_min_units_p[TYPE_COUNT])(void* t,
                                        const size_t v_dim,
                                        const void* v_values) = {
    [Int]     = _min_int_p,
    [Real]    = _min_real_p,
    [Complex] = _min_cmpl_p,

};

static void _min_parallel(void* t, const Vector* v)
{
    _min_units_p[v->dt](t, v->dim, v->values);
}

static void _min_scalar(void* t, const Vector* v)
{
    _min_units_s[v->dt](t, v->dim, v->values);
}

void Vector_min(void* target, const Vector* v)
{
    if (_parallel_condition(v->dim * elem_size(v->dt)))
    {
        _min_parallel(target, v);
        return;
    }
    _min_scalar(target, v);
    return;
}

static void _insertion_int(void* v_values, ssize_t low, size_t high)
{
    ACCESS_VOID(int_t, v_values_t, v_values);
    for (size_t i = low + 1; i <= high; ++i)
    {
        int_t key = v_values_t[i];
        ssize_t j = i - 1;

        while (j >= low && v_values_t[j] > key)
        {
            v_values_t[j + 1] = v_values_t[j];
            j--;
        }
        v_values_t[j + 1] = key;
    }
}

static void _insertion_real(void* v_values, ssize_t low, size_t high)
{
    ACCESS_VOID(real_t, v_values_t, v_values);
    for (size_t i = low + 1; i <= high; ++i)
    {
        real_t key = v_values_t[i];
        ssize_t j  = i - 1;

        while (j >= low && v_values_t[j] > key)
        {
            v_values_t[j + 1] = v_values_t[j];
            j--;
        }
        v_values_t[j + 1] = key;
    }
}

static void (*_insertion_units[TYPE_COUNT])(void* v_values,
                                            ssize_t low,
                                            size_t high) = {
    [Int]  = _insertion_int,
    [Real] = _insertion_real,
};

static void _insertion_sort(Vector* v, ssize_t low, size_t high)
{
    _insertion_units[v->dt](v->values, low, high);
}

static void _swap_int(void* v_values, const size_t i, const size_t j)
{
    ACCESS_VOID(int_t, v_values_t, v_values);
    int_t tmp     = v_values_t[i];
    v_values_t[i] = v_values_t[j];
    v_values_t[j] = tmp;
}

static void _swap_real(void* v_values, const size_t i, const size_t j)
{
    ACCESS_VOID(real_t, v_values_t, v_values);
    real_t tmp    = v_values_t[i];
    v_values_t[i] = v_values_t[j];
    v_values_t[j] = tmp;
}

static void (*_swap_units[3])(void* v_values,
                              const size_t i,
                              const size_t j) = { [Int]  = _swap_int,
                                                  [Real] = _swap_real };

static void _swap(Vector* v, const size_t i, const size_t j)
{
    _swap_units[v->dt](v->values, i, j);
}

static void _heapify_int(void* v_values,
                         size_t l,
                         size_t n,
                         size_t r,
                         ssize_t low,
                         size_t* largest)
{
    ACCESS_VOID(int_t, v_values_t, v_values);
    int_t val_low_l   = v_values_t[low + l];
    int_t val_low_lar = v_values_t[low + *largest];
    int_t val_low_r   = v_values_t[low + r];
    if (l < n && val_low_l > val_low_lar)
    {
        *largest = l;
    }
    if (r < n && val_low_r > val_low_lar)
    {
        *largest = r;
    }
}

static void _heapify_real(void* v_values,
                          size_t l,
                          size_t n,
                          size_t r,
                          ssize_t low,
                          size_t* largest)
{

    ACCESS_VOID(real_t, v_values_t, v_values);
    real_t val_low_l   = v_values_t[low + l];
    real_t val_low_lar = v_values_t[low + *largest];
    real_t val_low_r   = v_values_t[low + r];

    if (l < n && val_low_l > val_low_lar)
    {
        *largest = l;
    }
    if (r < n && val_low_r > val_low_lar)
    {
        *largest = r;
    }
}

static void (*_heapify_units[TYPE_COUNT])(void* v_values,
                                          size_t l,
                                          size_t n,
                                          size_t r,
                                          ssize_t low,
                                          size_t* largest) = {
    [Int]  = _heapify_int,
    [Real] = _heapify_real,
};

static void _heapify(Vector* v, ssize_t low, const size_t n, const size_t i)
{
    size_t largest = i;
    size_t l       = 2 * i + 1;
    size_t r       = 2 * i + 2;

    _heapify_units[v->dt](v->values, l, n, r, low, &largest);

    if (largest != i)
    {
        _swap(v, low + i, low + largest);
        _heapify(v, low, n, largest);
    }
}

static void _heapsort(Vector* v, ssize_t low, size_t high)
{
    const size_t n = high - low + 1;

    for (ssize_t i = n / 2 - 1; i >= 0; i--)
    {
        _heapify(v, low, n, i);
    }

    for (ssize_t i = n - 1; i > 0; i--)
    {
        _swap(v, low, low + i);

        _heapify(v, low, i, 0);
    }
}

static void _part_int(Vector* v, size_t* i, ssize_t low, size_t high)
{
    ACCESS_VOID(int_t, v_values, v->values);
    int_t pivot = v_values[high];
    *i          = low - 1;

    for (size_t j = low; j <= high - 1; j++)
    {
        if (v_values[j] < pivot)
        {
            (*i)++;
            _swap(v, *i, j);
        }
    }
}

static void _part_real(Vector* v, size_t* i, ssize_t low, size_t high)
{
    ACCESS_VOID(real_t, v_values, v->values);
    real_t pivot = v_values[high];
    *i           = low - 1;

    for (size_t j = low; j <= high - 1; j++)
    {
        if (v_values[j] < pivot)
        {
            (*i)++;
            _swap(v, *i, j);
        }
    }
}

static void (*_part_units[TYPE_COUNT])(Vector* v,
                                       size_t* i,
                                       ssize_t low,
                                       size_t high) = {
    [Int]  = _part_int,
    [Real] = _part_real,
};

static size_t _partition(Vector* v, ssize_t low, size_t high)
{
    size_t i;
    _part_units[v->dt](v, &i, low, high);

    _swap(v, i + 1, high);
    return i + 1;
}

static void _introsort(Vector* v, ssize_t low, size_t high, int depth_limit)
{
    if (low >= high)
    {
        return;
    }
    const size_t n = high - low + 1;
    if (n <= 32)
    {
        _insertion_sort(v, low, high);
        return;
    }
    if (depth_limit == 0)
    {
        _heapsort(v, low, high);
        return;
    }
    else
    {
        const size_t p = _partition(v, low, high);

        _introsort(v, low, p - 1, depth_limit - 1);
        _introsort(v, p + 1, high, depth_limit - 1);
    }
}

int Vector_sort_inplace(Vector* v)
{
    if (v->dt == Complex)
    {
        Log_log(
          "Math error in Vector_sort_inplace(), cannot sort complex numbers",
          LOG_RT_ERROR);
        return VECTOR_DIMENSION_ERROR;
    }
    if (v->dim == 0)
    {
        return VECTOR_SUCCESS;
    }
    int depth_limit = 2 * floor(log2(v->dim));
    _introsort(v, 0, v->dim - 1, depth_limit);
    return VECTOR_SUCCESS;
}

int Vector_sort(Vector* target, const Vector* v)
{
    Vector_copy(target, v);
    return Vector_sort_inplace(target);
}

void Vector_init_prng(const int seed)
{
    if (prng.aux == 0 && prng.state == 0)
    {
        prng = PRNG_State_init(seed);
    }
}

static real_t _norm_int(const Vector* v)
{
    int_t sum                = 0;
    int_t* restrict v_values = v->values;
    FMA(v->dim, sum, v_values[n], v_values[n]);
    return sqrt((real_t)sum);
}

static real_t _norm_real(const Vector* v)
{
    real_t sum                = 0;
    real_t* restrict v_values = v->values;
    FMA(v->dim, sum, v_values[n], v_values[n]);
    return sqrt(sum);
}

static real_t _norm_cmpl(const Vector* v)
{
    complex_t sum                = 0;
    complex_t* restrict v_values = v->values;
    FMA(v->dim, sum, v_values[n], creal(v_values[n]) - cimag(v_values[n]) * I);
    return sqrt(creal(sum));
}

static real_t (*_norm_units[TYPE_COUNT])(const Vector* v) = {
    [Int]     = _norm_int,
    [Real]    = _norm_real,
    [Complex] = _norm_cmpl,
};

real_t Vector_norm(const Vector* v)
{
    return _norm_units[v->dt](v);
}

static void _add_int(size_t dim, void* t_vals, void* v_vals)
{
    int_t* restrict a = t_vals;
    int_t* restrict b = v_vals;

#pragma omp simd
    ADD(dim, a[n], b[n]);
}

static void _add_real(size_t dim, void* t_vals, void* v_vals)
{
    real_t* restrict a = t_vals;
    real_t* restrict b = v_vals;

#pragma omp simd
    ADD(dim, a[n], b[n]);
}

static void _add_cmpl(size_t dim, void* t_vals, void* v_vals)
{
    complex_t* restrict a = t_vals;
    complex_t* restrict b = v_vals;

#pragma omp simd
    ADD(dim, a[n], b[n]);
}

static void (*_add_units[TYPE_COUNT])(size_t dim,
                                      void* t_vals,
                                      void* v_vals) = {
    [Int]     = _add_int,
    [Real]    = _add_real,
    [Complex] = _add_cmpl,
};

int Vector_add(Vector* target, const Vector* v)
{
    if (!Vector_dimension_match(target, v))
    {
        Log_log("Dimension error in Vector_add()", LOG_RT_ERROR);
        return VECTOR_DIMENSION_ERROR;
    }

    DataType dt = MAX(target->dt, v->dt);
    _add_units[dt](v->dim, target->values, v->values);

    return VECTOR_SUCCESS;
}

static int _dot_int_s(double* target, const Vector* u, const Vector* v)
{
    int_t s                  = 0;
    int_t* restrict u_values = u->values;
    int_t* restrict v_values = v->values;

#pragma omp simd
    FMA(u->dim, s, u_values[n], v_values[n]);
    *target = s;
    return VECTOR_SUCCESS;
}

static int _dot_int_p(double* target, const Vector* u, const Vector* v)
{
    int_t s                  = 0;
    int_t* restrict u_values = u->values;
    int_t* restrict v_values = v->values;

#pragma omp parallel for reduction(+ : s)
    FMA(u->dim, s, u_values[n], v_values[n]);
    *target = s;
    return VECTOR_SUCCESS;
}

static int _dot_real_s(double* target, const Vector* u, const Vector* v)
{
    real_t s                  = 0;
    real_t* restrict u_values = u->values;
    real_t* restrict v_values = v->values;

#pragma omp simd
    FMA(u->dim, s, u_values[n], v_values[n]);
    *target = s;
    return VECTOR_SUCCESS;
}

static int _dot_real_p(double* target, const Vector* u, const Vector* v)
{
    real_t s                  = 0;
    real_t* restrict u_values = u->values;
    real_t* restrict v_values = v->values;

#pragma omp parallel for reduction(+ : s)
    FMA(u->dim, s, u_values[n], v_values[n]);
    *target = s;
    return VECTOR_SUCCESS;
}

static int _dot_cmpl_s(double* target, const Vector* u, const Vector* v)
{
    complex_t s                  = 0;
    complex_t* restrict u_values = u->values;
    complex_t* restrict v_values = v->values;

#pragma omp simd
    FMA(u->dim, s, u_values[n], v_values[n]);
    *target = s;
    return VECTOR_SUCCESS;
}

static int _dot_cmpl_p(double* target, const Vector* u, const Vector* v)
{
    complex_t s                  = 0;
    complex_t* restrict u_values = u->values;
    complex_t* restrict v_values = v->values;

#pragma omp parallel for reduction(+ : s)
    FMA(u->dim, s, u_values[n], v_values[n]);
    *target = s;
    return VECTOR_SUCCESS;
}

static int (*_dot_units_s[TYPE_COUNT])(double* target,
                                       const Vector* u,
                                       const Vector* v) = {
    [Int]     = _dot_int_s,
    [Real]    = _dot_real_s,
    [Complex] = _dot_cmpl_s
};

static int (*_dot_units_p[TYPE_COUNT])(double* target,
                                       const Vector* u,
                                       const Vector* v) = {
    [Int]     = _dot_int_p,
    [Real]    = _dot_real_p,
    [Complex] = _dot_cmpl_p
};

static int _dot_scalar(double* target, const Vector* u, const Vector* v)
{
    DataType dt = MAX(u->dt, v->dt);
    return _dot_units_s[dt](target, u, v);
}

static int _dot_parallel(double* target, const Vector* u, const Vector* v)
{
    DataType dt = MAX(u->dt, v->dt);
    return _dot_units_p[dt](target, u, v);
}

int Vector_dot(double* target, const Vector* u, const Vector* v)
{
    if (!Vector_dimension_match(u, v))
    {
        Log_log("Dimension error in Vector_sub()", LOG_RT_ERROR);
        return VECTOR_DIMENSION_ERROR;
    }
    if (_parallel_condition(v->dim * elem_size(v->dt)))
    {
        return _dot_parallel(target, u, v);
    }
    return _dot_scalar(target, u, v);
}

static void _sub_int(size_t dim, void* t_vals, void* v_vals)
{
    int_t* restrict a = t_vals;
    int_t* restrict b = v_vals;

#pragma omp simd
    SUB(dim, a[n], b[n]);
}

static void _sub_real(size_t dim, void* t_vals, void* v_vals)
{
    real_t* restrict a = t_vals;
    real_t* restrict b = v_vals;

#pragma omp simd
    SUB(dim, a[n], b[n]);
}

static void _sub_cmpl(size_t dim, void* t_vals, void* v_vals)
{
    complex_t* restrict a = t_vals;
    complex_t* restrict b = v_vals;

#pragma omp simd
    SUB(dim, a[n], b[n]);
}

static void (*_sub_units[TYPE_COUNT])(size_t dim,
                                      void* t_vals,
                                      void* v_vals) = {
    [Int]     = _sub_int,
    [Real]    = _sub_real,
    [Complex] = _sub_cmpl,
};

int Vector_sub(Vector* target, const Vector* v)
{
    if (!Vector_dimension_match(target, v))
    {
        Log_log("Dimension error in Vector_sub()", LOG_RT_ERROR);
        return VECTOR_DIMENSION_ERROR;
    }
    DataType dt = MAX(v->dt, target->dt);
    _sub_units[dt](v->dim, target->values, v->values);

    return VECTOR_SUCCESS;
}

static void _scale_int(size_t dim, void* values, const void* lambda_ptr)
{
    ACCESS_VOID(int_t, lambda, lambda_ptr);
    int_t* values_t = values;
#pragma omp simd
    SCALE(dim, values_t[n], *lambda);
}

static void _scale_real(size_t dim, void* values, const void* lambda_ptr)
{
    ACCESS_VOID(real_t, lambda, lambda_ptr);
    real_t* values_t = values;
#pragma omp simd
    SCALE(dim, values_t[n], *lambda);
}

static void _scale_cmpl(size_t dim, void* values, const void* lambda_ptr)
{
    ACCESS_VOID(real_t, lambda, lambda_ptr);
    real_t* values_t = values;
#pragma omp simd
    SCALE(dim, values_t[n], *lambda);
}

static void (*_scale_units[TYPE_COUNT])(size_t dim,
                                        void* values,
                                        const void* lambda_ptr) = {
    [Int]     = _scale_int,
    [Real]    = _scale_real,
    [Complex] = _scale_cmpl,
};

int Vector_scale(Vector* target, const void* lambda)
{
    _scale_units[target->dt](target->dim, target->values, lambda);
    return VECTOR_SUCCESS;
}

static void _bc_add_int(size_t dim, void* t_vals, const void* addend)
{
    int_t* restrict a = t_vals;
    const int_t* b    = addend;

#pragma omp simd
    ADD(dim, a[n], *b);
}

static void _bc_add_real(size_t dim, void* t_vals, const void* addend)
{
    real_t* restrict a = t_vals;
    const real_t* b    = addend;

#pragma omp simd
    ADD(dim, a[n], *b);
}

static void _bc_add_cmpl(size_t dim, void* t_vals, const void* addend)
{
    complex_t* restrict a = t_vals;
    const complex_t* b    = addend;

#pragma omp simd
    ADD(dim, a[n], *b);
}

static void (*_bc_add_units[TYPE_COUNT])(size_t dim,
                                         void* t_vals,
                                         const void* addend) = {
    [Int]     = _bc_add_int,
    [Real]    = _bc_add_real,
    [Complex] = _bc_add_cmpl,
};

int Vector_broadcast_add(Vector* target, const void* a)
{
    _bc_add_units[target->dt](target->dim, target->values, a);
    return VECTOR_SUCCESS;
}

static void _negate_int(void* target, const void* source)
{
    ACCESS_VOID(int_t, s_val, source);
    *(int_t*)target = -*s_val;
}

static void _negate_real(void* target, const void* source)
{
    ACCESS_VOID(real_t, s_val, source);
    *(real_t*)target = -*s_val;
}

static void _negate_cmpl(void* target, const void* source)
{
    ACCESS_VOID(complex_t, s_val, source);
    *(complex_t*)target = -*s_val;
}

static void (*_negate_units[TYPE_COUNT])(void* target, const void* source) = {
    [Int]     = _negate_int,
    [Real]    = _negate_real,
    [Complex] = _negate_cmpl
};

int Vector_broadcast_sub(Vector* target, const void* a)
{
    void* neg_a = alloca(elem_size(target->dt));
    _negate_units[target->dt](neg_a, a);
    return Vector_broadcast_add(target, neg_a);
}

// TODO: reactivate
// int Vector_gradient(Vector* grad, const Vector* x, double (*f)(const
// Vector*))
// {
//     const size_t N   = Vector_get_dim(x);
//     const double f_x = f(x);
//     Vector_prepare_target(grad, N, x->dt);
//     for (size_t n = 0; n < N; n++)
//     {
//         double x_orig = x->values[n];
//         x->values[n]  = x_orig + EPS;
//         double f_plus = f(x);
//
//         x->values[n]   = x_orig - EPS;
//         double f_minux = f(x);
//         x->values[n]   = x_orig;
//
//         grad->values[n] = (f_plus - f_minux) / (2.0 * EPS);
//     }
//
//     return VECTOR_SUCCESS;
// }

// int Vector_gradient_maximize(Vector* target,
//                              const Vector* x,
//                              double (*f)(const Vector*),
//                              double stepsize)
// {
//
//     Vector grad = Vector_new(0, 0.0);
//     Vector_prepare_target(target, x->dim);
//     Vector_copy(target, x);
//
//     Vector_gradient(&grad, x, f);
//     double norm_grad = Vector_norm(&grad);
//     Vector grad_tmp  = Vector_new_copy(&grad);
//     Vector_scale(&grad, stepsize);
//
//     double last_f_of_x = f(x);
//     Vector test_step   = Vector_new_copy(x);
//     int add_status     = Vector_add(&test_step, &grad);
//     int current_status = 0;
//     current_status += add_status;
//     double current_f_of_x = f(&test_step);
//
//     int count = 0;
//     while (count < MAX_STEP && norm_grad >= GRAD_EPS)
//     {
//         int a = 0;
//         while (current_f_of_x < last_f_of_x)
//         {
//             stepsize *= 0.5;
//             Vector_copy(&grad, &grad_tmp);
//             Vector_scale(&grad, stepsize);
//             Vector_copy(&test_step, target);
//
//             current_status += Vector_add(&test_step, &grad);
//             current_f_of_x = f(&test_step);
//             if (a >= 20)
//             {
//                 break;
//             }
//             a++;
//         }
//
//         if (current_f_of_x >= last_f_of_x)
//         {
//             current_status += Vector_add(&test_step, &grad);
//             double longer_f_of_x = f(&test_step);
//             if (longer_f_of_x > current_f_of_x)
//             {
//                 stepsize *= 2;
//                 current_f_of_x = longer_f_of_x;
//             }
//             else
//             {
//                 current_status += Vector_sub(&test_step, &grad);
//                 Vector_scale(&grad, 0.5);
//             }
//         }
//
//         Vector_copy(target, &test_step);
//         last_f_of_x = current_f_of_x;
//         Vector_gradient(&grad, target, f);
//         Vector_norm(&grad);
//         Vector_copy(&grad_tmp, &grad);
//         Vector_scale(&grad, stepsize);
//         current_status += Vector_add(&test_step, &grad);
//         current_f_of_x = f(&test_step);
//
//         if (current_status)
//         {
//             Log_log("A dimension error happened in Vector_gradient_maximize",
//                     LOG_RT_ERROR);
//             return VECTOR_DIMENSION_ERROR;
//         }
//         count++;
//     }
//     Vector_prepare_target(target, x->dim);
//     memcpy(target->values, target->values, target->dim * sizeof(double));
//     Vector_free(&grad);
//     Vector_free(&grad_tmp);
//     Vector_free(&test_step);
//
//     return VECTOR_SUCCESS;
// }
//
// static double (*minimize_f)(const Vector*);
//
// static double neg_wrapper(const Vector* x)
// {
//     return -minimize_f(x);
// }
//
// int Vector_gradient_minimize(Vector* target,
//                              const Vector* x,
//                              double (*f)(const Vector*),
//                              double stepsize)
// {
//     minimize_f = f;
//     return Vector_gradient_maximize(target, x, neg_wrapper, stepsize);
// }
