#include "../include/vector.h"
#include "../include/logging.h"
#include "../include/prng.h"
#include <assert.h>
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

void Vector_prepare_target(Vector* target, const size_t dim)
{
    if (Vector_is_empty(target))
    {
        *target = Vector_zeros(dim);
        return;
    }

    if (target->dim == dim)
    {
        return;
    }

    Log_log("Target vector shape mismatch, reallocating", LOG_WARNING);

    Vector_free(target);
    *target = Vector_zeros(dim);
}

Vector Vector_new(const size_t dim, const double init_val)
{
    Vector v = Vector_zeros(dim);
#pragma omp simd
    for (size_t i = 0; i < dim; i++)
    {
        v.values[i] = init_val;
    }
    return v;
}

Vector Vector_new_vals(const size_t dim, const double* init_vals)
{
    double* vals = calloc(dim, sizeof(double));
    memcpy(vals, init_vals, dim * sizeof(double));
    Vector v = { vals, dim };
    return v;
}

Vector Vector_new_random_normal(const size_t dim,
                                const double mean,
                                const double variance)
{
    Vector_init_prng(NO_SEED);
    Vector v = Vector_zeros(dim);
    for (size_t n = 0; n < dim; n++)
    {
        v.values[n] = PRNG_State_normal(&prng, mean, variance);
    }
    return v;
}

Vector Vector_new_random_uniform(const size_t dim,
                                 const double min,
                                 const double max)
{
    Vector_init_prng(NO_SEED);
    Vector v = Vector_zeros(dim);
    for (size_t n = 0; n < dim; n++)
    {
        v.values[n] = PRNG_State_random_double_range(&prng, min, max);
    }
    return v;
}

Vector Vector_new_copy(const Vector* v)
{
    Vector res;

    res.values = calloc(v->dim, sizeof(double));
    memcpy(res.values, v->values, v->dim * sizeof(double));
    res.dim = v->dim;

    return res;
}

int _all_close_scalar(const Vector* u, const Vector* v)
{
    for (size_t n = 0; n < u->dim; n++)
    {
        if (fabs(u->values[n] - v->values[n]) > EPS)
        {
            return 0;
        }
    }
    return 1;
}

int _all_close_parallel(const Vector* u, const Vector* v)
{
    int all_close = 1;
#pragma omp parallel for
    for (size_t n = 0; n < u->dim; n++)
    {
        if (fabs(u->values[n] - v->values[n]) > EPS)
        {
            all_close = 0;
        }
    }
    return all_close;
}

int Vector_all_close(const Vector* u, const Vector* v)
{
    if (v->dim != u->dim)
    {
        return 0;
    }

    if (v->dim > PARALLEL_THRESHOLD)
    {
        return _all_close_parallel(u, v);
    }

    return _all_close_scalar(u, v);
}

void Vector_copy(Vector* target, const Vector* v)
{
    Vector_prepare_target(target, v->dim);
    memcpy(target->values, v->values, v->dim * sizeof(double));
    target->dim = v->dim;
}

Vector Vector_zeros_like(const Vector* v)
{
    Vector res;
    const size_t dim = v->dim;
    res.values       = calloc(dim, sizeof(double));
    res.dim          = dim;
    return res;
}

Vector Vector_zeros(const size_t dim)
{
    Vector res = { 0 };
    if (dim > 0)
    {
        res.values = calloc(dim, sizeof(double));
        if (!res.values)
        {
            Log_log("Error allocating data for Vector_zeros()", LOG_ERROR);
            return (Vector){ 0 };
        }
        res.dim = dim;
    }
    return res;
}

Vector Vector_ones(const size_t dim)
{
    return Vector_new(dim, 1.0);
}

double Vector_at(const Vector* v, const size_t index)
{
    return v->values[index];
}

int Vector_get_at(double* target, const Vector* v, const size_t index)
{
    if (index >= v->dim)
    {
        Log_log("Dimension error in Vector_get_at()", LOG_ERROR);
        return VECTOR_DIMENSION_ERROR;
    }
    *target = v->values[index];

    return VECTOR_SUCCESS;
}

int Vector_set_item(Vector* v, const size_t index, const double val)
{
    if (index >= v->dim)
    {
        Log_log("Dimension error in Vector_set_item()", LOG_ERROR);
        return VECTOR_DIMENSION_ERROR;
    }
    v->values[index] = val;
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

void Vector_print(Vector* v)
{
    printf("( ");
    for (int i = 0; i < v->dim; i++)
    {
        const double val = v->values[i];
        printf("%f ", val);
    }
    printf(")");
}

double _max_parallel(const Vector* v)
{
    double max = -INFINITY;

#pragma omp parallel for
    for (size_t i = 0; i < v->dim; i++)
    {
        if (v->values[i] > max)
        {
            max = v->values[i];
        }
    }
    return max;
}

double _max_scalar(const Vector* v)
{
    double max = -INFINITY;

    for (size_t i = 0; i < v->dim; i++)
    {
        if (v->values[i] > max)
        {
            max = v->values[i];
        }
    }
    return max;
}

double Vector_max(const Vector* v)
{
    if (v->dim > PARALLEL_THRESHOLD)
    {
        return _max_parallel(v);
    }
    return _max_scalar(v);
}

double _min_parallel(const Vector* v)
{
    double min = INFINITY;

#pragma omp parallel for
    for (size_t i = 0; i < v->dim; i++)
    {
        if (v->values[i] < min)
        {
            min = v->values[i];
        }
    }
    return min;
}

double _min_scalar(const Vector* v)
{
    double min = INFINITY;

    for (size_t i = 0; i < v->dim; i++)
    {
        if (v->values[i] < min)
        {
            min = v->values[i];
        }
    }
    return min;
}

double Vector_min(const Vector* v)
{
    if (v->dim > PARALLEL_THRESHOLD)
    {
        return _min_parallel(v);
    }
    return _min_scalar(v);
}

void _insertion_sort(Vector* v, size_t low, size_t high)
{
    for (size_t i = low + 1; i <= high; ++i)
    {
        const double key = v->values[i];
        ssize_t j        = i - 1;

        while (j >= (ssize_t)low && v->values[j] > key)
        {
            v->values[j + 1] = v->values[j];
            j--;
        }
        v->values[j + 1] = key;
    }
}

void _swap(Vector* v, const size_t i, const size_t j)
{
    double tmp   = v->values[i];
    v->values[i] = v->values[j];
    v->values[j] = tmp;
}

void _heapify(Vector* v, size_t low, const size_t n, const size_t i)
{
    size_t largest = i;
    size_t l       = 2 * i + 1;
    size_t r       = 2 * i + 2;
    if (l < n && v->values[l] > v->values[largest])
    {
        largest = l;
    }
    if (r < n && v->values[r] > v->values[largest])
    {
        largest = r;
    }

    if (largest != i)
    {
        // swap
        _swap(v, low + i, low + largest);

        _heapify(v, low, n, largest);
    }
}

void _heapsort(Vector* v, size_t low, size_t high)
{
    const size_t n = high - low + 1;

    for (ssize_t i = n / 2 - 1; i >= low; i--)
    {
        _heapify(v, low, n, i);
    }

    for (ssize_t i = n - 1; i > low; i--)
    {
        _swap(v, 0, i);

        _heapify(v, low, i, 0);
    }
}

size_t _partition(Vector* v, size_t low, size_t high)
{
    double pivot = v->values[high];
    size_t i     = low - 1;

    for (size_t j = low; j <= high - 1; j++)
    {
        if (v->values[j] < pivot)
        {
            i++;
            _swap(v, i, j);
        }
    }

    _swap(v, i + 1, high);
    return i + 1;
}

void _introsort(Vector* v, size_t low, size_t high, int depth_limit)
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

double Vector_norm(const Vector* v)
{
    double sum                = 0.0;
    double* restrict v_values = v->values;
    for (size_t n = 0; n < v->dim; n++)
    {
        sum += v_values[n] * v_values[n];
    }

    return sqrt(sum);
}

int Vector_add(Vector* target, const Vector* v)
{
    if (!Vector_dimension_match(target, v))
    {
        Log_log("Dimension error in Vector_add()", LOG_ERROR);
        return VECTOR_DIMENSION_ERROR;
    }

#pragma omp simd
    for (size_t n = 0; n < v->dim; n++)
    {
        target->values[n] += v->values[n];
    }

    return VECTOR_SUCCESS;
}

int _dot_scalar(double* target, const Vector* u, const Vector* v)
{
    double s                  = 0.0;
    double* restrict u_values = u->values;
    double* restrict v_values = v->values;

#pragma omp simd
    for (size_t i = 0; i < u->dim; i++)
    {
        s += u_values[i] * v_values[i];
    }
    *target = s;
    return VECTOR_SUCCESS;
}

int _dot_parallel(double* target, const Vector* u, const Vector* v)
{
    double s = 0.0;

#pragma omp parallel for reduction(+ : s)
    for (size_t i = 0; i < u->dim; i++)
    {
        s += u->values[i] * v->values[i];
    }
    *target = s;
    return VECTOR_SUCCESS;
}

int Vector_dot(double* target, const Vector* u, const Vector* v)
{
    if (!Vector_dimension_match(u, v))
    {
        Log_log("Dimension error in Vector_sub()", LOG_ERROR);
        return VECTOR_DIMENSION_ERROR;
    }
    if (u->dim > PARALLEL_THRESHOLD)
    {
        return _dot_parallel(target, u, v);
    }
    return _dot_scalar(target, u, v);
}

int Vector_sub(Vector* target, const Vector* v)
{
    if (!Vector_dimension_match(target, v))
    {
        Log_log("Dimension error in Vector_sub()", LOG_ERROR);
        return VECTOR_DIMENSION_ERROR;
    }

#pragma omp simd
    for (size_t n = 0; n < v->dim; n++)
    {
        target->values[n] -= v->values[n];
    }

    return VECTOR_SUCCESS;
}

int Vector_scale(Vector* target, const double lambda)
{
#pragma omp simd
    for (size_t n = 0; n < target->dim; n++)
    {
        target->values[n] *= lambda;
    }
    return VECTOR_SUCCESS;
}

int Vector_broadcast_add(Vector* target, const double a)
{
    for (size_t n = 0; n < target->dim; n++)
    {
        target->values[n] += a;
    }
    return VECTOR_SUCCESS;
}

int Vector_broadcast_sub(Vector* target, const double a)
{
    return Vector_broadcast_add(target, -a);
}

int Vector_gradient(Vector* grad, const Vector* x, double (*f)(const Vector*))
{
    const size_t N   = Vector_get_dim(x);
    const double f_x = f(x);
    Vector_prepare_target(grad, N);
    for (size_t n = 0; n < N; n++)
    {
        double x_orig = x->values[n];
        x->values[n]  = x_orig + EPS;
        double f_plus = f(x);

        x->values[n]   = x_orig - EPS;
        double f_minux = f(x);
        x->values[n]   = x_orig;

        grad->values[n] = (f_plus - f_minux) / (2.0 * EPS);
    }

    return VECTOR_SUCCESS;
}

int Vector_gradient_maximize(Vector* target,
                             const Vector* x,
                             double (*f)(const Vector*),
                             double stepsize)
{

    Vector grad = Vector_new(0, 0.0);
    Vector_prepare_target(target, x->dim);
    Vector_copy(target, x);

    Vector_gradient(&grad, x, f);
    double norm_grad = Vector_norm(&grad);
    Vector grad_tmp  = Vector_new_copy(&grad);
    Vector_scale(&grad, stepsize);

    double last_f_of_x = f(x);
    Vector test_step   = Vector_new_copy(x);
    int add_status     = Vector_add(&test_step, &grad);
    int current_status = 0;
    current_status += add_status;
    double current_f_of_x = f(&test_step);

    int count = 0;
    while (count < MAX_STEP && norm_grad >= GRAD_EPS)
    {
        int a = 0;
        while (current_f_of_x < last_f_of_x)
        {
            stepsize *= 0.5;
            Vector_copy(&grad, &grad_tmp);
            Vector_scale(&grad, stepsize);
            Vector_copy(&test_step, target);

            current_status += Vector_add(&test_step, &grad);
            current_f_of_x = f(&test_step);
            if (a >= 20)
            {
                break;
            }
            a++;
        }

        if (current_f_of_x >= last_f_of_x)
        {
            current_status += Vector_add(&test_step, &grad);
            double longer_f_of_x = f(&test_step);
            if (longer_f_of_x > current_f_of_x)
            {
                stepsize *= 2;
                current_f_of_x = longer_f_of_x;
            }
            else
            {
                current_status += Vector_sub(&test_step, &grad);
                Vector_scale(&grad, 0.5);
            }
        }

        Vector_copy(target, &test_step);
        last_f_of_x = current_f_of_x;
        Vector_gradient(&grad, target, f);
        Vector_norm(&grad);
        Vector_copy(&grad_tmp, &grad);
        Vector_scale(&grad, stepsize);
        current_status += Vector_add(&test_step, &grad);
        current_f_of_x = f(&test_step);

        if (current_status)
        {
            Log_log("A dimension error happened in Vector_gradient_maximize",
                    LOG_ERROR);
            return VECTOR_DIMENSION_ERROR;
        }
        count++;
    }
    Vector_prepare_target(target, x->dim);
    memcpy(target->values, target->values, target->dim * sizeof(double));
    Vector_free(&grad);
    Vector_free(&grad_tmp);
    Vector_free(&test_step);

    return VECTOR_SUCCESS;
}

static double (*minimize_f)(const Vector*);

static double neg_wrapper(const Vector* x)
{
    return -minimize_f(x);
}

int Vector_gradient_minimize(Vector* target,
                             const Vector* x,
                             double (*f)(const Vector*),
                             double stepsize)
{
    minimize_f = f;
    return Vector_gradient_maximize(target, x, neg_wrapper, stepsize);
}
