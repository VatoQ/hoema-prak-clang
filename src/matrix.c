// #define DEBUG
#define _POSIX_C_SOURCE 200112L
#include "../include/matrix.h"
#include "../include/logging.h"
#include "../include/prng.h"
#include "../include/vector.h"
#include <complex.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INITIALIZER(n, vals_arr, val_ptr)                                      \
    for (size_t i = 0; i < n; i++)                                             \
    vals_arr[i] = *val_ptr

static PRNG_State prng = { 0 };

int Matrix_is_empty(const Matrix* M)
{
    return ((M->values == NULL) || (M->m == 0 && M->n == 0));
}

static void Matrix_prepare_target(Matrix* target,
                                  const size_t m,
                                  const size_t n,
                                  DataType dt)
{
    if (Matrix_is_empty(target))
    {
        *target = Matrix_new(m, n, dt);
        return;
    }

    if (target->m == m && target->n == n && target->dt == dt)
    {
        return;
    }
    Log_log("Target matrix shape mismatch, reallocating", LOG_RT_WARNING);
    Matrix_free(target);
    *target = Matrix_new(m, n, dt);
}

size_t _elem_size(DataType dt)
{
    switch (dt)
    {
        case Int:
        {
            return sizeof(long);
        }
        case Real:
        {
            return sizeof(double);
        }
        case Complex:
        {
            return sizeof(complex double);
        }
        default:
        {
            return 1;
        }
    }
}

Matrix Matrix_new(const size_t m, const size_t n, DataType dt)
{
    Matrix res           = { 0 };
    const size_t dt_size = elem_size(dt);
    if (m != 0 && n != 0)
    {
        if (Log_get_verbosity() == LOG_VERB_ALL &&
            Log_info_threshold(m * n, dt_size))
        {
            char buf[128];
            double mibi_byte_size =
              1.0 * n * n * dt_size / (double)(1024 * 1024);
            snprintf(buf,
                     128,
                     "Allocated a new matrix of shape (%zu, %zu), %.2f MiB ",
                     m,
                     n,
                     mibi_byte_size);
            Log_log(buf, LOG_RT_INFO);
        }

        void* vals = NULL;
        if (posix_memalign((void**)&vals, 32, m * n * dt_size) != 0)
        {
            Log_log("Error allocating memory in Matrix_new()", LOG_RT_ERROR);
            return (Matrix){ 0 };
        }

        res.m      = m;
        res.n      = n;
        res.values = vals;
        res.dt     = dt;
    }
    return res;
}

Matrix Matrix_like(const Matrix* M)
{
    Matrix new = Matrix_new(M->m, M->n, M->dt);
    return new;
}

Matrix Matrix_zeros(const size_t m, const size_t n, DataType dt)
{
    Matrix M = Matrix_new(m, n, dt);
    memset(M.values, 0, m * n * elem_size(dt));
    return M;
}

Matrix Matrix_new_vals(const size_t m,
                       const size_t n,
                       const void* init_vals,
                       DataType dt)
{
    Matrix M = Matrix_new(m, n, dt);
    memcpy(M.values, init_vals, m * n * elem_size(dt));
    M.m  = m;
    M.n  = n;
    M.dt = dt;
    return M;
}

void Matrix_init_prng(const int seed)
{
    if (prng.aux != 0 || prng.state != 0)
    {
        return;
    }
    prng = PRNG_State_init(seed);
    if (Log_get_verbosity() == LOG_VERB_ALL)
    {
        char buf[128];
        snprintf(buf, 128, "Matrix PRNG initialized with seed %zu", prng.state);
        Log_log(buf, LOG_RT_INFO);
    }
}

static void _set_normal_int(void* values,
                            size_t size,
                            PRNG_State* prng,
                            double mean,
                            double variance)
{
    ACCESS_VOID(int_t, values_t, values);
    for (size_t i = 0; i < size; i++)
    {
        values_t[i] = (int_t)PRNG_State_normal(prng, mean, variance);
    }
}

static void _set_normal_real(void* values,
                             size_t size,
                             PRNG_State* prng,
                             double mean,
                             double variance)
{
    ACCESS_VOID(real_t, values_t, values);
    for (size_t i = 0; i < size; i++)
    {
        values_t[i] = (real_t)PRNG_State_normal(prng, mean, variance);
    }
}

static void _set_normal_compl(void* values,
                              size_t size,
                              PRNG_State* prng,
                              double mean,
                              double variance)
{
    ACCESS_VOID(complex_t, values_t, values);
    for (size_t i = 0; i < size; i++)
    {
        double a    = PRNG_State_normal(prng, mean, variance);
        double b    = PRNG_State_normal(prng, mean, variance);
        values_t[i] = a + b * I;
    }
}

static void (*_normal_setters[TYPE_COUNT])(void* values,
                                           size_t size,
                                           PRNG_State* prng,
                                           double mean,
                                           double variance) = {
    [Int]     = _set_normal_int,
    [Real]    = _set_normal_real,
    [Complex] = _set_normal_compl
};

Matrix Matrix_new_random_normal(const size_t m,
                                const size_t n,
                                const double mean,
                                const double variance,
                                DataType dt)
{

    Matrix_init_prng(NO_SEED);
    if (Log_get_verbosity() == LOG_VERB_ALL &&
        Log_info_threshold(m * n, sizeof(double)))
    {
        char buf[128];
        double mibi_byte_size =
          1.0 * n * n * sizeof(double) / (double)(1024 * 1024);
        snprintf(
          buf,
          128,
          "New random normal matrix constructed with shape (%zu, %zu) %f MiB",
          m,
          n,
          mibi_byte_size);
        Log_log(buf, LOG_RT_INFO);
    }

    Matrix M = Matrix_new(m, n, dt);
    _normal_setters[dt](M.values, m * n, &prng, mean, variance);
    return M;
}

static void _set_uniform_int(void* values,
                             size_t size,
                             PRNG_State* prng,
                             double min,
                             double max)
{
    for (size_t i = 0; i < size; i++)
    {
        void* ptr = values + i;
        long* val = (long*)ptr;
        *val      = (long)PRNG_State_random_double_range(prng, min, max);
    }
}

static void _set_uniform_real(void* values,
                              size_t size,
                              PRNG_State* prng,
                              double min,
                              double max)
{
    for (size_t i = 0; i < size; i++)
    {
        void* ptr   = values + i;
        double* val = (double*)ptr;
        *val        = (double)PRNG_State_random_double_range(prng, min, max);
    }
}

static void _set_uniform_compl(void* values,
                               size_t size,
                               PRNG_State* prng,
                               double min,
                               double max)
{
    for (size_t i = 0; i < size; i++)
    {
        void* ptr           = values + i;
        complex double* val = (complex double*)ptr;
        double a            = PRNG_State_random_double_range(prng, min, max);
        double b            = PRNG_State_random_double_range(prng, min, max);
        *val                = a + b * I;
    }
}

static void (*_uniform_setters[TYPE_COUNT])(void* values,
                                            size_t size,
                                            PRNG_State* prng,
                                            double mean,
                                            double variance) = {
    [Int]     = _set_normal_int,
    [Real]    = _set_normal_real,
    [Complex] = _set_normal_compl
};

Matrix Matrix_new_random_uniform(const size_t m,
                                 const size_t n,
                                 const double min,
                                 const double max,
                                 DataType dt)
{
    Matrix_init_prng(NO_SEED);

    if (Log_get_verbosity() == LOG_VERB_ALL &&
        Log_info_threshold(m * n, sizeof(double)))
    {
        char buf[128];
        double mibi_byte_size =
          1.0 * n * n * sizeof(double) / (double)(1024 * 1024);
        snprintf(
          buf,
          128,
          "New random uniform matrix constructed with shape (%zu, %zu), %f MiB",
          m,
          n,
          mibi_byte_size);
        Log_log(buf, LOG_RT_INFO);
    }
    Matrix M = Matrix_new(m, n, 0);
    _uniform_setters[dt](M.values, m * n, &prng, min, max);
    return M;
}

Matrix Matrix_new_random_symmetric(const size_t n, DataType dt)
{
    // TODO: Vector datatypes
    Matrix M = Matrix_zeros(n, n, dt);
    memset(M.values, 0, n * n * elem_size(dt));
    Matrix tmp = Matrix_zeros_like(&M, dt);

    Vector v = Vector_zeros(0, dt);

    for (size_t i = 0; i < n; i++)
    {
        v             = Vector_new_random_normal(n, 0, 1, dt);
        double norm_v = 1. / Vector_norm(&v);

        Vector_scale(&v, &norm_v);
        Matrix_Vector_outer(&tmp, &v, &v);
        Matrix_add(&M, &tmp);

        Vector_free(&v);
    }
    if (Log_get_verbosity() == LOG_VERB_ALL &&
        Log_info_threshold(n * n, elem_size(dt)))
    {
        char buf[128];
        double mibi_byte_size =
          1.0 * n * n * elem_size(dt) / (double)(1024 * 1024);
        snprintf(buf,
                 128,
                 "New random symmetric matrix constructed with shape (%zu, "
                 "%zu), %.2f MiB",
                 n,
                 n,
                 mibi_byte_size);
        Log_log(buf, LOG_RT_INFO);
    }
    return M;
}

static void _zero_init_int(void* values, const size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        void* ptr = values + i;
        long* val = (long*)val;
        *val      = 0;
    }
}

static void _zero_init_real(void* values, const size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        void* ptr   = values + i;
        double* val = (double*)val;
        *val        = 0;
    }
}

static void _zero_init_complex(void* values, const size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        void* ptr           = values + i;
        complex double* val = (complex double*)val;
        *val                = 0;
    }
}

static void (*_zero_initializers[TYPE_COUNT])(void* values,
                                              const size_t size) = {
    [Int]     = _zero_init_int,
    [Real]    = _zero_init_real,
    [Complex] = _zero_init_complex
};

Matrix Matrix_zeros_like(const Matrix* M, DataType dt)
{
    Matrix new = Matrix_like(M);
    memset(M->values, 0, M->m * M->n * elem_size(dt));

    // _zero_initializers[dt](new.values, M->m * M->n);
    return new;
}

void _diag_int(void* M_values, const size_t n, const void* vals)
{
    ACCESS_VOID(int_t, M_values_t, M_values);
    ACCESS_VOID(int_t, source, vals);
    for (size_t i = 0; i < n; i++)
    {
        M_values_t[i * n + i] = source[i];
    }
}

void _diag_real(void* M_values, const size_t n, const void* vals)
{
    ACCESS_VOID(real_t, M_values_t, M_values);
    ACCESS_VOID(real_t, source, vals);
    for (size_t i = 0; i < n; i++)
    {
        M_values_t[i * n + i] = source[i];
    }
}

void _diag_cmpl(void* M_values, const size_t n, const void* vals)
{
    ACCESS_VOID(complex_t, M_values_t, M_values);
    ACCESS_VOID(complex_t, source, vals);
    for (size_t i = 0; i < n; i++)
    {
        M_values_t[i * n + i] = source[i];
    }
}

void (*_diag_helper_workers[TYPE_COUNT])(void* M_values,
                                         const size_t n,
                                         const void* vals) = {
    [Int]     = _diag_int,
    [Real]    = _diag_real,
    [Complex] = _diag_cmpl
};

/**
 * @brief Set a given array of values to the diagonal entries of a matrix.
 *
 * @param `M` Target matrix.
 * @param `n` number of values in `val`.
 * @param `val` array of initial values.
 */
static void _diag_helper(Matrix* M,
                         const size_t n,
                         const double* val,
                         DataType dt)
{
    int status = 0;
    _diag_helper_workers[dt](M->values, n, val);
    if (status != MATRIX_SUCCESS * n)
    {
        Log_log("An unknown error has occured at "
                "Matrix_diag()\nReplacing M with zeros",
                LOG_RT_ERROR);
        Matrix tmp = Matrix_zeros_like(M, dt);
        Matrix_free(M);
        *M = tmp;
    }
}

Matrix Matrix_diag(const Vector* v)
{
    const size_t n = v->dim;

    Matrix M = Matrix_zeros(n, n, v->dt);
    _diag_helper(&M, n, v->values, v->dt);
    if (Log_get_verbosity() == LOG_VERB_ALL &&
        Log_info_threshold(n * n, sizeof(double)))
    {
        Log_log("Diagonal Matrix constructed from vector", LOG_RT_INFO);
    }
    return M;
}

Matrix Matrix_diag_val(const size_t n, const void* val, const DataType dt)
{
    Vector v = Vector_new(n, val, dt);
    Matrix M = Matrix_diag(&v);
    Vector_free(&v);
    return M;
}

void Matrix_copy(Matrix* target, const Matrix* M)
{
    Matrix_prepare_target(target, M->m, M->n, M->dt);
    memcpy(target->values, M->values, M->m * M->n * elem_size(M->dt));
    target->m  = M->m;
    target->n  = M->n;
    target->dt = M->dt;
}

static void _set_at_int(void* M_values, const size_t index, const void* value)
{
    long* v      = (long*)value;
    void* ptr    = M_values + index;
    long* target = (long*)ptr;
    *target      = *v;
}

static void _set_at_real(void* M_values, const size_t index, const void* value)
{
    double* v      = (double*)value;
    void* ptr      = M_values + index;
    double* target = (double*)ptr;
    *target        = *v;
}

static void _set_at_cmpl(void* M_values, const size_t index, const void* value)
{
    complex double* v      = (complex double*)value;
    void* ptr              = M_values + index;
    complex double* target = (complex double*)ptr;
    *target                = *v;
}

void (*_set_at_helpers[TYPE_COUNT])(void* M_values,
                                    const size_t index,
                                    const void* value) = {
    [Int]     = _set_at_int,
    [Real]    = _set_at_real,
    [Complex] = _set_at_cmpl,
};

int Matrix_set_at(Matrix* M, const size_t m, const size_t n, const void* value)
{
    if (m >= M->m || n >= M->n)
    {
        Log_log("Index out of range in Matrix_set_at()!", LOG_RT_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }
    const size_t index = m * M->n + n;
    _set_at_helpers[M->dt](M->values, index, value);

    // TODO: M->values[index]   = value;

    return MATRIX_SUCCESS;
}

static void _get_at_int(const void* M_values, const size_t index, void* target)
{
    ACCESS_VOID(int_t, M_value, M_values + index);
    ASSIGN_UNTYPED(int_t, target, M_value);
}

static void _get_at_real(const void* M_values, const size_t index, void* target)
{
    ACCESS_VOID(real_t, M_value, M_values + index);
    ASSIGN_UNTYPED(real_t, target, M_value);
}

static void _get_at_cmpl(const void* M_values, const size_t index, void* target)
{
    ACCESS_VOID(complex_t, M_value, M_values + index);
    ASSIGN_UNTYPED(complex_t, target, M_value);
}

static void (*_get_at_helpers[TYPE_COUNT])(const void* M_values,
                                           const size_t index,
                                           void* target) = {
    [Int]     = _get_at_int,
    [Real]    = _get_at_real,
    [Complex] = _get_at_cmpl,
};

int Matrix_get_at(void* target, const Matrix* M, const size_t m, const size_t n)
{
    if (m >= M->m || n >= M->n)
    {
        Log_log("Index out of range in Matrix_get_at()!", LOG_RT_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }

    const size_t index = m * M->n + n;
    _get_at_helpers[M->dt](M->values, index, target);

    return MATRIX_SUCCESS;
}

static bool _all_close_int(const Matrix* A, const Matrix* B)
{
    for (size_t i = 0; i < A->m * A->n; i++)
    {
        void* ptr1  = A->values + i;
        void* ptr2  = B->values + i;
        long* ptr1i = (long*)ptr1;
        long* ptr2i = (long*)ptr2;
        if (*ptr1i == *ptr2i)
        {
            return false;
        }
    }
    return true;
}

static bool _all_close_real(const Matrix* A, const Matrix* B)
{
    for (size_t i = 0; i < A->m * A->n; i++)
    {
        void* ptr1    = A->values + i;
        void* ptr2    = B->values + i;
        double* ptr1d = (double*)ptr1;
        double* ptr2d = (double*)ptr2;
        if (fabs(*ptr1d - *ptr2d) > EPS)
        {
            return false;
        }
    }
    return true;
}

static bool _all_close_cmpl(const Matrix* A, const Matrix* B)
{
    for (size_t i = 0; i < A->m * A->n; i++)
    {
        void* ptr1            = A->values + i;
        void* ptr2            = B->values + i;
        complex double* ptr1d = (complex double*)ptr1;
        complex double* ptr2d = (complex double*)ptr2;
        if (cabs(*ptr1d - *ptr2d) > EPS)
        {
            return false;
        }
    }
    return true;
}

static bool (*_all_close_helpers[TYPE_COUNT])(const Matrix* A,
                                              const Matrix* B) = {
    [Int]     = _all_close_int,
    [Real]    = _all_close_real,
    [Complex] = _all_close_cmpl
};

int Matrix_all_close(const Matrix* A, const Matrix* B)
{
    if (A->m != B->m || A->n != B->n)
    {
        return 0;
    }

    DataType dt = (A->dt > B->dt) ? A->dt : B->dt;

    return _all_close_helpers[dt](A, B);
}

void Matrix_free(Matrix* M)
{
    free(M->values);
    M->values = NULL;
    M->m = 0, M->n = 0, M->dt = 0;
}

static void _max_int(void* target, const Matrix* M)
{
    long max = -LONG_MAX;
    for (int i = 0; i < M->m * M->n; i++)
    {
        void* ptr = M->values + i;
        long* val = (long*)ptr;
        if (*val > max)
        {
            max = *val;
        }
    }
    long* target_val = (long*)target;
    *target_val      = max;
}

static void _max_real(void* target, const Matrix* M)
{
    double max = -INFINITY;
    for (int i = 0; i < M->m * M->n; i++)
    {
        void* ptr   = M->values + i;
        double* val = (double*)ptr;
        if (*val > max)
        {
            max = *val;
        }
    }
    double* target_val = (double*)target;
    *target_val        = max;
}

static void _max_cmpl(void* target, const Matrix* M)
{
    double max = -INFINITY;
    for (int i = 0; i < M->m * M->n; i++)
    {
        void* ptr           = M->values + i;
        complex double* val = (complex double*)ptr;
        if (cabs(*val) > max)
        {
            max = *val;
        }
    }
    double* target_val = (double*)target;
    *target_val        = max;
}

static void (*_max_helpers[TYPE_COUNT])(void* target, const Matrix* M) = {
    [Int]     = _max_int,
    [Real]    = _max_real,
    [Complex] = _max_cmpl,
};

void Matrix_max(void* target, const Matrix* M)
{
    double max = -INFINITY;
    _max_helpers[M->dt](target, M);
}

static void _min_int(void* target, const Matrix* M)
{
    long min = LONG_MAX;
    for (int i = 0; i < M->m * M->n; i++)
    {
        void* ptr = M->values + i;
        long* val = (long*)ptr;
        if (*val < min)
        {
            min = *val;
        }
    }
    long* target_val = (long*)target;
    *target_val      = min;
}

static void _min_real(void* target, const Matrix* M)
{
    double min = INFINITY;
    for (int i = 0; i < M->m * M->n; i++)
    {
        void* ptr   = M->values + i;
        double* val = (double*)ptr;
        if (*val < min)
        {
            min = *val;
        }
    }
    double* target_val = (double*)target;
    *target_val        = min;
}

static void _min_cmpl(void* target, const Matrix* M)
{
    double max = INFINITY;
    for (int i = 0; i < M->m * M->n; i++)
    {
        void* ptr           = M->values + i;
        complex double* val = (complex double*)ptr;
        if (cabs(*val) < max)
        {
            max = *val;
        }
    }
    double* target_val = (double*)target;
    *target_val        = max;
}

static void (*_min_helpers[TYPE_COUNT])(void* target, const Matrix* M) = {
    [Int]     = _min_int,
    [Real]    = _min_real,
    [Complex] = _min_cmpl,
};

void Matrix_min(void* target, const Matrix* M)
{
    double min = INFINITY;
    _min_helpers[M->dt](target, M);
}

static void _print_int(const Matrix* M)
{
    const size_t m = M->m, n = M->n;
    int status = 0;
    ACCESS_VOID(int_t, M_values, M->values);
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            if (j == 0 && i == 0)
            {
                printf("/ ");
            }
            else if (j == 0 && i == m - 1)
            {
                printf("\\ ");
            }
            else if (j == 0)
            {
                printf("| ");
            }

            int_t val = M_values[i * m + j];
            char* sep = (val > 0) ? "  " : " ";
            printf("%zu%s", val, sep);

            if (j == n - 1 && i == 0)
            {
                printf("\\");
            }
            else if (j == n - 1 && i == m - 1)
            {
                printf("/");
            }
            else if (j == n - 1)
            {
                printf("|");
            }
        }
        printf("\n");
    }
    if (status != MATRIX_SUCCESS * m * n)
    {
        Log_log("An unknown problem at Matrix_print", LOG_RT_WARNING);
    }
}

static void _print_real(const Matrix* M)
{
    const size_t m = M->m, n = M->n;
    int status = 0;
    ACCESS_VOID(real_t, M_values, M->values);
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            if (j == 0 && i == 0)
            {
                printf("/ ");
            }
            else if (j == 0 && i == m - 1)
            {
                printf("\\ ");
            }
            else if (j == 0)
            {
                printf("| ");
            }

            real_t val = M_values[i * m + j];
            char* sep  = (val > 0) ? "  " : " ";
            printf("%f%s", val, sep);

            if (j == n - 1 && i == 0)
            {
                printf("\\");
            }
            else if (j == n - 1 && i == m - 1)
            {
                printf("/");
            }
            else if (j == n - 1)
            {
                printf("|");
            }
        }
        printf("\n");
    }
    if (status != MATRIX_SUCCESS * m * n)
    {
        Log_log("An unknown problem at Matrix_print", LOG_RT_WARNING);
    }
}

static void _print_cmpl(const Matrix* M)
{
    const size_t m = M->m, n = M->n;
    int status = 0;
    ACCESS_VOID(complex_t, M_values, M->values);
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            if (j == 0 && i == 0)
            {
                printf("/ ");
            }
            else if (j == 0 && i == m - 1)
            {
                printf("\\ ");
            }
            else if (j == 0)
            {
                printf("| ");
            }

            complex_t val   = M_values[i * m + j];
            const char* sep = (creal(val) > 0) ? "  " : " ";
            PRINT_COMPLEX(val);
            printf("%s", sep);

            if (j == n - 1 && i == 0)
            {
                printf("\\");
            }
            else if (j == n - 1 && i == m - 1)
            {
                printf("/");
            }
            else if (j == n - 1)
            {
                printf("|");
            }
        }
        printf("\n");
    }
    if (status != MATRIX_SUCCESS * m * n)
    {
        Log_log("An unknown problem at Matrix_print", LOG_RT_WARNING);
    }
}

static void (*_printers[TYPE_COUNT])(const Matrix* M) = {
    [Int]     = _print_int,
    [Real]    = _print_real,
    [Complex] = _print_cmpl,
};

void Matrix_print(const Matrix* M)
{
    const size_t m = M->m, n = M->n;
    int status = 0;

    _printers[M->dt](M);
}

static void _frobenius_helper_int(void* target, const Matrix* M)
{
    long s = 0;
    for (size_t i = 0; i < M->m * M->n; i++)
    {
        void* ptr1 = M->values + i;
        void* ptr2 = M->values + i;
        long* val1 = (long*)ptr1;
        long* val2 = (long*)ptr2;

        s += *val1 * *val2;
    }
    memcpy(target, &s, elem_size(Int));
}

static void _frobenius_helper_real(void* target, const Matrix* M)
{
    double s = 0;
    for (size_t i = 0; i < M->m * M->n; i++)
    {
        void* ptr1   = M->values + i;
        void* ptr2   = M->values + i;
        double* val1 = (double*)ptr1;
        double* val2 = (double*)ptr2;

        s += *val1 * *val2;
    }
    memcpy(target, &s, elem_size(Real));
}

static void _frobenius_helper_cmpl(void* target, const Matrix* M)
{
    complex double s = 0;
    for (size_t i = 0; i < M->m * M->n; i++)
    {
        void* ptr1           = M->values + i;
        void* ptr2           = M->values + i;
        complex double* val1 = (complex double*)ptr1;
        complex double* val2 = (complex double*)ptr2;

        s += *val1 * *val2;
    }
    memcpy(target, &s, elem_size(Complex));
}

static void (*_frobenius_helpers[TYPE_COUNT])(void* target, const Matrix* M) = {
    [Int]     = _frobenius_helper_int,
    [Real]    = _frobenius_helper_real,
    [Complex] = _frobenius_helper_cmpl
};

static void _spectral_helper(void* target, const Matrix* M)
{
    Vector eigvals = Vector_zeros(M->n, M->dt);
    Matrix_eigvals(&eigvals, M, true);
    // TODO: correct eigval logic
    void* eig_ptr     = eigvals.values + M->n - 1;
    double* eig_ptr_t = (double*)eig_ptr;
    memcpy(target, eig_ptr_t, elem_size(M->dt));

    Vector_free(&eigvals);
}

void Matrix_norm(void* target, const Matrix* M, NormType nt)
{
    double result = 0;
    switch (nt)
    {
        case FROBENIUS:
        {
            _frobenius_helpers[M->dt](target, M);
            break;
        }
        case SPECTRAL:
        {
            //_spectral_helper(target, M);
            break;
        }
        default:
        {
            break;
        }
    }
}

static void _add_sub_int(operator_e op, Matrix* target, const Matrix* M)
{
    ACCESS_VOID(int_t, target_values, target->values);
    ACCESS_VOID(int_t, M_values, M->values);
    switch (op)
    {
        case Add:
        {
            ADD(M->m * M->n, target_values[n], M_values[n]);
            break;
        }
        case Sub:
        {
            SUB(M->m * M->n, target_values[n], M_values[n]);
            break;
        }
        default:
        {
            char buf[128];
            snprintf(buf,
                     sizeof(buf),
                     "Invalid value encountered in _add_sub_int() Got op=%d, "
                     "expected 0 <= op < %d",
                     op,
                     OPERATOR_COUNT);
            Log_log(buf, LOG_RT_WARNING);
            break;
        }
    }
}

static void _add_sub_real(operator_e op, Matrix* target, const Matrix* M)
{
    ACCESS_VOID(real_t, target_values, target->values);
    ACCESS_VOID(real_t, M_values, M->values);

    switch (op)
    {
        case Add:
        {
            ADD(M->m * M->n, target_values[n], M_values[n]);
            break;
        }
        case Sub:
        {
            SUB(M->m * M->n, target_values[n], M_values[n]);
            break;
        }
        default:
        {
            char buf[128];
            snprintf(buf,
                     sizeof(buf),
                     "Invalid value encountered in _add_sub_real() Got op=%d, "
                     "expected 0 <= op < %d",
                     op,
                     OPERATOR_COUNT);
            Log_log(buf, LOG_RT_WARNING);
            break;
        }
    }
}

static void _add_sub_cmpl(operator_e op, Matrix* target, const Matrix* M)
{
    ACCESS_VOID(complex_t, target_values, target->values);
    ACCESS_VOID(complex_t, M_values, M->values);

    switch (op)
    {
        case Add:
        {
            ADD(M->m * M->n, target_values[n], M_values[n]);
            break;
        }
        case Sub:
        {
            SUB(M->m * M->n, target_values[n], M_values[n]);
            break;
        }
        default:
        {
            char buf[128];
            snprintf(
              buf,
              sizeof(buf),
              "Invalid value encountered in _add_sub_complex() Got op=%d, "
              "expected 0 <= op < %d",
              op,
              OPERATOR_COUNT);
            Log_log(buf, LOG_RT_WARNING);
            break;
        }
    }
}

static void (*_add_sub_units[TYPE_COUNT])(operator_e op,
                                          Matrix* target,
                                          const Matrix* M) = {
    [Int]     = _add_sub_int,
    [Real]    = _add_sub_real,
    [Complex] = _add_sub_cmpl,
};

int Matrix_add(Matrix* target, const Matrix* M)
{
    if (target->m != M->m || target->n != M->n)
    {
        Log_log("Index out of range in Matrix_add()!", LOG_RT_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }
    DataType dt = (target->dt > M->dt) ? target->dt : M->dt;
    _add_sub_units[dt](Add, target, M);
    return MATRIX_SUCCESS;
}

int Matrix_sub(Matrix* target, const Matrix* M)
{
    if (target->m != M->m || target->n != M->n)
    {
        Log_log("Index out of range in Matrix_sub()!", LOG_RT_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }
    DataType dt = (target->dt > M->dt) ? target->dt : M->dt;
    _add_sub_units[dt](Sub, target, M);
    return MATRIX_SUCCESS;
}

static void _scale_int(Matrix* target, const complex double lambda)
{
    const size_t N        = target->m * target->n;
    const size_t lambda_w = round(creal(lambda));
    ACCESS_VOID(int_t, target_values, target->values);
    for (int n = 0; n < N; n++)
    {
        target_values[n] *= lambda_w;
    }
}

static void _scale_real(Matrix* target, const complex double lambda)
{
    const size_t N        = target->m * target->n;
    const double lambda_w = creal(lambda);
    ACCESS_VOID(real_t, target_values, target->values);
    for (int n = 0; n < N; n++)
    {
        target_values[n] *= lambda_w;
    }
}

static void _scale_cmpl(Matrix* target, const complex double lambda)
{
    const size_t N = target->m * target->n;
    ACCESS_VOID(real_t, target_values, target->values);
    for (int n = 0; n < N; n++)
    {
        target_values[n] *= lambda;
    }
}

static void (*_scale_units[TYPE_COUNT])(Matrix* target,
                                        const complex double lambda) = {
    [Int]     = _scale_int,
    [Real]    = _scale_real,
    [Complex] = _scale_cmpl,
};

void Matrix_scale(Matrix* target, const complex double lambda)
{
    const size_t N = target->m * target->n;
    _scale_units[target->dt](target, lambda);
}

static int _invert_n_n_matrix(Matrix* target, const Matrix* M)
{
    // Assume M is n x n
    const size_t n = M->n;

    // Make a working copy of M
    Matrix M_copy = Matrix_new(0, 0, Real);
    Matrix_copy(&M_copy, M);

    // Initialize target as identity
    Vector ones = Vector_ones(n, M->dt);
    Matrix tmp  = Matrix_diag(&ones);
    Matrix_free(target);
    *target = tmp;
    Vector_free(&ones);

    double* A = M_copy.values;  // n x n
    double* B = target->values; // n x n

    // Gauss–Jordan elimination: [A | I] -> [I | A^{-1}]
    for (size_t k = 0; k < n; ++k)
    {
        // Pivot element
        double pivot = A[k * n + k];
        if (fabs(pivot) < EPS)
        {
            Matrix_free(&M_copy);
            return MATRIX_MATH_ERROR;
        }

        double inv_pivot = 1.0 / pivot;

        // Scale pivot row in A and B
#pragma omp simd aligned(A, B : 32)
        for (size_t j = 0; j < n; ++j)
        {
            A[k * n + j] *= inv_pivot;
            B[k * n + j] *= inv_pivot;
        }

        // Eliminate all other rows
        for (size_t i = 0; i < n; ++i)
        {
            if (i == k)
                continue;

            double lambda = A[i * n + k]; // factor to eliminate column k
            if (fabs(lambda) < EPS)
            {
                // If lambda is ~0, row already has 0 in column k; skip
                continue;
            }

#pragma omp simd aligned(A, B : 32)
            for (size_t j = 0; j < n; ++j)
            {
                A[i * n + j] -= lambda * A[k * n + j];
                B[i * n + j] -= lambda * B[k * n + j];
            }
        }
    }

    Matrix_free(&M_copy);
    return MATRIX_SUCCESS;
}

// Computes LU decomposition of M (n×n) into target.
// target will contain both L and U in compact form:
// - U is in the upper triangle (including diagonal)
// - L is in the lower triangle (unit diagonal implied)
// Returns MATRIX_MATH_ERROR if a zero pivot is encountered.
static int _lu_decompose(Matrix* target, const Matrix* M)
{
    const size_t n = M->n;

    // Copy M into target (LU will be built in-place)
    // Matrix_free(target);
    Matrix_copy(target, M);

    double* A = target->values; // n×n matrix in row-major

    for (size_t k = 0; k < n; ++k)
    {
        double pivot = A[k * n + k];
        if (fabs(pivot) < EPS)
            return MATRIX_MATH_ERROR;

        // Compute U row k (already present)
        // Compute L column k (below pivot)
#pragma omp parallel for
        for (size_t i = k + 1; i < n; ++i)
        {
            A[i * n + k] /= pivot; // L(i,k)

            double lik = A[i * n + k];

#pragma omp simd aligned(A : 32)
            for (size_t j = k + 1; j < n; ++j)
            {
                A[i * n + j] -= lik * A[k * n + j];
            }
        }
    }

    return MATRIX_SUCCESS;
}

static void _forward_sub(double* LU, double* y, const double* b, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        double sum = b[i];
#pragma omp simd
        for (size_t j = 0; j < i; ++j)
            sum -= LU[i * n + j] * y[j];

        y[i] = sum; // L has unit diagonal
    }
}

static void _backward_sub(double* LU, double* x, const double* y, size_t n)
{
    for (long i = n - 1; i >= 0; --i)
    {
        double sum = y[i];
#pragma omp simd
        for (size_t j = i + 1; j < n; ++j)
            sum -= LU[i * n + j] * x[j];

        x[i] = sum / LU[i * n + i];
    }
}

static int _invert_via_lu(Matrix* target, const Matrix* M)
{
    const size_t n = M->n;

    Matrix LU = Matrix_new(0, 0, 0);
    if (_lu_decompose(&LU, M) != MATRIX_SUCCESS)
        return MATRIX_MATH_ERROR;

    Matrix_free(target);
    *target = Matrix_new(n, n, 0.0);

    double* inv = target->values;
    double* A   = LU.values;

    double* y = malloc(n * sizeof(double));
    double* x = malloc(n * sizeof(double));

    for (size_t col = 0; col < n; ++col)
    {
        // b = e_col (unit vector)
        memset(y, 0, n * sizeof(double));
        memset(x, 0, n * sizeof(double));

        double* b = y;
        b[col]    = 1.0;

        _forward_sub(A, y, b, n);
        _backward_sub(A, x, y, n);

        // Write column of inverse
        for (size_t row = 0; row < n; ++row)
            inv[row * n + col] = x[row];
    }

    free(y);
    free(x);
    Matrix_free(&LU);

    return MATRIX_SUCCESS;
}

int _2_x_2_int(Matrix* target, const Matrix* M, void** ptrs)
{
    long *a, *b, *c, *d;
    for (char i = 0; i < 4; i++)
    {
        ptrs[i] = M->values + i;
        a       = (long*)ptrs[i];
    }

    const long denom = *a * *d - *b * *c;

    if (denom == 0)
    {
        Log_log("Math error in Matrix_inverse(), 2x2 case. Matrix is not "
                "invertable.",
                LOG_RT_ERROR);
        return MATRIX_MATH_ERROR;
    }

    const double scalar = 1.0 / denom;
    double init_vals[4] = {
        scalar * *d, scalar * -*b, scalar * -*c, scalar * *a
    };
    bool all_int = true;
    for (char i = 0; i < 4; i++)
    {
        if (fabs(init_vals[i] - (long)init_vals) > EPS)
        {
            all_int = false;
            break;
        }
    }
    // If all values are integers, no need to
    // generalize to a Real valued Matrix
    if (all_int)
    {
        Matrix_prepare_target(target, 2, 2, Int);
        for (char i = 0; i < 4; i++)
        {
            long val = (long)init_vals[i];
            memcpy(target->values + i, &val, elem_size(Int));
        }
        return MATRIX_SUCCESS;
    }

    Matrix_prepare_target(target, 2, 2, Real);
    memcpy(target->values, init_vals, 4 * elem_size(Int));
    return MATRIX_SUCCESS;
}

int _2_x_2_real(Matrix* target, const Matrix* M, void** ptrs)
{
    Matrix_prepare_target(target, 2, 2, Real);
    double *a, *b, *c, *d;
    for (char i = 0; i < 4; i++)
    {
        ptrs[i] = M->values + i;
        a       = (double*)ptrs[i];
    }

    const double denom = *a * *d - *b * *c;

    if (fabs(denom) < EPS)
    {
        Log_log("Math error in Matrix_inverse(), 2x2 case. Matrix is not "
                "invertable.",
                LOG_RT_ERROR);
        return MATRIX_MATH_ERROR;
    }

    const double scalar = 1.0 / denom;
    double init_vals[4] = {
        scalar * *d, scalar * -*b, scalar * -*c, scalar * *a
    };

    memcpy(target->values, init_vals, 4 * elem_size(Real));
    return MATRIX_SUCCESS;
}

int _2_x_2_cmpl(Matrix* target, const Matrix* M, void** ptrs)
{
    Matrix_prepare_target(target, 2, 2, Complex);
    complex double *a, *b, *c, *d;
    for (char i = 0; i < 4; i++)
    {
        ptrs[i] = M->values + i;
        a       = (complex double*)ptrs[i];
    }

    const complex double denom = *a * *d - *b * *c;

    if (cabs(denom) < EPS)
    {
        Log_log("Math error in Matrix_inverse(), 2x2 case. Matrix is not "
                "invertable.",
                LOG_RT_ERROR);
        return MATRIX_MATH_ERROR;
    }

    const complex double scalar = 1.0 / denom;
    complex double init_vals[4] = {
        scalar * *d, scalar * -*b, scalar * -*c, scalar * *a
    };

    memcpy(target->values, init_vals, 4 * elem_size(Complex));
    return MATRIX_SUCCESS;
}

int (*_2_x_2_units[TYPE_COUNT])(Matrix* target,
                                const Matrix* M,
                                void** ptrs) = {
    [Int]     = _2_x_2_int,
    [Real]    = _2_x_2_real,
    [Complex] = _2_x_2_cmpl,
};

int Matrix_inverse(Matrix* target, const Matrix* M)
{
    if (M->m != M->n)
    {
        Log_log("Cannot invert matrix where n!=m", LOG_RT_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }
    if (M->m == 2 && M->n == 2)
    {
        void* ptrs[4] = { NULL };
        _2_x_2_units[M->dt](target, M, ptrs);
    }

    Matrix_prepare_target(target, M->m, M->n, M->dt);
    int status = _invert_via_lu(target, M);
    if (status != MATRIX_SUCCESS)
    {
        Log_log("Unknown math error in Matrix_inverse()", LOG_RT_ERROR);
        return MATRIX_MATH_ERROR;
    }

    return MATRIX_SUCCESS;
}

static void _vector_dot_int(Vector* target, const Matrix* M, const Vector* v)
{
    const size_t DIM = M->m;
    const size_t N   = M->n;
    Vector_prepare_target(target, DIM, M->dt);
    int_t* restrict M_values      = M->values;
    int_t* restrict v_values      = v->values;
    int_t* restrict target_values = target->values;
    int_t sum                     = 0.0;
    int_t M_val, V_val;

#pragma omp parallel for
    for (size_t d = 0; d < DIM; d++)
    {
        sum = 0.0;
#pragma omp simd aligned(M_values, v_values : 32)
        for (size_t n = 0; n < N; n++)
        {
            M_val = M_values[d * M->n + n];
            V_val = v_values[n];
            sum += M_val * V_val;
        }
        target_values[d] = sum;
    }
}

static void _vector_dot_real(Vector* target, const Matrix* M, const Vector* v)
{
    const size_t DIM = M->m;
    const size_t N   = M->n;
    Vector_prepare_target(target, DIM, M->dt);
    real_t* restrict M_values      = M->values;
    real_t* restrict v_values      = v->values;
    real_t* restrict target_values = target->values;
    real_t sum                     = 0.0;
    real_t M_val, V_val;

#pragma omp parallel for
    for (size_t d = 0; d < DIM; d++)
    {
        sum = 0.0;
#pragma omp simd aligned(M_values, v_values : 32)
        for (size_t n = 0; n < N; n++)
        {
            M_val = M_values[d * M->n + n];
            V_val = v_values[n];
            sum += M_val * V_val;
        }
        target_values[d] = sum;
    }
}

static void _vector_dot_cmpl(Vector* target, const Matrix* M, const Vector* v)
{
    const size_t DIM = M->m;
    const size_t N   = M->n;
    Vector_prepare_target(target, DIM, M->dt);
    complex_t* restrict M_values      = M->values;
    complex_t* restrict v_values      = v->values;
    complex_t* restrict target_values = target->values;
    complex_t sum                     = 0.0;
    complex_t M_val, V_val;

#pragma omp parallel for
    for (size_t d = 0; d < DIM; d++)
    {
        sum = 0.0;
#pragma omp simd aligned(M_values, v_values : 32)
        for (size_t n = 0; n < N; n++)
        {
            M_val = M_values[d * M->n + n];
            V_val = v_values[n];
            sum += M_val * V_val;
        }
        target_values[d] = sum;
    }
}

static void (*_vector_dot_units[TYPE_COUNT])(Vector* target,
                                             const Matrix* M,
                                             const Vector* v) = {
    [Int]     = _vector_dot_int,
    [Real]    = _vector_dot_real,
    [Complex] = _vector_dot_cmpl,
};

int Matrix_Vector_dot(Vector* target, const Matrix* M, const Vector* v)
{
    if (M->n != v->dim)
    {
        Log_log("Index out of range in Matrix_Vector_dot()", LOG_RT_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }
    DataType dt = MAX(M->dt, v->dt);
    _vector_dot_units[dt](target, M, v);
    return MATRIX_SUCCESS;
}

static void _matrix_dot_int(Matrix* target, const Matrix* M1, const Matrix* M2)
{
    int_t* restrict M1_values     = M1->values;
    int_t* restrict M2_values     = M2->values;
    int_t* restrict target_values = target->values;
    int_t sum                     = 0.0;
    int_t a, b;
    size_t K = M1->n;

#pragma omp parallel for collapse(2)
    for (size_t m = 0; m < target->m; m++)
    {
        for (size_t n = 0; n < target->n; n++)
        {
            sum = 0.0;

#pragma omp simd aligned(M1_values, M2_values, target_values : 32)
            for (size_t o = 0; o < M1->n; o++)
            {
                a = M1_values[m * M1->n + o];
                b = M2_values[o * M2->n + n];
                sum += a * b;
            }
            target_values[m * target->n + n] = sum;
        }
    }
}

static void _matrix_dot_real(Matrix* target, const Matrix* M1, const Matrix* M2)
{
    real_t* restrict M1_values     = M1->values;
    real_t* restrict M2_values     = M2->values;
    real_t* restrict target_values = target->values;
    real_t sum                     = 0.0;
    real_t a, b;
    size_t K = M1->n;

#pragma omp parallel for collapse(2)
    for (size_t m = 0; m < target->m; m++)
    {
        for (size_t n = 0; n < target->n; n++)
        {
            sum = 0.0;

#pragma omp simd aligned(M1_values, M2_values, target_values : 32)
            for (size_t o = 0; o < M1->n; o++)
            {
                a = M1_values[m * M1->n + o];
                b = M2_values[o * M2->n + n];
                sum += a * b;
            }
            target_values[m * target->n + n] = sum;
        }
    }
}

static void _matrix_dot_cmpl(Matrix* target, const Matrix* M1, const Matrix* M2)
{
    complex_t* restrict M1_values     = M1->values;
    complex_t* restrict M2_values     = M2->values;
    complex_t* restrict target_values = target->values;
    complex_t sum                     = 0.0;
    complex_t a, b;
    size_t K = M1->n;

#pragma omp parallel for collapse(2)
    for (size_t m = 0; m < target->m; m++)
    {
        for (size_t n = 0; n < target->n; n++)
        {
            sum = 0.0;

#pragma omp simd aligned(M1_values, M2_values, target_values : 32)
            for (size_t o = 0; o < M1->n; o++)
            {
                a = M1_values[m * M1->n + o];
                b = M2_values[o * M2->n + n];
                sum += a * b;
            }
            target_values[m * target->n + n] = sum;
        }
    }
}

static void (*_matrix_dot_units[TYPE_COUNT])(Matrix* target,
                                             const Matrix* M1,
                                             const Matrix* M2) = {
    [Int]     = _matrix_dot_int,
    [Real]    = _matrix_dot_real,
    [Complex] = _matrix_dot_cmpl,
};

int Matrix_Matrix_dot(Matrix* target, const Matrix* M1, const Matrix* M2)
{
    if (M1->n != M2->m)
    {
        Log_log("Index out of range in Matrix_Matrix_dot()", LOG_RT_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }
    Matrix_prepare_target(target, M1->n, M2->m, M2->dt);
    DataType dt = MAX(M1->dt, M2->dt);
    _matrix_dot_units[dt](target, M1, M2);
    return MATRIX_SUCCESS;
}

static void _jacobi_real(Matrix* target,
                         const size_t N,
                         const size_t M,
                         Vector* x_eps,
                         Vector* f_x_eps,
                         const Vector* f_x,
                         Vector (*f)(const Vector*))
{
    real_t* x_eps_values = x_eps->values;

    real_t fn_x, fn_x_eps;
    for (size_t m = 0; m < M; m++)
    {
        for (size_t n = 0; n < N; n++)
        {
            x_eps_values[n] += MATRIX_EPS;
            *f_x_eps = f(x_eps);
            x_eps_values[n] -= MATRIX_EPS;
            ASSIGN_TYPED(real_t, &fn_x, f_x->values + m);
            ASSIGN_TYPED(real_t, &fn_x_eps, f_x_eps->values + m);

            real_t value = (fn_x_eps - fn_x) / MATRIX_EPS;
            memcpy(target->values + m * target->n + n, &value, elem_size(Real));
        }
    }
}

static void _jacobi_cmpl(Matrix* target,
                         const size_t M,
                         const size_t N,
                         Vector* x_eps,
                         Vector* f_x_eps,
                         const Vector* f_x,
                         Vector (*f)(const Vector*))
{
    complex_t* x_eps_values = x_eps->values;
    complex_t fn_x, fn_x_eps;

    for (size_t m = 0; m < M; m++)
    {
        for (size_t n = 0; n < N; n++)
        {
            x_eps_values[n] += MATRIX_EPS + MATRIX_EPS * I;
            *f_x_eps = f(x_eps);
            x_eps_values[n] -= MATRIX_EPS + MATRIX_EPS * I;
            ASSIGN_TYPED(complex_t, &fn_x, f_x->values + m);
            ASSIGN_TYPED(complex_t, &fn_x_eps, f_x_eps->values + m);

            real_t value = (fn_x_eps - fn_x) / MATRIX_EPS;
            memcpy(
              target->values + m * target->n + n, &value, elem_size(Complex));
        }
    }
}

static void (*_jacobi_workers[TYPE_COUNT])(Matrix* target,
                                           const size_t N,
                                           const size_t M,
                                           Vector* x_eps,
                                           Vector* f_x_eps,
                                           const Vector* f_x,
                                           Vector (*f)(const Vector*)) = {
    [Real]    = _jacobi_real,
    [Complex] = _jacobi_cmpl,
};

int Matrix_jacobi(Matrix* target, const Vector* x, Vector (*f)(const Vector* x))
{
    const Vector f_x = f(x);
    const size_t M   = f_x.dim;
    const size_t N   = x->dim;
    Vector f_x_eps   = Vector_zeros(M, x->dt);
    Matrix_prepare_target(target, M, N, x->dt);
    Vector x_eps = Vector_zeros_like(x);
    Vector_copy(&x_eps, x);

    _jacobi_workers[f_x.dt](target, M, N, &x_eps, &f_x_eps, &f_x, f);

    return MATRIX_SUCCESS;
}

static void _solve_mv_p1_real(Vector* target,
                              const Matrix* M,
                              const Vector* v,
                              void* ptr,
                              Matrix* M_copy)
{
    ACCESS_VOID(real_t, target_values, target->values);
    for (size_t m = 0; m < M->m; m++)
    {
        ptr           = M_copy->values + m * M_copy->n + m;
        real_t lambda = 1.0 / *(real_t*)ptr;
        ACCESS_VOID(real_t, row_vals, M_copy->values + m * M_copy->n);
        Vector row = Vector_new_vals(M->n, row_vals, M_copy->dt);
        Vector_scale(&row, &lambda);
        memcpy(M_copy->values + m * M_copy->n,
               row.values,
               M_copy->n * elem_size(Real));

        target_values[m] *= lambda;
        real_t target_val = target_values[m];

        for (size_t n = m + 1; n < M->m; n++)
        {
            // ptr = M_copy->values + n * M_copy->n + m;

            ASSIGN_TYPED(real_t, &lambda, M_copy->values + n * M_copy->n + m);
            Vector_scale(&row, &lambda);
            target_val *= lambda;
            row_vals     = M_copy->values + n * M->n;
            Vector row_n = Vector_new_vals(M_copy->n, row_vals, Real);
            Vector_sub(&row_n, &row);

            memcpy(M_copy->values + n * M_copy->n,
                   row_n.values,
                   M->n * elem_size(Real));
            target_values[n] -= target_val;
            real_t lambda_inv = 1.0 / lambda;

            Vector_scale(&row, &lambda_inv);
            target_val *= lambda_inv;

            Vector_free(&row_n);
        }
        Vector_free(&row);
    }
}

static void _solve_mv_p1_cmpl(Vector* target,
                              const Matrix* M,
                              const Vector* v,
                              void* ptr,
                              Matrix* M_copy)
{
    ACCESS_VOID(complex_t, target_values, target->values);
    for (size_t m = 0; m < M->m; m++)
    {
        ptr              = M_copy->values + m * M_copy->n + m;
        complex_t lambda = 1.0 / *(real_t*)ptr;
        ACCESS_VOID(complex_t, row_vals, M_copy->values + m * M_copy->n);
        Vector row = Vector_new_vals(M->n, row_vals, M_copy->dt);
        Vector_scale(&row, &lambda);
        memcpy(M_copy->values + m * M_copy->n,
               row.values,
               M_copy->n * elem_size(Complex));

        target_values[m] *= lambda;
        complex_t target_val = target_values[m];

        for (size_t n = m + 1; n < M->m; n++)
        {
            ptr = M_copy->values + n * M_copy->n + m;

            ASSIGN_TYPED(
              complex_t, &lambda, M_copy->values + n * M_copy->n + m);
            Vector_scale(&row, &lambda);
            target_val *= lambda;
            row_vals     = M_copy->values + n * M->n;
            Vector row_n = Vector_new_vals(M_copy->n, row_vals, Complex);
            Vector_sub(&row_n, &row);

            memcpy(M_copy->values + n * M_copy->n,
                   row_n.values,
                   M->n * elem_size(Complex));
            target_values[n] -= target_val;
            complex_t lambda_inv = 1.0 / lambda;

            Vector_scale(&row, &lambda_inv);
            target_val *= lambda_inv;

            Vector_free(&row_n);
        }
        Vector_free(&row);
    }
}

static void (*_solve_mv_p1_units[TYPE_COUNT])(Vector* target,
                                              const Matrix* M,
                                              const Vector* v,
                                              void* ptr,
                                              Matrix* M_copy) = {
    [Real]    = _solve_mv_p1_real,
    [Complex] = _solve_mv_p1_cmpl,
};

static void _solve_mv_p2_real(Vector* target,
                              const Matrix* M,
                              const Vector* v,
                              void* ptr,
                              Matrix* M_copy)
{
    ACCESS_VOID(real_t, target_values, target->values);
    for (long m = M_copy->m - 1; m >= 0; m--)
    {
        real_t lambda;
        ACCESS_VOID(real_t, row_vals, M_copy->values + m * M_copy->n);

        Vector row        = Vector_new_vals(M->n, row_vals, M_copy->dt);
        real_t target_val = target_values[m];
        for (long n = m - 1; n >= 0; n--)
        {
            ptr = M_copy->values + n * M_copy->n + m;
            ASSIGN_TYPED(real_t, &lambda, M_copy->values + n * M_copy->n + m);
            Vector_scale(&row, &lambda);
            target_val *= lambda;

            row_vals     = (real_t*)M_copy->values + n * M->n;
            Vector row_n = Vector_new_vals(M_copy->n, row_vals, M_copy->dt);
            Vector_sub(&row_n, &row);
            memcpy(M_copy->values + n * M_copy->n,
                   row_n.values,
                   M_copy->n * elem_size(Real));
            target_values[n] -= target_val;
            real_t lambda_inv = 1.0 / lambda;

            Vector_scale(&row, &lambda_inv);
            target_val /= lambda;

            Vector_free(&row_n);
        }
        Vector_free(&row);
    }
}

static void _solve_mv_p2_cmpl(Vector* target,
                              const Matrix* M,
                              const Vector* v,
                              void* ptr,
                              Matrix* M_copy)
{
    ACCESS_VOID(complex_t, target_values, target->values);
    for (long m = M_copy->m - 1; m >= 0; m--)
    {
        complex_t lambda;
        ACCESS_VOID(complex_t, row_vals, M_copy->values + m * M_copy->n);

        Vector row           = Vector_new_vals(M->n, row_vals, M_copy->dt);
        complex_t target_val = target_values[m];
        for (long n = m - 1; n >= 0; n--)
        {
            ptr = M_copy->values + n * M_copy->n + m;
            ASSIGN_TYPED(
              complex_t, &lambda, M_copy->values + n * M_copy->n + m);
            Vector_scale(&row, &lambda);
            target_val *= lambda;

            row_vals     = (complex_t*)M_copy->values + n * M->n;
            Vector row_n = Vector_new_vals(M_copy->n, row_vals, M_copy->dt);
            Vector_sub(&row_n, &row);
            memcpy(M_copy->values + n * M_copy->n,
                   row_n.values,
                   M_copy->n * elem_size(Complex));
            target_values[n] -= target_val;
            complex_t lambda_inv = 1.0 / lambda;

            Vector_scale(&row, &lambda_inv);
            target_val /= lambda;

            Vector_free(&row_n);
        }
        Vector_free(&row);
    }
}

static void (*_solve_mv_p2_units[TYPE_COUNT])(Vector* target,
                                              const Matrix* M,
                                              const Vector* v,
                                              void* ptr,
                                              Matrix* M_copy) = {
    [Real]    = _solve_mv_p2_real,
    [Complex] = _solve_mv_p2_cmpl,
};

int Matrix_Vector_solve(Vector* target, const Matrix* M, const Vector* v)
{
    if (M->n != v->dim || M->m != M->n)
    {
        Log_log("Dimension error in Matrix_Vector_solve()", LOG_RT_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }
    Matrix M_copy = Matrix_new(M->m, M->n, 0);
    Matrix_copy(&M_copy, M);
    Vector_copy(target, v);
    void* ptr = NULL;

    // Phase one
    DataType dt = MAX(M->dt, v->dt);
    _solve_mv_p1_units[dt](target, M, v, ptr, &M_copy);

    // Phase two
    _solve_mv_p2_units[dt](target, M, v, ptr, &M_copy);

    Matrix_free(&M_copy);
    return MATRIX_SUCCESS;
}

int Matrix_column_pivot(Matrix* A, size_t k, size_t* P)
{
    size_t n = A->n;
    size_t m = A->m;

    if (k >= n)
    {
        Log_log("Index out of range in Matrix_column_pivot()", LOG_RT_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }

    // Find column with largest norm among k..n-1
    double best_norm = -INFINITY;
    size_t best_j    = k;

    for (size_t j = k; j < n; j++)
    {
        double norm = 0.0;
        for (size_t i = 0; i < m; i++)
        {
            void* ptr = A->values + i * n + j;
            norm += *(double*)ptr * *(double*)ptr;
        }

        if (norm > best_norm)
        {
            best_norm = norm;
            best_j    = j;
        }
    }

    // Swap columns k and best_j
    if (best_j != k)
    {
        for (size_t i = 0; i < m; i++)
        {
            void* ptr          = A->values + i * n + k;
            double* tmp        = (double*)ptr;
            void* A_i_n_k_ptr  = A->values + i * n + k;
            void* A_i_n_bj_ptr = A->values + i * n + best_j;
            double* best_j_val = (double*)A_i_n_bj_ptr;
            // A->values[i * n + k]      = A->values[i * n + best_j];
            // A->values[i * n + best_j] = tmp;
            memcpy(A_i_n_k_ptr, best_j_val, elem_size(Real));
            memcpy(A_i_n_bj_ptr, tmp, elem_size(Real));
        }

        // Update permutation vector
        size_t tmp = P[k];
        P[k]       = P[best_j];
        P[best_j]  = tmp;
    }

    return MATRIX_SUCCESS;
}

int Matrix_Matrix_dot_jordan(Matrix* target, const Matrix* M1, const Matrix* M2)
{
    if (M1->m != M1->n || M2->m != M2->n || M1->m != M2->m)
    {
        Log_log("Jordan multiplication requires symmetrical matrices",
                LOG_RT_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }
    Matrix_prepare_target(target, M1->m, M1->n, M1->dt);
    Matrix AB = Matrix_new(0, 0, 0.0);
    Matrix BA = Matrix_new(0, 0, 0.0);
    Matrix_Matrix_dot(&AB, M1, M2);
    Matrix_Matrix_dot(&BA, M2, M1);
    Matrix_scale(&AB, 0.5);
    Matrix_scale(&BA, 0.5);
    Matrix_add(&AB, &BA);
    Matrix_add(target, &AB);
    Matrix_free(&AB);
    Matrix_free(&BA);
    return MATRIX_SUCCESS;
}

// int Matrix_QR(Matrix* Q, Matrix* R, const Matrix* A)
// {
//     size_t m = A->m;
//     size_t n = A->n;
//     Matrix_copy(R, A);
//     complex double* ones = malloc(A->n * elem_size(Complex));
//     Matrix Unit          = Matrix_diag_val(A->n, ones, Complex);
//     free(ones);
//     Matrix_copy(Q, &Unit);
//     Matrix_free(&Unit);
//
//     for (size_t k = 0; k < A->n; k++)
//     {
//         size_t rows = m - k;
//
//         Vector x = Vector_zeros(rows, Real);
//         for (size_t i = 0; i < rows; i++)
//         {
//             void* R_ptr   = R->values + (k + i) * n + k;
//             void* x_ptr   = x.values + i;
//             double* R_val = (double*)R_ptr;
//             // x.values[i]   = R->values[(k + i) * n + k];
//             memcpy(x_ptr, R_val, elem_size(Real));
//         }
//
//         double norm_x = Vector_norm(&x);
//
//         double sign   = (x.values[0] >= 0 ? 1 : -1);
//
//         Vector e1    = Vector_zeros(rows, Real);
//         e1.values[0] = 1;
//
//         Vector v = Vector_zeros_like(&x);
//         Vector_copy(&v, &x);
//         Vector_scale(&e1, sign * norm_x);
//         Vector_add(&v, &e1);
//
//         Vector_free(&e1);
//         Vector_scale(&v, 1.0 / Vector_norm(&v));
//
//         for (size_t j = k; j < n; j++)
//         {
//             double dot = 0.0;
//             for (size_t i = 0; i < rows; i++)
//             {
//                 double* v_ptr = v.values + i;
//                 double* R_ptr = R->values + (k + i) * n + k;
//                 dot += *v_ptr * *R_ptr;
//             }
//             for (size_t i = 0; i < rows; i++)
//             {
//                 double* v_val = (double*)v.values + i;
//                 double* R_val = (double*)R->values + (k + i) * n + k;
//                 *R_val -= *v_val;
//                 // R->values[(k + i) * n + j] -= 2 * v.values[i] * dot;
//             }
//         }
//
//         for (size_t i = 0; i < m; i++)
//         {
//             double dot = 0.0;
//             for (size_t r = 0; r < rows; r++)
//             {
//                 double* Q_val = Q->values + i * n + (k + r);
//                 double* v_val = v.values + r;
//                 dot += *Q_val * *v_val;
//             }
//
//             for (size_t r = 0; r < rows; r++)
//             {
//                 double* Q_val = Q->values + i * n + (k + r);
//                 double* v_val = v.values + r;
//                 *Q_val -= 2 * dot * *v_val;
//                 // Q->values[i * n + (k + r)] -= 2 * dot * v.values[r];
//             }
//         }
//         Vector_free(&v);
//         Vector_free(&x);
//     }
//
//     return MATRIX_SUCCESS;
// }
//
// void _Matrix_Matrix_dot_active_raw(double* restrict H_values,
//                                    const double* restrict R_values,
//                                    const double* restrict Q_values,
//                                    const size_t active,
//                                    const size_t m,
//                                    const size_t n)
// {
//     double* tmp_values = NULL;
//     posix_memalign((void**)&tmp_values, 32, m * n * sizeof(double));
//
//     for (size_t i = 0; i < active; i++)
//     {
// #pragma omp simd aligned(R_values, Q_values : 32)
//         for (size_t j = 0; j < active; j++)
//         {
//             double sum = 0.0;
//             for (size_t k = 0; k < active; k++)
//                 sum += R_values[i * n + k] * Q_values[k * n + j];
//
//             tmp_values[i * n + j] = sum;
//         }
//     }
//
//     // Copy back only active block
// #pragma omp simd aligned(H_values, tmp_values : 32)
//     for (size_t i = 0; i < active; i++)
//         for (size_t j = 0; j < active; j++)
//             H_values[i * n + j] = tmp_values[i * n + j];
//
//     free(tmp_values);
// }
//
// void _Matrix_QR_hessenberg_raw(double* restrict Q_values,
//                                double* restrict R_values,
//                                double* restrict H_values,
//                                size_t active,
//                                const size_t n)
// {
//     for (size_t i = 0; i < n; i++)
//     {
//         for (size_t j = 0; j < n; j++)
//         {
//             Q_values[i * n + j] = (i == j ? 1.0 : 0.0);
//         }
//     }
//     memcpy(R_values, H_values, n * n * sizeof(double));
//     for (size_t i = 0; i < active - 1; i++)
//     {
//         double a = R_values[i * n + i];
//         double b = R_values[(i + 1) * n + i];
//
//         if (fabs(b) < 1e-15)
//         {
//             continue;
//         }
//
//         double r = hypot(a, b);
//         double c = a / r;
//         double s = -b / r;
// #pragma omp simd aligned(R_values : 32)
//         for (size_t j = i; j < active; j++)
//         {
//             double t1                 = R_values[i * n + j];
//             double t2                 = R_values[(i + 1) * n + j];
//             R_values[i * n + j]       = c * t1 - s * t2;
//             R_values[(i + 1) * n + j] = s * t1 + c * t2;
//         }
//
// #pragma omp simd aligned(Q_values : 32)
//         for (size_t j = 0; j < active; j++)
//         {
//             double t1                 = Q_values[j * n + i];
//             double t2                 = Q_values[j * n + (i + 1)];
//             Q_values[j * n + i]       = c * t1 - s * t2;
//             Q_values[j * n + (i + 1)] = s * t1 + c * t2;
//         }
//     }
// }
//
// int _eigvals_two(Vector* eigenvalues, const Matrix* M)
// {
//     const double* a = M->values + 0;
//     const double* b = M->values + 1;
//     const double* c = M->values + 2;
//     const double* d = M->values + 3;
//
//     const double p     = -(*a + *d);
//     const double q     = *a * *d - *b * *c;
//     const double inner = 0.25 * p * p - q;
//     Vector_prepare_target(eigenvalues, 2);
//     if (inner < 0.0)
//     {
//         Log_log("complex eigenvalues encountered", LOG_RT_WARNING);
//         return MATRIX_MATH_ERROR;
//     }
//
//     const double x1        = -p * 0.5 - sqrt(inner);
//     const double x2        = -p * 0.5 + sqrt(inner);
//     eigenvalues->values[0] = x1;
//     eigenvalues->values[1] = x2;
//     return MATRIX_SUCCESS;
// }
//
// double _wilkinson_shift(const Matrix* H, size_t active)
// {
//     size_t n = H->n;
//
//     double* a = H->values + (active - 2) * n + (active - 2);
//     double* b = H->values + (active - 2) * n + (active - 1);
//     double* c = H->values + (active - 1) * n + (active - 2);
//     double* d = H->values + (active - 1) * n + (active - 1);
//
//     double tr        = (*a + *d) * 0.5;
//     double det       = (*a - *d) * 0.5;
//     double disc_term = det * det + *b * *c;
//     if (disc_term < 0.0)
//     {
//         disc_term = 0.0;
//     }
//     double disc = sqrt(disc_term);
//
//     double lambda1 = tr + disc;
//     double lambda2 = tr - disc;
//
//     return (fabs(lambda1 - *d) < fabs(lambda2 - *d)) ? lambda1 : lambda2;
// }
//
// int _eigvals_big(Vector* eigenvalues, const Matrix* M, bool sort)
// {
//     const size_t max_iters = 10000;
//     const double eps       = 1e-12;
//
//     Matrix H = Matrix_zeros_like(M, Complex);
//     Matrix_Hessenberg(&H, M); // Q^T M Q → upper Hessenberg
//     Matrix Q;
//     Q.m      = H.m;
//     Q.n      = H.n;
//     Q.values = NULL;
//     posix_memalign((void**)&Q.values, 32, Q.m * Q.n * sizeof(double));
//
//     Matrix R;
//     R.m      = H.m;
//     R.n      = H.n;
//     R.values = NULL;
//     posix_memalign((void**)&R.values, 32, Q.m * Q.n * sizeof(double));
//
//     size_t n      = H.n;
//     size_t active = n;
//
//     size_t total_iters        = 0;
//     double* restrict H_values = H.values;
//     double* restrict Q_values = Q.values;
//     double* restrict R_values = R.values;
//
//     while (active > 1)
//     {
//         for (size_t iter = 0; iter < max_iters; iter++)
//         {
//             total_iters++;
//             double sub = fabs(H_values[(active - 1) * n + (active - 2)]);
//
//             if (sub < eps)
//             {
//                 break;
//             }
//
//             double mu = _wilkinson_shift(&H, active);
//
// #pragma omp simd
//             for (size_t i = 0; i < active; i++)
//             {
//                 // H.values[i * n + i] -= mu;
//                 H_values[i * n + i] -= mu;
//             }
//             // Matrix_QR_hessenberg(&Q, &R, &H, active);
//             _Matrix_QR_hessenberg_raw(
//               Q_values, R_values, H_values, active, Q.n);
//             // Matrix_Matrix_dot_active(&H, &R, &Q, active);
//             _Matrix_Matrix_dot_active_raw(
//               H_values, R_values, Q_values, active, H.m, H.n);
//
// #pragma omp simd
//             for (size_t i = 0; i < active; i++)
//             {
//                 H_values[i * n + i] += mu;
//             }
//             bool converged = true;
//             for (size_t i = 0; i < active - 1; i++)
//             {
//                 if (fabs(H_values[(i + 1) * n + i]) > eps)
//                 {
//                     converged = false;
//                     break;
//                 }
//             }
//             if (converged)
//             {
//                 break;
//             }
//         }
//
//         if (fabs(H_values[(active - 1) * n + (active - 2)]) < eps)
//         {
//             active--;
//             continue;
//         }
//         else
//         {
//             break;
//         }
//     }
//     Vector_prepare_target(eigenvalues, n);
// #pragma omp simd
//     for (size_t i = 0; i < n; i++)
//     {
//         eigenvalues->values[i] = H_values[i * n + i];
//     }
//     if (sort)
//     {
//         Vector_sort_inplace(eigenvalues);
//     }
//
//     Matrix_free(&Q);
//     Matrix_free(&R);
//     Matrix_free(&H);
//
//     return MATRIX_SUCCESS;
// }
//
// int Matrix_eigvals(Vector* eigenvalues, const Matrix* M, bool sort)
// {
//     if (M->n == 2)
//     {
//         return _eigvals_two(eigenvalues, M);
//     }
//
//     return _eigvals_big(eigenvalues, M, sort);
// }

static void _outer_int_s(Matrix* target, const Vector* a, const Vector* b)
{
    ACCESS_VOID(int_t, a_values, a->values);
    ACCESS_VOID(int_t, b_values, b->values);
    ACCESS_VOID(int_t, target_values, target->values);

    for (size_t m = 0; m < a->dim; m++)
    {
        for (size_t n = 0; n < b->dim; n++)
        {
            int_t value                      = a_values[m] * b_values[n];
            target_values[m * target->n + n] = value;
        }
    }
}

static void _outer_real_s(Matrix* target, const Vector* a, const Vector* b)
{
    ACCESS_VOID(real_t, a_values, a->values);
    ACCESS_VOID(real_t, b_values, b->values);
    ACCESS_VOID(real_t, target_values, target->values);

    for (size_t m = 0; m < a->dim; m++)
    {
        for (size_t n = 0; n < b->dim; n++)
        {
            real_t value                     = a_values[m] * b_values[n];
            target_values[m * target->n + n] = value;
        }
    }
}

static void _outer_cmpl_s(Matrix* target, const Vector* a, const Vector* b)
{
    ACCESS_VOID(complex_t, a_values, a->values);
    ACCESS_VOID(complex_t, b_values, b->values);
    ACCESS_VOID(complex_t, target_values, target->values);

    for (size_t m = 0; m < a->dim; m++)
    {
        for (size_t n = 0; n < b->dim; n++)
        {
            real_t value                     = a_values[m] * b_values[n];
            target_values[m * target->n + n] = value;
        }
    }
}

static void (*_outer_units_s[TYPE_COUNT])(Matrix* target,
                                          const Vector* a,
                                          const Vector* b) = {
    [Int]     = _outer_int_s,
    [Real]    = _outer_real_s,
    [Complex] = _outer_cmpl_s,
};

static void _outer_int_p(Matrix* target, const Vector* a, const Vector* b)
{
    ACCESS_VOID(int_t, a_values, a->values);
    ACCESS_VOID(int_t, b_values, b->values);
    ACCESS_VOID(int_t, target_values, target->values);

#pragma omp parallel for // parallel outer basically always faster
    for (size_t m = 0; m < a->dim; m++)
    {
        for (size_t n = 0; n < b->dim; n++)
        {
            int_t value                      = a_values[m] * b_values[n];
            target_values[m * target->n + n] = value;
        }
    }
}

static void _outer_real_p(Matrix* target, const Vector* a, const Vector* b)
{
    ACCESS_VOID(real_t, a_values, a->values);
    ACCESS_VOID(real_t, b_values, b->values);
    ACCESS_VOID(real_t, target_values, target->values);

#pragma omp parallel for // parallel outer basically always faster
    for (size_t m = 0; m < a->dim; m++)
    {
        for (size_t n = 0; n < b->dim; n++)
        {
            real_t value                     = a_values[m] * b_values[n];
            target_values[m * target->n + n] = value;
        }
    }
}

static void _outer_cmpl_p(Matrix* target, const Vector* a, const Vector* b)
{
    ACCESS_VOID(complex_t, a_values, a->values);
    ACCESS_VOID(complex_t, b_values, b->values);
    ACCESS_VOID(complex_t, target_values, target->values);

#pragma omp parallel for // parallel outer basically always faster
    for (size_t m = 0; m < a->dim; m++)
    {
        for (size_t n = 0; n < b->dim; n++)
        {
            real_t value                     = a_values[m] * b_values[n];
            target_values[m * target->n + n] = value;
        }
    }
}

static void (*_outer_units_p[TYPE_COUNT])(Matrix* target,
                                          const Vector* a,
                                          const Vector* b) = {
    [Int]     = _outer_int_p,
    [Real]    = _outer_real_p,
    [Complex] = _outer_cmpl_p,
};

int _outer_scalar(Matrix* target, const Vector* a, const Vector* b)
{
    DataType dt = MAX(a->dt, b->dt);
    _outer_units_s[dt](target, a, b);
    return MATRIX_SUCCESS;
}

int _outer_parallel(Matrix* target, const Vector* a, const Vector* b)
{
    DataType dt = MAX(a->dt, b->dt);
    _outer_units_p[dt](target, a, b);
    return MATRIX_SUCCESS;
}

int Matrix_Vector_outer(Matrix* target, const Vector* a, const Vector* b)
{
    Matrix_prepare_target(target, a->dim, b->dim, MAX(a->dt, b->dt));
    if (a->dim * b->dim > 200 * 200)
    {
        return _outer_parallel(target, a, b);
    }
    return _outer_scalar(target, a, b);
}

// deprecated, gets outperformed by naiive outer product
int Matrix_Vector_outer_square(Matrix* target, const Vector* v)
{
    return Matrix_Vector_outer(target, v, v);
}

static void _hess_1_real(Matrix* H, const Matrix* A)
{
    size_t m = A->m;
    size_t n = A->n;
    Matrix_copy(H, A);
    // Matrix_prepare_target(H, m, n);
    real_t* restrict H_values = H->values;

    for (size_t k = 0; k < m - 2; k++)
    {
        size_t rows = m - (k + 1);

        Vector x;
        x.values = NULL;
        // x.values = malloc(rows * sizeof(double));
        posix_memalign((void**)&x.values, 32, rows * elem_size(Real));
        x.dim        = rows;
        real_t normx = 0.0;
        ACCESS_VOID(real_t, x_values, x.values);

        for (size_t i = 0; i < rows; i++)
        {
            real_t val  = H_values[(k + 1 + i) * n + k];
            x_values[i] = val;
            normx += val * val;
        }
        if (normx == 0.0)
        {
            Vector_free(&x);
            continue;
        }
        normx = sqrt(normx);

        real_t sign = (x_values[0] >= 0.0) ? 1.0 : -1.0;

        Vector e1 = Vector_zeros(rows, Real);
        ACCESS_VOID(real_t, e1_values, e1.values);
        e1_values[0] = 1.0;

        Vector v;
        // v.values = malloc(x.dim * sizeof(double));
        v.values = NULL;
        posix_memalign((void**)&v.values, 32, x.dim * elem_size(Real));
        v.dim = x.dim;
        Vector_copy(&v, &x);
        real_t prod = sign * normx;
        Vector_scale(&e1, &prod);
        Vector_add(&v, &e1);
        Vector_free(&e1);
        ACCESS_VOID(real_t, v_values, v.values);

        real_t vnorm = 0.0;

        for (size_t i = 0; i < v.dim; i++)
        {
            vnorm += v_values[i] * v_values[i];
        }
        vnorm = sqrt(vnorm);

        // Vector_scale(&v, 1.0 / vnorm);
        for (size_t i = 0; i < v.dim; i++)
        {
            v_values[i] /= vnorm;
        }

        for (size_t j = k; j < n; j++)
        {
            real_t dot = 0.0;
            for (size_t i = 0; i < rows; i++)
            {
                dot += v_values[i] * H_values[(k + 1 + i) * n + j];
            }

            for (size_t i = 0; i < rows; i++)
            {
                H_values[(k + 1 + i) * n + j] -= 2.0 * v_values[i] * dot;
            }
        }

        for (size_t i = 0; i < m; i++)
        {
            real_t dot = 0.0;
            for (size_t r = 0; r < rows; r++)
            {
                dot += H_values[i * n + (k + 1 + r)] * v_values[r];
            }
            for (size_t r = 0; r < rows; r++)
            {
                H_values[i * n + (k + 1 + r)] -= 2.0 * dot * v_values[r];
            }
        }
        Vector_free(&v);
        Vector_free(&x);
    }
}

static void _hess_1_cmpl(Matrix* H, const Matrix* A)
{
    size_t m = A->m;
    size_t n = A->n;
    Matrix_copy(H, A);
    // Matrix_prepare_target(H, m, n);
    complex_t* restrict H_values = H->values;

    for (size_t k = 0; k < m - 2; k++)
    {
        size_t rows = m - (k + 1);

        Vector x;
        x.values = NULL;
        // x.values = malloc(rows * sizeof(double));
        posix_memalign((void**)&x.values, 32, rows * elem_size(Complex));
        x.dim        = rows;
        real_t normx = 0.0;
        ACCESS_VOID(complex_t, x_values, x.values);

        for (size_t i = 0; i < rows; i++)
        {
            complex_t val      = H_values[(k + 1 + i) * n + k];
            x_values[i]        = val;
            complex_t val_conj = creal(val) - cimag(val) * I;
            normx += creal(val * val_conj);
        }
        if (normx == 0.0)
        {
            Vector_free(&x);
            continue;
        }
        normx = sqrt(normx);

        complex_t sign = (creal(x_values[0]) >= 0.0) ? 1.0 : -1.0;

        Vector e1 = Vector_zeros(rows, Real);
        ACCESS_VOID(complex_t, e1_values, e1.values);
        e1_values[0] = 1.0;

        Vector v;
        // v.values = malloc(x.dim * sizeof(double));
        v.values = NULL;
        posix_memalign((void**)&v.values, 32, x.dim * elem_size(Complex));
        v.dim = x.dim;
        Vector_copy(&v, &x);
        complex_t prod = sign * normx;
        Vector_scale(&e1, &prod);
        Vector_add(&v, &e1);
        Vector_free(&e1);
        ACCESS_VOID(complex_t, v_values, v.values);

        complex_t vnorm = 0.0;

        for (size_t i = 0; i < v.dim; i++)
        {
            vnorm += v_values[i] * v_values[i];
        }
        vnorm = sqrt(vnorm);

        // Vector_scale(&v, 1.0 / vnorm);
        for (size_t i = 0; i < v.dim; i++)
        {
            v_values[i] /= vnorm;
        }

        for (size_t j = k; j < n; j++)
        {
            complex_t dot = 0.0;
            for (size_t i = 0; i < rows; i++)
            {
                dot += v_values[i] * H_values[(k + 1 + i) * n + j];
            }

            for (size_t i = 0; i < rows; i++)
            {
                H_values[(k + 1 + i) * n + j] -= 2.0 * v_values[i] * dot;
            }
        }

        for (size_t i = 0; i < m; i++)
        {
            complex_t dot = 0.0;
            for (size_t r = 0; r < rows; r++)
            {
                dot += H_values[i * n + (k + 1 + r)] * v_values[r];
            }
            for (size_t r = 0; r < rows; r++)
            {
                H_values[i * n + (k + 1 + r)] -= 2.0 * dot * v_values[r];
            }
        }
        Vector_free(&v);
        Vector_free(&x);
    }
}

static void (*_hess_1_units[TYPE_COUNT])(Matrix* H, const Matrix* A) = {
    [Real]    = _hess_1_real,
    [Complex] = _hess_1_cmpl,
};

int Matrix_Hessenberg(Matrix* H, const Matrix* A)
{
    if (A->m != A->n)
    {
        return MATRIX_DIMENSION_ERROR;
    }
    _hess_1_units[A->dt](H, A);

    return MATRIX_SUCCESS;
}

static void _qr_hess_real(Matrix* Q,
                          Matrix* R,
                          Matrix* H,
                          size_t active,
                          const size_t n)
{
    ACCESS_VOID(real_t, R_values, R->values);
    ACCESS_VOID(real_t, Q_values, Q->values);
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            real_t* Q_val = Q_values + i * n + j;
            *Q_val        = (i == j ? 1.0 : 0.0);
        }
    }
    Matrix_copy(R, H);
    for (size_t i = 0; i < active - 1; i++)
    {
        real_t* a = R_values + i * n + i;
        real_t* b = R_values + (i + 1) * n + i;

        if (fabs(*b) < 1e-15)
            continue;

        // Compute Givens rotation
        real_t r = hypot(*a, *b);
        real_t c = *a / r;
        real_t s = -*b / r;

        // Apply Givens rotation to R (right multiplication)
        for (size_t j = i; j < active; j++)
        {
            real_t* t1     = R_values + i * n + j;
            real_t* t2     = R_values + (i + 1) * n + j;
            real_t* R_val1 = R_values + i * n + j;
            real_t* R_val2 = R_values + (i + 1) * n + j;
            *R_val1        = c * *t1 - s * *t2;
            *R_val2        = s * *t1 + c * *t2;
        }

        // Apply Givens rotation to Q (accumulate Q)
        for (size_t j = 0; j < active; j++)
        {
            real_t* t1     = Q_values + j * n + i;
            real_t* t2     = Q_values + j * n + (i + 1);
            real_t* Q_val1 = Q_values + j * n + i;
            real_t* Q_val2 = Q_values + j * n + (i + 1);
            *Q_val1        = c * *t1 - s * *t2;
            *Q_val2        = s * *t1 + c * *t2;
        }
    }
}

static void _qr_hess_cmpl(Matrix* Q,
                          Matrix* R,
                          Matrix* H,
                          size_t active,
                          const size_t n)
{
    ACCESS_VOID(complex_t, R_values, R->values);
    ACCESS_VOID(complex_t, Q_values, Q->values);
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            complex_t* Q_val = Q_values + i * n + j;
            *Q_val           = (i == j ? 1.0 : 0.0);
        }
    }
    Matrix_copy(R, H);
    for (size_t i = 0; i < active - 1; i++)
    {
        complex_t* a = R_values + i * n + i;
        complex_t* b = R_values + (i + 1) * n + i;

        if (cabs(*b) < 1e-15)
            continue;

        // Compute Givens rotation
        complex_t r = hypot(*a, *b);
        complex_t c = *a / r;
        complex_t s = -*b / r;

        // Apply Givens rotation to R (right multiplication)
        for (size_t j = i; j < active; j++)
        {
            complex_t* t1     = R_values + i * n + j;
            complex_t* t2     = R_values + (i + 1) * n + j;
            complex_t* R_val1 = R_values + i * n + j;
            complex_t* R_val2 = R_values + (i + 1) * n + j;
            *R_val1           = c * *t1 - s * *t2;
            *R_val2           = s * *t1 + c * *t2;
        }

        // Apply Givens rotation to Q (accumulate Q)
        for (size_t j = 0; j < active; j++)
        {
            complex_t* t1     = Q_values + j * n + i;
            complex_t* t2     = Q_values + j * n + (i + 1);
            complex_t* Q_val1 = Q_values + j * n + i;
            complex_t* Q_val2 = Q_values + j * n + (i + 1);
            *Q_val1           = c * *t1 - s * *t2;
            *Q_val2           = s * *t1 + c * *t2;
        }
    }
}

static void (*_qr_hess_units[TYPE_COUNT])(Matrix* Q,
                                          Matrix* R,
                                          Matrix* H,
                                          size_t active,
                                          const size_t n) = {
    [Real]    = _qr_hess_real,
    [Complex] = _qr_hess_cmpl,
};

void Matrix_QR_hessenberg(Matrix* Q, Matrix* R, Matrix* H, size_t active)
{
    const size_t n = H->n;
    _qr_hess_units[H->dt](Q, R, H, active, n);
}

static void _dot_active_real(Matrix* H,
                             const Matrix* R,
                             const Matrix* Q,
                             size_t active,
                             const size_t n)
{
    Matrix tmp;
    tmp.values = NULL;
    posix_memalign((void**)&tmp.values, 32, H->m * H->n * elem_size(Real));
    tmp.m = H->m;
    tmp.n = H->n;
    ACCESS_VOID(real_t, R_values, R->values);
    ACCESS_VOID(real_t, Q_values, Q->values);
    ACCESS_VOID(real_t, H_values, H->values);
    ACCESS_VOID(real_t, tmp_values, tmp.values);

    for (size_t i = 0; i < active; i++)
    {
        for (size_t j = 0; j < active; j++)
        {
            real_t sum = 0.0;
            for (size_t k = 0; k < active; k++)
            {
                real_t* R_val = R_values + i * n + k;
                real_t* Q_val = Q_values + k * n + j;
                sum += *R_val * *Q_val;
            }
            real_t* t_ptr = tmp_values + i * n + j;

            *t_ptr = sum;
        }
    }

    for (size_t i = 0; i < active; i++)
    {
#pragma omp simd
        for (size_t j = 0; j < active; j++)
        {
            real_t* t_val = tmp_values + i * n + j;
            real_t* H_val = H_values + i * n + j;

            *H_val = *t_val;
        }
    }

    Matrix_free(&tmp);
}

static void _dot_active_cmpl(Matrix* H,
                             const Matrix* R,
                             const Matrix* Q,
                             size_t active,
                             const size_t n)
{
    Matrix tmp;
    tmp.values = NULL;
    posix_memalign((void**)&tmp.values, 32, H->m * H->n * elem_size(Complex));
    tmp.m = H->m;
    tmp.n = H->n;
    ACCESS_VOID(complex_t, R_values, R->values);
    ACCESS_VOID(complex_t, Q_values, Q->values);
    ACCESS_VOID(complex_t, H_values, H->values);
    ACCESS_VOID(complex_t, tmp_values, tmp.values);

    for (size_t i = 0; i < active; i++)
    {
        for (size_t j = 0; j < active; j++)
        {
            complex_t sum = 0.0;
            for (size_t k = 0; k < active; k++)
            {
                complex_t* R_val = R_values + i * n + k;
                complex_t* Q_val = Q_values + k * n + j;
                sum += *R_val * *Q_val;
            }
            complex_t* t_ptr = tmp_values + i * n + j;

            *t_ptr = sum;
        }
    }

    for (size_t i = 0; i < active; i++)
    {
#pragma omp simd
        for (size_t j = 0; j < active; j++)
        {
            complex_t* t_val = tmp_values + i * n + j;
            complex_t* H_val = H_values + i * n + j;
            *H_val           = *t_val;
        }
    }

    Matrix_free(&tmp);
}

static void (*_dot_active_units[TYPE_COUNT])(Matrix* H,
                                             const Matrix* R,
                                             const Matrix* Q,
                                             size_t active,
                                             const size_t n) = {
    [Real]    = _dot_active_real,
    [Complex] = _dot_active_cmpl,
};

void Matrix_Matrix_dot_active(Matrix* H,
                              const Matrix* R,
                              const Matrix* Q,
                              size_t active)
{
    size_t n = H->n;
    _dot_active_units[Q->dt](H, R, Q, active, n);
}

static void _inner_dot_int(void* target, const Matrix* A, const Matrix* B)
{
    int_t sum = 0;
    ACCESS_VOID(int_t, A_values, A->values);
    ACCESS_VOID(int_t, B_values, B->values);

#pragma omp parallel for
    FMA(A->m * A->n, sum, A_values[n], B_values[n]);

    ASSIGN_UNTYPED(int_t, target, &sum);
}

static void _inner_dot_real(void* target, const Matrix* A, const Matrix* B)
{
    real_t sum = 0;
    ACCESS_VOID(real_t, A_values, A->values);
    ACCESS_VOID(real_t, B_values, B->values);

#pragma omp parallel for
    FMA(A->m * A->n, sum, A_values[n], B_values[n]);

    ASSIGN_UNTYPED(real_t, target, &sum);
}

static void _inner_dot_cmpl(void* target, const Matrix* A, const Matrix* B)
{
    complex_t sum = 0;
    ACCESS_VOID(complex_t, A_values, A->values);
    ACCESS_VOID(complex_t, B_values, B->values);

#pragma omp parallel for
    FMA(A->m * A->n, sum, A_values[n], B_values[n]);
    ASSIGN_UNTYPED(complex_t, target, &sum);
}

static void (*_inner_dot_units[TYPE_COUNT])(void* target,
                                            const Matrix* A,
                                            const Matrix* B) = {
    [Int]     = _inner_dot_int,
    [Real]    = _inner_dot_real,
    [Complex] = _inner_dot_cmpl,
};

int Matrix_inner_dot(void* target, const Matrix* A, const Matrix* B)
{
    if (A->m != B->m || A->n != B->n)
    {
        return MATRIX_DIMENSION_ERROR;
    }
    DataType dt = MAX(A->dt, B->dt);
    _inner_dot_units[dt](target, A, B);

    return MATRIX_SUCCESS;
}

static void _hadamard_dot_int(Matrix* target, const Matrix* A, const Matrix* B)
{
    const size_t M = A->m, N = A->n;
    ACCESS_VOID(int_t, A_values, A->values);
    ACCESS_VOID(int_t, B_values, B->values);
    ACCESS_VOID(int_t, target_values, target->values);

#pragma omp parallel for
    for (size_t i = 0; i < M * N; i++)
    {
        target_values[i] = A_values[i] * B_values[i];
    }
}

static void _hadamard_dot_real(Matrix* target, const Matrix* A, const Matrix* B)
{
    const size_t M = A->m, N = A->n;
    ACCESS_VOID(real_t, A_values, A->values);
    ACCESS_VOID(real_t, B_values, B->values);
    ACCESS_VOID(real_t, target_values, target->values);

#pragma omp parallel for
    for (size_t i = 0; i < M * N; i++)
    {
        target_values[i] = A_values[i] * B_values[i];
    }
}

static void _hadamard_dot_cmpl(Matrix* target, const Matrix* A, const Matrix* B)
{
    const size_t M = A->m, N = A->n;
    ACCESS_VOID(complex_t, A_values, A->values);
    ACCESS_VOID(complex_t, B_values, B->values);
    ACCESS_VOID(complex_t, target_values, target->values);

#pragma omp parallel for
    for (size_t i = 0; i < M * N; i++)
    {
        target_values[i] = A_values[i] * B_values[i];
    }
}

static void (*_hadamard_dot_units[TYPE_COUNT])(Matrix* target,
                                               const Matrix* A,
                                               const Matrix* B) = {
    [Int]     = _hadamard_dot_int,
    [Real]    = _hadamard_dot_real,
    [Complex] = _hadamard_dot_cmpl,
};

int Matrix_Hadamard_dot(Matrix* target, const Matrix* A, const Matrix* B)
{
    if (A->m != B->m || A->n != B->n)
    {
        return MATRIX_DIMENSION_ERROR;
    }
    DataType dt = MAX(A->dt, B->dt);
    Matrix_prepare_target(target, A->m, A->n, dt);
    _hadamard_dot_units[dt](target, A, B);
    return MATRIX_SUCCESS;
}
