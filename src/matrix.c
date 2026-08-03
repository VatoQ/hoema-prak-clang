#include "../include/matrix.h"
#include "../include/logging.h"
#include "../include/prng.h"
#include "../include/vector.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int Matrix_is_empty(const Matrix* M)
{
    return ((M->values == NULL) || (M->m == 0 && M->n == 0));
}

/**
 * @brief Prepares `target` for further processing. A `target` with matching
 * dimension will remain unchanged.
 *
 * @param `target` Target container for matrix operations.
 * @param `m` Rows of target matrix.
 * @param `n` Columns of target matrix.
 */
static void Matrix_prepare_target(Matrix* target,
                                  const size_t m,
                                  const size_t n)
{
    if (Matrix_is_empty(target))
    {
        *target = Matrix_new(m, n, 0.0);
        return;
    }

    if (target->m == m && target->n == n)
    {
        return;
    }
    Log_log("Target matrix shape mismatch, reallocating", LOG_WARNING);
    Matrix_free(target);
    *target = Matrix_new(m, n, 0.0);
}

Matrix Matrix_new(const size_t m, const size_t n, const double init_val)
{
    Matrix res = { 0 };
    if (m != 0 && n != 0)
    {
        double* vals = calloc(m * n, sizeof(double));
        if (!vals)
        {
            Log_log("Error allocating memory in Matrix_new()", LOG_ERROR);
            return (Matrix){ 0 };
        }
        if (fabs(init_val) > EPS)
        {
            for (size_t i = 0; i < m * n; i++)
            {
                vals[i] = init_val;
            }
        }
        res.m      = m;
        res.n      = n;
        res.values = vals;
    }
    return res;
}

Matrix Matrix_new_vals(const size_t m, const size_t n, const double* init_vals)
{
    Matrix M = Matrix_new(m, n, 0);
    memcpy(M.values, init_vals, m * n * sizeof(double));
    M.m = m;
    M.n = n;
    return M;
}

Matrix Matrix_new_random_normal(const size_t m,
                                const size_t n,
                                const double mean,
                                const double variance)
{
    PRNG_State prng = PRNG_State_init(NO_SEED);

    Matrix M = Matrix_new(m, n, 0.0);
    for (size_t i = 0; i < m * n; i++)
    {
        M.values[i] = PRNG_State_normal(&prng, mean, variance);
    }
    return M;
}

Matrix Matrix_new_random_uniform(const size_t m,
                                 const size_t n,
                                 const double min,
                                 const double max)
{
    PRNG_State prng = PRNG_State_init(NO_SEED);

    Matrix M = Matrix_new(m, n, 0);
    for (size_t i = 0; i < m * n; i++)
    {
        M.values[i] = PRNG_State_random_double_range(&prng, min, max);
    }
    return M;
}

Matrix Matrix_zeros_like(const Matrix* M)
{
    return Matrix_new(M->m, M->n, 0);
}

/**
 * @brief Set a given array of values to the diagonal entries of a matrix.
 *
 * @param `M` Target matrix.
 * @param `n` number of values in `val`.
 * @param `val` array of initial values.
 */
static void _diag_helper(Matrix* M, const size_t n, const double* val)
{
    int status = 0;
    for (size_t i = 0; i < n; i++)
    {
        status += Matrix_set_at(M, i, i, val[i]);
    }
    if (status != MATRIX_SUCCESS * n)
    {
        Log_log("An unknown error has occured at "
                "Matrix_diag()\nReplacing M with zeros",
                LOG_ERROR);
        Matrix tmp = Matrix_zeros_like(M);
        Matrix_free(M);
        *M = tmp;
    }
}

Matrix Matrix_diag(const Vector* v)
{
    const size_t n = v->dim;

    Matrix M = Matrix_new(n, n, 0.0);
    _diag_helper(&M, n, v->values);
    return M;
}

Matrix Matrix_diag_val(const size_t n, const double val)
{
    Matrix M     = Matrix_new(n, n, 0.0);
    double* vals = calloc(n, sizeof(double));
    for (size_t i = 0; i < n; i++)
    {
        vals[i] = val;
    }
    _diag_helper(&M, n, vals);
    free(vals);
    return M;
}

void Matrix_copy(Matrix* target, const Matrix* M)
{
    Matrix_prepare_target(target, M->m, M->n);
    target->values = calloc(M->m * M->n, sizeof(double));
    memcpy(target->values, M->values, M->m * M->n * sizeof(double));
    target->m = M->m;
    target->n = M->n;
}

int Matrix_set_at(Matrix* M, const size_t m, const size_t n, const double value)
{
    if (m >= M->m || n >= M->n)
    {
        Log_log("Index out of range in Matrix_set_at()!", LOG_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }
    const size_t index = m * M->n + n;
    M->values[index]   = value;

    return MATRIX_SUCCESS;
}

int Matrix_get_at(double* target,
                  const Matrix* M,
                  const size_t m,
                  const size_t n)
{
    if (m >= M->m || n >= M->n)
    {
        Log_log("Index out of range in Matrix_get_at()!", LOG_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }

    const size_t index = m * M->n + n;
    *target            = M->values[index];
    return MATRIX_SUCCESS;
}

int Matrix_all_close(const Matrix* A, const Matrix* B)
{
    if (A->m != B->m || A->n != B->n)
    {
        return 0;
    }

    for (size_t i = 0; i < A->m * A->n; i++)
    {
        if (fabs(A->values[i] - B->values[i]) > EPS)
        {
            return 0;
        }
    }
    return 1;
}

void Matrix_free(Matrix* M)
{
    free(M->values);
    M->values = NULL;
    M->m = 0, M->n = 0;
}

void Matrix_print(const Matrix* M)
{
    const size_t m = M->m, n = M->n;
    int status = 0;
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            double val = 0;
            status += Matrix_get_at(&val, M, i, j);
            printf("%f ", val);
        }
        printf("\n");
    }
    if (status != MATRIX_SUCCESS * m * n)
    {
        Log_log("An unknown problem at Matrix_print", LOG_WARNING);
    }
}

double _Frobenius_helper(const Matrix* M)
{
    double s = 0.0;

    for (size_t i = 0; i < M->m * M->n; i++)
    {
        s += M->values[i] * M->values[i];
    }
    return sqrt(s);
}

double _Spectral_helper(const Matrix* M)
{
    Vector b      = Vector_new_random_normal(M->n, 0, 1);
    Vector tmp    = Vector_zeros_like(&b);
    Vector b_prev = Vector_zeros_like(&b);
    double lambda = 0.0;
    double denom  = 1;

    do
    {
        Vector_copy(&b_prev, &b);
        Matrix_Vector_dot(&tmp, M, &b);
        Vector_copy(&b, &tmp);
        double b_norm = Vector_norm(&b);
        Vector_scale(&b, 1.0 / b_norm);

        Matrix_Vector_dot(&tmp, M, &b);

        Vector_dot(&lambda, &tmp, &b);
        Vector_dot(&denom, &b, &b);
        lambda /= denom;

    } while (!Vector_all_close(&b_prev, &b));
    Vector_free(&b);
    Vector_free(&b_prev);
    Vector_free(&tmp);

    return lambda;
}

double Matrix_norm(const Matrix* M, NormType nt)
{
    double result = 0;
    switch (nt)
    {
        case FROBENIUS:
        {
            return _Frobenius_helper(M);
        }
        case SPECTRAL:
        {
            return _Spectral_helper(M);
        }
        default:
        {
            return 0;
        }
    }
}

int Matrix_add(Matrix* target, const Matrix* M)
{
    if (target->m != M->m || target->n != M->n)
    {
        Log_log("Index out of range in Matrix_add()!", LOG_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }
    for (size_t n = 0; n < M->m * M->n; n++)
    {
        target->values[n] += M->values[n];
    }
    return MATRIX_SUCCESS;
}

int Matrix_sub(Matrix* target, const Matrix* M)
{
    if (target->m != M->m || target->n != M->n)
    {
        Log_log("Index out of range in Matrix_sub()!", LOG_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }
    for (size_t n = 0; n < M->m * M->n; n++)
    {
        target->values[n] -= M->values[n];
    }
    return MATRIX_SUCCESS;
}

void Matrix_scale(Matrix* target, const double lambda)
{
    const size_t N = target->m * target->n;
    for (int n = 0; n < N; n++)
    {
        target->values[n] *= lambda;
    }
}

static int _invert_n_n_matrix(Matrix* target, const Matrix* M)
{
    // TODO: Error code handling
    Matrix M_copy = Matrix_new(0, 0, 0.0);
    Matrix_copy(&M_copy, M);
    Vector ones = Vector_new(M->n, 1);
    *target     = Matrix_diag(&ones);
    // Phase one, make lower right triangle
    for (size_t m = 0; m < M->m; m++)
    {
        double lambda;
        Matrix_get_at(&lambda, &M_copy, m, m);
        lambda                  = 1.0 / lambda;
        double* row_vals        = M_copy.values + m * M->n;
        double* row_target_vals = target->values + m * M->n;
        Vector row              = Vector_new_vals(M->n, row_vals);
        Vector row_target       = Vector_new_vals(M->n, row_target_vals);
        Vector_scale(&row, lambda);
        Vector_scale(&row_target, lambda);
        memcpy(
          target->values + m * M->n, row_target.values, M->n * sizeof(double));
        memcpy(M_copy.values + m * M->n, row.values, M->n * sizeof(double));

        for (size_t n = m + 1; n < M->n; n++)
        {
            Matrix_get_at(&lambda, &M_copy, n, m);
            Vector_scale(&row, lambda);
            Vector_scale(&row_target, lambda);

            row_vals            = M_copy.values + n * M->n;
            row_target_vals     = target->values + n * M->n;
            Vector row_n        = Vector_new_vals(M->n, row_vals);
            Vector row_target_n = Vector_new_vals(M->n, row_target_vals);

            Vector_sub(&row_target_n, &row_target);
            Vector_sub(&row_n, &row);

            memcpy(target->values + n * M->n,
                   row_target_n.values,
                   M->n * sizeof(double));
            memcpy(
              M_copy.values + n * M->n, row_n.values, M->n * sizeof(double));

            Vector_scale(&row, 1 / lambda);
            Vector_scale(&row_target, 1 / lambda);

            Vector_free(&row_n);
            Vector_free(&row_target_n);
        }

        Vector_free(&row);
        Vector_free(&row_target);
    }

    // Phase two, fill upper right triangle.
    for (long m = M->m - 1; m >= 0; m--)
    {
        double lambda;
        double* row_vals        = M_copy.values + m * M->n;
        double* row_target_vals = target->values + m * M->n;
        Vector row              = Vector_new_vals(M->n, row_vals);
        Vector row_target       = Vector_new_vals(M->n, row_target_vals);
        memcpy(
          target->values + m * M->n, row_target.values, M->n * sizeof(double));
        memcpy(M_copy.values + m * M->n, row.values, M->n * sizeof(double));
        for (long n = m - 1; n >= 0; n--)
        {
            Matrix_get_at(&lambda, &M_copy, n, m);
            Vector_scale(&row, lambda);
            Vector_scale(&row_target, lambda);

            row_vals            = M_copy.values + n * M->n;
            row_target_vals     = target->values + n * M->n;
            Vector row_n        = Vector_new_vals(M->n, row_vals);
            Vector row_target_n = Vector_new_vals(M->n, row_target_vals);

            Vector_sub(&row_target_n, &row_target);
            Vector_sub(&row_n, &row);

            memcpy(target->values + n * M->n,
                   row_target_n.values,
                   M->n * sizeof(double));
            memcpy(
              M_copy.values + n * M->n, row_n.values, M->n * sizeof(double));

            Vector_scale(&row, 1 / lambda);
            Vector_scale(&row_target, 1 / lambda);
        }
    }

    Matrix_free(&M_copy);

    return MATRIX_SUCCESS;
}

int Matrix_inverse(Matrix* target, const Matrix* M)
{
    Matrix_prepare_target(target, M->m, M->n);
    if (M->m != M->n)
    {
        Log_log("Cannot invert matrix where n!=m", LOG_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }
    if (M->m == 2 && M->n == 2)
    {
        double a, b, c, d;
        int status = 0;
        status += Matrix_get_at(&a, M, 0, 0);
        status += Matrix_get_at(&b, M, 0, 1);
        status += Matrix_get_at(&c, M, 1, 0);
        status += Matrix_get_at(&d, M, 1, 1);
        if (status != MATRIX_SUCCESS * 4)
        {
            Log_log("Dimension error in Matrix_inverse(), 2x2 case.",
                    LOG_ERROR);
            return MATRIX_DIMENSION_ERROR;
        }
        const double denom = a * d - b * c;
        if (fabs(denom) < EPS)
        {
            Log_log("Math error in Matrix_inverse(), 2x2 case. Matrix is not "
                    "invertable.",
                    LOG_ERROR);
            return MATRIX_MATH_ERROR;
        }
        const double scalar = 1 / denom;
        double init_vals[4] = { d, -b, -c, a };

        Matrix tmp = Matrix_new_vals(M->m, M->n, init_vals);
        Matrix_scale(&tmp, scalar);
        Matrix_copy(target, &tmp);
        return MATRIX_SUCCESS;
    }

    int status = _invert_n_n_matrix(target, M);
    if (status != MATRIX_SUCCESS)
    {
        Log_log("Unknown math error in Matrix_inverse()", LOG_ERROR);
        return MATRIX_MATH_ERROR;
    }

    return MATRIX_SUCCESS;
}

int Matrix_Vector_dot(Vector* target, const Matrix* M, const Vector* v)
{
    if (M->n != v->dim)
    {
        Log_log("Index out of range in Matrix_Vector_dot()", LOG_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }
    const size_t DIM = M->m;
    const size_t N   = M->n;
    Vector_prepare_target(target, DIM);
    int checksum = 0;

    for (size_t d = 0; d < DIM; d++)
    {
        double sum = 0.0;
        for (size_t n = 0; n < N; n++)
        {
            double M_val;
            checksum += Matrix_get_at(&M_val, M, d, n);
            double V_val = Vector_at(v, n);
            sum += M_val * V_val;
        }
        Vector_set_item(target, d, sum);
    }

    if (checksum != DIM * N * MATRIX_SUCCESS)
    {
        Log_log("Unknown error in Matrix_Vector_dot()", LOG_ERROR);
        return MATRIX_BASIC_ERROR;
    }

    return MATRIX_SUCCESS;
}

int Matrix_Matrix_dot(Matrix* target, const Matrix* M1, const Matrix* M2)
{
    if (M1->n != M2->m)
    {
        Log_log("Index out of range in Matrix_Matrix_dot()", LOG_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }

    Matrix_prepare_target(target, M1->n, M2->m);

    *target    = Matrix_new(M1->m, M2->n, 0);
    int status = 0;

    for (size_t m = 0; m < target->m; m++)
    {
        for (size_t n = 0; n < target->n; n++)
        {
            double sum = 0.0;

            for (size_t o = 0; o < M1->n; o++)
            {
                double a, b;
                status += Matrix_get_at(&a, M1, m, o);
                status += Matrix_get_at(&b, M2, o, n);
                sum += a * b;
            }
            Matrix_set_at(target, m, n, sum);
        }
    }
    if (status != target->m * M1->n * 2 * MATRIX_SUCCESS)
    {
        Log_log("Unknown error in Matrix_Matrix_dot()", LOG_ERROR);
        return MATRIX_BASIC_ERROR;
    }
    return MATRIX_SUCCESS;
}

int Matrix_jacobi(Matrix* target, const Vector* x, Vector (*f)(const Vector* x))
{
    const Vector f_x = f(x);
    const size_t M   = f_x.dim;
    const size_t N   = x->dim;
    Vector f_x_eps   = Vector_zeros(M);
    Matrix_prepare_target(target, M, N);
    Vector x_eps = Vector_zeros_like(x);
    Vector_copy(&x_eps, x);
    double fn_x, fn_x_eps;

    int status = 0;
    for (size_t m = 0; m < M; m++)
    {

        for (size_t n = 0; n < N; n++)
        {
            x_eps.values[n] += MATRIX_EPS;
            f_x_eps = f(&x_eps);
            x_eps.values[n] -= MATRIX_EPS;
            fn_x     = f_x.values[m];
            fn_x_eps = f_x_eps.values[m];

            status +=
              Matrix_set_at(target, m, n, (fn_x_eps - fn_x) / MATRIX_EPS);
        }
    }
    if (status != M * N * MATRIX_SUCCESS)
    {
        Log_log("Index error occured during Matrix_jacobi()", LOG_ERROR);
        return MATRIX_BASIC_ERROR;
    }

    return MATRIX_SUCCESS;
}

int Matrix_Matrix_dot_jordan(Matrix* target, const Matrix* M1, const Matrix* M2)
{
    if (M1->m != M1->n || M2->m != M2->n || M1->m != M2->m)
    {
        Log_log("Jordan multiplication requires symmetrical matrices",
                LOG_ERROR);
        return MATRIX_DIMENSION_ERROR;
    }
    Matrix_prepare_target(target, M1->m, M1->n);
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
