#include "matrix.h"
#include "vector.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Matrix Matrix_new(const size_t m, const size_t n, const double init_val)
{
    double* vals = calloc(m * n, sizeof(double));
    if (fabs(init_val) > EPS)
    {
        for (size_t i = 0; i < m * n; i++)
        {
            vals[i] = init_val;
        }
    }
    Matrix res;
    res.m      = m;
    res.n      = n;
    res.values = vals;
    return res;
}

Matrix Matrix_new_vals(const size_t m, const size_t n, const double* init_vals)
{
    Matrix M;
    M.values = calloc(m * n, sizeof(double));
    // memcpy(M.values, init_vals, m * n * sizeof(double));
    for (size_t i = 0; i < m * n; i++)
    {

        M.values[i] = init_vals[i];
        // printf("val %ld: %f\n", i, M.values[i]);
    }

    M.m = m;
    M.n = n;
    return M;
}

Matrix Matrix_zeros_like(const Matrix* M)
{
    Matrix m;
    m.values = calloc(M->m * M->n, sizeof(double));
    m.m      = M->m;
    m.n      = M->n;

    return m;
}

void _diag_helper(Matrix* M, const size_t n, const double* val)
{
    int status = 0;
    for (size_t i = 0; i < n; i++)
    {
        status += Matrix_set_at(M, i, i, val[i]);
    }
    if (status != MATRIX_BASIC_SUCCESS * n)
    {
        perror("ERROR: An unknown error has occured at "
               "Matrix_diag()\nReplacing M with zeros\n");
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
    if (target->m != 0 || target->n != 0)
    {
        Matrix_free(target);
    }
    target->values = calloc(M->m * M->n, sizeof(double));
    memcpy(target->values, M->values, M->m * M->n * sizeof(double));
    target->m = M->m;
    target->n = M->n;
}

int Matrix_set_at(Matrix* M, const size_t m, const size_t n, const double value)
{
    if (m >= M->m || n >= M->n)
    {
        return MATRIX_DIMENSION_ERROR;
    }
    const size_t index = m * M->n + n;
    M->values[index]   = value;

    return MATRIX_BASIC_SUCCESS;
}

int Matrix_get_at(double* target,
                  const Matrix* M,
                  const size_t m,
                  const size_t n)
{
    if (m >= M->m || n >= M->n)
    {
        return MATRIX_DIMENSION_ERROR;
    }

    const size_t index = m * M->n + n;
    *target            = M->values[index];
    return MATRIX_BASIC_SUCCESS;
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
}

void Matrix_scale(Matrix* target, const double lambda)
{
    const size_t N = target->m * target->n;
    for (int n = 0; n < N; n++)
    {
        target->values[n] *= lambda;
    }
}

int _reverse_n_n_matrix(Matrix* target, const Matrix* M)
{
    *target = Matrix_zeros_like(M);
    for (size_t m = 0; m < M->m; m++)
    {
        double lambda;
        Matrix_get_at(&lambda, M, m, m);
        double* row_vals = target->values + m * M->n;
        Vector row       = Vector_new_vals(M->m, row_vals);
        Vector_scale(&row, 1.0 / lambda);
        memcpy(target->values + m * M->m, row.values, n * sizeof(double));
        for (size_t n = m + 1; n < M->n; n++)
        {
        }
    }

    return MATRIX_MATH_SUCCESS;
}

int Matrix_inverse(Matrix* target, const Matrix* M)
{
    if (target->m != 0 && target->n != 0)
    {
        Matrix_free(target);
    }
    if (M->m != M->n)
    {
        perror("Cannot invert matrix where n!=m\n");
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
        if (status != MATRIX_BASIC_SUCCESS * 4)
        {
            return MATRIX_DIMENSION_ERROR;
        }
        const double scalar = 1 / (a * d - b * c);
        double init_vals[4] = { d, -b, -c, a };

        Matrix tmp = Matrix_new_vals(M->m, M->n, init_vals);
        Matrix_scale(&tmp, scalar);
        Matrix_copy(target, &tmp);
        return MATRIX_MATH_SUCCESS;
    }

    // TODO: Other inverse methods

    return MATRIX_DIMENSION_ERROR;
}

int Matrix_Vector_dot(Vector* target, const Matrix* M, const Vector* v)
{
    if (M->m != v->dim)
    {
        return MATRIX_DIMENSION_ERROR;
    }
    const size_t DIM = M->m;
    const size_t N   = M->n;
    Vector tmp       = Vector_new(DIM, 0.0);
    int checksum     = 0;

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
        Vector_set_item(&tmp, d, sum);
    }

    if (checksum != DIM * N * MATRIX_BASIC_SUCCESS)
    {
        return MATRIX_BASIC_ERROR;
    }
    Vector_copy(target, &tmp);

    return MATRIX_MATH_SUCCESS;
}
