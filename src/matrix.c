#include "../include/matrix.h"
#include "../include/logging.h"
#include "../include/prng.h"
#include "../include/vector.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int Matrix_is_empty(const Matrix* M)
{
    return ((M->values == NULL) || (M->m == 0 && M->n == 0));
}

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
        M->values[i * M->n + i] = val[i];
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

double Matrix_max(const Matrix* M)
{
    double max = -INFINITY;
    for (int i = 0; i < M->m * M->n; i++)
    {
        if (M->values[i] > max)
        {
            max = M->values[i];
        }
    }
    return max;
}

double Matrix_min(const Matrix* M)
{
    double min = INFINITY;
    for (int i = 0; i < M->m * M->n; i++)
    {
        if (M->values[i] < min)
        {
            min = M->values[i];
        }
    }
    return min;
}

void Matrix_print(const Matrix* M)
{
    const size_t m = M->m, n = M->n;
    int status = 0;
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            double val = M->values[i * M->n + j];
            // status += Matrix_get_at(&val, M, i, j);
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
    Vector eigvals = Vector_zeros(M->n);
    Matrix_eigvals(&eigvals, M);
    double max = -INFINITY;
    for (size_t i = 0; i < eigvals.dim; i++)
    {
        if (fabs(eigvals.values[i]) > max)
        {
            max = fabs(eigvals.values[i]);
        }
    }
    Vector_free(&eigvals);
    return max;
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
        double lambda = M_copy.values[m * M_copy.n + m];
        // Matrix_get_at(&lambda, &M_copy, m, m);
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
            // Matrix_get_at(&lambda, &M_copy, n, m);
            lambda = M_copy.values[n * M_copy.n + m];
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
            // Matrix_get_at(&lambda, &M_copy, n, m);
            lambda = M_copy.values[n * M_copy.n + m];
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

        a = M->values[0];
        b = M->values[1];
        c = M->values[2];
        d = M->values[3];
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
    int checksum              = 0;
    double* restrict M_values = M->values;
    double* restrict v_values = v->values;

#pragma omp parallel for
    for (size_t d = 0; d < DIM; d++)
    {
        double sum = 0.0;
#pragma omp simd
        for (size_t n = 0; n < N; n++)
        {
            double M_val = M_values[d * M->n + n];
            double V_val = v_values[n];
            sum += M_val * V_val;
        }
        target->values[d] = sum;
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

    int status                     = 0;
    double* restrict M1_values     = M1->values;
    double* restrict M2_values     = M2->values;
    double* restrict target_values = target->values;

    for (size_t m = 0; m < target->m; m++)
    {
#pragma omp parallel for
        for (size_t n = 0; n < target->n; n++)
        {
            double sum = 0.0;

#pragma omp simd
            for (size_t o = 0; o < M1->n; o++)
            {
                double a, b;
                a = M1_values[m * M1->n + o];
                b = M2_values[o * M2->n + n];
                sum += a * b;
            }
            target_values[m * target->n + n] = sum;
        }
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

    for (size_t m = 0; m < M; m++)
    {
        for (size_t n = 0; n < N; n++)
        {
            x_eps.values[n] += MATRIX_EPS;
            f_x_eps = f(&x_eps);
            x_eps.values[n] -= MATRIX_EPS;
            fn_x     = f_x.values[m];
            fn_x_eps = f_x_eps.values[m];

            target->values[m * target->n + n] = (fn_x_eps - fn_x) / MATRIX_EPS;
        }
    }

    return MATRIX_SUCCESS;
}

int Matrix_Vector_solve(Vector* target, const Matrix* M, const Vector* v)
{
    if (M->n != v->dim || M->m != M->n)
    {
        return MATRIX_DIMENSION_ERROR;
    }
    Matrix M_copy = Matrix_new(M->m, M->n, 0);
    Matrix_copy(&M_copy, M);
    Vector_copy(target, v);

    // Phase one
    for (size_t m = 0; m < M->m; m++)
    {
        double lambda    = 1.0 / M_copy.values[m * M_copy.n + m];
        double* row_vals = M_copy.values + m * M_copy.n;
        Vector row       = Vector_new_vals(M->n, row_vals);
        Vector_scale(&row, lambda);
        memcpy(
          M_copy.values + m * M_copy.n, row.values, M_copy.n * sizeof(double));

        target->values[m] *= lambda;
        double target_val = target->values[m];

        for (size_t n = m + 1; n < M->m; n++)
        {
            lambda = M_copy.values[n * M_copy.n + m];
            Vector_scale(&row, lambda);
            target_val *= lambda;
            row_vals     = M_copy.values + n * M->n;
            Vector row_n = Vector_new_vals(M_copy.n, row_vals);
            Vector_sub(&row_n, &row);

            memcpy(M_copy.values + n * M_copy.n,
                   row_n.values,
                   M->n * sizeof(double));
            target->values[n] -= target_val;

            Vector_scale(&row, 1 / lambda);
            target_val /= lambda;

            Vector_free(&row_n);
        }
        Vector_free(&row);
    }

    // Phase two
    for (long m = M_copy.m - 1; m >= 0; m--)
    {
        double lambda;
        double* row_vals  = M_copy.values + m * M_copy.n;
        Vector row        = Vector_new_vals(M->n, row_vals);
        double target_val = target->values[m];
        for (long n = m - 1; n >= 0; n--)
        {
            lambda = M_copy.values[n * M_copy.n + m];
            Vector_scale(&row, lambda);
            target_val *= lambda;

            row_vals     = M_copy.values + n * M->n;
            Vector row_n = Vector_new_vals(M_copy.n, row_vals);
            Vector_sub(&row_n, &row);
            memcpy(M_copy.values + n * M_copy.n,
                   row_n.values,
                   M_copy.n * sizeof(double));
            target->values[n] -= target_val;

            Vector_scale(&row, 1 / lambda);
            target_val /= lambda;

            Vector_free(&row_n);
        }
        Vector_free(&row);
    }
    Matrix_free(&M_copy);
    return MATRIX_SUCCESS;
}

int Matrix_column_pivot(Matrix* A, size_t k, size_t* P)
{
    size_t n = A->n;
    size_t m = A->m;

    if (k >= n)
        return MATRIX_DIMENSION_ERROR;

    // Find column with largest norm among k..n-1
    double best_norm = -INFINITY;
    size_t best_j    = k;

    for (size_t j = k; j < n; j++)
    {
        double norm = 0.0;
        for (size_t i = 0; i < m; i++)
            norm += A->values[i * n + j] * A->values[i * n + j];

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
            double tmp                = A->values[i * n + k];
            A->values[i * n + k]      = A->values[i * n + best_j];
            A->values[i * n + best_j] = tmp;
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

int Matrix_QR(Matrix* Q, Matrix* R, const Matrix* A)
{
    // Matrix_prepare_target(Q, A->m, A->n);
    // Matrix_prepare_target(R, A->m, A->n);
    // Matrix outer = Matrix_new(A->n, A->n, 0);
    size_t m = A->m;
    size_t n = A->n;
    Matrix_copy(R, A);
    Matrix I = Matrix_diag_val(A->n, 1);
    Matrix_copy(Q, &I);
    // Matrix H   = Matrix_zeros_like(&I);
    // Matrix tmp = Matrix_zeros_like(A);
    // Matrix A_copy = Matrix_zeros_like(A);

    for (size_t k = 0; k < A->n; k++)
    {
        size_t rows = m - k;

        Vector x = Vector_zeros(rows);
        for (size_t i = 0; i < rows; i++)
        {
            x.values[i] = R->values[(k + i) * n + k];
        }

        double norm_x = Vector_norm(&x);
        double sign   = (x.values[0] >= 0 ? 1 : -1);

        Vector e1    = Vector_zeros(rows);
        e1.values[0] = 1;

        Vector v = Vector_zeros_like(&x);
        Vector_copy(&v, &x);
        Vector_scale(&e1, sign * norm_x);
        Vector_add(&v, &e1);

        Vector_free(&e1);
        Vector_scale(&v, 1.0 / Vector_norm(&v));

        for (size_t j = k; j < n; j++)
        {
            double dot = 0.0;
            for (size_t i = 0; i < rows; i++)
            {
                dot += v.values[i] * R->values[(k + i) * n + j];
            }
            for (size_t i = 0; i < rows; i++)
            {
                R->values[(k + i) * n + j] -= 2 * v.values[i] * dot;
            }
        }

        for (size_t i = 0; i < m; i++)
        {
            double dot = 0.0;
            for (size_t r = 0; r < rows; r++)
            {
                dot += Q->values[i * n + (k + r)] * v.values[r];
            }

            for (size_t r = 0; r < rows; r++)
            {
                Q->values[i * n + (k + r)] -= 2 * dot * v.values[r];
            }
        }
        Vector_free(&v);
        Vector_free(&x);
    }

    return MATRIX_SUCCESS;
}

int _eigvals_two(Vector* eigenvalues, const Matrix* M)
{
    const double a = M->values[0];
    const double b = M->values[1];
    const double c = M->values[2];
    const double d = M->values[3];

    const double p     = -(a + d);
    const double q     = a * d - b * c;
    const double inner = 0.25 * p * p - q;
    Vector_prepare_target(eigenvalues, 2);
    if (inner < 0.0)
    {
        Log_log("complex eigenvalues encountered", LOG_WARNING);
        return MATRIX_MATH_ERROR;
    }

    const double x1        = -p * 0.5 + sqrt(inner);
    const double x2        = -p * 0.5 - sqrt(inner);
    eigenvalues->values[0] = x1;
    eigenvalues->values[1] = x2;
    return MATRIX_SUCCESS;
}

double _wilkinson_shift(const Matrix* H, size_t active)
{
    size_t n = H->n;

    double a = H->values[(active - 2) * n + (active - 2)];
    double b = H->values[(active - 2) * n + (active - 1)];
    double c = H->values[(active - 1) * n + (active - 2)];
    double d = H->values[(active - 1) * n + (active - 1)];

    double tr        = (a + d) * 0.5;
    double det       = (a - d) * 0.5;
    double disc_term = det * det + b * c;
    if (disc_term < 0.0)
    {
        disc_term = 0.0;
    }
    double disc = sqrt(disc_term);

    double lambda1 = tr + disc;
    double lambda2 = tr - disc;

    return (fabs(lambda1 - d) < fabs(lambda2 - d)) ? lambda1 : lambda2;
}

int _eigvals_big(Vector* eigenvalues, const Matrix* M)
{
    printf("Enter _eigvals_big with\n");
    Matrix_print(M);
    printf("\n");

    const size_t max_iters = 10000;
    const double eps       = 1e-12;

    Matrix H = Matrix_zeros_like(M);
    Matrix_Hessenberg(&H, M); // Q^T M Q → upper Hessenberg
    printf("Hessenberg Matrix: \n");
    Matrix_print(&H);
    printf("\n");
    Matrix Q = Matrix_zeros_like(&H);
    Matrix R = Matrix_zeros_like(&H);

    size_t n      = H.n;
    size_t active = n;

    size_t total_iters = 0;

    while (active > 1)
    {
        // printf("Active: %zu\n", active);
        for (size_t iter = 0; iter < max_iters; iter++)
        {
            total_iters++;
            double sub = fabs(H.values[(active - 1) * n + (active - 2)]);

            if (sub < eps)
            {
                break;
            }

            double mu = _wilkinson_shift(&H, active);

            // for (size_t i = 0; i < active; i++)
            // {
            //     H.values[i * n + i] -= mu;
            // }

            Matrix_QR(&Q, &R, &H);

            Matrix_Matrix_dot(&H, &R, &Q);

            // for (size_t i = 0; i < active; i++)
            // {
            //     H.values[i * n + i] += mu;
            // }
            bool converged = true;
            for (size_t i = 0; i < active - 1; i++)
            {
                if (fabs(H.values[(i + 1) * n + i]) > eps)
                {
                    converged = false;
                    break;
                }
            }
            if (converged)
            {
                break;
            }
        }

        if (fabs(H.values[(active - 1) * n + (active - 2)]) < eps)
        {
            active--;
            continue;
        }
        else
        {
            break;
        }
    }
    printf("After %zu iterations, got\n", total_iters);
    Matrix_print(&H);
    printf("\n");

    Vector_prepare_target(eigenvalues, n);
    for (size_t i = 0; i < n; i++)
    {
        eigenvalues->values[i] = H.values[i * n + i];
    }
    Matrix_free(&Q);
    Matrix_free(&R);
    Matrix_free(&H);

    return MATRIX_SUCCESS;
}

int Matrix_eigvals(Vector* eigenvalues, const Matrix* M)
{
    if (M->n == 2)
    {
        return _eigvals_two(eigenvalues, M);
    }

    return _eigvals_big(eigenvalues, M);
}

int Matrix_Vector_outer(Matrix* target, const Vector* a, const Vector* b)
{
    Matrix_prepare_target(target, a->dim, b->dim);

    for (size_t m = 0; m < a->dim; m++)
    {
        for (size_t n = 0; n < b->dim; n++)
        {
            target->values[m * target->n + n] = a->values[m] * b->values[n];
        }
    }

    return VECTOR_SUCCESS;
}

int Matrix_Hessenberg(Matrix* H, const Matrix* A)
{
    size_t m = A->m;
    size_t n = A->n;
    if (n != m)
    {
        return MATRIX_DIMENSION_ERROR;
    }

    Matrix_copy(H, A);

    for (size_t k = 0; k < m - 2; k++)
    {
        size_t rows = m - (k + 1);

        Vector x = Vector_zeros(rows);
        for (size_t i = 0; i < rows; i++)
        {
            x.values[i] = H->values[(k + 1 + i) * n + k];
        }
        double normx = Vector_norm(&x);
        if (normx == 0.0)
        {
            Vector_free(&x);
            continue;
        }

        double sign = (x.values[0] >= 0.0) ? 1.0 : -1.0;

        Vector e1    = Vector_zeros(rows);
        e1.values[0] = 1.0;

        Vector v = Vector_zeros_like(&x);
        Vector_copy(&v, &x);
        Vector_scale(&e1, sign * normx);
        Vector_add(&v, &e1);
        Vector_free(&e1);

        double vnorm = Vector_norm(&v);
        Vector_scale(&v, 1.0 / vnorm);

        for (size_t j = k; j < n; j++)
        {
            double dot = 0.0;
            for (size_t i = 0; i < rows; i++)
            {
                dot += v.values[i] * H->values[(k + 1 + i) * n + j];
            }

            for (size_t i = 0; i < rows; i++)
            {
                H->values[(k + 1 + i) * n + j] -= 2.0 * v.values[i] * dot;
            }
        }

        for (size_t i = 0; i < m; i++)
        {
            double dot = 0.0;
            for (size_t r = 0; r < rows; r++)
            {
                dot += H->values[i * n + (k + 1 + r)] * v.values[r];
            }
            for (size_t r = 0; r < rows; r++)
            {
                H->values[i * n + (k + 1 + r)] -= 2.0 * dot * v.values[r];
            }
        }
        Vector_free(&v);
        Vector_free(&x);
    }
    return MATRIX_SUCCESS;
}

void Matrix_QR_hessenberg(Matrix* Q, Matrix* R, Matrix* H, size_t active)
{
    size_t n = H->n;

    // Initialize Q = I
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            Q->values[i * n + j] = (i == j ? 1.0 : 0.0);

    // Copy H into R (we will transform R into upper triangular)
    Matrix_copy(R, H);

    for (size_t i = 0; i < active - 1; i++)
    {
        double a = R->values[i * n + i];
        double b = R->values[(i + 1) * n + i];

        if (fabs(b) < 1e-15)
            continue;

        // Compute Givens rotation
        double r = hypot(a, b);
        double c = a / r;
        double s = -b / r;

        // Apply Givens rotation to R (right multiplication)
        for (size_t j = i; j < active; j++)
        {
            double t1                  = R->values[i * n + j];
            double t2                  = R->values[(i + 1) * n + j];
            R->values[i * n + j]       = c * t1 - s * t2;
            R->values[(i + 1) * n + j] = s * t1 + c * t2;
        }

        // Apply Givens rotation to Q (accumulate Q)
        for (size_t j = 0; j < active; j++)
        {
            double t1                  = Q->values[j * n + i];
            double t2                  = Q->values[j * n + (i + 1)];
            Q->values[j * n + i]       = c * t1 - s * t2;
            Q->values[j * n + (i + 1)] = s * t1 + c * t2;
        }
    }
}

void Matrix_Matrix_dot_active(Matrix* H,
                              const Matrix* R,
                              const Matrix* Q,
                              size_t active)
{
    size_t n = H->n;

    Matrix tmp = Matrix_zeros_like(H);

    for (size_t i = 0; i < active; i++)
    {
        for (size_t j = 0; j < active; j++)
        {
            double sum = 0.0;
            for (size_t k = 0; k < active; k++)
                sum += R->values[i * n + k] * Q->values[k * n + j];

            tmp.values[i * n + j] = sum;
        }
    }

    // Copy back only active block
    for (size_t i = 0; i < active; i++)
        for (size_t j = 0; j < active; j++)
            H->values[i * n + j] = tmp.values[i * n + j];

    Matrix_free(&tmp);
}
