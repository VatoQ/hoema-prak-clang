// Uncomment for detailed output
#define TEST_NO_SIGNAL_HANDLING
// #define DEBUG
#include "../include/matrix.h"
#include "../include/vector.h"
#include "acutest.h"
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#ifndef N_C
#define N_C 5
#endif

void test_matrix_create(void)
{
    Matrix m = Matrix_zeros(3, 3, Real);
    TEST_CHECK(m.values != NULL);
    TEST_CHECK(m.m == 3);
    TEST_CHECK(m.n == 3);

    for (size_t i = 0; i < 9; i++)
    {
        double* ptr = m.values + i;

        TEST_CHECK(*ptr == 0.0);
    }

    Matrix_free(&m);
}

void test_matrix_symmetric(void)
{
    const size_t m = 2;
    Matrix M       = Matrix_new_random_symmetric(m, Real);
    ACCESS_VOID(real_t, M_values, M.values);
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = i + 1; j < m; j++)
        {
            const double a = M_values[i * m + j];
            const double b = M_values[j * m + i];
            TEST_CHECK(fabs(a - b) < EPS);
        }
    }

    Matrix_free(&M);
}

void test_matrix_add(void)
{
    real_t initvals1[8 * 5];
    real_t initvals2[8 * 5];
    for (size_t i = 0; i < 8 * 5; i++)
    {
        initvals1[i] = 4.5;
        initvals2[i] = 3.2;
    }

    Matrix A = Matrix_new_vals(8, 5, initvals1, Real);
    Matrix B = Matrix_new_vals(8, 5, initvals2, Real);

    int status = Matrix_add(&B, &A);

    ACCESS_VOID(real_t, B_values, B.values);
    for (size_t i = 0; i < 8 * 5; i++)
    {
        const real_t a = B_values[i];
        TEST_CHECK(fabs(a - 7.7) < EPS);
    }

    TEST_CHECK(status == MATRIX_SUCCESS);

    Matrix C = Matrix_new(5, 8, 2.3);

    status = Matrix_add(&C, &B);

    TEST_CHECK(status == MATRIX_DIMENSION_ERROR);

    Matrix_free(&A);
    Matrix_free(&B);
    Matrix_free(&C);
}
void test_matrix_sub(void)
{
    real_t initvals1[8 * 5];
    real_t initvals2[8 * 5];
    real_t initvals3[8 * 5];
    for (size_t i = 0; i < 8 * 5; i++)
    {
        initvals1[i] = 4.5;
        initvals2[i] = 3.2;
        initvals3[i] = 2.3;
    }

    Matrix A = Matrix_new_vals(8, 5, initvals1, Real);
    Matrix B = Matrix_new_vals(8, 5, initvals2, Real);
    Matrix C = Matrix_new_vals(5, 8, initvals3, Real);

    int status = Matrix_sub(&A, &B);
    ACCESS_VOID(real_t, A_values, A.values);

    for (size_t i = 0; i < 8 * 5; i++)
    {
        const real_t a = A_values[i];

        TEST_CHECK(fabs(a - 1.3) < EPS);
    }

    TEST_CHECK(status == MATRIX_SUCCESS);

    status = Matrix_add(&C, &B);

    TEST_CHECK(status == MATRIX_DIMENSION_ERROR);

    Matrix_free(&A);
    Matrix_free(&B);
    Matrix_free(&C);
}

void test_matrix_scale(void)
{
    const real_t val            = 51.34;
    const real_t lambda         = 0.23;
    real_t init_vals[N_C * N_C] = { val };
    for (size_t i = 0; i < 5 * 5; i++)
    {
        init_vals[i] = val;
    }
    Matrix M = Matrix_new_vals(N_C, N_C, init_vals, Real);
    Matrix_scale(&M, lambda);
    ACCESS_VOID(real_t, M_values, M.values);
    for (size_t i = 0; i < N_C * N_C; i++)
    {
        const real_t a = M_values[i];
        TEST_CHECK(fabs(a - val * lambda) < EPS);
    }
    Matrix_free(&M);
}

void test_matrix_inverse(void)
{
    double a = 0.523, b = -0.048, c = -0.048, d = 0.916;
    double vals[]     = { a, b, c, d };
    Matrix M          = Matrix_new_vals(2, 2, vals, Real);
    double scalar     = 1 / (a * d - b * c);
    double vals_inv[] = { d, -b, -c, a };
    Matrix M_inv      = Matrix_new(0, 0, Real);

    int status = Matrix_inverse(&M_inv, &M);
    TEST_CHECK(status == MATRIX_SUCCESS);
    ACCESS_VOID(real_t, M_values, M.values);
    ACCESS_VOID(real_t, M_inv_values, M_inv.values);

    if (status == MATRIX_SUCCESS)
    {
        for (size_t i = 0; i < 4; i++)
        {
            const double a = M_values[i];
            const double b = M_inv_values[i];

            TEST_CHECK(fabs(a - scalar * b) < EPS);
        }
    }

    vals[0] = 10.0, vals[1] = 5.0, vals[2] = 6.0, vals[3] = 3.0;
    Matrix_free(&M);
    Matrix_free(&M_inv);
    M      = Matrix_new_vals(2, 2, vals, Real);
    M_inv  = Matrix_new(0, 0, Real);
    status = Matrix_inverse(&M_inv, &M);
    TEST_CHECK(status == MATRIX_MATH_ERROR);
    Matrix_free(&M);
    Matrix_free(&M_inv);

    const size_t N = 100;
    M              = Matrix_new_random_symmetric(N, Real);
    M_inv          = Matrix_zeros_like(&M, Real);
    Matrix B       = Matrix_zeros_like(&M, Real);
    Vector v       = Vector_ones(N, Real);
    Matrix unit    = Matrix_diag(&v);
    status         = Matrix_inverse(&M_inv, &M);

    TEST_CHECK(status == MATRIX_SUCCESS);
    Matrix_Matrix_dot(&B, &M, &M_inv);
    TEST_CHECK(Matrix_all_close(&unit, &B));
    Matrix_free(&M);
    Matrix_free(&M_inv);
    Matrix_free(&B);
    Matrix_free(&unit);
    Vector_free(&v);
}

void test_matrix_set_get(void)
{
    Matrix M = Matrix_new(4, 4, Real);

    double set_val = 9.81;
    int status     = Matrix_set_at(&M, 2, 3, &set_val);
    TEST_CHECK(status == MATRIX_SUCCESS);

    double val = 0.0;
    status     = Matrix_get_at(&val, &M, 2, 3);
    TEST_CHECK(status == MATRIX_SUCCESS);
    TEST_CHECK(fabs(val - 9.81) < EPS);

    set_val = 1.0;
    status  = Matrix_set_at(&M, 10, 1, &set_val);
    TEST_CHECK(status == MATRIX_DIMENSION_ERROR);

    set_val = 99;
    status  = Matrix_get_at(&val, &M, 0, 99);
    TEST_CHECK(status == MATRIX_DIMENSION_ERROR);

    Matrix_free(&M);
}

void test_matrix_copy(void)
{
    real_t vals[]   = { 1, 2, 3, 4, 5, 6 };
    real_t zeros[6] = { 0 };
    Matrix A        = Matrix_new_vals(2, 3, vals, Real);
    Matrix B        = Matrix_new(0, 0, Real);

    Matrix_copy(&B, &A);

    TEST_CHECK(B.m == 2);
    TEST_CHECK(B.n == 3);
    ACCESS_VOID(real_t, B_values, B.values);

    for (size_t i = 0; i < 6; i++)
    {
        const real_t a = B_values[i];
        TEST_CHECK(fabs(a - vals[i]) < EPS);
    }

    Matrix_free(&A);
    Matrix_free(&B);
}

void test_matrix_zeros_like(void)
{
    double vals2[6] = { 3.14 };
    Matrix A        = Matrix_new_vals(5, 7, vals2, Real);
    Matrix Z        = Matrix_zeros_like(&A, Real);

    TEST_CHECK(Z.m == 5);
    TEST_CHECK(Z.n == 7);

    for (size_t i = 0; i < 5 * 7; i++)
    {
        double* ptr = (double*)Z.values + i;
        TEST_CHECK(*ptr == 0.0);
    }

    Matrix_free(&A);
    Matrix_free(&Z);
}

void test_matrix_diag(void)
{
    real_t vals[3] = { 1.0, 2.0, 3.0 };
    Vector v       = Vector_new_vals(3, vals, Real);

    Matrix D = Matrix_diag(&v);

    TEST_CHECK(D.m == 3);
    TEST_CHECK(D.n == 3);
    ACCESS_VOID(real_t, D_values, D.values);

    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            real_t val = D_values[i * 3 + j];
            DEBUG_PRINT("%f ", val);
            if (i == j)
            {
                TEST_CHECK(fabs(val - vals[i]) < EPS);
            }
            else
            {
                TEST_CHECK(val == 0.0);
            }
        }
        DEBUG_PRINT("\n");
    }

    Vector_free(&v);
    Matrix_free(&D);
}

void test_matrix_diag_val(void)
{
    const size_t N = 4;
    real_t val     = 7.5;
    Matrix D       = Matrix_diag_val(N, &val, Real);

    TEST_CHECK(D.m == N);
    TEST_CHECK(D.n == N);
    ACCESS_VOID(real_t, D_values, D.values);
    DEBUG_CODE(Matrix_print(&D));

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            const real_t a = D_values[i * N + j];
            if (i == j)
            {
                TEST_CHECK(fabs(a - val) < EPS);
            }
            else
            {
                TEST_CHECK(a == 0.0);
            }
        }
    }

    Matrix_free(&D);
}

void test_matrix_free(void)
{
    Matrix A = Matrix_new(5, 5, 2.0);
    Matrix_free(&A);

    TEST_CHECK(A.values == NULL);
    TEST_CHECK(A.m == 0);
    TEST_CHECK(A.n == 0);
    TEST_CHECK(A.dt == 0);
}

void test_matrix_vector_dot(void)
{
    real_t mvals[] = { 1, 2, 3, 4, 5, 6 };
    Matrix M       = Matrix_new_vals(2, 3, mvals, Real);

    real_t vvals[] = { 1, 2, 3 };
    Vector v       = Vector_new_vals(3, vvals, Real);

    Vector out = Vector_zeros(2, Real);

    int status = Matrix_Vector_dot(&out, &M, &v);
    ACCESS_VOID(real_t, out_values, out.values);
    TEST_CHECK(status == MATRIX_SUCCESS);
    DEBUG_PRINT("Vector in vector_dot():\n");
    DEBUG_CODE(Vector_print(&out));

    TEST_CHECK(fabs(out_values[0] - (1 * 1 + 2 * 2 + 3 * 3)) < EPS);
    TEST_CHECK(fabs(out_values[1] - (4 * 1 + 5 * 2 + 6 * 3)) < EPS);

    SCALE(5, status, 4);
    Vector_free(&v);
    Vector_free(&out);
    Matrix_free(&M);
}

void test_matrix_matrix_dot(void)
{
    real_t Avals[] = { 1, 2, 3, 4 };
    real_t Bvals[] = { 5, 6, 7, 8 };

    Matrix A = Matrix_new_vals(2, 2, Avals, Real);
    Matrix B = Matrix_new_vals(2, 2, Bvals, Real);
    Matrix C = Matrix_new(2, 2, Real);

    int status = Matrix_Matrix_dot(&C, &A, &B);
    TEST_CHECK(status == MATRIX_SUCCESS);
    ACCESS_VOID(real_t, C_values, C.values);
    real_t a = C_values[0];
    real_t b = C_values[1];
    real_t c = C_values[2];
    real_t d = C_values[3];

    TEST_CHECK(fabs(a - (1 * 5 + 2 * 7)) < EPS);
    TEST_CHECK(fabs(b - (1 * 6 + 2 * 8)) < EPS);
    TEST_CHECK(fabs(c - (3 * 5 + 4 * 7)) < EPS);
    TEST_CHECK(fabs(d - (3 * 6 + 4 * 8)) < EPS);

    Matrix_free(&A);
    Matrix_free(&B);
    Matrix_free(&C);
}

void test_matrix_matrix_dot_jordan(void)
{
    real_t Avals[] = { 2, 0, 0, 3 };
    real_t Bvals[] = { 4, 0, 0, 5 };

    Matrix A = Matrix_new_vals(2, 2, Avals, Real);
    Matrix B = Matrix_new_vals(2, 2, Bvals, Real);
    Matrix J = Matrix_new(0, 0, Real);

    int status = Matrix_Matrix_dot_jordan(&J, &A, &B);
    TEST_CHECK(status == MATRIX_SUCCESS);
    ACCESS_VOID(real_t, J_values, J.values);
    real_t a = J_values[0];
    real_t b = J_values[1];
    real_t c = J_values[2];
    real_t d = J_values[3];

    TEST_CHECK(fabs(a - 0.5 * (2 * 4 + 4 * 2)) < EPS);
    TEST_CHECK(fabs(d - 0.5 * (3 * 5 + 5 * 3)) < EPS);

    TEST_CHECK(b == 0.0);
    TEST_CHECK(c == 0.0);

    Matrix_free(&A);
    Matrix_free(&B);
    Matrix_free(&J);
}

void matrix_test_norm(void)
{

    double values[10] = { 0 };
    for (size_t i = 0; i < 10; i++)
    {
        values[i] = 4 * i;
    }
    Vector v                = Vector_new_vals(10, values, Real);
    Matrix M                = Matrix_diag(&v);
    const double max_lambda = 4 * 9;
    double calc_lambda;
    Matrix_norm(&calc_lambda, &M, SPECTRAL);

    TEST_CHECK(fabs(max_lambda - calc_lambda) < 1e-5);
}

// void test_matrix_eigenvalues(void)
// {
//     size_t N        = 5;
//     double values[] = { 1, 2, 3, 4, 5 };
//     Vector init_v   = Vector_new_vals(N, values, Real);
//     Matrix M        = Matrix_diag(&init_v);
//     Vector eigs     = Vector_zeros(N, Real);
//     ACCESS_VOID(real_t, eigs_values, eigs.values);
//     ACCESS_VOID(real_t, init_v_values, init_v.values);
//
//     Matrix_eigvals(&eigs, &M, true);
//     for (size_t i = 0; i < N; i++)
//     {
//         TEST_CHECK(fabs(eigs_values[i] - init_v_values[i]) < EPS);
//     }
//
//     Vector_free(&init_v);
//     Matrix_free(&M);
//     Vector_free(&eigs);
//
//     N = 3;
//
//     double values2[] = { 3, 2, 1, 2, 4, 5, 1, 5, 6 };
//     M                = Matrix_new_vals(N, N, values2, Real);
//     eigs             = Vector_zeros(N, Real);
//
//     Matrix_eigvals(&eigs, &M, true);
//
//     double true_eigs[] = { -0.3784815, 2.72913544, 10.64934606 };
//     eigs_values        = eigs.values;
//     for (size_t i = 0; i < N; i++)
//     {
//         TEST_CHECK(fabs(true_eigs[i] - eigs_values[i]) < EPS);
//     }
//     Matrix_free(&M);
//     Vector_free(&eigs);
// }

void test_matrix_outer(void)
{
    const size_t N = 3;
    Vector v1      = Vector_new_random_normal(N, 0, 1, Real);
    Vector v2      = Vector_new_random_normal(N, 0, 1, Real);
    Matrix M       = Matrix_new(N, N, Real);

    int status = Matrix_Vector_outer(&M, &v1, &v2);
    ACCESS_VOID(real_t, v1_values, v1.values);
    ACCESS_VOID(real_t, v2_values, v2.values);
    ACCESS_VOID(real_t, M_values, M.values);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            const real_t true_val = v1_values[i] * v2_values[j];
            real_t val            = M_values[i * N + j];
            DEBUG_PRINT("true: %f, val: %f\n", true_val, val);
            TEST_CHECK(fabs(true_val - val) < EPS);
        }
    }

    Vector x = Vector_new_random_normal(N, 0, 1, Real);
    status   = Matrix_Vector_outer_square(&M, &x);

    ACCESS_VOID(real_t, x_values, x.values);
    M_values = (real_t*)M.values;
    DEBUG_CODE(Matrix_print(&M));

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            const double true_val = x_values[i] * x_values[j];
            real_t val            = M_values[i * N + j];
            DEBUG_PRINT("true: %f, val: %f\n", true_val, val);

            TEST_CHECK(fabs(true_val - val) < EPS);
        }
    }
    Vector_free(&x);
    Vector_free(&v1);
    Vector_free(&v2);
    Matrix_free(&M);
}
TEST_LIST = { { "matrix_create", test_matrix_create },
              { "test_matrix_symmetric", test_matrix_symmetric },
              { "matrix_add", test_matrix_add },
              { "matrix_sub", test_matrix_sub },
              { "matrix_scale", test_matrix_scale },
              { "matrix_set_get", test_matrix_set_get },
              { "matrix_copy", test_matrix_copy },
              { "matrix_zeros_like", test_matrix_zeros_like },
              { "matrix_diag", test_matrix_diag },
              // { "test_matrix_inverse", test_matrix_inverse },
              { "matrix_diag_val", test_matrix_diag_val },
              { "matrix_free", test_matrix_free },
              { "matrix_vector_dot", test_matrix_vector_dot },
              { "matrix_matrix_dot", test_matrix_matrix_dot },
              { "matrix_matrix_dot_jordan", test_matrix_matrix_dot_jordan },
              // { "matrix_test_norm", matrix_test_norm },
              // { "test_matrix_eigenvalues", test_matrix_eigenvalues },
              { "test_matrix_outer", test_matrix_outer },
              { NULL, NULL } };
