#include "../include/matrix.h"
#include "../include/vector.h"
#include "acutest.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

void test_matrix_create(void)
{
    Matrix m = Matrix_new(3, 3, 0);
    TEST_CHECK(m.values != NULL);
    TEST_CHECK(m.m == 3);
    TEST_CHECK(m.n == 3);

    for (size_t i = 0; i < 9; i++)
    {
        TEST_CHECK(m.values[i] == 0.0);
    }

    Matrix_free(&m);
}

void test_matrix_add(void)
{
    Matrix A = Matrix_new(8, 5, 4.5);
    Matrix B = Matrix_new(8, 5, 3.2);

    int status = Matrix_add(&B, &A);

    for (size_t i = 0; i < 8 * 5; i++)
    {
        TEST_CHECK(fabs(B.values[i] - 7.7) < EPS);
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
    Matrix A = Matrix_new(8, 5, 4.5);
    Matrix B = Matrix_new(8, 5, 3.2);

    int status = Matrix_sub(&A, &B);

    for (size_t i = 0; i < 8 * 5; i++)
    {
        TEST_CHECK(fabs(A.values[i] - 1.3) < EPS);
    }

    TEST_CHECK(status == MATRIX_SUCCESS);

    Matrix C = Matrix_new(5, 8, 2.3);

    status = Matrix_add(&C, &B);

    TEST_CHECK(status == MATRIX_DIMENSION_ERROR);

    Matrix_free(&A);
    Matrix_free(&B);
    Matrix_free(&C);
}

void test_matrix_scale(void)
{
    const double val    = 51.34;
    const double lambda = 0.23;
    Matrix M            = Matrix_new(10, 10, val);
    Matrix_scale(&M, lambda);
    for (size_t i = 0; i < 10 * 10; i++)
    {
        TEST_CHECK(fabs(M.values[i] - val * lambda) < EPS);
    }
    Matrix_free(&M);
}

void test_matrix_inverse(void)
{
    double a = 0.523, b = -0.048, c = -0.048, d = 0.916;
    double vals[]     = { a, b, c, d };
    Matrix M          = Matrix_new_vals(2, 2, vals);
    double scalar     = 1 / (a * d - b * c);
    double vals_inv[] = { d, -b, -c, a };
    Matrix M_inv      = Matrix_new(0, 0, 0.0);

    int status = Matrix_inverse(&M_inv, &M);
    TEST_CHECK(status == MATRIX_SUCCESS);

    if (status == MATRIX_SUCCESS)
    {
        for (size_t i = 0; i < 4; i++)
        {
            TEST_CHECK(fabs(M_inv.values[i] - scalar * vals_inv[i]) < EPS);
        }
    }

    vals[0] = 10.0, vals[1] = 5.0, vals[2] = 6.0, vals[3] = 3.0;
    Matrix_free(&M);
    Matrix_free(&M_inv);
    M      = Matrix_new_vals(2, 2, vals);
    M_inv  = Matrix_new(0, 0, 0.0);
    status = Matrix_inverse(&M_inv, &M);
    TEST_CHECK(status == MATRIX_MATH_ERROR);
    Matrix_free(&M);
    Matrix_free(&M_inv);

    M        = Matrix_new_random_normal(20, 20, 0, 1);
    M_inv    = Matrix_zeros_like(&M);
    Matrix B = Matrix_zeros_like(&M);
    Vector v = Vector_ones(20);
    Matrix I = Matrix_diag(&v);
    status   = Matrix_inverse(&M_inv, &M);
    TEST_CHECK(status == MATRIX_SUCCESS);
    Matrix_Matrix_dot(&B, &M, &M_inv);
    TEST_CHECK(Matrix_all_close(&I, &B));
    Matrix_free(&M);
    Matrix_free(&M_inv);
    Matrix_free(&B);
    Matrix_free(&I);
    Vector_free(&v);
}

void test_matrix_set_get(void)
{
    Matrix M = Matrix_new(4, 4, 0.0);

    int status = Matrix_set_at(&M, 2, 3, 9.81);
    TEST_CHECK(status == MATRIX_SUCCESS);

    double val = 0.0;
    status     = Matrix_get_at(&val, &M, 2, 3);
    TEST_CHECK(status == MATRIX_SUCCESS);
    TEST_CHECK(fabs(val - 9.81) < EPS);

    status = Matrix_set_at(&M, 10, 1, 1.0);
    TEST_CHECK(status == MATRIX_DIMENSION_ERROR);

    status = Matrix_get_at(&val, &M, 0, 99);
    TEST_CHECK(status == MATRIX_DIMENSION_ERROR);

    Matrix_free(&M);
}

void test_matrix_copy(void)
{
    double vals[] = { 1, 2, 3, 4, 5, 6 };
    Matrix A      = Matrix_new_vals(2, 3, vals);
    Matrix B      = Matrix_new(0, 0, 0.0);

    Matrix_copy(&B, &A);

    TEST_CHECK(B.m == 2);
    TEST_CHECK(B.n == 3);

    for (size_t i = 0; i < 6; i++)
    {
        TEST_CHECK(fabs(B.values[i] - vals[i]) < EPS);
    }

    Matrix_free(&A);
    Matrix_free(&B);
}

void test_matrix_zeros_like(void)
{
    Matrix A = Matrix_new(5, 7, 3.14);
    Matrix Z = Matrix_zeros_like(&A);

    TEST_CHECK(Z.m == 5);
    TEST_CHECK(Z.n == 7);

    for (size_t i = 0; i < 5 * 7; i++)
    {
        TEST_CHECK(Z.values[i] == 0.0);
    }

    Matrix_free(&A);
    Matrix_free(&Z);
}

void test_matrix_diag(void)
{
    double vals[] = { 1.0, 2.0, 3.0 };
    Vector v      = Vector_new_vals(3, vals);

    Matrix D = Matrix_diag(&v);

    TEST_CHECK(D.m == 3);
    TEST_CHECK(D.n == 3);

    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            if (i == j)
                TEST_CHECK(fabs(D.values[i * 3 + j] - vals[i]) < EPS);
            else
                TEST_CHECK(D.values[i * 3 + j] == 0.0);

    Vector_free(&v);
    Matrix_free(&D);
}

void test_matrix_diag_val(void)
{
    Matrix D = Matrix_diag_val(4, 7.5);

    TEST_CHECK(D.m == 4);
    TEST_CHECK(D.n == 4);

    for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 4; j++)
            if (i == j)
                TEST_CHECK(fabs(D.values[i * 4 + j] - 7.5) < EPS);
            else
                TEST_CHECK(D.values[i * 4 + j] == 0.0);

    Matrix_free(&D);
}

void test_matrix_free(void)
{
    Matrix A = Matrix_new(5, 5, 2.0);
    Matrix_free(&A);

    TEST_CHECK(A.values == NULL);
    TEST_CHECK(A.m == 0);
    TEST_CHECK(A.n == 0);
}

void test_matrix_vector_dot(void)
{
    double mvals[] = { 1, 2, 3, 4, 5, 6 };
    Matrix M       = Matrix_new_vals(2, 3, mvals);

    double vvals[] = { 1, 2, 3 };
    Vector v       = Vector_new_vals(3, vvals);

    Vector out = Vector_new(2, 0.0);

    int status = Matrix_Vector_dot(&out, &M, &v);
    TEST_CHECK(status == MATRIX_SUCCESS);

    TEST_CHECK(fabs(out.values[0] - (1 * 1 + 2 * 2 + 3 * 3)) < EPS);
    TEST_CHECK(fabs(out.values[1] - (4 * 1 + 5 * 2 + 6 * 3)) < EPS);

    Vector_free(&v);
    Vector_free(&out);
    Matrix_free(&M);
}

void test_matrix_matrix_dot(void)
{
    double Avals[] = { 1, 2, 3, 4 };
    double Bvals[] = { 5, 6, 7, 8 };

    Matrix A = Matrix_new_vals(2, 2, Avals);
    Matrix B = Matrix_new_vals(2, 2, Bvals);
    Matrix C = Matrix_new(0, 0, 0.0);

    int status = Matrix_Matrix_dot(&C, &A, &B);
    TEST_CHECK(status == MATRIX_SUCCESS);

    TEST_CHECK(fabs(C.values[0] - (1 * 5 + 2 * 7)) < EPS);
    TEST_CHECK(fabs(C.values[1] - (1 * 6 + 2 * 8)) < EPS);
    TEST_CHECK(fabs(C.values[2] - (3 * 5 + 4 * 7)) < EPS);
    TEST_CHECK(fabs(C.values[3] - (3 * 6 + 4 * 8)) < EPS);

    Matrix_free(&A);
    Matrix_free(&B);
    Matrix_free(&C);
}

void test_matrix_matrix_dot_jordan(void)
{
    double Avals[] = { 2, 0, 0, 3 };
    double Bvals[] = { 4, 0, 0, 5 };

    Matrix A = Matrix_new_vals(2, 2, Avals);
    Matrix B = Matrix_new_vals(2, 2, Bvals);
    Matrix J = Matrix_new(0, 0, 0.0);

    int status = Matrix_Matrix_dot_jordan(&J, &A, &B);
    TEST_CHECK(status == MATRIX_SUCCESS);

    TEST_CHECK(fabs(J.values[0] - 0.5 * (2 * 4 + 4 * 2)) < EPS);
    TEST_CHECK(fabs(J.values[3] - 0.5 * (3 * 5 + 5 * 3)) < EPS);

    TEST_CHECK(J.values[1] == 0.0);
    TEST_CHECK(J.values[2] == 0.0);

    Matrix_free(&A);
    Matrix_free(&B);
    Matrix_free(&J);
}

void matrix_test_norm(void)
{

    double values[100] = { 0 };
    for (size_t i = 0; i < 100; i++)
    {
        values[i] = 4 * i;
    }
    Vector v                 = Vector_new_vals(100, values);
    Matrix M                 = Matrix_diag(&v);
    const double max_lambda  = 4 * 99;
    const double calc_lambda = Matrix_norm(&M, SPECTRAL);
    TEST_CHECK(fabs(max_lambda - calc_lambda) < 1e-5);
}

TEST_LIST = { { "matrix_create", test_matrix_create },
              { "matrix_add", test_matrix_add },
              { "matrix_sub", test_matrix_sub },
              { "matrix_scale", test_matrix_scale },
              { "matrix_set_get", test_matrix_set_get },
              { "matrix_copy", test_matrix_copy },
              { "matrix_zeros_like", test_matrix_zeros_like },
              { "matrix_diag", test_matrix_diag },
              { "test_matrix_inverse", test_matrix_inverse },
              { "matrix_diag_val", test_matrix_diag_val },
              { "matrix_free", test_matrix_free },
              { "matrix_vector_dot", test_matrix_vector_dot },
              { "matrix_matrix_dot", test_matrix_matrix_dot },
              { "matrix_matrix_dot_jordan", test_matrix_matrix_dot_jordan },
              { "matrix_test_norm", matrix_test_norm },
              { NULL, NULL } };
