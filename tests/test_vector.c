#include "../include/vector.h"
#include "acutest.h"
#include <math.h>
#include <stddef.h>

void test_vector_create(void)
{
    Vector v = Vector_new(5, 3.14);

    TEST_CHECK(v.values != NULL);
    TEST_CHECK(v.dim == 5);

    for (size_t i = 0; i < 5; i++)
        TEST_CHECK(fabs(v.values[i] - 3.14) < EPS);

    Vector_free(&v);
}

void test_vector_new_vals(void)
{
    double vals[] = { 1, 2, 3, 4 };
    Vector v      = Vector_new_vals(4, vals);

    TEST_CHECK(v.dim == 4);

    for (size_t i = 0; i < 4; i++)
        TEST_CHECK(fabs(v.values[i] - vals[i]) < EPS);

    Vector_free(&v);
}

void test_vector_new_copy(void)
{
    double vals[] = { 5, 6, 7 };
    Vector v      = Vector_new_vals(3, vals);
    Vector c      = Vector_new_copy(&v);

    TEST_CHECK(c.dim == 3);

    for (size_t i = 0; i < 3; i++)
        TEST_CHECK(fabs(c.values[i] - vals[i]) < EPS);

    Vector_free(&v);
    Vector_free(&c);
}

void test_vector_zeros_like(void)
{
    Vector v = Vector_new(4, 9.0);
    Vector z = Vector_zeros_like(&v);

    TEST_CHECK(z.dim == 4);

    for (size_t i = 0; i < 4; i++)
        TEST_CHECK(z.values[i] == 0.0);

    Vector_free(&v);
    Vector_free(&z);
}

void test_vector_zeros_ones(void)
{
    Vector z = Vector_zeros(8);
    Vector o = Vector_ones(8);

    for (size_t i = 0; i < 8; i++)
    {
        TEST_CHECK(z.values[i] == 0.0);
        TEST_CHECK(o.values[i] == 1.0);
    }

    Vector_free(&z);
    Vector_free(&o);
}

void test_vector_copy(void)
{
    double vals[] = { 2.5, 3.5 };
    Vector v      = Vector_new_vals(2, vals);
    Vector t      = Vector_new(0, 0.0);

    Vector_copy(&t, &v);

    TEST_CHECK(t.dim == 2);
    TEST_CHECK(fabs(t.values[0] - 2.5) < EPS);
    TEST_CHECK(fabs(t.values[1] - 3.5) < EPS);

    Vector_free(&v);
    Vector_free(&t);
}

void test_vector_get_set(void)
{
    Vector v = Vector_new(4, 0.0);

    int status = Vector_set_item(&v, 2, 9.81);
    TEST_CHECK(status == VECTOR_SUCCESS);

    double val = 0.0;
    status     = Vector_get_at(&val, &v, 2);
    TEST_CHECK(status == VECTOR_SUCCESS);
    TEST_CHECK(fabs(val - 9.81) < EPS);

    status = Vector_set_item(&v, 10, 1.0);
    TEST_CHECK(status == VECTOR_DIMENSION_ERROR);

    status = Vector_get_at(&val, &v, 99);
    TEST_CHECK(status == VECTOR_DIMENSION_ERROR);

    Vector_free(&v);
}

void test_vector_norm(void)
{
    double vals[] = { 3, 4 };
    Vector v      = Vector_new_vals(2, vals);

    double n = Vector_norm(&v);
    TEST_CHECK(fabs(n - 5.0) < EPS);

    Vector_free(&v);
}

void test_vector_add_sub(void)
{
    double a_vals[] = { 1, 2, 3 };
    double b_vals[] = { 4, 5, 6 };

    Vector A = Vector_new_vals(3, a_vals);
    Vector B = Vector_new_vals(3, b_vals);

    int status = Vector_add(&A, &B);
    TEST_CHECK(status == VECTOR_SUCCESS);

    TEST_CHECK(fabs(A.values[0] - 5) < EPS);
    TEST_CHECK(fabs(A.values[1] - 7) < EPS);
    TEST_CHECK(fabs(A.values[2] - 9) < EPS);

    status = Vector_sub(&A, &B);
    TEST_CHECK(status == VECTOR_SUCCESS);

    TEST_CHECK(fabs(A.values[0] - 1) < EPS);
    TEST_CHECK(fabs(A.values[1] - 2) < EPS);
    TEST_CHECK(fabs(A.values[2] - 3) < EPS);

    Vector_free(&A);
    Vector_free(&B);
}

void test_vector_scale(void)
{
    Vector v = Vector_new(3, 2.0);

    int status = Vector_scale(&v, 0.5);
    TEST_CHECK(status == VECTOR_SUCCESS);

    for (size_t i = 0; i < 3; i++)
        TEST_CHECK(fabs(v.values[i] - 1.0) < EPS);

    Vector_free(&v);
}

void test_vector_broadcast_add(void)
{
    Vector v = Vector_new(4, 1.0);

    int status = Vector_broadcast_add(&v, 2.0);
    TEST_CHECK(status == VECTOR_SUCCESS);

    for (size_t i = 0; i < 4; i++)
        TEST_CHECK(fabs(v.values[i] - 3.0) < EPS);

    Vector_free(&v);
}

void test_vector_dot(void)
{
    double vals_u[] = { 2, 3, 4, 5 };
    double vals_v[] = { 3, 4, 1, 4 };
    Vector u        = Vector_new_vals(4, vals_u);
    Vector v        = Vector_new_vals(4, vals_v);
    Vector w        = Vector_new(6, 3.14);
    double dot      = 0.0;
    double sink     = 0.0;
    int status      = Vector_dot(&dot, &u, &v);
    int err_status  = Vector_dot(&sink, &u, &w);
    TEST_CHECK(fabs(dot - (2 * 3 + 3 * 4 + 4 + 5 * 4)) < EPS);
    TEST_CHECK(status == VECTOR_SUCCESS);
    TEST_CHECK(err_status == VECTOR_DIMENSION_ERROR);
    Vector_free(&u);
    Vector_free(&v);
    Vector_free(&w);

    Vector v1 = Vector_new_random_normal(200000, 0, 1);
    Vector v2 = Vector_new_random_normal(200000, 0, 1);
    status    = Vector_dot(&dot, &v1, &v2);
    double s  = 0.0;
    for (size_t i = 0; i < v1.dim; i++)
    {
        s += v1.values[i] * v2.values[i];
    }
    TEST_CHECK(fabs(s - dot) < EPS);
    Vector_free(&v1);
    Vector_free(&v2);
}

static double test_func_square(const Vector* x)
{
    double s = 0.0;
    for (size_t i = 0; i < x->dim; i++)
        s += -x->values[i] * x->values[i];
    s += 0.5;
    return s;
}

static double test_func_square_min(const Vector* x)
{
    double s = 0.0;
    for (size_t i = 0; i < x->dim; i++)
    {
        s += x->values[i] * x->values[i];
    }
    s -= 0.5;
    return s;
}

void test_vector_gradient(void)
{
    double vals[] = { 1.0, 2.0 };
    Vector x      = Vector_new_vals(2, vals);
    Vector grad   = Vector_new(2, 0.0);

    int status = Vector_gradient(&grad, &x, test_func_square);
    TEST_CHECK(status == VECTOR_SUCCESS);

    TEST_CHECK(fabs(grad.values[0] + 2.0) < EPS);
    TEST_CHECK(fabs(grad.values[1] + 4.0) < EPS);

    Vector_free(&x);
    Vector_free(&grad);
}

void test_vector_gradient_maximize(void)
{
    const int n    = 25;
    Vector x       = Vector_new(n, 0.5);
    double f_x     = test_func_square(&x);
    Vector out     = Vector_zeros_like(&x);
    int status     = Vector_gradient_maximize(&out, &x, test_func_square, 0.1);
    double f_x_max = test_func_square(&out);

    TEST_CHECK(status == VECTOR_SUCCESS);
    TEST_CHECK(f_x_max >= f_x);

    Vector_free(&x);
    Vector_free(&out);
}

void test_vector_gradient_minimize(void)
{
    const int n = 25;

    Vector x   = Vector_new(n, 0.5);
    double f_x = test_func_square_min(&x);
    Vector out = Vector_zeros_like(&x);
    int status = Vector_gradient_minimize(&out, &x, test_func_square_min, 0.1);
    double f_x_min = test_func_square_min(&out);

    TEST_CHECK(status == VECTOR_SUCCESS);

    TEST_CHECK(f_x >= f_x_min);

    Vector_free(&x);
    Vector_free(&out);
}

void test_vector_sort(void)
{
    const size_t N = 1000000;
    Vector v       = Vector_new_random_uniform(N, 0, 1);
    Vector_sort_inplace(&v);

    for (size_t i = 0; i < N - 1; i++)
    {
        TEST_CHECK(v.values[i] <= v.values[i + 1]);
    }

    Vector_free(&v);
}

TEST_LIST = { { "vector_create", test_vector_create },
              { "vector_new_vals", test_vector_new_vals },
              { "vector_new_copy", test_vector_new_copy },
              { "vector_zeros_like", test_vector_zeros_like },
              { "vector_zeros_ones", test_vector_zeros_ones },
              { "vector_copy", test_vector_copy },
              { "vector_get_set", test_vector_get_set },
              { "vector_norm", test_vector_norm },
              { "vector_add_sub", test_vector_add_sub },
              { "vector_dot", test_vector_dot },
              { "vector_scale", test_vector_scale },
              { "vector_broadcast_add", test_vector_broadcast_add },
              { "vector_gradient", test_vector_gradient },
              { "vector_gradient_maximize", test_vector_gradient_maximize },
              { "vector_gradient_minimize", test_vector_gradient_minimize },
              { "test_vector_sort", test_vector_sort },
              { NULL, NULL } };
