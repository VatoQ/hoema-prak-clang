#define DEBUG
#include "../include/vector.h"
#include "acutest.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>

/* ============================================================
   Helpers for typed access
   ============================================================ */

static double get_real(const Vector* v, size_t i)
{
    switch (v->dt)
    {
        case Int:
            return (real_t)((int_t*)v->values)[i];
        case Real:
            return ((real_t*)v->values)[i];
        case Complex:
            return creal(((complex_t*)v->values)[i]);
        default:
            TEST_CHECK_(0, "Unknown DataType");
            return 0.0;
    }
}

static complex double get_complex(const Vector* v, size_t i)
{
    if (v->dt == Complex)
        return ((complex_t*)v->values)[i];
    TEST_CHECK_(0, "get_complex called on non-complex vector");
    return 0.0 + 0.0 * I;
}

static void set_real(Vector* v, size_t i, double x)
{
    switch (v->dt)
    {
        case Int:
            ((int_t*)v->values)[i] = (int)x;
            break;
        case Real:
            ((real_t*)v->values)[i] = x;
            break;
        case Complex:
            ((complex_t*)v->values)[i] = x + 0.0 * I;
            break;
        default:
            TEST_CHECK_(0, "Unknown DataType");
    }
}

/* ============================================================
   Construction tests
   ============================================================ */

void test_new_vals_real(void)
{
    real_t init[3] = { 1.0, 2.0, 3.0 };
    Vector v       = Vector_new_vals(3, init, Real);

    TEST_CHECK(v.dim == 3);
    ACCESS_VOID(real_t, v_values, v.values);
    for (size_t i = 0; i < 3; i++)
    {
        real_t val = v_values[i];
        DEBUG_PRINT("a: %f\n", val);
        TEST_CHECK(fabs(val - init[i]) < EPS);
    }

    Vector_free(&v);
}

void test_new_vals_int(void)
{
    int_t init[4] = { 10, 20, 30, 40 };
    Vector v      = Vector_new_vals(4, init, Int);

    TEST_CHECK(v.dim == 4);
    ACCESS_VOID(int_t, v_values, v.values);
    for (size_t i = 0; i < 4; i++)
    {
        TEST_CHECK(v_values[i] == init[i]);
    }

    Vector_free(&v);
}

void test_new_vals_complex(void)
{
    complex double init[3] = { 1.0 + 2.0 * I, -3.0 + 0.5 * I, 0.0 + 7.0 * I };
    Vector v               = Vector_new_vals(3, init, Complex);

    TEST_CHECK(v.dim == 3);
    ACCESS_VOID(complex_t, v_values, v.values);
    for (size_t i = 0; i < 3; i++)
    {
        TEST_CHECK(v_values[i] == init[i]);
    }

    Vector_free(&v);
}

void test_copy(void)
{
    real_t init[3] = { 9.0, 8.0, 7.0 };
    Vector v       = Vector_new_vals(3, init, Real);
    Vector c       = Vector_new_copy(&v);

    TEST_CHECK(c.dim == 3);
    ACCESS_VOID(real_t, v_values, v.values);
    for (size_t i = 0; i < 3; i++)
    {
        TEST_CHECK(v_values[i] == init[i]);
    }

    Vector_free(&v);
    Vector_free(&c);
}

/* ============================================================
   Accessors
   ============================================================ */

void test_get_set_item_real(void)
{
    Vector v = Vector_zeros(3, Real);

    real_t x;
    TEST_CHECK(Vector_get_at(&x, &v, 1) == VECTOR_SUCCESS);
    TEST_CHECK(x == 0.0);

    real_t newval = 5.5;
    TEST_CHECK(Vector_set_item(&v, 1, &newval) == VECTOR_SUCCESS);
    ACCESS_VOID(real_t, v_values, v.values);
    TEST_CHECK(v_values[1] == 5.5);

    TEST_CHECK(Vector_get_at(&x, &v, 99) == VECTOR_DIMENSION_ERROR);
    TEST_CHECK(Vector_set_item(&v, 99, &newval) == VECTOR_DIMENSION_ERROR);

    Vector_free(&v);
}

void test_get_set_item_int(void)
{
    Vector v = Vector_zeros(3, Int);

    int_t x;
    TEST_CHECK(Vector_get_at(&x, &v, 1) == VECTOR_SUCCESS);
    TEST_CHECK(x == 0.0);

    int_t newval = 7.0;
    TEST_CHECK(Vector_set_item(&v, 1, &newval) == VECTOR_SUCCESS);
    ACCESS_VOID(int_t, v_values, v.values);
    TEST_CHECK(v_values[1] == 7);

    Vector_free(&v);
}

void test_get_set_item_complex(void)
{
    Vector v = Vector_zeros(3, Complex);

    complex double c = 2.0 + 3.0 * I;
    TEST_CHECK(Vector_set_item(&v, 1, &c) == VECTOR_SUCCESS);
    ACCESS_VOID(complex_t, v_values, v.values);
    TEST_CHECK(v_values[1] == c);

    Vector_free(&v);
}

/* ============================================================
   all_close
   ============================================================ */

void test_all_close_real(void)
{
    double a[3] = { 1.0, 2.0, 3.0 };
    double b[3] = { 1.0 + 1e-9, 2.0 - 1e-9, 3.0 };

    Vector u = Vector_new_vals(3, a, Real);
    Vector v = Vector_new_vals(3, b, Real);

    TEST_CHECK(Vector_all_close(&u, &v) == 1);

    b[1] = 2.1;
    Vector_free(&v);
    v = Vector_new_vals(3, b, Real);

    TEST_CHECK(Vector_all_close(&u, &v) == 0);

    Vector_free(&u);
    Vector_free(&v);
}

/* ============================================================
   Random vectors (Real only)
   ============================================================ */

void test_random_uniform_real(void)
{
    Vector_init_prng(123);

    Vector v = Vector_new_random_uniform(100, -1.0, 1.0, Real);

    ACCESS_VOID(real_t, v_values, v.values);
    for (size_t i = 0; i < 100; i++)
    {
        double x = v_values[i];
        TEST_CHECK(x >= -1.0 && x <= 1.0);
    }

    Vector_free(&v);
}

void test_random_normal_real(void)
{
    Vector_init_prng(123);

    Vector v = Vector_new_random_normal(200, 0.0, 1.0, Real);

    double sum = 0.0;
    ACCESS_VOID(real_t, v_values, v.values);
    for (size_t i = 0; i < 200; i++)
        sum += v_values[i];

    double mean = sum / 200.0;
    TEST_CHECK(fabs(mean) < 0.3);

    Vector_free(&v);
}

/* ============================================================
   Math operations
   ============================================================ */

void test_norm_real(void)
{
    real_t a[3] = { 3.0, 4.0, 12.0 };
    Vector v    = Vector_new_vals(3, a, Real);

    real_t n = Vector_norm(&v);
    TEST_CHECK(fabs(n - 13.0) < 1e-12);

    Vector_free(&v);
}

void test_norm_complex(void)
{
    complex double a[2] = { 3.0 + 4.0 * I, 12.0 + 0.0 * I };
    Vector v            = Vector_new_vals(2, a, Complex);

    double n = Vector_norm(&v);
    TEST_CHECK(fabs(n - 13.0) < 1e-12);

    Vector_free(&v);
}

void test_add_sub_real(void)
{
    real_t a[3] = { 1.0, 2.0, 3.0 };
    real_t b[3] = { 4.0, 5.0, 6.0 };

    Vector u = Vector_new_vals(3, a, Real);
    Vector v = Vector_new_vals(3, b, Real);

    TEST_CHECK(Vector_add(&u, &v) == VECTOR_SUCCESS);
    TEST_CHECK(get_real(&u, 0) == 5.0);
    TEST_CHECK(get_real(&u, 1) == 7.0);
    TEST_CHECK(get_real(&u, 2) == 9.0);

    TEST_CHECK(Vector_sub(&u, &v) == VECTOR_SUCCESS);
    TEST_CHECK(get_real(&u, 0) == 1.0);
    TEST_CHECK(get_real(&u, 1) == 2.0);
    TEST_CHECK(get_real(&u, 2) == 3.0);

    Vector_free(&u);
    Vector_free(&v);
}

void test_broad_add_sub(void)
{
    const size_t N            = 9;
    const complex_t init_val  = 3.0 + 4.0 * I;
    const complex_t broad_val = -2.0 + 0.5 * I;
    Vector u                  = Vector_new(N, &init_val, Complex);
    Vector v                  = Vector_zeros_like(&v);
    Vector_copy(&v, &u);
    Vector_broadcast_add(&u, &broad_val);
    Vector_broadcast_sub(&v, &broad_val);

    ACCESS_VOID(complex_t, u_values, u.values);
    ACCESS_VOID(complex_t, v_values, v.values);

    for (size_t i = 0; i < N; i++)
    {
        TEST_CHECK(u_values[i] == init_val + broad_val);
        TEST_CHECK(v_values[i] == init_val - broad_val);
    }
    Vector_free(&v);
    Vector_free(&u);
}

void test_dot_real(void)
{
    real_t a[3] = { 1.0, 2.0, 3.0 };
    real_t b[3] = { 4.0, -1.0, 2.0 };

    Vector u = Vector_new_vals(3, a, Real);
    Vector v = Vector_new_vals(3, b, Real);

    real_t dp;
    TEST_CHECK(Vector_dot(&dp, &u, &v) == VECTOR_SUCCESS);
    TEST_CHECK(fabs(dp - (1 * 4 + 2 * (-1) + 3 * 2)) < 1e-12);

    Vector_free(&u);
    Vector_free(&v);
}

void test_dot_complex(void)
{
    complex_t a[2] = { 1.0 + 2.0 * I, 3.0 + 4.0 * I };
    complex_t b[2] = { 2.0 + 0.0 * I, -1.0 + 1.0 * I };

    Vector u = Vector_new_vals(2, a, Complex);
    Vector v = Vector_new_vals(2, b, Complex);

    complex_t dp;
    TEST_CHECK(Vector_dot(&dp, &u, &v) == VECTOR_SUCCESS);

    complex double expected = (a[0] * b[0]) + (a[1] * b[1]);

    TEST_CHECK(fabs(creal(dp) - creal(expected)) < 1e-12);

    Vector_free(&u);
    Vector_free(&v);
}

/* ============================================================
   Sorting (Real only)
   ============================================================ */

void test_sort_real(void)
{
    real_t a[5] = { 5, 1, 4, 3, 2 };
    Vector v    = Vector_new_vals(5, a, Real);

    Vector sorted = Vector_zeros(5, Real);
    TEST_CHECK(Vector_sort(&sorted, &v) == VECTOR_SUCCESS);

    ACCESS_VOID(real_t, sorted_values, sorted.values);
    for (size_t i = 0; i < 5; i++)
    {
        TEST_CHECK(sorted_values[i] == (real_t)(i + 1));
    }

    TEST_CHECK(Vector_sort_inplace(&v) == VECTOR_SUCCESS);
    ACCESS_VOID(real_t, v_values, v.values);
    for (size_t i = 0; i < 5; i++)
    {
        TEST_CHECK(v_values[i] == (real_t)(i + 1));
    }

    Vector_free(&v);
    Vector_free(&sorted);

    const size_t count = 5000;
    const double min   = 10;
    const double max   = 1000;
    Vector large       = Vector_new_random_uniform(count, min, max, Real);
    Vector_sort_inplace(&large);

    ACCESS_VOID(real_t, large_values, large.values);

    for (size_t i = 0; i < count - 1; i++)
    {
        TEST_CHECK(large_values[i] <= large_values[i + 1]);
    }
    Vector_free(&large);
}

/* ============================================================
   Max / Min (Real only)
   ============================================================ */

void test_max_min_real(void)
{
    real_t a[5] = { -1, 10, 3, 7, 2 };
    Vector v    = Vector_new_vals(5, a, Real);

    real_t mx, mn;
    Vector_max(&mx, &v);
    Vector_min(&mn, &v);

    TEST_CHECK(mx == 10.0);
    TEST_CHECK(mn == -1.0);

    Vector_free(&v);
}

void test_scale(void)
{
    const size_t N           = 100;
    const complex_t lambda   = -2.51 + 0.74 * I;
    const complex_t init_val = 5.15 + 2.0 * I;
    Vector v                 = Vector_new(N, &init_val, Complex);

    Vector_scale(&v, &lambda);
    ACCESS_VOID(complex_t, v_values, v.values);
    for (size_t i = 0; i < N; i++)
    {
        TEST_CHECK(cabs(v_values[i] - lambda * init_val) < EPS);
    }
    Vector_free(&v);
}

void test_dot(void)
{
    const size_t N = 1000;
    Vector u       = Vector_new_random_normal(N, 0, 100, Int);
    Vector v       = Vector_new_random_normal(N, 0, 100, Int);
    ACCESS_VOID(int_t, u_values, u.values);
    ACCESS_VOID(int_t, v_values, v.values);

    real_t dot;
    Vector_dot(&dot, &u, &v);
    real_t dot_expected = 0;
    for (size_t i = 0; i < N; i++)
    {
        dot_expected += u_values[i] * v_values[i];
    }
    TEST_CHECK(fabs(dot - dot_expected) < EPS);
}

/* ============================================================
   Test list
   ============================================================ */

TEST_LIST = { { "new_vals_real", test_new_vals_real },
              { "new_vals_int", test_new_vals_int },
              { "new_vals_complex", test_new_vals_complex },
              { "copy", test_copy },
              { "get_set_item_real", test_get_set_item_real },
              { "get_set_item_int", test_get_set_item_int },
              { "get_set_item_complex", test_get_set_item_complex },
              { "all_close_real", test_all_close_real },
              { "random_uniform_real", test_random_uniform_real },
              { "random_normal_real", test_random_normal_real },
              { "norm_real", test_norm_real },
              { "norm_complex", test_norm_complex },
              { "add_sub_real", test_add_sub_real },
              { "test_broad_add_sub", test_broad_add_sub },
              { "dot_real", test_dot_real },
              { "dot_complex", test_dot_complex },
              { "sort_real", test_sort_real },
              { "max_min_real", test_max_min_real },
              { "test_scale", test_scale },
              { "test_dot", test_dot },
              { NULL, NULL } };
