#include "../include/vector.h"
#include "acutest.h"
#include <math.h>
#include <stddef.h>

/* ============================================================
   Helpers for typed access
   ============================================================ */

static double get_real(const Vector* v, size_t i)
{
    switch (v->dt)
    {
        case Int:
            return (double)((int*)v->values)[i];
        case Real:
            return ((double*)v->values)[i];
        case Complex:
            return creal(((complex double*)v->values)[i]);
        default:
            TEST_CHECK_(0, "Unknown DataType");
            return 0.0;
    }
}

static complex double get_complex(const Vector* v, size_t i)
{
    if (v->dt == Complex)
        return ((complex double*)v->values)[i];
    TEST_CHECK_(0, "get_complex called on non-complex vector");
    return 0.0 + 0.0 * I;
}

static void set_real(Vector* v, size_t i, double x)
{
    switch (v->dt)
    {
        case Int:
            ((int*)v->values)[i] = (int)x;
            break;
        case Real:
            ((double*)v->values)[i] = x;
            break;
        case Complex:
            ((complex double*)v->values)[i] = x + 0.0 * I;
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
    double init[3] = { 1.0, 2.0, 3.0 };
    Vector v       = Vector_new_vals(3, init, Real);

    TEST_CHECK(v.dim == 3);
    for (size_t i = 0; i < 3; i++)
        TEST_CHECK(get_real(&v, i) == init[i]);

    Vector_free(&v);
}

void test_new_vals_int(void)
{
    int init[4] = { 10, 20, 30, 40 };
    Vector v    = Vector_new_vals(4, init, Int);

    TEST_CHECK(v.dim == 4);
    for (size_t i = 0; i < 4; i++)
        TEST_CHECK(get_real(&v, i) == init[i]);

    Vector_free(&v);
}

void test_new_vals_complex(void)
{
    complex double init[3] = { 1.0 + 2.0 * I, -3.0 + 0.5 * I, 0.0 + 7.0 * I };
    Vector v               = Vector_new_vals(3, init, Complex);

    TEST_CHECK(v.dim == 3);
    for (size_t i = 0; i < 3; i++)
        TEST_CHECK(get_complex(&v, i) == init[i]);

    Vector_free(&v);
}

void test_copy(void)
{
    double init[3] = { 9.0, 8.0, 7.0 };
    Vector v       = Vector_new_vals(3, init, Real);
    Vector c       = Vector_new_copy(&v);

    TEST_CHECK(c.dim == 3);
    for (size_t i = 0; i < 3; i++)
        TEST_CHECK(get_real(&c, i) == init[i]);

    Vector_free(&v);
    Vector_free(&c);
}

/* ============================================================
   Accessors
   ============================================================ */

void test_get_set_item_real(void)
{
    Vector v = Vector_zeros(3, Real);

    double x;
    TEST_CHECK(Vector_get_at(&x, &v, 1) == VECTOR_SUCCESS);
    TEST_CHECK(x == 0.0);

    double newval = 5.5;
    TEST_CHECK(Vector_set_item(&v, 1, &newval) == VECTOR_SUCCESS);
    TEST_CHECK(get_real(&v, 1) == 5.5);

    TEST_CHECK(Vector_get_at(&x, &v, 99) == VECTOR_DIMENSION_ERROR);
    TEST_CHECK(Vector_set_item(&v, 99, &newval) == VECTOR_DIMENSION_ERROR);

    Vector_free(&v);
}

void test_get_set_item_int(void)
{
    Vector v = Vector_zeros(3, Int);

    double x;
    TEST_CHECK(Vector_get_at(&x, &v, 1) == VECTOR_SUCCESS);
    TEST_CHECK(x == 0.0);

    int newval = 7;
    TEST_CHECK(Vector_set_item(&v, 1, &newval) == VECTOR_SUCCESS);
    TEST_CHECK(get_real(&v, 1) == 7.0);

    Vector_free(&v);
}

void test_get_set_item_complex(void)
{
    Vector v = Vector_zeros(3, Complex);

    complex double c = 2.0 + 3.0 * I;
    TEST_CHECK(Vector_set_item(&v, 1, &c) == VECTOR_SUCCESS);
    TEST_CHECK(get_complex(&v, 1) == c);

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

    for (size_t i = 0; i < 100; i++)
    {
        double x = get_real(&v, i);
        TEST_CHECK(x >= -1.0 && x <= 1.0);
    }

    Vector_free(&v);
}

void test_random_normal_real(void)
{
    Vector_init_prng(123);

    Vector v = Vector_new_random_normal(200, 0.0, 1.0, Real);

    double sum = 0.0;
    for (size_t i = 0; i < 200; i++)
        sum += get_real(&v, i);

    double mean = sum / 200.0;
    TEST_CHECK(fabs(mean) < 0.3);

    Vector_free(&v);
}

/* ============================================================
   Math operations
   ============================================================ */

void test_norm_real(void)
{
    double a[3] = { 3.0, 4.0, 12.0 };
    Vector v    = Vector_new_vals(3, a, Real);

    double n = Vector_norm(&v);
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
    double a[3] = { 1.0, 2.0, 3.0 };
    double b[3] = { 4.0, 5.0, 6.0 };

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

void test_dot_real(void)
{
    double a[3] = { 1.0, 2.0, 3.0 };
    double b[3] = { 4.0, -1.0, 2.0 };

    Vector u = Vector_new_vals(3, a, Real);
    Vector v = Vector_new_vals(3, b, Real);

    double dp;
    TEST_CHECK(Vector_dot(&dp, &u, &v) == VECTOR_SUCCESS);
    TEST_CHECK(fabs(dp - (1 * 4 + 2 * (-1) + 3 * 2)) < 1e-12);

    Vector_free(&u);
    Vector_free(&v);
}

void test_dot_complex(void)
{
    complex double a[2] = { 1.0 + 2.0 * I, 3.0 + 4.0 * I };
    complex double b[2] = { 2.0 + 0.0 * I, -1.0 + 1.0 * I };

    Vector u = Vector_new_vals(2, a, Complex);
    Vector v = Vector_new_vals(2, b, Complex);

    double dp;
    TEST_CHECK(Vector_dot(&dp, &u, &v) == VECTOR_SUCCESS);

    complex double expected = (a[0] * b[0]) + (a[1] * b[1]);

    TEST_CHECK(fabs(dp - creal(expected)) < 1e-12);

    Vector_free(&u);
    Vector_free(&v);
}

/* ============================================================
   Sorting (Real only)
   ============================================================ */

void test_sort_real(void)
{
    double a[5] = { 5, 1, 4, 3, 2 };
    Vector v    = Vector_new_vals(5, a, Real);

    Vector sorted = Vector_zeros(5, Real);
    TEST_CHECK(Vector_sort(&sorted, &v) == VECTOR_SUCCESS);

    for (size_t i = 0; i < 5; i++)
        TEST_CHECK(get_real(&sorted, i) == (double)(i + 1));

    TEST_CHECK(Vector_sort_inplace(&v) == VECTOR_SUCCESS);
    for (size_t i = 0; i < 5; i++)
        TEST_CHECK(get_real(&v, i) == (double)(i + 1));

    Vector_free(&v);
    Vector_free(&sorted);
}

/* ============================================================
   Max / Min (Real only)
   ============================================================ */

void test_max_min_real(void)
{
    double a[5] = { -1, 10, 3, 7, 2 };
    Vector v    = Vector_new_vals(5, a, Real);

    double mx, mn;
    Vector_max(&mx, &v);
    Vector_min(&mn, &v);

    TEST_CHECK(mx == 10.0);
    TEST_CHECK(mn == -1.0);

    Vector_free(&v);
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
              { "dot_real", test_dot_real },
              { "dot_complex", test_dot_complex },
              { "sort_real", test_sort_real },
              { "max_min_real", test_max_min_real },
              { NULL, NULL } };
