#include "vector.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Vector Vector_new(const size_t dim, const double init_val)
{
    double* vals = calloc(dim, sizeof(double));
    Vector v     = { vals, dim };
    if (fabs(init_val - EPS) > EPS)
    {
        for (size_t i = 0; i < dim; i++)
        {
            v.values[i] = init_val;
        }
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

Vector Vector_new_copy(const Vector* v)
{
    Vector res;

    res.values = calloc(v->dim, sizeof(double));
    memcpy(res.values, v->values, v->dim * sizeof(double));
    res.dim = v->dim;

    return res;
}

void Vector_copy(Vector* target, const Vector* v)
{
    if (target->dim != 0)
    {
        Vector_free(target);
    }
    target->values = calloc(v->dim, sizeof(double));
    memcpy(target->values, v->values, v->dim * sizeof(double));
    target->dim = v->dim;
}

double Vector_at(const Vector* v, const size_t index)
{
    return v->values[index];
}

void Vector_set_item(Vector* v, const size_t index, const double val)
{
    v->values[index] = val;
}

size_t Vector_dim(const Vector* v)
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

double Vector_norm(const Vector* v)
{
    double sum = 0.0;
    for (size_t n = 0; n < v->dim; n++)
    {
        sum += v->values[n];
    }

    return sqrt(sum);
}

int Vector_add(Vector* target, const Vector* v)
{
    if (Vector_dim(target) != Vector_dim(v))
    {
        return VECTOR_DIMENSION_ERROR;
    }

#pragma omp simd
    for (size_t n = 0; n < v->dim; n++)
    {
        target->values[n] += v->values[n];
    }

    return VECTOR_MATH_SUCCESS;
}

int Vector_sub(Vector* target, const Vector* v)
{
    if (Vector_dim(target) != Vector_dim(v))
    {
        return VECTOR_DIMENSION_ERROR;
    }

#pragma omp simd
    for (size_t n = 0; n < v->dim; n++)
    {
        target->values[n] -= v->values[n];
    }

    return VECTOR_MATH_SUCCESS;
}

int Vector_scale(Vector* target, const double lambda)
{
    for (size_t n = 0; n < target->dim; n++)
    {
        target->values[n] *= lambda;
    }
    return VECTOR_MATH_SUCCESS;
}

int Vector_broadcast_add(Vector* target, const double a)
{
    for (size_t n = 0; n < target->dim; n++)
    {
        target->values[n] += a;
    }
    return VECTOR_MATH_SUCCESS;
}

int Vector_gradient(Vector* grad, Vector* x, double (*f)(const Vector*))
{
    if (Vector_dim(grad) != 0)
    {
        Vector_free(grad);
    }

    const size_t N   = Vector_dim(x);
    grad->values     = calloc(N, sizeof(double));
    grad->dim        = N;
    const double f_x = f(x);
    for (size_t n = 0; n < N; n++)
    {
        x->values[n] += EPS;
        const double f_x_h = f(x);
        x->values[n] -= EPS;
        grad->values[n] = (f_x_h - f_x) / EPS;
    }

    return VECTOR_MATH_SUCCESS;
}

int Vector_gradient_maximize(Vector* target,
                             Vector* x,
                             double (*f)(const Vector*),
                             double stepsize)
{
    // printf("\tEnter maximize...\n");

    Vector grad = Vector_new(0, 0.0);

    // printf("\tInit vars...\n");
    Vector_gradient(&grad, x, f);
    double norm_grad = Vector_norm(&grad);
    Vector grad_tmp  = Vector_new_copy(&grad);
    Vector_scale(&grad, stepsize);

    double last_f_of_x = f(x);
    Vector test_step   = Vector_new_copy(x);
    int add_status     = Vector_add(&test_step, &grad);
    bool once_error    = false;
    once_error |= add_status;
    double current_f_of_x = f(&test_step);
    // printf("\tInit complete.\n");

    // printf("\t grad init: ");
    // Vector_print(&grad);
    // printf(", test_step: ");
    // Vector_print(&test_step);
    // printf("\n");

    int count = 0;
    while (count < MAX_STEP && norm_grad >= GRAD_EPS)
    {
        // printf("\t\tIteration %d\n", count);
        // printf("\t\tCurrent x: ");
        // Vector_print(x);
        // printf("current grad: ");
        // Vector_print(&grad);
        // printf("\n");
        int a = 0;
        while (current_f_of_x < last_f_of_x)
        {
            // printf(
            //   "\t\t\tcurrent_f_of_x = %f, last_f_of_x = %f, stepsize = %f\n",
            //   current_f_of_x,
            //   last_f_of_x,
            //   stepsize);
            // printf("\t\t\ttest_step: ");
            // Vector_print(&test_step);
            stepsize *= 0.5;
            Vector_copy(&grad, &grad_tmp);
            Vector_scale(&grad, stepsize);
            // printf(", grad: ");
            // Vector_print(&grad);
            // printf(", stepsize: %f, cur_f: %f, last_f: %f\n",
            //        stepsize,
            //        current_f_of_x,
            //        last_f_of_x);
            Vector_copy(&test_step, x);

            once_error |= Vector_add(&test_step, &grad);
            current_f_of_x = f(&test_step);
            if (a >= 20)
            {
                break;
            }
            a++;
        }

        if (current_f_of_x >= last_f_of_x)
        {
            // printf("\t\tcase bigger \n");
            once_error |= Vector_add(&test_step, &grad);
            double longer_f_of_x = f(&test_step);
            if (longer_f_of_x > current_f_of_x)
            {
                stepsize *= 2;
                current_f_of_x = longer_f_of_x;
            }
            else
            {
                once_error |= Vector_sub(&test_step, &grad);
                Vector_scale(&grad, 0.5);
            }
        }

        Vector_copy(x, &test_step);
        // printf("\n");
        last_f_of_x = current_f_of_x;
        Vector_gradient(&grad, x, f);
        Vector_norm(&grad);
        Vector_copy(&grad_tmp, &grad);
        Vector_scale(&grad, stepsize);
        once_error |= Vector_add(&test_step, &grad);
        current_f_of_x = f(&test_step);
        // printf("\t\tIter end: -----------\n");
        // printf("\t\tx: ");
        // Vector_print(x);
        // printf(", test_step: ");
        // Vector_print(&test_step);
        // printf(", grad: ");
        // Vector_print(&grad);
        // printf("\n");

        if (once_error)
        {
            perror("A dimension error happened in Vector_gradient_maximize\n");
            return VECTOR_DIMENSION_ERROR;
        }
        count++;
    }
    if (target->dim != 0)
    {
        Vector_free(target);
    }

    target->values = calloc(x->dim, sizeof(double));
    memcpy(target->values, x->values, x->dim * sizeof(double));
    target->dim = x->dim;

    Vector_free(&grad);
    Vector_free(&grad_tmp);
    Vector_free(&test_step);

    return VECTOR_MATH_SUCCESS;
}

static double (*minimize_f)(const Vector*);

static double neg_wrapper(const Vector* x)
{
    return -minimize_f(x);
}

int Vector_gradient_minimize(Vector* target,
                             Vector* x,
                             double (*f)(const Vector*),
                             double stepsize)
{
    minimize_f = f;
    return Vector_gradient_maximize(target, x, neg_wrapper, stepsize);
}
