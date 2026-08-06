#include "../include/matrix.h"
#include "performance_tracker.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct
{
    Matrix* out;
    const Matrix* in;
} inverse_args;

void inverse_callback(void* ctx)
{
    inverse_args* a = ctx;
    Matrix_inverse(a->out, a->in);
}

typedef struct
{
    Vector* eigen;
    const Matrix* M;
    bool sort;

} eigen_args;

void eigen_callback(void* ctx)
{
    eigen_args* a = ctx;
    Matrix_eigvals(a->eigen, a->M, a->sort);
}

typedef struct
{
    Matrix* target;
    const Matrix *M1, *M2;
} m_dot_m_args;

void m_dot_m_callback(void* ctx)
{
    m_dot_m_args* a = ctx;
    Matrix_Matrix_dot(a->target, a->M1, a->M2);
}

typedef struct
{
    Vector* target;
    const Matrix* M;
    const Vector* v;
} m_dot_v_args;

void m_dot_v_callback(void* ctx)
{
    m_dot_v_args* a = ctx;
    Matrix_Vector_dot(a->target, a->M, a->v);
}

typedef struct
{
    Matrix* target;
    const Vector *u, *v;

} outer_args;

void outer_callback(void* ctx)
{
    outer_args* a = ctx;
    Matrix_Vector_outer(a->target, a->u, a->v);
}

void outer_square_callback(void* ctx)
{

    outer_args* a = ctx;
    Matrix_Vector_outer_square(a->target, a->u);
}

int main(int argc, char** argv)
{
    const size_t runs   = 6;
    const size_t ignore = 2;

    const size_t m     = 150;
    const size_t n     = m;
    const size_t FLOPS = (size_t)(8.0 / 3.0 * n * n * n);
    Matrix M           = Matrix_new_random_symmetric(m);
    Matrix M_inv       = Matrix_zeros_like(&M);
    inverse_args ia    = { &M_inv, &M };
    callable_t c       = { inverse_callback, &ia };

    track_performance(runs, ignore, FLOPS, m, "Matrix_inverse()", c);

    Vector eigenvalues       = Vector_new(m, 0);
    eigen_args ea            = { &eigenvalues, &M, false };
    callable_t call          = { eigen_callback, &ea };
    const size_t eigen_flops = 300 * m * m * m;

    track_performance(runs, ignore, eigen_flops, m, "Matrix_eigvals()", call);
    Vector_free(&eigenvalues);

    Matrix_free(&M);
    size_t new_m        = 3 * m;
    M                   = Matrix_new_random_symmetric(new_m);
    Vector v            = Vector_new_random_normal(new_m, 0, 1);
    Vector target       = Vector_zeros_like(&v);
    m_dot_v_args mva    = { &target, &M, &v };
    callable_t call_mva = { m_dot_v_callback, &mva };
    track_performance(
      runs, ignore, 2 * new_m * new_m, new_m, "Matrix_Vector_dot()", call_mva);

    Vector_free(&v);
    Vector_free(&target);

    Matrix M2      = Matrix_new_random_symmetric(new_m);
    Matrix M2M_res = Matrix_zeros_like(&M2);

    m_dot_m_args mma    = { &M2M_res, &M, &M2 };
    callable_t call_mma = { m_dot_m_callback, &mma };

    track_performance(runs,
                      ignore,
                      2 * new_m * new_m * new_m,
                      new_m,
                      "Matrix_Matrix_dot() ",
                      call_mma);

    new_m = 900;

    Vector outer_left  = Vector_new_random_normal(new_m, 0, 1);
    Vector outer_right = Vector_new_random_normal(new_m, 0, 1);
    Matrix Outer_res   = Matrix_new(new_m, new_m, 0);

    outer_args oa      = { &Outer_res, &outer_left, &outer_right };
    callable_t call_oa = { outer_callback, &oa };
    track_performance(
      runs, ignore, new_m * new_m, new_m, "Matrix_Vector_outer()", call_oa);

    size_t flops_outer  = n * (n + 1) / 2;
    outer_args osa      = { &Outer_res, &outer_left, &outer_left };
    callable_t call_osa = { outer_square_callback, &osa };

    track_performance(runs,
                      ignore,
                      flops_outer,
                      new_m,
                      "Matrix_Vector_outer_square()",
                      call_osa);

    Matrix_free(&Outer_res);
    Vector_free(&outer_left);
    Vector_free(&outer_right);

    Matrix_free(&M2);
    Matrix_free(&M2M_res);

    Matrix_free(&M_inv);
    Matrix_free(&M);
    Vector_free(&v);
    return EXIT_SUCCESS;
}
