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

// void eigen_callback(void* ctx)
// {
//     eigen_args* a = ctx;
//     Matrix_eigvals(a->eigen, a->M, a->sort);
// }

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

void m_hadamard_m_callback(void* ctx)
{
    m_dot_m_args* a = ctx;
    Matrix_Hadamard_dot(a->target, a->M1, a->M2);
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
    double* target;
    const Matrix *A, *B;
} m_inner_args;

void m_inner_callback(void* ctx)
{
    m_inner_args* a = ctx;
    Matrix_inner_dot(a->target, a->A, a->B);
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

    // Inverse

    const size_t runs   = 15;
    const size_t ignore = 4;

    const size_t m     = 100;
    const size_t n     = m;
    const size_t FLOPS = (size_t)(8.0 / 3.0 * n * n * n);
    Matrix M           = Matrix_new_random_symmetric(m, Real);
    Matrix M_inv       = Matrix_zeros_like(&M, Real);
    inverse_args ia    = { &M_inv, &M };
    callable_t c       = { inverse_callback, &ia };

    track_performance(runs, ignore, FLOPS, m, "Matrix_inverse()", c);

    // Eigvals

    Matrix_free(&M_inv);
    // Vector eigenvalues       = Vector_new(m, 0, Real);
    // eigen_args ea            = { &eigenvalues, &M, false };
    // callable_t call          = { eigen_callback, &ea };
    // const size_t eigen_flops = 300 * m * m * m;

    // track_performance(runs, ignore, eigen_flops, m, "Matrix_eigvals()",
    // call); Vector_free(&eigenvalues);

    // M_dot_V

    Matrix_free(&M);
    size_t new_m        = m * 3;
    M                   = Matrix_new_random_symmetric(new_m, Real);
    Vector v            = Vector_new_random_normal(new_m, 0, 1, Real);
    Vector target       = Vector_zeros_like(&v);
    m_dot_v_args mva    = { &target, &M, &v };
    callable_t call_mva = { m_dot_v_callback, &mva };
    track_performance(
      runs, ignore, 2 * new_m * new_m, new_m, "Matrix_Vector_dot()", call_mva);

    Vector_free(&v);
    Vector_free(&target);

    // M_DOT_M

    Matrix M2      = Matrix_new_random_symmetric(new_m, Real);
    Matrix M2M_res = Matrix_zeros_like(&M2, Real);

    m_dot_m_args mma    = { &M2M_res, &M, &M2 };
    callable_t call_mma = { m_dot_m_callback, &mma };

    track_performance(runs,
                      ignore,
                      2 * new_m * new_m * new_m,
                      new_m,
                      "Matrix_Matrix_dot() ",
                      call_mma);

    callable_t call_had = { m_hadamard_m_callback, &mma };
    track_performance(
      runs, ignore, new_m * new_m, new_m * new_m, "Hadamard product", call_had);

    double inner_target = 0.0;
    m_inner_args mia    = { &inner_target, &M, &M2 };
    callable_t call_mia = { m_inner_callback, &mia };
    track_performance(runs,
                      ignore,
                      2 * new_m * new_m,
                      new_m * new_m,
                      "Matrix_inner_dot()",
                      call_mia);

    Matrix_free(&M2);
    Matrix_free(&M2M_res);

    new_m = 250;

    // Outer

    Vector outer_left  = Vector_new_random_normal(new_m, 0, 1, Real);
    Vector outer_right = Vector_new_random_normal(new_m, 0, 1, Real);
    Matrix Outer_res   = Matrix_new(new_m, new_m, Real);

    outer_args oa      = { &Outer_res, &outer_left, &outer_right };
    callable_t call_oa = { outer_callback, &oa };
    track_performance(
      runs, ignore, new_m * new_m, new_m, "Matrix_Vector_outer()", call_oa);

    size_t flops_outer = new_m * (new_m + 1) / 2;
    if (new_m > 110)
    {
        flops_outer = new_m * new_m;
    }
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

    return EXIT_SUCCESS;
}
