#ifndef MATRIX_H
#define MATRIX_H

#include "vector.h"
#define MATRIX_MATH_SUCCESS 10
#define MATRIX_BASIC_SUCCESS 20
#define MATRIX_DIMENSION_ERROR -10
#define MATRIX_BASIC_ERROR -20

#include <stddef.h>
typedef struct
{
    double* values;
    size_t m, n;
} Matrix;

//~- --------------- -~//
//~- Basic Functions -~//
//~- --------------- -~//

Matrix Matrix_new(const size_t m, const size_t n, const double init_val);
Matrix Matrix_new_vals(const size_t m, const size_t n, const double* init_vals);
Matrix Matrix_zeros_like(const Matrix* M);
Matrix Matrix_diag(const Vector* v);
Matrix Matrix_diag_val(const size_t n, const double val);
void Matrix_copy(Matrix* target, const Matrix* M);
int Matrix_set_at(Matrix* M,
                  const size_t m,
                  const size_t n,
                  const double value);
int Matrix_get_at(double* target,
                  const Matrix* M,
                  const size_t m,
                  const size_t n);
void Matrix_free(Matrix* M);
void Matrix_print(const Matrix* M);

//~- --------------- -~//
//~- Math  Functions -~//
//~- --------------- -~//

void Matrix_scale(Matrix* target, const double lambda);
int Matrix_inverse(Matrix* target, const Matrix* M);
int Matrix_Vector_dot(Vector* target, const Matrix* M, const Vector* v);

#endif // MATRIX_H
