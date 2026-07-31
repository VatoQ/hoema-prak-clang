#ifndef MATRIX_H
#define MATRIX_H

#define MATRIX_MATH_SUCCESS 10
#define MATRIX_BASIC_SUCCESS 20
#define MATRIX_DIMENSION_ERROR -10
#define MATRIX_BASIC_ERROR -20
#define MATRIX_MATH_ERROR -30
#define MATRIX_EPS 10e-4

#include "vector.h"
#include <stddef.h>

/**
 * @brief Type for mathematical matrix operations.
 *
 * - `values` A contiguous array of matrix values.
 *
 * - `m` Rows of the matrix.
 *
 * - `n` Columns of the matrix.
 */
typedef struct
{
    double* values;
    size_t m, n;
} Matrix;

///////////////////////////////////
//~- ------------------------- -~//
//~-      Basic Functions      -~//
//~- ------------------------- -~//
///////////////////////////////////

/**
 * @brief Construct a new matrix in \[ℝ^{m \times n}\]
 *
 * @param `m` number of rows.
 * @param `n` Number of columns.
 * @param `init_val` Initial value.
 * @return Matrix with the shape { `init_val` (`m` * `n` times) }
 */
Matrix Matrix_new(const size_t m, const size_t n, const double init_val);
/**
 * @brief Construct a new matrix in \[ℝ^{m \times n}\]
 *
 * @param `m` number of rows.
 * @param `n` Number of columns.
 * @param `init_vals` Array of initial values.
 * @return Matrix with the shape { `init_vals[0]` ... `init_vals[m * n - 1]`  }
 */
Matrix Matrix_new_vals(const size_t m, const size_t n, const double* init_vals);
/**
 * @brief Construct a Matrix with the shape of `M` filled with zeros.
 *
 * @param `M` Matrix to imitate.
 * @return New Matrix with the shape { 0.0, (`M->m` * `M->n` times) }
 */
Matrix Matrix_zeros_like(const Matrix* M);
/**
 * @brief Construct a diagonal matrix with the values of `v`.
 *
 * @param `v` Vector after which to model the diagonal matrix.
 * @return New Matrix with the values of `v` along the diagonal.
 */
Matrix Matrix_diag(const Vector* v);
/**
 * @brief Construct a diagonal matrix.
 *
 * @param `n` Number of columns and rows: A diagonal matrix is symmetrical.
 * @return New Matrix with the shape nxn with `val` along the diagonal.
 */
Matrix Matrix_diag_val(const size_t n, const double val);
/**
 * @brief Copy `M` into `target`.
 *
 * @param `target` Target container. Will be overwritten and all data in it will
 * be lost.
 * @param `M` Matrix to be copied.
 */
void Matrix_copy(Matrix* target, const Matrix* M);
/**
 * @brief Set a `value` to a given position in the matrix `M`.
 *
 * @param `M` Matrix to be accessed.
 * @param `m` Row at which to set `value`.
 * @param `n` Column at which to set `value`.
 * @param `value` Value to be set to `M`.
 * @return `MATRIX_DIMENSION_ERROR` or `MATRIX_BASIC_SUCCESS`
 */
int Matrix_set_at(Matrix* M,
                  const size_t m,
                  const size_t n,
                  const double value);
/**
 * @brief Get a value of `M` at a given position.
 *
 * @param `target` Value at `m`x`n` is stored here.
 * @param `M` Matrix at which to access the value.
 * @param `m` Row at which to get `value`.
 * @param `n` Column at which to get `value`
 * @return `MATRIX_DIMENSION_ERROR` or `MATRIX_BASIC_SUCCESS`
 */
int Matrix_get_at(double* target,
                  const Matrix* M,
                  const size_t m,
                  const size_t n);
/**
 * @brief Deallocate data within `M`. Sets `m` and `n` to `0`.
 *
 * @param `M` Matrix to be deallocated.
 */
void Matrix_free(Matrix* M);
/**
 * @brief Print `M` to the console.
 *
 * @param `M` Matrix to be printed.
 */
void Matrix_print(const Matrix* M);

///////////////////////////////////
//~- ------------------------- -~//
//~-      Math  Functions      -~//
//~- ------------------------- -~//
///////////////////////////////////

/**
 * @brief Scale `target` by `lambda`.
 *
 * @param `target` Matrix to be scaled.
 * @param `lambda` Real scalar.
 */
void Matrix_scale(Matrix* target, const double lambda);
/**
 * @brief Invert a matrix if it is invertable.
 *
 * @param target `M^-1` is stored here.
 * @param M Matrix to be inverted.
 * @return `MATRIX_DIMENSION_ERROR` or `MATRIX_MATH_ERROR` or
 * `MATRIX_MATH_SUCCESS`
 */
int Matrix_inverse(Matrix* target, const Matrix* M);
/**
 * @brief Matrix vector multiplication into `target`.
 *
 * @param `target` Result of `M @ v` is stored here.
 * @param `M` Matrix of the operation.
 * @param `v` Vector of the operation.
 * @return `MATRIX_DIMENSION_ERROR` or `MATRIX_BASIC_ERROR` or
 * `MATRIX_MATH_SUCCESS`
 */
int Matrix_Vector_dot(Vector* target, const Matrix* M, const Vector* v);
/**
 * @brief Matrix matrix multiplication into `target`.
 *
 * @param `target` Result of `M @ v` is stored here.
 * @param `M` Matrix of the operation.
 * @param `v` Vector of the operation.
 * @return `MATRIX_DIMENSION_ERROR` or `MATRIX_BASIC_ERROR` or
 * `MATRIX_MATH_SUCCESS`
 */
int Matrix_Matrix_dot(Matrix* target, const Matrix* M1, const Matrix* M2);
/**
 * @brief Numerical approximation of the jacobi matrix of ` f(x)`.
 *
 * @param `target` Container where to store the jacobi matrix.
 * @param `x` Point at which to evaluate the jacobi matrix.
 * @param `f(x)` f:R^n->R^m
 * @return `MATRIX_BASIC_ERROR` or `MATRIX_MATH_SUCCESS`
 */
int Matrix_jacobi(Matrix* target,
                  const Vector* x,
                  Vector (*f)(const Vector* x));

#endif // MATRIX_H
