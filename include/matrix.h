#ifndef MATRIX_H
#define MATRIX_H

///////////////////////////////////////////////////////////
//~- ------------------------------------------------- -~//
//~-      Matrix Module - Design and Memory Model      -~//
//~- ------------------------------------------------- -~//
///////////////////////////////////////////////////////////
/*
 * Memory Model:
 * -------------
 * - All constructors (Matrix_new, Matrix_diag, etc.) allocate new memory.
 * - All math functions treat `target` as an output container. If `target` is
 *   non-empty, it will be freed and reallocated internally.
 * - Users never need to pre-size `target` for math functions.
 *     - Note: It is strongly recommended to always initialize
 *       new matrices using a constructor. Even if they are empty.
 *          - Example: `Matrix new_empty_matrix = Matrix_new(0,0,0.0);`
 *       Non constructed Matrices may lead to undefined behaviour.
 * - Users must call Matrix_free() on any matrix they no longer need.
 *
 * Error Handling:
 * ---------------
 * - All functions return a MatrixStatus code.
 * - MATRIX_SUCCESS (0) indicates success.
 * - Non-zero codes indicate dimension errors or numerical errors.
 *
 * Empty Matrix:
 * -------------
 * - A matrix is considered empty if (values == NULL) or (m == 0) or (n == 0).
 * - Empty matrices are safe to pass as `target`: they will be allocated.
 *
 *
 */

////////////////////////////////
//~- ---------------------- -~//
//~-      STATUS CODES      -~//
//~- ---------------------- -~//
////////////////////////////////

#include <stdbool.h>
#define MATRIX_EPS 1e-4

typedef enum
{
    MATRIX_SUCCESS         = 0,
    MATRIX_DIMENSION_ERROR = 1,
    MATRIX_MATH_ERROR      = 2,
    MATRIX_BASIC_ERROR     = 3,
} MatrixStatus;

typedef enum
{
    FROBENIUS,
    SPECTRAL,
} NormType;

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

void Matrix_init_prng(const int seed);

/**
 * @brief Construct a new Matrix with normally distributed initial values.
 *
 * @param `m` Rows of the new matrix.
 * @param `n` Columns of the new matrix.
 * @param `mean` Mean of the distribution.
 * @param `variance` Variance of the distribution.
 * @return Matrix with the shape { \[N(mean, variance)\] (`m * n` times)}
 */
Matrix Matrix_new_random_normal(const size_t m,
                                const size_t n,
                                const double mean,
                                const double variance);

/**
 * @brief Construct a new matrix with uniformly distributed values.
 *
 * @param `m` Rows of the new matrix.
 * @param `n` Columns of the new matrix.
 * @param `min` Smallest value in the distribution.
 * @param `max` Largest value in the distribution.
 * @return Matrix with uniformly distributed values.
 */
Matrix Matrix_new_random_uniform(const size_t m,
                                 const size_t n,
                                 const double min,
                                 const double max);

Matrix Matrix_new_random_symmetric(const size_t n);

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

int Matrix_all_close(const Matrix* A, const Matrix* B);
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

double Matrix_max(const Matrix* M);

double Matrix_min(const Matrix* M);

///////////////////////////////////
//~- ------------------------- -~//
//~-      Math  Functions      -~//
//~- ------------------------- -~//
///////////////////////////////////

/**
 * @brief Prepares `target` for further processing. A `target` with matching
 * dimension will remain unchanged.
 *
 * @param `target` Target container for matrix operations.
 * @param `m` Rows of target matrix.
 * @param `n` Columns of target matrix.
 */
static void Matrix_prepare_target(Matrix* target,
                                  const size_t m,
                                  const size_t n);

double Matrix_norm(const Matrix* M, NormType nt);

/**
 * @brief Add `M` to `target`. Equivalent to `target += M`.
 *
 * @param `target` Matrix where the sum is stored.
 * @param `M` Addend.
 * @return `MATRIX_DIMENSION_ERROR` or `MATRIX_MATH_SUCCESS`
 */
int Matrix_add(Matrix* target, const Matrix* M);
/**
 * @brief Subtract `M` from `target`. Equivalent to `target -= M`.
 *
 * @param `target` Matrix where the difference is stored.
 * @param `M` Subtrahend.
 * @return `MATRIX_DIMENSION_ERROR` or `MATRIX_MATH_SUCCESS`
 */
int Matrix_sub(Matrix* target, const Matrix* M);
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

/**
 * @brief Solve the quation \[Mx=v\], \[x\] being stored in `target`.
 *
 * @param `target` Store the vector found to satisfy the equation here.
 * @param `M` Matrix of the equation.
 * @param `v` Right hand side vector.
 * @return Status code.
 */
int Matrix_Vector_solve(Vector* target, const Matrix* M, const Vector* v);

/**
 * @brief Solve the quation \[MX=N\], \[X\] being stored in `target`.
 *
 * @param `target` Store the matrix found to satisfy the equation here.
 * @param `M` Matrix of the equation.
 * @param `N` Right hand side vector.
 * @return Status code.
 */
int Matrix_Matrix_solve(Matrix* target, const Matrix* M, const Matrix* N);

int Matrix_column_pivot(Matrix* A, size_t k, size_t* P);

/**
 * @brief Symmetrizes the product of two symmetric matrices. Used to turn
 * symmetric matrices into a ring. Results in \[target=0.5*(M1*M2 + M2*M1)\]
 *
 * @param target Target matrix, jordan product is stored here.
 * @param M1 First matrix.
 * @param M2 Second matrix.
 * @return `MATRIX_DIMENSION_ERROR` or `MATRIX_MATH_SUCCESS`
 */
int Matrix_Matrix_dot_jordan(Matrix* target,
                             const Matrix* M1,
                             const Matrix* M2);

/**
 * @brief QR factorization, where `Q` is a unitary matrix and `R` a upperhand
 * triangle matrix.
 *
 * @param Q Target for Q.
 * @param R Target for R.
 * @param A Matrix to be factorized.
 * @return Status Code.
 */
int Matrix_QR(Matrix* Q, Matrix* R, const Matrix* A);

/**
 * @brief Store the eigenvalues of `M` to `eigenvalues`.
 *
 * @param eigenvalues Target container.
 * @param M Matrix to inspect.
 * @return Status Code.
 */
int Matrix_eigvals(Vector* eigenvalues, const Matrix* M, bool sort);

/**
 * @brief Compute the outer product of two given vectors into the matrix
 * `target`.
 *
 * @param target Target container.
 * @param a Left hand vector.
 * @param b Right hand vector.
 * @return Status code.
 */
int Matrix_Vector_outer(Matrix* target, const Vector* a, const Vector* b);

/**
 * @brief Compute the Hassenberg reduction of the matrix `A` into `H`.
 *
 * @param H Target container.
 * @param A Matrix to be reduced.
 * @return Status code.
 */
int Matrix_Hessenberg(Matrix* H, const Matrix* A);

void Matrix_QR_hessenberg(Matrix* Q, Matrix* R, Matrix* H, size_t active);

void Matrix_Matrix_dot_active(Matrix* H,
                              const Matrix* R,
                              const Matrix* Q,
                              size_t active);

#endif // MATRIX_H
