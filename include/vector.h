#ifndef VECTOR_H
#define VECTOR_H

#define ZERO_INIT 0.0
#define GRAD_EPS 10e-5
#define MAX_STEP 25

//~- ------------ -~//
//~- STATUS_CODES -~//
//~- ------------ -~//

typedef enum
{
    VECTOR_SUCCESS,
    VECTOR_DIMENSION_ERROR,
} VectorStatus;

#include "config.h"
#include <stddef.h>

#define EPS (sizeof(real_t) < 8 ? 1e-4 : 1e-8)
/**
 * Type for mathematical vector operations.
 *
 * - `values` array of double precision floats.
 *
 * - `dim` dimension of the vector.
 */
typedef struct
{
    void* values;
    size_t dim;
    DataType dt;
} Vector;

typedef struct
{
    double (*f)(const Vector*);
} NegContext;

///////////////////////////////////
//~- ------------------------- -~//
//~-      Basic functions      -~//
//~- ------------------------- -~//
///////////////////////////////////

void Vector_prepare_target(Vector* target, const size_t dim, const DataType dt);

Vector Vector_new(const size_t dim, const void* init_val, const DataType dt);
/**
 * @brief Constructs a new vector with a specified dimension and values.
 *
 * @param `dim` Dimension of the new vector.
 * @param `init_vals` Array of initial values. Assumed to be on the stack.
 * @return A new constructed vector with shape { `init_vals[0]` ...
 * `init_vals[dim-1]` }.
 */
Vector Vector_new_vals(const size_t dim, const void* init_vals, DataType dt);

Vector Vector_new_random_normal(const size_t dim,
                                const double mean,
                                const double variance,
                                const DataType dt);

Vector Vector_new_random_uniform(const size_t dim,
                                 const double min,
                                 const double max,
                                 const DataType dt);
/**
 * @brief Copies the vector `v` and returns a new vector.
 *
 * @param `v` Vector to be copied.
 * @return A new vector with all entries in v.
 */
Vector Vector_new_copy(const Vector* v);
/**
 * @brief Constructs a new vector containing `v->dim` zeros.
 *
 * @param `v` Vector whos shape the new vector must have.
 * @return A new vector with the shape { 0.0, (`v->dim` times) }
 */
Vector Vector_zeros_like(const Vector* v);

Vector Vector_zeros(const size_t dim, const DataType dt);

Vector Vector_ones(const size_t dim, const DataType dt);
/**
 * @brief Copy `v` into `target`.
 *
 * @param `target` Will inherit `v->dim` and all values.
 * @param `v` Source vector to be copied.
 */
void Vector_copy(Vector* target, const Vector* v);

int Vector_all_close(const Vector* u, const Vector* v);

int Vector_at(void* target, const Vector* v, const size_t index);
/**
 * @brief Set the value of `v` of a specified `index` to `target`.
 *
 * @param target Value is stored here.
 * @param v Vector to be accessed.
 * @param index Index of the data
 * @return `VECTOR_DIMENSION_ERROR` or `VECTOR_SUCCESS`
 */
int Vector_get_at(void* target, const Vector* v, const size_t index);

int Vector_set_item(Vector* v, const size_t index, const void* val);
/**
 * @brief Get the dimension of a vector.
 *
 * @param v Vector to get the dimension from.
 * @return Dimension of `v`.
 */
size_t Vector_get_dim(const Vector* v);
/**
 * @brief Deallocate data in `v`
 *
 * @param v Vector to be deallocated and reset do 0.
 */
void Vector_free(Vector* v);
/**
 * @brief Print `v` to the console.
 *
 * @param `v` Vector to be displayed.
 */
void Vector_print(Vector* v);

void Vector_max(void* target, const Vector* v);

void Vector_min(void* target, const Vector* v);

int Vector_sort_inplace(Vector* v);

int Vector_sort(Vector* target, const Vector* v);

void Vector_init_prng(const int seed);

/////////////////////////////////////////
//~- ------------------------------ -~//
//~-      Mathematic Functions      -~//
//~- ------------------------------ -~//
/////////////////////////////////////////

real_t Vector_norm(const Vector* v);
/**
 * @brief Add `v` to `target`. Equivalent to `target += v`, if they were atomic.
 *
 * @param `target` Target vector
 * @param `v` summand
 * @return `VECTOR_DIMENSION_ERROR` if the dimensions of the vectors don't
 * match, otherwise `VECTOR_SUCCESS`.
 */
int Vector_add(Vector* target, const Vector* v);
/**
 * @brief Subtract `v` from `target`. Equivalent to `target -= v`, if they were
 * atomic.
 *
 * @param `target` Target vector
 * @param `v` subtrahend
 * @return `VECTOR_DIMENSION_ERROR` if the dimensions of the vectors don't
 * match, otherwise `VECTOR_SUCCESS`.
 */
int Vector_sub(Vector* target, const Vector* v);
/**
 * @brief Scale `v` by `lambda`.
 *
 * @param `target` Vector to be scaled.
 * @param `lambda` Scalar.
 * @return `VECTOR_SUCCESS`
 */
int Vector_scale(Vector* target, const void* lambda);
/**
 * @brief Add `a` to every item in `target`.
 *
 * @param `target` Target vector.
 * @param `a` addend.
 * @return `VECTOR_SUCCESS`
 */
int Vector_broadcast_add(Vector* target, const void* a);

/**
 * @brief Subtract `a` from every item in `target`.
 *
 * @param `target` Target vector.
 * @param `a` subtrahend.
 * @return `VECTOR_SUCCESS`
 */
int Vector_broadcast_sub(Vector* target, const void* a);

/**
 * @brief Store the dot product `<u,v>` to `target`.
 *
 * @param target Target of dot product.
 * @param u First vector.
 * @param v Second vector.
 * @return `VECTOR_SUCCESS` or `VECTOR_DIMENSION_ERROR`
 */
int Vector_dot(void* target, const Vector* u, const Vector* v);

// /**
//  * @brief Numerical approximation of \[\nabla(f(x))\].
//  *
//  * @param `grad` Target vector. The gradient is stored here.
//  * @param `x` Point at which the gradient is evaluated.
//  * @param `f(x)` Function that accepts a vector and returns a real number.
//  * @return `VECTOR_SUCCESS`
//  */
// int Vector_gradient(Vector* grad,
//                     const Vector* x,
//                     double (*f)(const Vector* x));
// /**
//  * @brief Gradient ascent method to maximize \[f(x)\].
//  *
//  * @param `target` X value of the local maximum is stored here.
//  * @param `x` Starting point of the maximizer.
//  * @param `f(x)` Function to be maximized.
//  * @param stepsize Initial stepsize of the gradient ascent.
//  * @return `VECTOR_DIMENSION_ERROR` or `VECTOR_SUCCESS`
//  */
// int Vector_gradient_maximize(Vector* target,
//                              const Vector* x,
//                              double (*f)(const Vector*),
//                              double stepsize);
// /**
//  * @brief Gradient descent method to maximize \[f(x)\].
//  *
//  * @param `target` X value of the local maximum is stored here.
//  * @param `x` Starting point of the maximizer.
//  * @param `f(x)` Function to be maximized.
//  * @param stepsize Initial stepsize of the gradient descent.
//  * @return `VECTOR_DIMENSION_ERROR` or `VECTOR_SUCCESS`
//  */
// int Vector_gradient_minimize(Vector* target,
//                              const Vector* x,
//                              double (*f)(const Vector*),
//                              double stepsize);

#endif // VECTOR_H
