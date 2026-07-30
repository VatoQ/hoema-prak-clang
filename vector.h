#ifndef VECTOR_H
#define VECTOR_H

#define ZERO_INIT 0.0
#define EPS 10e-8
#define GRAD_EPS 10e-5
#define MAX_STEP 25

//~- ------------ -~//
//~- STATUS_CODES -~//
//~- ------------ -~//
#define VECTOR_MATH_SUCCESS 0
#define VECTOR_DIMENSION_ERROR 1

#include <stddef.h>
typedef struct
{
    double* values;
    size_t dim;
} Vector;

typedef struct
{
    double (*f)(const Vector*);
} NegContext;

//~- --------------- -~//
//~- Basic functions -~//
//~- --------------- -~//
Vector Vector_new(const size_t dim, const double init_val);
Vector Vector_new_vals(const size_t dim, const double* init_vals);
Vector Vector_new_copy(const Vector* v);
void Vector_copy(Vector* target, const Vector* v);
double Vector_at(const Vector* v, const size_t index);
void Vector_set_item(Vector* v, const size_t index, const double val);
size_t Vector_dim(const Vector* v);
void Vector_free(Vector* v);
void Vector_print(Vector* v);

//~- -------------------- -~//
//~- Mathematic Functions -~//
//~- -------------------- -~//
double Vector_norm(const Vector* v);
int Vector_add(Vector* target, const Vector* v);
int Vector_sub(Vector* target, const Vector* v);
int Vector_scale(Vector* target, const double lambda);
int Vector_broadcast_add(Vector* target, const double a);
int Vector_gradient(Vector* grad, Vector* x, double (*f)(const Vector* x));
int Vector_gradient_maximize(Vector* target,
                             Vector* x,
                             double (*f)(const Vector*),
                             double stepsize);
int Vector_gradient_minimize(Vector* target,
                             Vector* x,
                             double (*f)(const Vector*),
                             double stepsize);

#endif // VECTOR_H
