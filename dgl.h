#ifndef DGL_H
#define DGL_H

#define DGL_SYSTEM 1
#define DGL_NTH_ORDER 2

#define DGL_BASIC_ERROR -10
#define DGL_MATH_ERROR -20
#define DGL_BASIC_SUCCESS 10
#define DGL_MATH_SUCCESS 20
#include "vector.h"

typedef struct
{
    double x_order;
    Vector x_system;
    int dgl_type;
} DGL_Result;

typedef struct
{
    int dgl_type;

    Vector (*f_dgl_system)(const Vector* y, double x);

    double (*f_dgl_nth_order)(const Vector* y, double x);

} DGL;

DGL DGL_new(void* function, int type);

void DGL_Result_free(DGL_Result* dgl_result);
#endif // DGL_H
