#ifndef CONFIG_H
#define CONFIG_H

// Macros that may be overwritten in the project can be defined here.

#include <complex.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEBUG

#ifdef DEBUG
#define IF_DEBUG(code)                                                         \
    do                                                                         \
    {                                                                          \
        code;                                                                  \
    } while (0)

#define DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__);
#else
#define IF_DEBUIF_DEBUG(code)                                                  \
    do                                                                         \
    {                                                                          \
    } while (0)
#define DEBUG_PRINT(...)
#endif // DEBUG

#define LOG_INFO_THRESHOLD 100 * 1024 * 1024

#define ACCESS_VOID(typename, target, source)                                  \
    typename* target = (typename*)(source)

#define ASSIGN_TYPED(typename, target, source) *target = *(typename*)(source)

#define ASSIGN_UNTYPED(typename, target, source)                               \
    *(typename*)target = *(typename*)(source)

#define FMA(upper, target, a, b)                                               \
    for (size_t n = 0; n < upper; n++)                                         \
    target += (a) * (b)

#define ADD(upper, a, b)                                                       \
    for (size_t n = 0; n < upper; n++)                                         \
    a += b

#define SUB(upper, a, b)                                                       \
    for (size_t n = 0; n < upper; n++)                                         \
    a -= b

#define SCALE(upper, a, b)                                                     \
    for (size_t n = 0; n < upper; n++)                                         \
    a *= b

#define PARALLEL_THRESHOLD 100000000
#define PRINT_COMPLEX(v)                                                       \
    printf("%.2f%c%.2fi", creal(v), cimag(v) < 0 ? '-' : '+', fabs(cimag(v)))

#define MAX(a, b) (a > b) ? a : b

// typedef long int_t;
// typedef double real_t;
// typedef complex double complex_t;

#ifndef int_t
#define int_t long
#endif

#ifndef real_t
#define real_t double
#endif

#ifndef complex_t
#define complex_t complex double
#endif

typedef enum
{
    CONFIG_SUCCESS,
    CONFIG_ERROR,
} ConfigStatus;

typedef struct
{
    size_t lower, upper;
} parallel_thresholds;

typedef enum
{
    Add,
    Sub,
    OPERATOR_COUNT,
} operator_e;

size_t detect_L3_cache_size(void);

extern parallel_thresholds PARALLEL_THRESHOLDS;

void config_init(void);

typedef enum
{
    Int,
    Real,
    Complex,
    TYPE_COUNT,
} DataType;

size_t elem_size(DataType dt);

typedef enum
{
    Min,
    Max,
    Infty,
    LIMIT_COUNT,
} LimitType;

int get_limit(void* target, DataType dt, LimitType lt);

#endif // CONFIG_H
