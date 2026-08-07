#ifndef CONFIG_H
#define CONFIG_H

// Macros that may be overwritten in the project can be defined here.

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define LOG_INFO_THRESHOLD 100 * 1024 * 1024

#define PARALLEL_THRESHOLD 100000000
// #define VECTOR_PARALLEL_LOWER_THRESHOLD (15500 * sizeof(double))
// #define VECTOR_PARALLEL_UPPER_THRESHOLD (1000000 * sizeof(double))

typedef struct
{
    size_t lower, upper;
} parallel_thresholds;

size_t detect_L3_cache_size(void);

extern parallel_thresholds PARALLEL_THRESHOLDS;

void config_init(void);

#endif // CONFIG_H
