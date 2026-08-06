#ifndef MATH_LOG_H
#define MATH_LOG_H

#include "config.h"
#include <stdbool.h>
#include <stddef.h>

typedef enum
{
    LOG_MODE_NONE,
    LOG_MODE_STDERR,
    LOG_MODE_FILE
} LogMode;

typedef enum
{
    LOG_VERB_ERROR_ONLY,
    LOG_VERB_ERROR_WARNING,
    LOG_VERB_ALL,
} LogVerbosity;

typedef enum
{
    LOG_RT_ERROR,
    LOG_RT_WARNING,
    LOG_RT_INFO
} RecordType;

void Log_set_log_mode(LogMode mode, LogVerbosity lv);
void Log_set_log_file(const char* path);
void Log_log(const char* msg, RecordType rt);
LogVerbosity Log_get_verbosity(void);
bool Log_info_threshold(const size_t count, const size_t object_size);

#endif // MATH_LOG_H
