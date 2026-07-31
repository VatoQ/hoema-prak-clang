#ifndef MATH_LOG_H
#define MATH_LOG_H

typedef enum
{
    MATH_LOG_NONE,
    MATH_LOG_STDERR,
    MATH_LOG_FILE
} LogMode;

typedef enum
{
    LOG_ERROR,
    LOG_WARNING,
    LOG_INFO
} RecordType;

void Log_set_log_mode(LogMode mode);
void Log_set_log_file(const char* path);
void Log_log(const char* msg, RecordType rt);

#endif // MATH_LOG_H
