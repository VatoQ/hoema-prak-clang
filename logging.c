#include "logging.h"
#include <stdio.h>
#include <time.h>

#define RED "\x1b[31m"
#define YELLOW "\x1b[33m"
#define RESET "\x1b[0m"

static LogMode LOG_MODE = MATH_LOG_NONE;
static FILE* LOG_FILE   = NULL;

void Log_set_log_mode(LogMode mode)
{
    LOG_MODE = mode;
}

void Log_set_log_file(const char* path)
{
    LOG_FILE = fopen(path, "a");
    LOG_MODE = MATH_LOG_FILE;
}

static const char* Log_prefix(RecordType rt)
{
    switch (rt)
    {
        case LOG_ERROR:
            return "ERROR";
        case LOG_WARNING:
            return "WARNING";
        case LOG_INFO:
            return "INFO";
        default:
            return "UNKNOWN";
    }
}

const char* Log_color(RecordType rt)
{
    switch (rt)
    {
        case LOG_ERROR:
            return RED;
        case LOG_WARNING:
            return YELLOW;
        default:
            return RESET;
    }
}

void Log_log(const char* msg, RecordType rt)
{
    time_t now   = time(NULL);
    struct tm* t = localtime(&now);

    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", t);
    const char* prefix = Log_prefix(rt);
    const char* format = "%s[%s] %s: %s%s\n";
    // color [timestamp] PREFIX: message RESET

    if (LOG_MODE == MATH_LOG_STDERR)
    {
        fprintf(stderr, format, Log_color(rt), timestamp, prefix, msg, RESET);
    }
    else if (LOG_MODE == MATH_LOG_FILE && LOG_FILE)
    {
        fprintf(LOG_FILE, "[%s] %s: %s\n", timestamp, prefix, msg);
        fflush(LOG_FILE);
    }
}
