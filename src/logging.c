#include "../include/logging.h"
#include <stdio.h>
#include <time.h>

#define RED "\x1b[31m"
#define YELLOW "\x1b[33m"
#define RESET "\x1b[0m"

static LogVerbosity LOG_VERBOSITY = LOG_VERB_ERROR_ONLY;
static LogMode LOG_MODE           = LOG_MODE_NONE;
static FILE* LOG_FILE_PTR         = NULL;

void Log_set_log_mode(LogMode mode, LogVerbosity lv)
{
    LOG_MODE      = mode;
    LOG_VERBOSITY = lv;
}

void Log_set_log_file(const char* path)
{
    LOG_FILE_PTR = fopen(path, "a");
    LOG_MODE     = LOG_MODE_FILE;
}

static const char* Log_prefix(RecordType rt)
{
    switch (rt)
    {
        case LOG_RT_ERROR:
            return "ERROR";
        case LOG_RT_WARNING:
            return "WARNING";
        case LOG_RT_INFO:
            return "INFO";
        default:
            return "UNKNOWN";
    }
}

const char* Log_color(RecordType rt)
{
    switch (rt)
    {
        case LOG_RT_ERROR:
            return RED;
        case LOG_RT_WARNING:
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

    if (LOG_MODE == LOG_MODE_STDERR)
    {
        if (rt == LOG_RT_INFO && LOG_VERBOSITY < LOG_VERB_ALL)
        {
            return;
        }
        if (rt == LOG_RT_WARNING && LOG_VERBOSITY < LOG_VERB_ERROR_WARNING)
        {
            return;
        }

        fprintf(stderr, format, Log_color(rt), timestamp, prefix, msg, RESET);
    }
    else if (LOG_MODE == LOG_MODE_FILE && LOG_MODE_FILE)
    {
        if (rt == LOG_RT_INFO && LOG_VERBOSITY < LOG_VERB_ALL)
        {
            return;
        }
        if (rt == LOG_RT_WARNING && LOG_VERBOSITY < LOG_VERB_ERROR_WARNING)
        {
            return;
        }
        fprintf(LOG_FILE_PTR, "[%s] %s: %s\n", timestamp, prefix, msg);
        fflush(LOG_FILE_PTR);
    }
}
