#include "../include/config.h"

size_t detect_L3_cache_size(void)
{
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");

    if (!f)
    {
        return 0;
    }
    char buf[64];
    if (!fgets(buf, sizeof(buf), f))
    {
        fclose(f);
        return 0;
    }

    fclose(f);

    size_t size = strtoull(buf, NULL, 10);

    if (strstr(buf, "K"))
    {
        size *= 1024;
    }

    if (strstr(buf, "M"))
    {
        size *= 1024 * 1024;
    }

    return size;
}

parallel_thresholds PARALLEL_THRESHOLDS = { 15000, 1000000 };

void config_init(void)
{
    const size_t L3 = detect_L3_cache_size();
    if (L3 != 0)
    {
        PARALLEL_THRESHOLDS.upper = L3 * 2;
    }
}
