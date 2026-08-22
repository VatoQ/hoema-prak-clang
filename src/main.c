
#include "../include/logging.h"
#include "../include/matrix.h"
#include "../include/prng.h"
#include "../include/vector.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
    config_init();
    Log_set_log_mode(LOG_MODE_STDERR, LOG_VERB_ERROR_WARNING);
    const size_t N = 1000;
    Matrix M       = Matrix_new_random_symmetric(N, Real);

    return EXIT_SUCCESS;
}
