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
    Log_set_log_mode(MATH_LOG_STDERR);
    double vals[]        = { -6, 0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3 };
    Matrix M             = Matrix_new_vals(4, 4, vals);
    double spectral_norm = Matrix_norm(&M, SPECTRAL);

    printf("||M||_2 = %f\n", spectral_norm);
    Matrix_free(&M);
    return 0;
}
