#include "../include/complex_functions.h"
#include "../include/real_functions.h"

complex_t cfn_exp(real_t phi)
{
    return rfn_cos(phi) + rfn_sin(phi) * I;
}
