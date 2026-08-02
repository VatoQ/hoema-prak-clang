#include "../include/prng.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

PRNG_State PRNG_State_init(int seed)
{
    PRNG_State prng;
    size_t state = seed;
    if (seed == NO_SEED)
    {
        state = (size_t)time(NULL);
    }

    state ^= state >> 30;
    state *= D;
    state ^= state >> 27;
    state *= C;
    state ^= state >> 31;

    prng.state = state;
    prng.aux   = state ^ 0x94D049BB133111EBULL;

    return prng;
}

size_t PRNG_State_random(PRNG_State* prng)
{
    const size_t state = prng->state;
    size_t aux         = prng->aux;
    const size_t n     = sizeof(size_t) * 8;
    aux                = (aux % n) ?: 1;

    // State diffusion
    const size_t new_aux = (state << aux) | (state >> (n - aux));
    prng->aux            = new_aux;
    size_t new_state     = state + C;
    new_state            = (new_state ^ new_aux) * C;
    aux ^= new_aux;
    aux       = (aux % n) ?: 1;
    new_state = (new_state << aux) | (new_state >> (n - aux));
    new_state *= D;
    aux ^= new_aux;
    aux       = (aux % n) ?: 1;
    new_state = (new_state << aux) | (new_state >> (n - aux));
    new_state += C;
    prng->state = new_state;

    // Random result diffusion
    size_t z = new_state;
    z ^= z >> 30;
    z *= D;
    z ^= z >> 27;
    z *= C;
    z ^= z >> 31;
    return z;
}

double PRNG_State_random_double(PRNG_State* prng)
{
    const size_t randbits = PRNG_State_random(prng);
    const double x        = randbits / (double)(SIZE_MAX - 1);
    return x;
}

double PRNG_State_random_double_range(PRNG_State* prng, double min, double max)
{
    const double x = PRNG_State_random_double(prng);
    return min + x * (max - min);
}

double PRNG_State_normal(PRNG_State* prng,
                         const double mean,
                         const double variance)
{
    double disc = sqrt(mean * mean + 8.0 * variance * variance);

    double x_plus  = (mean + disc) / 2.0;
    double x_minus = (mean - disc) / 2.0;

    double v_plus  = x_plus * exp(-((x_plus - mean) * (x_plus - mean)) /
                                  (4.0 * variance * variance));
    double v_minus = x_minus * exp(-((x_minus - mean) * (x_minus - mean)) /
                                   (4.0 * variance * variance));

    for (;;)
    {
        double U1 = PRNG_State_random_double(prng);
        double U2 = PRNG_State_random_double(prng);

        double u = U1;
        double v = v_minus + (v_plus - v_minus) * U2;

        double x = v / u;

        double fx = exp(-0.5 * (x - mean) * (x - mean) / (variance * variance));
        if (u * u <= fx)
        {
            return x;
        }
    }
}

double PRNG_State_exponential(PRNG_State* prng, const double lambda)
{
    const double umax = 1.0;
    const double vmax = (2.0 / lambda) * exp(-1.0);

    for (;;)
    {
        double U1 = PRNG_State_random_double(prng);
        double U2 = PRNG_State_random_double(prng);

        double u = umax * U1;
        double v = vmax * U2;

        if (u <= 0.0)
        {
            continue;
        }

        double x  = v / u;
        double fx = exp(-lambda * x);

        if (u * u <= fx)
        {
            return x;
        }
    }
}
