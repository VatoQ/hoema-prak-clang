#include "../include/random.h"
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
