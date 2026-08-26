#ifndef RANDOM_H
#define RANDOM_H

#include <stddef.h>
#include <sys/types.h>
#define NO_SEED 0
#define C 0x9E3779B97F4A7C15ULL
#define D 0xBF58476D1CE4E5B9ULL

/**
 * @brief Two word PRNG. Contains `size_t state` and `size_t aux`.
 */
typedef struct
{
    size_t state;
    size_t aux;
} PRNG_State;

/**
 * @brief Initialize a PRNG state.
 *
 * @param seed Custom seed for the PRNG. Choose `NO_SEED` to get a random state.
 * @return A two word PRNG state. Contains `state` and `aux`.
 */
PRNG_State PRNG_State_init(int seed);

/**
 * @brief Generate a random 64 bit word. First the two states
 * get diffused, then the random word gets diffused.
 *
 * @param prng Two word PRNG state.
 * @return Random bit pattern.
 */
size_t PRNG_State_random(PRNG_State* prng);

/**
 * @brief Generate a uniformly distributed random integer in a specified range.
 *
 * @param prng Two state PRNG state.
 * @param low Lowest number in the range.
 * @param high Highest number in the range.
 * @return Random integer in (low, high).
 */
ssize_t PRNG_State_random_range(PRNG_State* prng, ssize_t low, ssize_t high);

/**
 * @brief Generate a random double precision floating point number
 * in `[0,1]`
 *
 * @param prng Two word PRNG state.
 * @return Random double in `[0,1]`
 */
double PRNG_State_random_double(PRNG_State* prng);

/**
 * @brief Generate a random double precision floating point number
 * in `[min, max]`
 *
 * @param prng Two word PRNG state.
 * @param min Minimum random float
 * @param max Maximum random float
 * @return Random float in `[min, max]`
 */
double PRNG_State_random_double_range(PRNG_State* prng, double min, double max);

/**
 * @brief Generate a normally distributed floating point number with specified
 * mean and variance.
 *
 * @param `prng` Two word PRNG state.
 * @param `mean` Mean/Expected value of the distribution.
 * @param `variance` Variance of the distribution.
 * @return Normally distributed floating point number.
 */
double PRNG_State_normal(PRNG_State* prng,
                         const double mean,
                         const double variance);

/**
 * @brief Generate an exponentially distributed floating point number.
 *
 * @param `prng` Two workd PRNG state.
 * @param `lambda` Scales the distribution
 * @return Random exponentially distributed floating point number.
 */
double PRNG_State_exponential(PRNG_State* prng, const double lambda);

#endif // RANDOM_H
