#ifndef FOURIER_H
#define FOURIER_H

#include <complex.h>
#include <stdbool.h>
#include <stddef.h>

#define TO true
#define FROM false

// #define DFT_SCALAR 0
// #define DFT_PARALLEL 1
// #define FFT 2

typedef enum
{
    DFT_SCALAR,
    DFT_PARALLEL,
    FFT
} FourierAlgorithm;

typedef struct
{
    double complex* data;
    size_t size;
} DataPoints;

/**
 * @brief Discrete complex fourier transformation. Can transform from
 * time domain to frequency domain and vice versa.
 *
 * @param target Target container for transformed data
 * @param source Data to be transformed
 * @param to time to frequency domain if `TO` (`true`), frequency to time if
 * `FROM` (`false`)
 * @return Status code depending on the success of the transformation.
 */
int fourier_transform(DataPoints* target, const DataPoints* source, bool to);

int fourier_worker_fft(DataPoints* target, const DataPoints* source, bool to);

/**
 * @brief Construct a new data set. Allocates `size` complex numbers.
 * All numbers are initialized as \[0+0i\]
 *
 * @param size number of complex number array to be allocated
 * @return new DataPoints struct
 */
DataPoints DataPoints_new(const size_t size);

/**
 * @brief Free allocated data from memory.
 *
 * @param data Data container to be deallocated.
 * @return Status code
 */
int DataPoints_free(DataPoints* data);

#endif // FOURIER_H
