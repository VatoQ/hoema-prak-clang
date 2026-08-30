#include "../include/fourier.h"

#include "../include/logging.h"
#include <complex.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief In place implementation of FFT
 *
 * @param data complex array of data to be transformed
 * @param N Size of the array
 * @param inverse Time to frequency if true, frequency to time if false
 */
void fourier_worker_fft_rec(double complex* data, size_t N, bool inverse)
{
    if (N == 1)
    {
        return;
    }
    size_t M = N / 2;

    double complex* even = malloc(M * sizeof(double complex));
    double complex* odd  = malloc(M * sizeof(double complex));

    for (size_t i = 0; i < M; i++)
    {
        even[i] = data[2 * i];
        odd[i]  = data[2 * i + 1];
    }

    fourier_worker_fft_rec(even, M, inverse);
    fourier_worker_fft_rec(odd, M, inverse);

    double sign  = inverse ? 1.0 : -1.0;
    double theta = sign * 2.0 * PI / N;

    double complex w  = 1.0;
    double complex tw = cos(theta) + sin(theta) * I;

    for (size_t k = 0; k < M; k++)
    {
        data[k]     = even[k] + w * odd[k];
        data[k + M] = even[k] - w * odd[k];
        w *= tw;
    }
    free(even);
    free(odd);
}

/**
 * @brief Performs a fast fourier transform (fft). Reallocates
 * source data to the next power of two. A consequence of this is
 * `target->size >= source->size`.
 *
 * @param target Target container for transformed data
 * @param source Data to be transformed
 * @return Status code.
 */
int fourier_worker_fft(DataPoints* target, const DataPoints* source, bool to)
{
    if (target->size != 0)
    {
        DataPoints_free(target);
    }
    const size_t N = source->size;

    size_t M = 1;
    while (M < N)
    {
        M <<= 1;
    }

    double complex* buf = calloc(M, sizeof(double complex));
    memcpy(buf, source->data, N * sizeof(double complex));

    fourier_worker_fft_rec(buf, M, to);
    double scale = 1.0 / sqrt(M);
    for (int m = 0; m < M; m++)
    {
        buf[m] *= scale;
    }

    target->data = calloc(M, sizeof(double complex));
    target->size = M;

    memcpy(target->data, buf, M * sizeof(double complex));

    free(buf);
    Log_log("Performed FFT.", LOG_RT_INFO);
    return FFT;
}

/**
 * @brief helper for naive fourier transform
 * This helper performs a scalar fourier transform.
 * Useful for small problem sizes.
 *
 * @param target Target container for transformed data
 * @param source Data to be transformed
 * @param sign Can be \[1.0\] or \[-1.0\], depending on type of transform.
 * @return status code
 */
int fourier_worker_scalar(DataPoints* target,
                          const DataPoints* source,
                          const double sign)
{
    const double tao       = PI * 2;
    const size_t N         = source->size;
    const double inv_sqrtN = 1 / sqrt(N);
    for (size_t a = 0; a < N; a++)
    {
        const double theta = sign * tao * a / N;
        const double c     = cos(theta);
        const double s     = sin(theta);

        const double complex tw = c + s * I;
        double complex omega    = 1.0 + 0.0 * I;
        double complex sum      = 0.0;

        for (size_t b = 0; b < N; b++)
        {
            sum += omega * source->data[b];
            omega *= tw;
        }
        sum *= inv_sqrtN;
        target->data[a] = sum;
    }
    return DFT_SCALAR;
}

/**
 * @brief helper for naive fourier transform
 * This helper performs a parallelized fourier transform.
 * Useful for larger problem sizes.
 *
 * @param target Target container for transformed data
 * @param source Data to be transformed
 * @param sign Can be \[1.0\] or \[-1.0\], depending on type of transform.
 * @return status code
 */
int fourier_worker_parallel(DataPoints* target,
                            const DataPoints* source,
                            const double sign)
{
    const double tao       = PI * 2;
    const size_t N         = source->size;
    const double inv_sqrtN = 1 / sqrt(N);
#pragma omp parallel for
    for (size_t a = 0; a < N; a++)
    {
        const double theta = sign * tao * a / N;
        const double c     = cos(theta);
        const double s     = sin(theta);

        const double complex tw = c + s * I;
        double complex omega    = 1.0 + 0.0 * I;
        double complex sum      = 0.0;

        for (size_t b = 0; b < N; b++)
        {
            sum += omega * source->data[b];
            omega *= tw;
        }
        sum *= inv_sqrtN;
        target->data[a] = sum;
    }
    return DFT_PARALLEL;
}

int fourier_transform(DataPoints* target, const DataPoints* source, bool to)
{
    const size_t N         = source->size;
    const double inv_sqrtN = 1.0 / sqrt(N);
    const double tao       = 2 * PI;
    double sign            = -1;
    if (!to)
    {
        sign *= -1;
    }
    if (target->size != 0)
    {
        DataPoints_free(target);
    }

    if (N < 30)
    {
        target->data = calloc(N, sizeof(double complex));
        target->size = N;
        return fourier_worker_scalar(target, source, sign);
    }
    return fourier_worker_fft(target, source, to);
}

int DataPoints_free(DataPoints* data)
{
    free(data->data);
    data->data = NULL;
    data->size = 0;
    return 0;
}

DataPoints DataPoints_new(const size_t size)
{
    double complex* data = calloc(size, sizeof(double complex));

    DataPoints dp;
    dp.data = data;
    dp.size = size;

    return dp;
}
