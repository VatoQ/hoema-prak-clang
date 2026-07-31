#include "../include/matrix.h"
#include "../include/vector.h"
#include "acutest.h"
#include <stddef.h>
#include <unistd.h>

void test_matrix_create(void)
{
    Matrix m = Matrix_new(3, 3, 0);
    TEST_CHECK(m.values != NULL);
    TEST_CHECK(m.m == 3);
    TEST_CHECK(m.n == 3);

    for (size_t i = 0; i < 9; i++)
    {
        TEST_CHECK(m.values[i] == 0.0);
    }

    Matrix_free(&m);
}

TEST_LIST = { { "matrix_create", test_matrix_create }, { NULL, NULL } };
