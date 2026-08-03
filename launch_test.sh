#!/bin/env fish

gcc -O3 -march=native -ffast-math \
    -Iinclude \
    tests/test_matrix.c \
    src/matrix.c src/vector.c src/logging.c src/prng.c \
    -o test -lm


./test
