#!/bin/env bash
gcc -O3 -march=native -ffast-math \
    -Iinclude \
    src/main.c \
    src/prng.c \
    src/vector.c \
    src/matrix.c \
    src/logging.c \
    -o main -lm

./main

