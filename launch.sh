#!/bin/env fish
gcc -O3 -march=native -ffast-math main.c vector.c matrix.c logging.c -o main -lm

time ./main

