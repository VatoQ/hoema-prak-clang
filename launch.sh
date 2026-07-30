#!/bin/env fish
gcc -O2 -march=native -ffast-math main.c vector.c matrix.c -o main -lm

time ./main

