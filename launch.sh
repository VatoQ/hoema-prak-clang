#!/bin/env fish
gcc -O2 main.c vector.c matrix.c -o main -lm

time ./main

