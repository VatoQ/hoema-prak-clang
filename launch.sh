#!/bin/env fish
gcc -O3 -march=native -ffast-math \ 
    -Iinclude \
    src/main.c src/vector.c src/matrix.c src/logging.c \
    -o main -lm

time ./main

