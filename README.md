# Matrix, Vector and Numerical Mathematics Library in Pure C

This repository contains a lightweight, self‑contained implementation
of matrix and vector operations written entirely in **ISO C**, 
without C++ classes, templates, operator overloading, or other 
modern abstractions. The project began as a personal exploration 
while working through exercises from the HöMa 2 Praktikum at 
FH Aachen, but it quickly grew into a standalone numerical 
toolkit with a clear design philosophy:

**Simple, explicit, predictable code — built from first principles.**

## Planned Features
- [x] matrix multiplication (including Jordan product)
- [x] matrix inversion for arbitrary n×n matrices
- [x] a minimal logging system
- [x] Pseudo random number generation.
- [x] a small unit‑testing setup using Acutest
- [x] Fourier transformations (DFT, FFT)
- [ ] Differential equations
- [ ] Stochastics (Monte Carlo simulations, random sampling)




# Build instructions

This project has been built on two systems:
- an HP ProBook running Manjaro Linux with an *AMD Ryzen 5 4500U*
- a personal workstation running CachyOS Linux with an *AMD Ryzen 5 5600X*

The instructions below apply to any comparable setup. Adjust paths,
compiler flags or shell commands as needed for your own environment.

## Requirements
- GCC or Clang
- POSIX-compatible shell (Fish, Bash, Zsh, etc.)
- Standard math library (`libm`)

## Building the project
A minimal build can be done directly from the command line.

Depending on the features used it may look like the following:

```fish
    gcc -O3 -march=native -ffast-math \
    src/main.c src/vector.c src/matrix.c src/logging.c \
    -o main -lm
```

Keep in mind that every feature uses the logging feature, 
so every build must include `logging.c`

## Using the provided `makefile`

Build `main.c` from the commandline using the proided `makefile`:
```fish
make
```

Build any unit test by passing the name of the test file as an argument:

```fish
make test_matrix
```

This will make an executable called `test_matrix` in the project root.

## Using provided build scripts

There are two shell scripts provided that
1. Build source files via makefile.
2. Set environment variables.
3. Launch the appropriate executable.
4. Are written in fish, so make sure it is installed and visible.

### `run_main.sh`

- Build all sources in `src/*` and `src/main.c` into the executable `main`. All compiler flags are specified by `makefile`.
- Set OpenMP environment flags.
- Execute `.main`.


### `run_tests.sh`
- Build all non main sources in `src/*` and all tests in `tests/*` into executables `test_*`.
- Iterate over all `test_*` files and execute.
- Give additional output if one or zero tests failed.

### `run_bench.sh`

- Build all non main sources in `src/*` and all benchmarks in `benchmarks/*` into executables `test_*`.
- Iterate over all `bench_*` files and execute.
- Give additional output.



# License & Usage


This project contains my personal C implementation of matrix, vector, and numerical routines originally inspired by exercises from the HöMa 2 Praktikum at FH Aachen.

I am publishing this code freely so that others can learn from it, study it, or use parts of it in their own projects.

**You may use, modify, and redistribute this code for any purpose — academic, personal, or commercial — as long as you include proper attribution.**

I do **not** accept responsibility for academic misuse: this repository is *not* a drop‑in solution for the original Praktikum, and I strongly encourage students to complete their own assignments.

The full license text is provided below.



## MIT License

# License & Usage


This project contains my personal C implementation of matrix, vector, and numerical routines originally inspired by exercises from the HöMa 2 Praktikum at FH Aachen.

I am publishing this code freely so that others can learn from it, study it, or use parts of it in their own projects.

**You may use, modify, and redistribute this code for any purpose — academic, personal, or commercial — as long as you include proper attribution.**

I do **not** accept responsibility for academic misuse: this repository is *not* a drop‑in solution for the original Praktikum, and I strongly encourage students to complete their own assignments.

The full license text is provided below.



## MIT License

Copyright &copy; 2026 VatoQ

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.



## Third‑Party Components

This project includes the header-only testing framework Acutest  
(&copy; 2013–2021, Acutest contributors), distributed under the MIT License.
The original license text is preserved in acutest.h.



