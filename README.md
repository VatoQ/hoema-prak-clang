# Numerical Math Library — C core, plus a small Python API

This repository began as a compact, self-contained implementation of
matrix, vector and numerical routines written in plain ISO C. Over time
I added tooling, tests and a small Python API and the repository's
purpose shifted from being a collection of university assignment
solutions to a general-purpose learning and experimentation workspace.

What this repository is now

- Core numerical routines and performance-sensitive code remain
  implemented in C.
- A growing Python API (and related scripts) provide a higher-level
  interface and make experimenting with the algorithms easier.
- The original HöMa Praktikum exercises are preserved for reference,
  but they are no longer the main focus of the project.

Design goals

- Simple, explicit, predictable implementation in C for the core
  algorithms.
- Easy-to-use Python bindings and helper tools for prototyping and
  experimentation.
- Clear tests and small, readable examples rather than large framework
  dependencies.

Quick status / roadmap

- C: matrix multiplication, inversion, basic linear algebra — maintained
  and exercised by unit tests.
- Python API: initial bindings and convenience wrappers (work in
  progress).
- Tests: small unit tests are present (Acutest) and are runnable via the
  Makefile.
- Planned: improve packaging for the Python API, add more examples,
  and expand numerical methods (ODE solvers, stochastic methods).

Building and running

Most of the performance-sensitive code is in C and can be built using
your system compiler (GCC or Clang) and the provided Makefile. Python
components require a Python 3 interpreter; see the relevant Python
scripts for any additional dependency notes.

Minimal C build example:

    gcc -O3 -march=native -ffast-math \
        src/main.c src/vector.c src/matrix.c src/logging.c \
        -o main -lm

Use the Makefile to build the main executable or the tests:

    make        # build main (default target)
    make test_matrix   # build a test executable

If you've been working with the repository previously: note that some
scripts or helper files for the Python API may be present in the repo
root or a `python/` subdirectory. Check the tree for `py`/`python`/
`bindings` folders to find the latest Python code.

Contributing and notes

This is my personal project and a place for learning. Contributions are
welcome — please open issues or pull requests for fixes, improvements or
suggestions. If you use code in academic work, treat this repository as
reference material rather than a drop-in solution for course
assignments.

License

This project is distributed under the MIT License — see the full text
below.

<!-- MIT License text -->

Copyright (c) 2026 VatoQ

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
