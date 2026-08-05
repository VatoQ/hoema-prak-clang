#!/bin/env fish

make

set -x OMP_PROC_BIND true
set -x OMP_PLACES cores

echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "OMP_PROC_BIND   = $OMP_PROC_BIND"
echo "OMP_PLACES      = $OMP_PLACES"
./main
