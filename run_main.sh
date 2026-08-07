#!/bin/env fish

make

set -x OMP_NUM_THREADS 6
set -x OMP_PROC_BIND true
set -x OMP_PLACES cores
set -x OMP_SCHEDULE static
set -x OMP_DYNAMIC false
set -x OMP_WAIT_POLICY active

echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "OMP_PROC_BIND   = $OMP_PROC_BIND"
echo "OMP_PLACES      = $OMP_PLACES"

bin/main
