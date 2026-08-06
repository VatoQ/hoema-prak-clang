#!/bin/env fish

make benchmarks


echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "OMP_PROC_BIND   = $OMP_PROC_BIND"
echo "OMP_PLACES      = $OMP_PLACES"

for f in bin/bench_*
    if test -f "$f" -a -x "$f"
        echo "Running benchmark $f"
        ./"$f"
        echo
    end
end
