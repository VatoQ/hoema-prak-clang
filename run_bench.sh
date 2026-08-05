#!/bin/env fish

make benchmarks



for f in bin/bench_*
    if test -f "$f" -a -x "$f"
        echo "Running benchmark $f"
        ./"$f"
        echo
    end
end
