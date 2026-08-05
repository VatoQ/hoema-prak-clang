#!/bin/env fish

make tests

for f in ./test_*

    if test -f "$f" -a -x "$f"
        echo "Running $f"
        ./"$f"
        echo

        if test $status -ne 0
            echo "Test $f failed with exit code $status"
            exit $status
        end
    end
end

echo "All tests passed."
