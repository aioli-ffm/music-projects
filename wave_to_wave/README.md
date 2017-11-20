

# NON-DEVELOPMENT TOOLS:

### C++ UNIT TESTING WITH CATCH2 `https://github.com/catchorg/Catch2`:

Catch2 is a modern, C++-native, header-only, test framework for unit-tests. The whole framework is contained in a single header, so the installation and usage is very straightforward:

1. Download the header from here: `https://raw.githubusercontent.com/CatchOrg/Catch2/master/single_include/catch.hpp` and put it into `<CATCH>`
2. In your `<UTEST_FILE.CPP>`, include the header with `#include <CATCH>/catch.hpp`
3. Compile with `g++ -std=c++11 -Wall -Wextra  <UTEST_FILE.CPP> -o bin/utest && ./bin/utest`

See the (tests/utest_example.cpp) file for an example, and the tutorial: `https://github.com/catchorg/Catch2/blob/master/docs/tutorial.md`


### C++ PLOTTING WITH MATPLOTLIB-CPP`https://github.com/lava/matplotlib-cpp`:

Like Catch2, matplotlib-cpp is a single-header plotting library built on the top of Python's matplotlib:

1. Download the header from here: `https://raw.githubusercontent.com/lava/matplotlib-cpp/master/matplotlibcpp.h` and put it into `<PLOT>`
2. In your `<PLOT_FILE.CPP>`, include the header with `#include <PLOT>/matplotlibcpp.h`
3. Compile with `g++ -std=c++11 -Wall -Wextra <PLOT_FILE.CPP> -I/usr/include/python2.7 -lpython2.7 -o ./bin/testplot && ./bin/testplot`

See the (tests/plot_example.cpp) file for an example. I didn't find any documentation whatsoever but the header file is somehow self-explanatory, and doesn't provide complex functionality.


### C++ BENCHMARKING WITH GOOGLE BENCHMARK LIBRARY `https://github.com/google/benchmark`:

The `third_party/google_benchmark` dependency contains two library files: `benchmark.h` and `libbenchmark.a`, which have been obtained as follows:

1. clone repo from `https://github.com/google/benchmark` into `<REPO>`
2. in `<REPO>`, do `mkdir build` and `cd build`
3. then `cmake -DCMAKE_BUILD_TYPE=Release ..` and `make`
4. find the `benchmark.h` and `libbenchmark.a` files, and copy them to `<LOCAL_LIB>` (look in ``<REPO>/include/benchmark` and `<REPO>/build/src`)

To use them for performing benchmarking in local files, two things are required:

1. In `<BENCHMARK_FILE.CPP>`, include the header: `#include "<LOCAL_LIB>/benchmark.h"`
2. compile and run `my_test.cpp` with `g++ -std=c++11 -Wall -Wextra -L<LOCAL_LIB> <BENCHMARK_FILE.CPP> -lbenchmark -lpthread -o bin/benchmark && ./bin/benchmark`

See the (tests/benchmark_example.cpp) file for an example. The repo's README offers a good introduction: `https://github.com/google/benchmark/blob/master/README.md`

