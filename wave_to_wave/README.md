# LICENSING:

This repository uses third-party libraries, but doesn't include them. The THIRD-PARTY LIBRARIES subsection say (among other things) how to get the files for the `third_party` directory.

* All the third-party libraries used are expected to be in the [third_party](third_party) directory. Each third-party library is acquired and installed by the released under its own individual license, which should be mentioned in the respective source files.

* The remaining files are part of this project and are **licensed under the GPLv3 or higher**, as stated in the [LICENSE](LICENSE) file. Again, **the [LICENSE](LICENSE) file of does NOT apply to the libraries located in the third-party directory**.



# RUN:

### Build program:
`g++ -O3 -std=c++11 -Wall -Wextra wave_to_wave.cpp -fopenmp -lfftw3f -lpython2.7 -o ./bin/test && ./bin/test -z -y wisdom -i 10`

### UTests:
In the tests directory: `for i in utest_*.cpp; do g++ -std=c++11 main_utest.o "$i" -o tests && ./tests -r compact; done`

### Benchmarks:
In the benchmarks directory: `mkdir -p bin && g++ -std=c++11 -Wall -Wextra -L../third_party/google_benchmark  benchmark_example.cpp -lbenchmark -lpthread -o bin/benchmark && ./bin/benchmark`


### Using Make (in development):

```bash
mkdir build
cd build
cmake ..
make
make test
```

# TODO:

### HIGHER PRIORITY

Development/Test:
- [ ] Add delay to signal, and overload operators to allow signal-to-signal arithmetic. Build and run utests.
- [ ] Add wav import-export functionality to signal class. Build and run utests.
- [ ] Add basic synth doing real morlet-wavelets (chi2 envelope). It should generate signal objects. Build and run utests.
- [ ] Make prototype of optimizer as a collection of convolvers plus a "result" signal. Benchmark and utest.

Cmake/Make related:

- [ ] cmake/make should generate all build-related files in the build dir, and the executable in the bin dir.
- [x] `make build` to compile the whole program, with main in `wave_to_wave.cpp`
- [ ] `make run-tests` (see explanation further, currently running with `for i in utest_*.cpp; do g++ -std=c++11 main_utest.o "$i" -o tests && ./tests -r compact; done`)
- [ ] `make clean` to clean all build-related and executable files (build and bin directories).
- [ ] `make run-benchmarks` with a dummy output for the moment
- [ ] `make` or `make all` should run clean, test, benchmark, and build in that order.

### LOWER PRIORITY
- [ ] Input parser: How to get the flags:values as a ` map<string, string>` in an elegant way?
- [ ] Input parser: How to check if no flag was activated, to plot the `-h` flag?
- [ ] Maybe look for a better input parser if this is difficult...
- [ ] Explicitly SIMDize SpectralConvolution and SpectralCorrelation funcs: https://github.com/VcDevel/Vc
- [ ] Benchmarking: test the kMinChunkSize of the convolver for many powers of 2 and output the best one to a config file in benchmark dir.
- [ ] Benchmarking: test with/without OpenMP for different routines and signal sizes.
- [ ] *Advanced*: allow cmake to grab the optimal results of benchmarking for the current system and use them at build stage.


# THIRD-PARTY LIBRARIES:

### STRUCTURE

This project requires third party libraries to work. At the moment, and since some of them (like google's benchmark) are system-specific, the `third_party` directory **is ignored and has to be incorporated to the project by hand**. This README documents how to incorporate them to this repository to be able to build the program successfully.

The structure for the third-party libraries of this `<REPO>` is expected to be the following:
```
<REPO>
  \
  | - include
  | - ... (the rest of the files that are not ignored)
  | - third_party
                 \
                 | - catch.hpp
                 | - cxxopts.hpp
                 | - matplotlibcpp.h
                 | - google_benchmark
                                     \
                                     | - benchmark.h
                                     | - libbenchmark.a  
```


### UNIT TESTING WITH CATCH2 `https://github.com/catchorg/Catch2`:

Catch2 is a modern, C++-native, header-only, test framework for unit-tests. The whole framework is contained in a single header, so the installation and usage is very straightforward:

1. Download the header from here: `https://raw.githubusercontent.com/CatchOrg/Catch2/master/single_include/catch.hpp` and put it into `<CATCH>`
2. In your `<UTEST_FILE.CPP>`, include the header with `#include <CATCH>/catch.hpp`
3. Compile with `g++ -std=c++11 -Wall -Wextra  <UTEST_FILE.CPP> -o bin/utest && ./bin/utest`

See the [test](test) directory for examples, and the tutorial: `https://github.com/catchorg/Catch2/blob/master/docs/tutorial.md`

#### NOTE:

The method explained above doesn't scalate well, as explained here: [](https://github.com/catchorg/Catch2/blob/master/docs/slow-compiles.md). It is preferred to compile a `main_utest.cpp` file including the catch header just once, and include its object file when compiling the other tests, speeding up the whole process:

1. In the test folder, compile the main test: `g++ -std=c++11 -Wall -Wextra  main_utest.cpp -c` That has only 2 lines as explained in the link.
2. Make your test files including the `catch.hpp` header (not the main file) and name them `utest_<NAME>.cpp`. **THEY SHOULD NOT HAVE THE `#define CATCH_CONFIG_MAIN` LINE**.
3. Compile and rum them all with `for i in utest_*.cpp; do g++ -std=c++11 main_utest.o "$i" -o tests && ./tests -r compact; done`


### PLOTTING WITH MATPLOTLIB-CPP`https://github.com/lava/matplotlib-cpp`:

Like Catch2, matplotlib-cpp is a single-header plotting library built on the top of Python's matplotlib:

1. Download the header from here: `https://raw.githubusercontent.com/lava/matplotlib-cpp/master/matplotlibcpp.h` and put it into `<PLOT>`
2. In your `<PLOT_FILE.CPP>`, include the header with `#include <PLOT>/matplotlibcpp.h`.
3. Compile with `g++ -std=c++11 -Wall -Wextra <PLOT_FILE.CPP> -I/usr/include/python2.7 -lpython2.7 -o ./bin/testplot && ./bin/testplot`


See the project's [README](https://github.com/lava/matplotlib-cpp/blob/master/README.md) for examples, I didn't find any further documentation but the header file is somehow self-explanatory, and doesn't provide complex functionality.

### INPUT PARSER WITH CXXOPTS `https://github.com/jarro2783/cxxopts`:

1. Download the header from here: `https://raw.githubusercontent.com/jarro2783/cxxopts/master/include/cxxopts.hpp` and put it into `<PARSER>`
2. In your `<MAIN_FILE.CPP>`, include the header with `#include <PARSER>/cxxopts.hpp`
3. Use in your main function as described in the docs, and compile `<MAIN_FILE.CPP>` as usual.

See the project's [README](https://github.com/jarro2783/cxxopts/blob/master/README.md) for examples.

### BENCHMARKING WITH GOOGLE BENCHMARK LIBRARY `https://github.com/google/benchmark`:

The `third_party/google_benchmark` dependency contains two library files: `benchmark.h` and `libbenchmark.a`, which have been obtained as follows:

1. clone repo from `https://github.com/google/benchmark` into `<REPO>`
2. in `<REPO>`, do `mkdir build` and `cd build`
3. then `cmake -DCMAKE_BUILD_TYPE=Release ..` and `make`
4. find the `benchmark.h` and `libbenchmark.a` files, and copy them to `<LOCAL_LIB>` (look in `<REPO>/include/benchmark` and `<REPO>/build/src`)

To use them for performing benchmarking in local files, two things are required:

1. In `<BENCHMARK_FILE.CPP>`, include the header: `#include "<LOCAL_LIB>/benchmark.h"`
2. compile and run `my_test.cpp` with `g++ -std=c++11 -Wall -Wextra -L<LOCAL_LIB> <BENCHMARK_FILE.CPP> -lbenchmark -lpthread -o bin/benchmark && ./bin/benchmark`

See the [benchmark](benchmark) directory for examples. The repo's README offers a good introduction: `https://github.com/google/benchmark/blob/master/README.md`

