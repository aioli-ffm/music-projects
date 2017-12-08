#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

// compile this:
// g++ -std=c++11 -Wall -Wextra main_utest.cpp -I../third_party -c

// run all tests in this directory:
// for i in utest_*.cpp; do g++ -std=c++11 main_utest.o "$i" -o tests -I../third_party -I../third_party/sndlib/src -L../third_party/sndlib -fopenmp -lfftw3f -lpython2.7 -lsndfile -lcsound64 && ./tests -r compact; done
