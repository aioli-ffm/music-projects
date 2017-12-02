#include "catch.hpp"

// STL INCLUDES
#include<string>
#include<vector>
#include<list>
// SYSTEM-INSTALLED LIBRARIES
#include <fftw3.h>
// LOCAL INCLUDES
#include "../include/helpers.hpp"
#include "../include/signal.hpp"


TEST_CASE("FloatSignal construction and destruction", "[FloatSignal]"){
  // make and fill a float array for testing purposes
  const size_t kTestSize = 27;
  float* arr = new float[kTestSize];
  for(size_t i=0; i<kTestSize; ++i){arr[i] = 2*i;}
  // use different constructors:
  FloatSignal fs1(kTestSize);
  FloatSignal fs2(arr, kTestSize);
  FloatSignal fs3(arr, kTestSize, 10, 10);

  SECTION("FloatSignal is iterable, and can be used by IterableToString"){
    // make a STL iterable and create a string from it and fs2
    std::vector<float> v(arr, arr+kTestSize);
    std::string v_str = IterableToString(v);
    std::string fs2_str = IterableToString(fs2);
    // check that both strings and all values are equal
    REQUIRE(v_str.compare(fs2_str) == 0);
    for(size_t i=0; i<kTestSize; ++i){
      REQUIRE(v[i] == fs2[i]);
    }
  }



  SECTION("FloatSignal works with iterators") {
    REQUIRE(fs3.getSize() == kTestSize+20);
  }

  // finally delete test array
  delete[] arr;
}


TEST_CASE("Testing AudioSignal class via FloatSignal", "[signal]"){

  const size_t kTestSize = 53;
  FloatSignal f1(kTestSize);
  for(size_t i=0; i<kTestSize; ++i){
    f1[i] = 2*i;
  }

  SECTION( "FloatSignal works with iterators" ) {
    REQUIRE(f1[0] == 0);
    REQUIRE(f1[1] == 2);
    REQUIRE(f1[2] == 4);
  }
}
