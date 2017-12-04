#include "catch.hpp"

// STL INCLUDES
#include<string>
#include<vector>
#include<list>
#include<complex>
// SYSTEM-INSTALLED LIBRARIES
#include <fftw3.h>
// LOCAL INCLUDES
#include "../include/helpers.hpp"
#include "../include/signal.hpp"



////////////////////////////////////////////////////////////////////////////////
/// TESTING THE FLOATSIGNAL CLASS
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("Testing the FloatSignal class", "[AudioSignal, FloatSignal]"){
  // make and fill a float array for testing purposes
  const size_t kTestSize = 27;
  float arr[kTestSize];
  for(size_t i=0; i<kTestSize; ++i){arr[i] = 2*i;}

  // use different constructors:
  FloatSignal fs1(kTestSize);
  FloatSignal fs2(arr, kTestSize);
  FloatSignal fs3(arr, kTestSize, 10, 10);

  SECTION("FloatSignal constructors, initialization, padding and operator[]"){
    for(size_t i=0; i<kTestSize; ++i){
      REQUIRE(fs1[i] == 0); // check that fs1 is initialized with zeros
      REQUIRE(fs2[i] == arr[i]); // check all values arr and fs2 are same
      REQUIRE(fs3[i+10] == arr[i]); // check all values arr and fs3 are same
    }
    // check padding:
    float sum_pad = 0;
    for(size_t i=0; i<10; ++i){
      sum_pad += abs(fs3[i]);
      sum_pad += abs(fs3[10+kTestSize+i]);
    }
    REQUIRE(sum_pad == 0); // sum of padded cells is zero
  }

  SECTION("FloatSignal is iterable, and can be used by IterableToString"){
    // make a STL iterable and create a string from it and fs2
    std::vector<float> v(arr, arr+kTestSize);
    std::string v_str = IterableToString(v);
    std::string fs2_str = IterableToString(fs2);
    REQUIRE(v_str.compare(fs2_str) == 0); // check both IterToString are equal
    for(auto& f : fs2){ // use some iterator loop
      auto i = &f - &fs2[0];
      REQUIRE(f == arr[i]);
    }
  }

  SECTION("FloatSignal getters and setters"){
    // getSize
    REQUIRE(fs1.getSize() == kTestSize);
    REQUIRE(fs2.getSize() == kTestSize);
    REQUIRE(fs3.getSize() == kTestSize+20);
    // getData
    REQUIRE(fs1.getData() != arr); // constructor makes a copy of arr
    REQUIRE(fs2.getData() != fs1.getData()); // constructor makes a copy of arr
    REQUIRE(*(fs2.getData()+1) == fs2[1]);
  }

  SECTION("FloatSignal-to-constant compound assignment operators"){
    SECTION("+= and -= operators"){
      float kOffset = 1000.1234;
      fs2 += kOffset;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == arr[i]+kOffset);
      }
      fs2 -= kOffset;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == arr[i]);
      }
    }
    SECTION("*= operator"){
      float kOffset = 12.345;
      fs2 *= kOffset;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == arr[i]*kOffset);
      }
    }
  }

  SECTION("FloatSignal-to-FloatSignal operators"){
    FloatSignal other(arr, kTestSize);
    SECTION("+= with itself and with another:"){
      // test
      fs2.addSignal(other, -2000);
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == arr[i]);
      }
      fs2.addSignal(other, 2000);
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == arr[i]);
      }
      fs2.addSignal(other);
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == 2*arr[i]);
      }
      // test that is possible to add to itself
      fs2.addSignal(other);
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == 3*arr[i]);
      }
      // test compound operator
      fs2 += other;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == 4*arr[i]);
      }
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// TESTING THE COMPLEXSIGNAL CLASS
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("Testing the ComplexSignal class", "[AudioSignal, ComplexSignal]"){
  // to avoid verbosity:
  std::complex<float> kComplexZero(0,0);
  // make and fill a complex array for testing purposes
  const size_t kTestSize = 27;
  std::complex<float> arr[kTestSize];
  for(size_t i=0; i<kTestSize; ++i){arr[i] = std::complex<float>(10*i, 10*i);}
  // float f_arr[kTestSize];
  // for(size_t i=0; i<kTestSize; ++i){f_arr[i] = 2*i;}

  // use different constructors:
  ComplexSignal cs1(kTestSize);
  ComplexSignal cs2(arr, kTestSize);
  ComplexSignal cs3(arr, kTestSize, 10, 10);

  SECTION("ComplexSignal constructors, initialization, padding and operator[]"){
    for(size_t i=0; i<kTestSize; ++i){
      REQUIRE(cs1[i] == kComplexZero);  // check that cs1 is initialized with 0s
      REQUIRE(cs2[i] == arr[i]); // check all values arr and cs2 are same
      REQUIRE(cs3[i+10] == arr[i]); // check all values arr and cs3 are same
    }
    // check padding:
    std::complex<float> sum_pad(0,0);
    for(size_t i=0; i<10; ++i){
      sum_pad += cs3[i];
      sum_pad += cs3[10+kTestSize+i];
    }
    REQUIRE(sum_pad == kComplexZero); // sum of padded cells is zero
  }

  SECTION("ComplexSignal is iterable, and can be used by IterableToString"){
    // make a STL iterable and create a string from it and fs2
    std::vector<std::complex<float> > v(arr, arr+kTestSize);
    std::string v_str = IterableToString(v);
    std::string cs2_str = IterableToString(cs2);
    REQUIRE(v_str.compare(cs2_str) == 0); // check both IterToString are equal
    for(auto& c : cs2){ // use some iterator loop
      auto i = &c - &cs2[0];
      REQUIRE(c == arr[i]);
    }
  }

  SECTION("ComplexSignal getters and setters"){
    // getSize
    REQUIRE(cs1.getSize() == kTestSize);
    REQUIRE(cs2.getSize() == kTestSize);
    REQUIRE(cs3.getSize() == kTestSize+20);
    // getData
    REQUIRE(cs1.getData() != arr); // constructor makes a copy of arr
    REQUIRE(cs2.getData() != cs1.getData()); // constructor makes a copy of arr
    REQUIRE(*(cs2.getData()+1) == cs2[1]);
  }

  SECTION("ComplexSignal-to-constant compound assignment operators"){
    SECTION("+= and -= operators"){
      std::complex<float> kOffset(1000.1234, 1000.1234);
      cs2 += kOffset;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]+kOffset);
      }
      cs2 -= kOffset;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]);
      }
    }
    SECTION("*= operator"){
      std::complex<float> kOffset(1000, 1000);
      cs2 *= kOffset;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]*kOffset);
      }
    }
  }
}
