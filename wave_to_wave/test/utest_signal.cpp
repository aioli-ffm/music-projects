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
  float* arr = new float[kTestSize];
  for(size_t i=0; i<kTestSize; ++i){arr[i] = 2*i;}
  // use different constructors:
  FloatSignal fs1(kTestSize);
  FloatSignal fs2(arr, kTestSize);
  FloatSignal fs3(arr, kTestSize, 10, 10);

  SECTION("FloatSignal constructors, padding and [] operator"){
    float sum = 0; // to check that fs1 is initialized with zeros
    for(size_t i=0; i<kTestSize; ++i){
      REQUIRE(fs2[i] == arr[i]); // check all values arr and fs2 are same
      REQUIRE(fs3[i+10] == arr[i]); // check all values arr and fs3 are same
      sum += abs(fs1[i]); // to check that fs1 is initialized with zeros
    }
    REQUIRE(sum == 0); // to check that fs1 is initialized with zeros
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
    REQUIRE(*(fs2.getData()+1) == 2);
  }

  SECTION("FloatSignal compound assignment operators"){
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

  // delete array to finish the FloatSignal test
  delete[] arr;
}



////////////////////////////////////////////////////////////////////////////////
/// TESTING THE COMPLEXSIGNAL CLASS
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("Testing the ComplexSignal class", "[AudioSignal, ComplexSignal]"){
  // make and fill a float array for testing purposes
  const size_t kTestSize = 27;
  std::complex<float> arr[kTestSize];// = fftwf_alloc_complex(kTestSize);
  for(size_t i=0; i<kTestSize; ++i){
    arr[i] = (2*i, 0);
  }
  //
  ComplexSignal cs1(kTestSize);
  cs1 += std::complex<float>(0, 1);
  // cs1 += 5;
  // cs1 -= std::complex<float>(3, 4);
  cs1 *= std::complex<float>(1, 2);
  cs1 *= 10;
  cs1.print();

  // SECTION("ComplexSignal constructor and [] operator"){
  //   float sum = 0; // to check that cs1 is initialized with zeros
  //   for(size_t i=0; i<kTestSize; ++i){
  //     sum += abs(cs1[i][REAL]);
  //     sum += abs(cs1[i][IMAG]);
  //   }
  //   REQUIRE(sum == 0); // to check that cs1 is initialized with zeros
  // }

  // SECTION("ComplexSignal is iterable, and can be used by IterableToString"){
  //   make a STL iterable and create a string from it and fs2
  //   std::vector<fftwf_complex> v(arr, arr+kTestSize);
  //   std::string v_str = IterableToString(v);
  //   std::string fs2_str = IterableToString(fs2);
  //   REQUIRE(v_str.compare(fs2_str) == 0); // check both IterToString are equal
  //   for(auto& f : fs2){ // use some iterator loop
  //     auto i = &f - &fs2[0];
  //     REQUIRE(f == arr[i]);
  //   }
  // }

  // SECTION("FloatSignal getters and setters"){
  //   // getSize
  //   REQUIRE(fs1.getSize() == kTestSize);
  //   REQUIRE(fs2.getSize() == kTestSize);
  //   REQUIRE(fs3.getSize() == kTestSize+20);
  //   // getData
  //   REQUIRE(fs1.getData() != arr); // constructor makes a copy of arr
  //   REQUIRE(fs2.getData() != fs1.getData()); // constructor makes a copy of arr
  //   REQUIRE(*(fs2.getData()+1) == 2);
  // }

  // SECTION("FloatSignal compound assignment operators"){
  //   SECTION("+= and -= operators"){
  //     float kOffset = 1000.1234;
  //     fs2 += kOffset;
  //     for(size_t i=0; i<kTestSize; ++i){
  //       REQUIRE(fs2[i] == arr[i]+kOffset);
  //     }
  //     fs2 -= kOffset;
  //     for(size_t i=0; i<kTestSize; ++i){
  //       REQUIRE(fs2[i] == arr[i]);
  //     }
  //   }
  //   SECTION("*= operator"){
  //     float kOffset = 12.345;
  //     fs2 *= kOffset;
  //     for(size_t i=0; i<kTestSize; ++i){
  //       REQUIRE(fs2[i] == arr[i]*kOffset);
  //     }
  //   }
  // }

  // // delete array to finish the FloatSignal test
  // // delete[] arr;
}
