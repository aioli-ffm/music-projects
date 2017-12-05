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
#include "../include/convolver.hpp"


////////////////////////////////////////////////////////////////////////////////
/// TESTING THE CONVOLVER CLASS
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("Testing the OverlapSaveConvolver class", "[OverlapSaveConvolver]"){
  // settings
  const size_t kSizeS = 10;
  const size_t kSizeP1 = 3;
  // create a test signal {1, 2, 3, ... kSizeS}
  float s_arr[kSizeS];
  for(size_t i=0; i<kSizeS; ++i){s_arr[i] = i+1;}
  FloatSignal s(s_arr, kSizeS);
  // create a test patch {1, 2, 3 ... kSizeP1}
  float p1_arr[kSizeP1];
  for(size_t i=0; i<kSizeP1; ++i){p1_arr[i]=i+1;}
  FloatSignal p1(p1_arr, kSizeP1);

  SECTION("Convolver constructor and init fields"){
    // instantiate convolver
    OverlapSaveConvolver xx(s, p1);
    // the convolution between s and p1
    FloatSignal testConv(kSizeS+kSizeP1-1);
    testConv[4] = 123;
    float kTestConv[]{1, 4, 10, 16, 22, 28, 34, 40, 46, 52, 47, 30};
    // the cross-correlation between s and p1
    float kTestXcorr[]{3, 8, 14, 20, 26, 32, 38, 44, 50, 56, 29, 10};
    // make and test convolution
    xx.executeConv();
    FloatSignal conv = xx.extractResult();
    conv.print("nowtest");
    for(size_t i=0; i<(kSizeS+kSizeP1-1); ++i){
      REQUIRE(Approx(conv[i]) == kTestConv[i]);
    }
    // make and test cross-correlation
    xx.executeXcorr();
    FloatSignal xcorr = xx.extractResult();
    for(size_t i=0; i<(kSizeS+kSizeP1-1); ++i){
      REQUIRE(Approx(xcorr[i]) == kTestXcorr[i]);
    }
  }


  SECTION("sandbox"){
    FloatSignal o([](long int x){return x+1;}, 6);
    FloatSignal m([](long int x){return 1;}, 3);

    Test x(o, m);

    o.print("oo");
    m.print("mm");

    x.makeXcorr().print("xcorr");

  }
}
