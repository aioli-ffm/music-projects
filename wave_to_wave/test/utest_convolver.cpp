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

// This test function returns the dot product between two signals, assuming
// that patch is smaller than sig. Offset is applied to patch, e.g
// dotprod += sig[i]*patch[i+offset]. If any index is out of bounds
float dotProdAt(FloatSignal &sig, FloatSignal &patch, const int offset){
  float result = 0;
  for(int i=0, j=offset, n=patch.getSize(), m=sig.getSize(); i<n; ++i, ++j){
    if(j>=0 && j<m){
      result += patch[i] * sig[j];
    }
  }
  return result;
}


TEST_CASE("Testing the OverlapSaveConvolver class", "[OverlapSaveConvolver]"){
  // create signals
  const size_t kSizeS = 50;
  const size_t kSizeP1 = 3;
  auto lin = [](const long int n)->float {return n+1;};
  auto const1 = [](const long int n)->float {return 1;};

  SECTION("test dotProdAt"){
    FloatSignal s(lin, kSizeS);
    FloatSignal p1(const1,  kSizeP1);
    REQUIRE(dotProdAt(s, p1, -1000) == 0);
    REQUIRE(dotProdAt(s, p1, -3) == 0);
    REQUIRE(dotProdAt(s, p1, -2) == 1);
    REQUIRE(dotProdAt(s, p1, -1) == 1+2);
    REQUIRE(dotProdAt(s, p1, 0) == 1+2+3);
    REQUIRE(dotProdAt(s, p1, 1) == 2+3+4);
    REQUIRE(dotProdAt(s, p1, 2) == 3+4+5);
    REQUIRE(dotProdAt(s, p1, kSizeS-1) == kSizeS);
    REQUIRE(dotProdAt(s, p1, kSizeS) == 0);
    REQUIRE(dotProdAt(s, p1, 1000) == 0);
  }

  SECTION("Convolver constructor and init fields"){
    FloatSignal s(lin, kSizeS);
    FloatSignal p1(lin, kSizeP1);
    FloatSignal p1_reversed(p1.getData(),p1.getSize());
    std::reverse(p1_reversed.begin(), p1_reversed.end());
    // instantiate convolver, and extract conv and xcorr
    OverlapSaveConvolver x(s, p1);
    x.executeConv();
    FloatSignal conv = x.extractResult();
    x.executeXcorr();
    FloatSignal xcorr = x.extractResult();
    // compare results with tests using dotProdAt
    for(size_t i=0, n=kSizeS+kSizeP1-1; i<n; ++i){
      REQUIRE(Approx(xcorr[i]) == dotProdAt(s, p1, i-kSizeP1+1));
      REQUIRE(Approx(conv[i]) == dotProdAt(s, p1_reversed, i-kSizeP1+1));
    }
  }
  SECTION("sandbox"){
    FloatSignal o([](long int x){return x+1;}, 44100*10/20);
    FloatSignal m([](long int x){return 1;}, 44100*1/20);

    // CrossCorrelator x(o, m);
    // for(size_t i=0; i<10000; ++i){
    //   x.makeXcorr();
    //   if(i%1000==0){
    //     std::cout << i << std::endl;
    //   }
    // }

    // Optimizer opt(o, 20);

  }
}
