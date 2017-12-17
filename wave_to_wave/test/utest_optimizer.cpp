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
#include "../include/optimizer.hpp"


////////////////////////////////////////////////////////////////////////////////
/// TESTING THE WAVTOWAVOPTIMIZER CLASS
////////////////////////////////////////////////////////////////////////////////

std::vector<std::pair<long int, float> > AbsoluteMaxCriterium(FloatSignal &fs){
  std::vector<std::pair<long int, float> > result;
  float* abs_max = std::max_element(fs.begin(), fs.end(), abs_compare<float>);
  result.push_back(std::pair<long int,float>(std::distance(fs.begin(), abs_max),
                                             *abs_max));
  return result;
};



TEST_CASE("test optimizer", "[optimizer]"){

  SECTION("test against a simple unit delta with different sizes"){
    for(size_t N=1000; N<10000; N+=1000){
      for(size_t M=100; M<=N/3; M+=N/7){
        std::cout << "testing for N="<<N<<", M="<<M<<std::endl;
        FloatSignal sig([](long int x){return x+1;}, N);
        FloatSignal patch([](long int x){return x==0;}, M);
        WavToWavOptimizer o(sig, AbsoluteMaxCriterium);
        // After exactly N iterations, the delta should optimize the given
        // signal with an per-sample error below 0.01%
        for(size_t i=0; i<N; ++i){
          o.step(patch);
        }
        REQUIRE(o.getResidualEnergy()/N < 0.0001);
      }
    }
  }

  SECTION("test against a redundant signal with a redundant delta"){
    for(size_t N=1000; N<10000; N+=1000){
      for(size_t M=100; M<=N/3; M+=N/7){
        std::cout << "testing for N="<<N<<", M="<<M<<std::endl;
        FloatSignal sig([](long int x){return x-(x%2==1);}, N); //0,0,2,2,...
        FloatSignal patch([](long int x){return x<2;}, M);
        WavToWavOptimizer o(sig, AbsoluteMaxCriterium);
        // After exactly N/2 iterations, the delta should optimize the given
        // signal with an per-sample error below 0.01%
        for(size_t i=0; i<N/2; ++i){
          o.step(patch);
        }
        REQUIRE(o.getResidualEnergy()/N < 0.0001);
      }
    }
  }


   SECTION("test against a delayed, redundant delta"){
    const size_t N = 23456;
    const size_t D = 100;
    for(size_t d=0; d<=D; d+=13){
      std::cout << "testing for delay="<<d<<std::endl;
      FloatSignal sig([](long int x){return x-(x%2==1);}, N); //0,0,2,2,..
      FloatSignal patch([=](long int x){return x==d||x==(d+1);}, 737);
      WavToWavOptimizer o(sig, AbsoluteMaxCriterium);
      // After exactly N/2 iterations, the delayed delta should optimize the
      // given signal with an per-sample error below 0.01%
      for(size_t i=0; i<N/2; ++i){
        o.step(patch);
      }
      REQUIRE(o.getResidualEnergy()/N < 0.0001);
    }
  }
}
