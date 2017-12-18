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

std::vector<std::pair<long int, float> > SingleMaxCriterium(FloatSignal &fs,
                                                              size_t patch_sz){
  std::vector<std::pair<long int, float> > result;
  float* abs_max = std::max_element(fs.begin(), fs.end(), abs_compare<float>);
  result.push_back(std::pair<long int,float>((abs_max-fs.begin())-patch_sz+1,
                                             *abs_max));
  return result;
};


// a single run of this function is much slower than SingleMaxCriterium, but
//
std::vector<std::pair<long int, float> > PopulateMaxCriterium(FloatSignal &fs,
                                                     size_t patch_sz){
  // fill result with indexes and sort them by descending fs[idx]
  std::vector<long int> sorted_idxs(fs.getSize());
  std::vector<long int>::iterator beg = sorted_idxs.begin();
  std::vector<long int>::iterator end = sorted_idxs.end();
  std::iota(beg, end, 0);
  std::sort(beg, end, [&fs](const long int a, const long int b)
            {return std::abs(fs[a]) > std::abs(fs[b]);});
  std::vector<std::pair<long int, float> > result;
  const long int idx = sorted_idxs[0];
  result.push_back(std::pair<long int,float>(idx-patch_sz+1,fs[idx]));
  return result;
};

TEST_CASE("test optimizer", "[optimizer]"){


  // SECTION("test against a redundant signal with a redundant delta"){
  //   size_t N = 44100*6;
  //   FloatSignal sig([](long int x){return x-(x%2==1);}, N);
  //   FloatSignal patch([](long int x){return x<2;}, N/10);
  //   const float kPatchEnergy = std::inner_product(patch.begin(), patch.end(),
  //                                                 patch.begin(),0.0f);
  //   WavToWavOptimizer o(sig, AbsoluteMaxCriterium);
  //   // After exactly N/2 iterations, the delta should optimize the given
  //   // signal with an per-sample error below 0.01%
  //   for(size_t i=1, max=N/2; i<=max; ++i){
  //     o.step(patch, kPatchEnergy, "myPatch 1 2 3");
  //     if(i%1000==0){
  //       std::cout <<"i="<< i <<"\tenergy="<< o.getResidualEnergy() << std::endl;
  //     }
  //   }
  //   REQUIRE(o.getResidualEnergy()/N < 0.0001);
  //   o.exportTxtSequence("/tmp/seq.txt");
  // }

  SECTION("test against a simple unit delta with different sizes"){
    for(size_t N=1000; N<10000; N+=1000){
      for(size_t M=100; M<=N/3; M+=N/7){
        std::cout << "testing for N="<<N<<", M="<<M<<std::endl;
        FloatSignal sig([](long int x){return x+1;}, N);
        FloatSignal patch([](long int x){return x==0;}, M);
        const float kPatchEnergy = std::inner_product(patch.begin(),patch.end(),
                                                      patch.begin(),0.0f);
        WavToWavOptimizer o(sig, [=](FloatSignal &fs){
            return PopulateMaxCriterium(fs, M);});
        // After exactly N iterations, the delta should optimize the given
        // signal with an per-sample error below 0.01%
        for(size_t i=0; i<N; ++i){
          o.step(patch, kPatchEnergy, "myPatch 1 2 3");
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
        const float kPatchEnergy = std::inner_product(patch.begin(),patch.end(),
                                                      patch.begin(),0.0f);
        WavToWavOptimizer o(sig, [=](FloatSignal &fs){
            return PopulateMaxCriterium(fs, M);});
        // After exactly N/2 iterations, the delta should optimize the given
        // signal with an per-sample error below 0.01%
        for(size_t i=0; i<N/2; ++i){
          o.step(patch, kPatchEnergy, "myPatch 1 2 3");
        }
        REQUIRE(o.getResidualEnergy()/N < 0.0001);
      }
    }
  }

   SECTION("test against a negative, multiplied, delayed, redundant delta"){
    const size_t N = 23456;
    const size_t M = 737;
    const size_t D = 100;
    for(size_t d=0; d<=D; d+=13){
      std::cout << "testing for delay="<<d<<std::endl;
      FloatSignal sig([](long int x){return x-(x%2==1);}, N); //0,0,2,2,..
      FloatSignal patch([=](long int x){return -12.34*(x==d||x==(d+1));}, M);
      const float kPatchEnergy = std::inner_product(patch.begin(),patch.end(),
                                                    patch.begin(),0.0f);
      WavToWavOptimizer o(sig, [=](FloatSignal &fs){
          return PopulateMaxCriterium(fs, M);});
      // After exactly N/2 iterations, the delayed delta should optimize the
      // given signal with an per-sample error below 0.01%
      for(size_t i=0; i<N/2; ++i){
        o.step(patch, kPatchEnergy, "myPatch 1 2 3");
      }
      REQUIRE(o.getResidualEnergy()/N < 0.0001);
    }
   }
}
