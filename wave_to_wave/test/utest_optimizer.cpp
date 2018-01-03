#include "catch.hpp"

// STL INCLUDES
#include<string>
#include<vector>
#include<list>
#include<complex>
#include<algorithm>
#include <random>
// SYSTEM-INSTALLED LIBRARIES
#include <fftw3.h>
// LOCAL INCLUDES
#include "../include/helpers.hpp"
#include "../include/signal.hpp"
#include "../include/optimizer.hpp"

//// GLOBALS:
std::random_device rand_device; // seed
std::mt19937 rand_engine(rand_device()); // engine

////////////////////////////////////////////////////////////////////////////////
/// TESTING THE WAVTOWAVOPTIMIZER CLASS
////////////////////////////////////////////////////////////////////////////////

std::vector<std::pair<long int, float> > SingleMaxCriterium(FloatSignal &fs,
                                                              size_t _){
  std::vector<std::pair<long int, float> > result;
  float* abs_max = std::max_element(fs.begin(), fs.end(), abs_compare<float>);
  result.push_back(std::pair<long int,float>(abs_max-fs.begin(), *abs_max));
  return result;
};


//
std::vector<std::pair<long int, float> > PopulateCriterium(FloatSignal &fs,
                                                           size_t patch_sz,
                                                           float eps=0.001){
  std::vector<std::pair<long int, float> > result;
  float* fs_begin = fs.begin();
  std::uniform_int_distribution<size_t> uni_dist(0,patch_sz);
  float* it = fs_begin+uni_dist(rand_engine);
  float* end = fs.end();
  for(; it<end; it+=patch_sz){
    // std::cout << "  " << it-fs_begin;
    it = std::max_element(it, std::min(end, it+patch_sz), abs_compare<float>);
    // std::cout << " " << it-fs_begin << " ||";
    if(*it>eps){
      result.push_back(std::pair<long int, float>(it-fs_begin, *it));
    }
  }
  // for_each(result.begin(), result.end(), [](auto x){
  //     std::cout << std::get<0>(x) << " ";});
  // fs.print("fs");
  return result;
};





// TEST_CASE("test optimizer with singlemax criterium", "[optimizer]"){

//   SECTION("test against a simple unit delta with different sizes"){
//     for(size_t N=1000; N<10000; N+=1000){
//       for(size_t M=100; M<=N/3; M+=N/7){
//         std::cout << "testing for N="<<N<<", M="<<M<<std::endl;
//         FloatSignal sig([](long int x){return x+1;}, N);
//         FloatSignal patch([](long int x){return x==0;}, M);
//         const float kPatchEnergy = std::inner_product(patch.begin(),patch.end(),
//                                                       patch.begin(),0.0f);
//         WavToWavOptimizer o(sig);
//         // After exactly N iterations, the delta should optimize the given
//         // signal with an per-sample error below 0.01%
//         for(size_t i=0; i<N; ++i){
//           o.step(patch, kPatchEnergy, [=](FloatSignal &fs){
//               return SingleMaxCriterium(fs, M);}, "test");
//         }
//         REQUIRE(o.getResidualEnergy()/N < 0.0001);
//       }
//     }
//   }

//   SECTION("test against a redundant signal with a redundant delta"){
//     for(size_t N=1000; N<10000; N+=1000){
//       for(size_t M=100; M<=N/3; M+=N/7){
//         std::cout << "testing for N="<<N<<", M="<<M<<std::endl;
//         FloatSignal sig([](long int x){return x-(x%2==1);}, N); //0,0,2,2,...
//         FloatSignal patch([](long int x){return x<2;}, M);
//         const float kPatchEnergy = std::inner_product(patch.begin(),patch.end(),
//                                                       patch.begin(),0.0f);
//         WavToWavOptimizer o(sig);
//         // After exactly N/2 iterations, the delta should optimize the given
//         // signal with an per-sample error below 0.01%
//         for(size_t i=0; i<N/2; ++i){
//           o.step(patch, kPatchEnergy, [=](FloatSignal &fs){
//               return SingleMaxCriterium(fs, M);}, "myPatch 1 2 3");
//         }
//         REQUIRE(o.getResidualEnergy()/N < 0.0001);
//       }
//     }
//   }

//    SECTION("test against a negative, multiplied, delayed, redundant delta"){
//     const size_t N = 23456;
//     const size_t M = 737;
//     const size_t D = 100;
//     for(size_t d=0; d<=D; d+=13){
//       std::cout << "testing for delay="<<d<<std::endl;
//       FloatSignal sig([](long int x){return x-(x%2==1);}, N); //0,0,2,2,..
//       FloatSignal patch([=](long int x){return -12.34*(x==d||x==(d+1));}, M);
//       const float kPatchEnergy = std::inner_product(patch.begin(),patch.end(),
//                                                     patch.begin(),0.0f);
//       WavToWavOptimizer o(sig);
//       // After exactly N/2 iterations, the delayed delta should optimize the
//       // given signal with an per-sample error below 0.01%
//       for(size_t i=0; i<N/2; ++i){
//         o.step(patch, kPatchEnergy, [=](FloatSignal &fs){
//             return SingleMaxCriterium(fs, M);}, "myPatch 1 2 3");
//       }
//       REQUIRE(o.getResidualEnergy()/N < 0.0001);
//     }
//    }
// }








TEST_CASE("test optimizer with populate+absmax criterium", "[optimizer]"){

  SECTION("test against a simple unit delta with different sizes 123"){
    for(size_t N=1000; N<10000; N+=1000){
      for(size_t M=100; M<=N/3; M+=N/7){
        std::cout << "testing for N="<<N<<", M="<<M<<std::endl;
        FloatSignal sig([](long int x){return x+1;}, N);
        FloatSignal patch([](long int x){return x==0;}, M);
        const float kPatchEnergy = std::inner_product(patch.begin(),patch.end(),
                                                      patch.begin(),0.0f);
        WavToWavOptimizer o(sig);
        // After below N iterations, the delta should optimize the given
        // signal with a per-sample error below 0.01%
        size_t i = 0;
        while(o.getResidualEnergy()/N >0.0001){
          o.step(patch, kPatchEnergy, [=](FloatSignal &fs){
              return PopulateCriterium(fs, M);}, "test");
          // if(i%50==0){
          //   std::cout << "i=" << i << " residual=" <<
          // o.getResidualEnergy() <<std::endl;//
          // }
          i++;
        }
        // o.getResidual()->r->plot("test123"); //
        REQUIRE(o.getResidualEnergy()/N <= 0.0001);
      }
    }
  }


  // SECTION("test against a simple unit delta with different sizes"){
  //   for(size_t N=1000; N<10000; N+=1000){
  //     for(size_t M=100; M<=N/3; M+=N/7){
  //       std::cout << "testing for N="<<N<<", M="<<M<<std::endl;

  //       // size_t N = 1000;
  //       // size_t M = 2;
  //       FloatSignal sig([](long int x){return x+1;}, N);
  //       FloatSignal patch([](long int x){return x==0;}, M);
  //       const float kPatchEnergy = std::inner_product(patch.begin(), patch.end(),
  //                                                     patch.begin(),0.0f);
  //       WavToWavOptimizer o(sig);
  //       // With the populate criterium, the optimization should be near
  //       // optimal after M+1 iterations:
  //       for(size_t i=0, max=N; i<=max; ++i){
  //         o.step(patch, kPatchEnergy,
  //            [=](FloatSignal &fs){return PopulateCriterium(fs, M);}, "test");
  //         std::cout <<"i="<<i<< " residual="<<o.getResidualEnergy() <<std::endl;
  //       }
  //       o.step(patch, kPatchEnergy,
  //              [=](FloatSignal &fs){return SingleMaxCriterium(fs, M);}, "test");
  //       o.getResidual()->r->print("test123");
  //       REQUIRE(o.getResidualEnergy()/N < 0.0001);
  //     }
  //   }
  // }


}
