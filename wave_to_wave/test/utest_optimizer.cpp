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


TEST_CASE("test optimizer", "[optimizer]"){
  const size_t N = 4100;
  size_t downsampling = 1;
  FloatSignal aaa([](long int x){return x+1;}, 4100);
  //FloatSignal aaa("country.wav");
  FloatSignal bbb([](long int x){return -10.12345*(x==0);}, 1230);
  auto opt_criterium = [](FloatSignal &fs){
    std::map<long int, float> result;
    float* abs_max = std::max_element(fs.begin(), fs.end(), abs_compare<float>);
    result.insert(std::pair<long int, float>(std::distance(fs.begin(), abs_max),
                                             *abs_max));
    return result;
  };
  WavToWavOptimizer o(aaa, opt_criterium);
  std::unique_ptr<FloatSignal> fs;
  for(size_t i=0; i<N; ++i){
    o.step(bbb);
    if(i%100==0){
      o.printResidualEnergy(i);
      // o.printResidual(i);
    }
  }
  // o.printResidual(44100);
  o.printResidualEnergy(44100);


  // SECTION("test speed"){
  //   const size_t N = 1000*1000;
  //   size_t downsampling = 100;
  //   FloatSignal aaa([](long int x){return x%2 == 0;}, 44100*60/downsampling);
  //   FloatSignal bbb([](long int x){return x%3 == 0;}, 44100*3/downsampling);
  //   OverlapSaveConvolver xxx(aaa, bbb, true, true, 2048);
  //   for(size_t i=0; i<N; ++i){
  //     if(i%5000==0){std::cout << "i was " << i << std::endl;}
  //     xxx.updateSignal(aaa);
  //     xxx.updatePatch(bbb);
  //     xxx.forwardSignal();
  //     xxx.forwardPatch();
  //     xxx.spectralConv();
  //     xxx.backwardSignal();
  //     FloatSignal ftest(aaa.getSize()+bbb.getSize()-1);
  //     xxx.extractConvolvedTo(ftest);
  //   }
  // }


}
