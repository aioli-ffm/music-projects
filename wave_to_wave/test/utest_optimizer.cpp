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

// CHUNKSIZE=2048
// 2000, 9 se cuelga en step 976
// 200, 200 se cuelga en el step 8
// 100, 100 se cuelga en el step 18, tambien para diferente sig
// 50, 50 se cuelga en step 39
// 49, 49 se cuelga en step 40
// 48, 48 se cuelga en step 41
// 47, 47 se cuelga en step 42
// 46, 46 se cuelga en step 43
// 45, 45 aparentemente no se cuelga

// case study: 47, 47 que se cuelga en 42 inclusive con valor 30
// si cambiamos chunksize a 4096, no se cuelga
// si cambiamos chunksize a 1, se cuelga 2+4+2+12+RESTO o sea todo el rato
//    pero si cambiamos #m=1, no se cuelga nada.
//    si cambiamos #m=2, se cuelga en step 44, value=5
//    si cambiamos #m=3, se cuelga en step 42, value=30
//    si cambiamos #m=4, se cuelga en step 42, value=30
//    si cambiamos #m=5, se cuelga en step 38, value=204
//    si cambiamos #m=6, se cuelga en step 38, value=204
//    si cambiamos #m=7, se cuelga en step 38, value=204
//    si cambiamos #m=8, se cuelga en step 38, value=204
//    si cambiamos #m=9, se cuelga en step 30, value=1496

// observacion: el bug afecta a la pipeline, en concreto va con el chunksize.
// habia observado que los valores de "prepadding" se llenan de garbage debido
// a la conv. circular. Asegurarse de q la funcion de update rellena el padding.

TEST_CASE("test optimizer", "[optimizer]"){
  const size_t N = 2001;
  size_t downsampling = 1;
  FloatSignal aaa([](long int x){return x+1;}, 2000);
  //FloatSignal aaa("country.wav");
  FloatSignal bbb([](long int x){return (x== 0);}, 3);
  auto opt_criterium = [](FloatSignal &fs){
    std::map<long int, float> result;
    float* abs_max = std::max_element(fs.begin(), fs.end());//, abs_compare);
    result.insert(std::pair<long int, float>(std::distance(fs.begin(), abs_max),
                                             *abs_max));
    return result;
  };
  WavToWavOptimizer o(aaa, opt_criterium);
  std::unique_ptr<FloatSignal> fs;
  for(size_t i=0; i<N; ++i){
    o.step(bbb);
    if(i%10==0){
      o.printResidualEnergy(i);
      // o.printResidual(i);
    }
  }


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
