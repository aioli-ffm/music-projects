#include "catch.hpp"

// STL INCLUDES

// LOCAL INCLUDES
#include "../include/signal.hpp"
#include "../include/synth.hpp"
#include "../include/convolver.hpp"



////////////////////////////////////////////////////////////////////////////////
/// TESTING THE SYNTH CLASS
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("Testing the AudioSignal default constructor", "[AudioSignal]"){

  // REQUIRE(1==1);
  // Test();
  
  size_t samplerate = 44100;
  double garbage;
  size_t env_ratio_resolution = 100;
  Chi2Server serv;
  
  FloatSignal fs(samplerate*1);
  double xx=0;
  for(size_t ii=0; ii<5; ++ii, xx+=0.1){
    if(ii%100==0){std::cout << ii << ">>>>  "  << std::endl;}
    Chi2Synth ss(serv, 44100, 442.0, samplerate, xx);
    // fs.addMultipliedSignal(ss, 1, ii*samplerate);
    ss.plot("hello", samplerate);
  }
  // fs.toWav("/tmp/pinheiro.wav", samplerate);
  //fs.plot("viva aznar", 44100);



  // FloatSignal* ss = serv.get(samplerate, 30.0, 2);
  // ss->plot("hello", samplerate);

  // for(double x=2.0; x<5; x+=0.01){
  //   FloatSignal* ss = serv.get(samplerate, 30.0, x);
  //   std::cout  << ">>>>  " << *std::min_element(ss->begin(), ss->end(),
  //                                               abs_compare<float>) << std::endl;
  // }

}

