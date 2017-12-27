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
  
  FloatSignal fs(samplerate*10);
  double xx=0;
  for(size_t ii=0; ii<5; ++ii, xx+=0.2){
    std::cout << "<<" << xx << std::endl;
    Chi2Synth ss(serv, samplerate, 442.0, samplerate, xx);
    // fs.addMultipliedSignal(ss, 1, ii*samplerate);

    // std::cout << ii << ">>>>  " << *std::max_element(ss.begin(), ss.end()) << std::endl;
    // std::cout <<ii << ">>>>  " << *std::min_element(ss.begin(), ss.end()) << std::endl;
    ss.plot("hello", samplerate);
    
  }
  // fs.toWav("/tmp/pinheiro.wav", samplerate);
  // fs.plot("viva aznar", 44100);



  // FloatSignal* ss = serv.get(samplerate, 30.0, 2);
  // ss->plot("hello", samplerate);

  // for(double x=2.0; x<5; x+=0.01){
  //   FloatSignal* ss = serv.get(samplerate, 30.0, x);
  //   std::cout  << ">>>>  " << *std::min_element(ss->begin(), ss->end(),
  //                                               abs_compare<float>) << std::endl;
  // }

}





// DF: 2.23241
// 0>>>>  0.366611
// 0>>>>  -0.366538
// DF: 2.50243
// 1>>>>  0.303226
// 1>>>>  -0.303207
// DF: 2.81614
// 2>>>>  0.259985
// 2>>>>  -0.259937
// DF: 3.18063
// 3>>>>  0.227435
// 3>>>>  -9.57116e+27
// DF: 3.60411
// 4>>>>  0.201514
// 4>>>>  -1.46024e+23
// DF: 4.09611
// 5>>>>  0.180293
// 5>>>>  -0.180305
// DF: 4.66774
// 6>>>>  0.162412
// 6>>>>  -0.162405
// DF: 5.33188
// 7>>>>  0.147074
// 7>>>>  -0.147076
// DF: 6.1035
// 8>>>>  0.133725
// 8>>>>  -0.133733
// DF: 7
// 9>>>>  0.122022
// 9>>>>  -0.122017
// Passed 1 test case (no assertions).
