#include "catch.hpp"

// STL INCLUDES

// LOCAL INCLUDES
#include "../include/signal.hpp"
#include "../include/synth.hpp"



////////////////////////////////////////////////////////////////////////////////
/// TESTING THE SYNTH CLASS
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("Testing the AudioSignal default constructor", "[AudioSignal]"){
  for(size_t i=0; i<10000; ++i){
    Chi2Synth s(1000, 440, 1);
    if(i%100==0){std::cout << "times: " << i <<std::endl;}
  }
}

