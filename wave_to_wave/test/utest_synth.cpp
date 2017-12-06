#include "catch.hpp"

// STL INCLUDES

// LOCAL INCLUDES
#include "../include/signal.hpp"
#include "../include/synth.hpp"



////////////////////////////////////////////////////////////////////////////////
/// TESTING THE SYNTH CLASS
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("Testing the AudioSignal default constructor", "[AudioSignal]"){
  Synth s;
  REQUIRE(1 == 1);
}

