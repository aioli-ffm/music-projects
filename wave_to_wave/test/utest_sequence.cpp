#include "catch.hpp"

// LOCAL INCLUDE
#include "../include/w2w.hpp"


////////////////////////////////////////////////////////////////////////////////
/// TESTING THE CHI2OPTIMIZER CLASS
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("test Chi2Optimizer", "[Sequence]"){
  RandGen rand;
  // test that with bigger patch crashes.
  SECTION("test"){
    REQUIRE(1==1);
    REQUIRE_THROWS_AS(throw std::runtime_error(""), std::runtime_error);
  }
}
