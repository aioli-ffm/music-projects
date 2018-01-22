#include "catch.hpp"

// LOCAL INCLUDE
#include "../include/w2w.hpp"


////////////////////////////////////////////////////////////////////////////////
/// 
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("testing the Sequence class", "[Sequence]"){
  SECTION("test empty constructor"){
    Sequence seq;
    Sequence seq2("test_sequence.txt");
    std::cout << seq.asString() << std::endl;
  }
  SECTION("test"){
    REQUIRE(1==1);
    REQUIRE_THROWS_AS(throw std::runtime_error(""), std::runtime_error);
  }
    
}
