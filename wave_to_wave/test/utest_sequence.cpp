#include "catch.hpp"

// LOCAL INCLUDE
#include "../include/w2w.hpp"


////////////////////////////////////////////////////////////////////////////////
/// 
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("testing the Sequence class", "[Sequence]"){
  SECTION("test empty constructor"){
    Sequence seq("test_sequence.txt");
    std::cout << seq.asString({{1,"hello"}, {3,"bye"}}) << std::endl;
  }
  SECTION("test"){
    REQUIRE(1==1);
    REQUIRE_THROWS_AS(throw std::runtime_error(""), std::runtime_error);
  }
    
}


