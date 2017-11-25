#include "catch.hpp"

#include "../include/helpers.hpp"

#include<string>
#include<vector>
#include<list>





TEST_CASE("Some test case", "[typecheck]"){


  std::vector<size_t> v{1,2,3,4,5};
  std::list<std::string> l{"hello",  "world"};

  REQUIRE(v[1]*10 == 20);

  SECTION( "basic test" ) {
    v.push_back(6);
    const std::string kVstr = "{1, 2, 3, 4, 5, 6}";
    REQUIRE( IterableToString(v) == kVstr);
    REQUIRE( IterableToString(v.begin(), v.end()) == kVstr);
  }
}
