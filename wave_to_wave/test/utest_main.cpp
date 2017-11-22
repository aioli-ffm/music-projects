#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file

#include<string>
#include<vector>
#include<list>

#include "../third_party/catch.hpp"
#include "../include/helpers.hpp"

// compile with:
// g++ -std=c++11 -Wall -Wextra  utest_main.cpp -o bin/utest && ./bin/utest
// https://stackoverflow.com/questions/4927676/implementing-make-check-or-make-test


TEST_CASE("IterableToString accepts collections and iterators", "[typecheck]"){

  // vector<std::string> c1({"foo", "bar"});
// vector<size_t> c2({1});
// list<double> c3({1,2,3,4,5});
// vector<bool> c4({false, true, false});
// list<int> c5;
// cout << IterableToString({1.23, 4.56, -789.0}) << endl;
// cout << IterableToString(c1) << endl;
// cout << IterableToString({"hello", "hello"}) << endl;
// cout << IterableToString(c2) << endl;
// cout << IterableToString(c3) << endl;
// cout << IterableToString(c4) << endl;
// cout << IterableToString(c5.begin(), c5.end()) << endl;

  std::vector<size_t> v{1,2,3,4,5};
  std::list<std::string> l{"hello",  "world"};

  REQUIRE(v[1]*10 == 20);

  SECTION( "basic test" ) {
    v.push_back(6);
    const std::string kVstr = "{1, 2, 3, 4, 5, 6}";
    REQUIRE( IterableToString(v) == kVstr);
    REQUIRE( IterableToString(v.begin(), v.end()) != kVstr);

  }
  // SECTION( "resizing smaller changes size but not capacity" ) {
  //     v.resize( 0 );

  //     REQUIRE( v.size() == 0 );
  //     REQUIRE( v.capacity() >= 5 );
  // }
  // SECTION( "reserving bigger changes capacity but not size" ) {
  //     v.reserve( 10 );

  //     REQUIRE( v.size() == 5 );
  //     REQUIRE( v.capacity() >= 10 );
  // }
  // SECTION( "reserving smaller does not change size or capacity" ) {
  //     v.reserve( 0 );

  //     REQUIRE( v.size() == 5 );
  //     REQUIRE( v.capacity() >= 5 );
  // }
}





TEST_CASE("CheckAllEqual accepts collections and iterators", "[typecheck]"){

  // vector<size_t> v1({});
// vector<double> v2({123.4, 123.4, 123.4});
// vector<bool> v3({false, false, false});
// vector<size_t> v4({1});
// vector<string> v5({"hello", "hello", "bye"});
// CheckAllEqual({3,3,3,3,3,3,3,3});
// CheckAllEqual(v1);
// CheckAllEqual(v2.begin(), v2.end());
// CheckAllEqual(v3);
// CheckAllEqual(v4);
// CheckAllEqual(v5.begin(), prev(v5.end()));
// CheckAllEqual(v5);


  std::vector<size_t> v{1,2,3,4,5};
  std::list<std::string> l{"hello",  "world"};

  REQUIRE(v[1]*10 == 20);

  SECTION( "basic test" ) {
    v.push_back(6);
    const std::string kVstr = "{1, 2, 3, 4, 5, 6}";
    REQUIRE( IterableToString(v) == kVstr);
    REQUIRE( IterableToString(v.begin(), v.end()) != kVstr);

  }
  // SECTION( "resizing smaller changes size but not capacity" ) {
  //     v.resize( 0 );

  //     REQUIRE( v.size() == 0 );
  //     REQUIRE( v.capacity() >= 5 );
  // }
  // SECTION( "reserving bigger changes capacity but not size" ) {
  //     v.reserve( 10 );

  //     REQUIRE( v.size() == 5 );
  //     REQUIRE( v.capacity() >= 10 );
  // }
  // SECTION( "reserving smaller does not change size or capacity" ) {
  //     v.reserve( 0 );

  //     REQUIRE( v.size() == 5 );
  //     REQUIRE( v.capacity() >= 5 );
  // }
}
