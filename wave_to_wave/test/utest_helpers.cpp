#include "catch.hpp"

// STL INCLUDES
#include<iostream>
#include<string>
#include<vector>
#include<list>
#include<complex>
#include<numeric>
#include<algorithm>
// LIB INCLUDES
#include<sndfile.h>
// LOCAL INCLUDES
#include "../include/helpers.hpp"
#include "../include/signal.hpp"

#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>

TEST_CASE("Pow2Ceil", "[helpers]"){
  size_t outputs[18]{0,1,2,4,4,8,8,8,8,16,16,16,16,16,16,16,16,32};
  for(size_t i=0; i<18; ++i){
    REQUIRE(Pow2Ceil(i) == outputs[i]);
  }
}


TEST_CASE("Gamma", "[helpers]"){
  size_t outputs[10]{1,2,6,24,120,720,5040,40320,362880,3628800};
  FloatSignal gamma([](const size_t x)->float{
      return (float)Gamma(0.1f+0.001*x);}, 1000*5);
  gamma.plot("gamma",1000, 0.25);
  for(size_t i=0; i<10; ++i){
    std::cout << i << " >> " << Gamma(i+1) << std::endl;
    REQUIRE(Approx(Gamma(i+2)) == outputs[i]);
  }
}

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
    REQUIRE( IterableToString(v.begin(), v.end()) == kVstr);

  }
}
