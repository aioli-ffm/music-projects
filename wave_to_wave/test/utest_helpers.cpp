#include "catch.hpp"

// LOCAL INCLUDE
#include "../include/w2w.hpp"



////////////////////////////////////////////////////////////////////////////////
/// TESTER GLOBALS
////////////////////////////////////////////////////////////////////////////////

const bool kPlot = false; // true if you wish the tests to generate plots

// Implementation of Chi2 using built-in gamma. Slower than the one implemented
// in helpers.hpp using lanczos approx, but used for testing
static double Chi2Test(double x, double df){
  double df_half = 0.5*df;
  double x_half = 0.5*x;
  return 0.5 * pow(x_half, df_half-1) / exp(x_half) / std::tgamma(df_half);
}

////////////////////////////////////////////////////////////////////////////////
/// TESTING THE IMPLEMENTED MATH
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("Testing freq<->midi conversion", "[helpers, math]"){
  REQUIRE(Approx(FreqToMidi(440)) == 69);
  REQUIRE(Approx(FreqToMidi(880)) == 81);
  REQUIRE(Approx(MidiToFreq(81)) == 880);
  REQUIRE(Approx(MidiToFreq(69)) == 440);
}

TEST_CASE("Pow2Ceil", "[helpers, math]"){
  size_t outputs[18]{0,1,2,4,4,8,8,8,8,16,16,16,16,16,16,16,16,32};
  for(size_t i=0; i<18; ++i){
    REQUIRE(Pow2Ceil(i) == outputs[i]);
  }
}

TEST_CASE("LinInterp", "[helpers, math]"){
  for(float i=0; i<=10; i+=0.5){
    REQUIRE(LinInterp(0, 10, i/10) == i);
  }
}

TEST_CASE("Factorial", "[helpers]"){
  size_t outputs[11]{1,1,2,6,24,120,720,5040,40320,362880,3628800};
  for(size_t i=0; i<11; ++i){
    REQUIRE(Factorial(i) == outputs[i]);
  }
}

TEST_CASE("DoubleFactorial", "[helpers]"){
  size_t outputs[14]{1,1,1,2,3,8,15,48,105,384,945,3840,10395,46080};
  for(size_t i=-1; i<13; ++i){
    REQUIRE(DoubleFactorial(i) == outputs[i]);
  }
}

TEST_CASE("Gamma double and size_t input", "[helpers]"){
  // first compare the plain factorial version with the lanczos for ints
  size_t outputs[10]{1,2,6,24,120,720,5040,40320,362880,3628800};
  for(size_t i=0; i<10; ++i){
    REQUIRE(Approx(Gamma((double(i+2)))) == outputs[i]);
    REQUIRE(Approx(Gamma((size_t)(i+2))) == outputs[i]);
  }
  // then compare the lanczos with the STL gamma
  for(double i=1; i<20; i+=0.1){
    REQUIRE(Approx(Gamma(i)) == std::tgamma(i));
  }
  // optional: plot
  if(kPlot){
    FloatSignal gamma_sig([](const size_t x)->float{
        return (float)Gamma(0.1f+0.001*x);}, 1000*5);
    gamma_sig.plot("gamma",1000, 0.25);
  }
}


TEST_CASE("Chi2(double, double)", "[helpers]"){
  const size_t kMaxRange = 20;
  const size_t kResolution = 100;
  const size_t kNumPoints = kMaxRange*kResolution;
  const double kEps = 2.2204460492503131e-16;
  // Testing Chi2(double, double) against Chi2Test(double, double)
  for(float k : {1.5f, 1.7f, 1.9f, 2.0f,3.0f, 4.0f, 6.0f, 9.0f}){
    FloatSignal chi2_sig([=](const size_t x)->float{
        return (float)Chi2(kEps+1.0*x/kResolution, k);}, kNumPoints);
    FloatSignal chi2test_sig([=](const size_t x)->float{
        return (float)Chi2Test(kEps+1.0*x/kResolution, k);}, kNumPoints);
    for(size_t i=0; i<kNumPoints; ++i){
      REQUIRE(chi2_sig[i] == chi2test_sig[i]);
    }
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










  // // speed measurements:
  // for(size_t i=0; i<1000*20; ++i){
  //   FloatSignal chi2_sig([=](const size_t x)->float{
  //       return (float)Chi2(1.0/kResolution*x, 1.234);}, kNumPoints);
  //   if(i%1000==0){
  //     std::cout << "Chi float" << i << std::endl;
  //   }
  // }
  // size_t kk = 3;
  // for(size_t i=0; i<1000*20; ++i){
  //   FloatSignal chi2_sig([=](const size_t x)->float{
  //       return (float)Chi2(1.0/kResolution*x, kk);}, kNumPoints);
  //   if(i%1000==0){
  //     std::cout << "Chi int" << i << std::endl;
  //   }
  // }

























// TEST_CASE("Test the RandGen methods", "[RandGen]"){
//   // For this tests, the chi2 method is used: first, sample from the
//   // distributions and fill the histograms a and b (just different datatypes).
//   // Then, calculate the chi2 score by the sum of (observed-expected)^2/expected
//   // for each bin. The likelihood of the observed distribution NOT being the
//   // expected is the integral from zero to the chi2 score on the chi2 curve
//   // for degrees_of_freedom = num_of_bins-1.
//   // **NOTE: the expected chi2score for a unif with B buckets is (B-1).
//   //   And the expected variance is 2(B-1). Therefore it is unfeasible to
//   // CONFIRM a randgen just by this means, since the p-value will fluctuate
//   // largely around an already ambiguous p-value.
//   RandGen rand;
//   const size_t kNumBuckets = 50;
//   const size_t kDegreesFreedom = kNumBuckets-1;
//   const size_t kNumSamples = kNumBuckets*1000;

//   FloatSignal histogram_a(kNumBuckets);
//   FloatSignal histogram_b(kNumBuckets);
//   SECTION("Test unifInt for different types"){
//     // fill the histograms a and b
//     for(size_t x=0; x<kNumSamples; ++x){
//       int r1 = rand.unifInt(0, (int)kNumBuckets-1);
//       size_t r2 = rand.unifInt((size_t)0, kNumBuckets-1);
//       histogram_a[r1] += 1;
//       histogram_b[r2] += 1;
//     }
//     // calculate the chi2 score for every histogram
//     const double kExpectedSamplesPerBucket = ((double)kNumSamples)/kNumBuckets;
//     histogram_a -= kExpectedSamplesPerBucket;
//     histogram_b -= kExpectedSamplesPerBucket;
//     double chi2value_a = Energy(histogram_a.begin(), histogram_a.end()) /
//       kExpectedSamplesPerBucket;
//     double chi2value_b = Energy(histogram_b.begin(), histogram_b.end()) /
//       kExpectedSamplesPerBucket;
//     // calculate the probability that the observed dist matched the expected one
//     double chi2integral_a = 0;
//     double chi2integral_b = 0;
//     for(double x=0; x<chi2value_a; x+=0.001){
//       chi2integral_a += Chi2(x, kDegreesFreedom);
//     }
//     for(double x=0; x<chi2value_b; x+=0.001){
//       chi2integral_b += Chi2(x, kDegreesFreedom);
//     }
//     chi2integral_a /= 1000.0;
//     chi2integral_b /= 1000.0;
//     // require the probability to be ??% (this is the problem, note above)
//     REQUIRE(1.0-chi2integral_a > 0.90);
//     REQUIRE(1.0-chi2integral_b > 0.90);
//   }
// }
