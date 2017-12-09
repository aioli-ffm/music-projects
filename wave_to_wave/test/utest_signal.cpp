#include "catch.hpp"

// STL INCLUDES
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


const bool kPlot = false;

////////////////////////////////////////////////////////////////////////////////
/// TESTING THE FLOATSIGNAL CLASS
////////////////////////////////////////////////////////////////////////////////


TEST_CASE("Testing plot", "[AudioSignal]"){
  auto sin_gen2 = [](const long int x)->float{return 2.34f*std::sin(0.001f*x);};
  auto spiral_gen = [](const long int x)->std::complex<float>{return 2.34f*exp(std::complex<float>(0, 0.001f*x));};
  FloatSignal fss(sin_gen2, 44100);
  ComplexSignal css(spiral_gen, 44100);

  FloatSignal pop("pop.wav");
  FloatSignal pop_short(pop.getData(), 22050);
  if(kPlot){
    fss.plot("a plot of 2.34*sin(x) between 0s and 1s", 44100);
    css.plot("a plot of 2.34*e**(ix) between 0s and 1s", 44100);
    pop_short.plot("pop song", 22050, 5);
  }
}


TEST_CASE("Testing the AudioSignal default constructor", "[AudioSignal]"){
  AudioSignal<int> as;
  REQUIRE(as.getSize() == 0);
  REQUIRE(as.getData() == nullptr);
}

TEST_CASE("Testing the FloatSignal class", "[AudioSignal, FloatSignal]"){
  // make and fill a float array for testing purposes (gen.constr. is preferred)
  const size_t kTestSize = 27;
  float arr[kTestSize];
  for(size_t i=0; i<kTestSize; ++i){arr[i] = 2*i;}

  // use different constructors:
  FloatSignal fs1(kTestSize);
  FloatSignal fs2(arr, kTestSize);
  FloatSignal fs3(arr, kTestSize, 10, 10);
  FloatSignal fs4(fs2.getData(), fs2.getSize());

  SECTION("FloatSignal constructors, initialization, padding and operator[]"){
    for(size_t i=0; i<kTestSize; ++i){
      REQUIRE(fs1[i] == 0); // check that fs1 is initialized with zeros
      REQUIRE(fs2[i] == arr[i]); // check all values arr and fs2 are same
      REQUIRE(fs2[i] == fs4[i]); // check all values fs2 and fs4 are same
      REQUIRE(fs3[i+10] == arr[i]); // check all values arr and fs3 are same
    }
    // check padding:
    float sum_pad = 0;
    for(size_t i=0; i<10; ++i){
      sum_pad += abs(fs3[i]);
      sum_pad += abs(fs3[10+kTestSize+i]);
    }
    REQUIRE(sum_pad == 0); // sum of padded cells is zero
  }

  SECTION("FloatSignal generative constructor"){
    auto lambda = [](const long int x)->float {return 1.23*(x-50);};
    FloatSignal fs(lambda, kTestSize);
    for(size_t i=0; i<kTestSize; ++i){
      REQUIRE(fs[i] == lambda(i));
    }
  }

  SECTION("Interaction between FloatSignal and libsndfile"){
    const std::string kWavPath = "/tmp/test.wav";
    const size_t kWavSize = 10000;
    // make a sin signal and save it to kWavPath (FORMAT_FLOAT to allow testing)
    auto sin_gen = [](const long int x)->float{0.001*std::sin(0.1f*(float)x);};
    FloatSignal fs(sin_gen, kWavSize);
    fs.toWav(kWavPath, 44100, SF_FORMAT_FLOAT);
    // load the wave from kWavPath and check that contents are still the same
    FloatSignal loaded(kWavPath);
    REQUIRE(loaded.getSize() == kWavSize);
    for(size_t i=0; i<kWavSize; ++i){
      REQUIRE(loaded[i] == sin_gen(i));
    }
  }

  SECTION("FloatSignal is iterable, and can be used by IterableToString"){
    // make a STL iterable and create a string from it and fs2
    std::vector<float> v(arr, arr+kTestSize);
    std::string v_str = IterableToString(v);
    std::string fs2_str = IterableToString(fs2);
    // check both IterToString are equal
    REQUIRE(v_str.compare(fs2_str) == 0);
    // check that foreach loops work with FloatSignal, as well as STL dotprod:
    float dotprod = std::inner_product(fs2.begin(), fs2.end(), fs2.begin(), 0);
    float check_dotprod = 0;
    for(auto& f : fs2){
      check_dotprod += f*f;
    }
    REQUIRE(dotprod == check_dotprod);
  }

  SECTION("FloatSignal getters and setters"){
    // getSize
    REQUIRE(fs1.getSize() == kTestSize);
    REQUIRE(fs2.getSize() == kTestSize);
    REQUIRE(fs3.getSize() == kTestSize+20);
    // getData
    REQUIRE(fs1.getData() != arr); // constructor makes a copy of arr
    REQUIRE(fs2.getData() != fs1.getData()); // constructor makes a copy of arr
    REQUIRE(fs2.getData() != fs4.getData()); // also copy-constructor
    REQUIRE(*(fs2.getData()+1) == fs2[1]);
  }

  SECTION("FloatSignal-to-constant compound assignment operators"){
    SECTION("+= and -= operators"){
      float kOffset = 1000.1234;
      fs2 += kOffset;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == arr[i]+kOffset);
      }
      fs2 -= kOffset;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == arr[i]);
      }
    }
    SECTION("*= operator"){
      float kOffset = 12.345;
      fs2 *= kOffset;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == arr[i]*kOffset);
      }
    }
  }

  SECTION("FloatSignal-to-FloatSignal operators"){
    SECTION("+= and -+, with itself and with another:"){
      FloatSignal other(arr, kTestSize);
      // this two should do nothing (since offset is out of bounds)
      fs2.addSignal(other, -44100*300);
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == arr[i]);
      }
      fs2.addSignal(other, 44100*300);
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == arr[i]);
      }
      // test adding and subtracting for offset=0
      fs2.addSignal(other);
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == 2*arr[i]);
      }
      fs2.subtractSignal(other);
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == arr[i]);
      }
      // test the same but with the operators
      fs2 += other;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == 2*arr[i]);
      }
      fs2 -= other;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == arr[i]);
      }
      // test the same but with itself
      fs2 += fs2;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == 2*arr[i]);
      }
      fs2 -= fs2;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == 0);
      }
      // test with some different offset
      fs2 *= 0;
      fs2.addSignal(other, -5);
      for(size_t i=0; i<kTestSize-5; ++i){
        REQUIRE(fs2[i] == arr[i+5]);
      }
    }
    SECTION("*=, brief test assuming most of += tests hold here"){
      FloatSignal other(arr, kTestSize);
      // test "restart" fs2
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == arr[i]);
      }
      // multiply fs2 by other, no offset
      fs2 *= other;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(fs2[i] == arr[i]*arr[i]);
      }
      // multiply other by itself, no offset
      other *= other;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(other[i] == arr[i]*arr[i]);
      }
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// TESTING THE COMPLEXSIGNAL CLASS
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("Testing the ComplexSignal class", "[AudioSignal, ComplexSignal]"){
  // to avoid verbosity:
  std::complex<float> kComplexZero(0,0);
  // make and fill a complex array for testing purposes
  const size_t kTestSize = 27;
  std::complex<float> arr[kTestSize];
  for(size_t i=0; i<kTestSize; ++i){arr[i] = std::complex<float>(10*i, 10*i);}
  // float f_arr[kTestSize];
  // for(size_t i=0; i<kTestSize; ++i){f_arr[i] = 2*i;}

  // use different constructors:
  ComplexSignal cs1(kTestSize);
  ComplexSignal cs2(arr, kTestSize);
  ComplexSignal cs3(arr, kTestSize, 10, 10);
  ComplexSignal cs4(cs2.getData(), cs2.getSize());

  SECTION("ComplexSignal constructors, initialization, padding and operator[]"){
    for(size_t i=0; i<kTestSize; ++i){
      REQUIRE(cs1[i] == kComplexZero);  // check that cs1 is initialized with 0s
      REQUIRE(cs2[i] == arr[i]); // check all values arr and cs2 are same
      REQUIRE(cs2[i] == cs4[i]); // check all values cs2 and cs4 are same
      REQUIRE(cs3[i+10] == arr[i]); // check all values arr and cs3 are same
    }
    // check padding:
    std::complex<float> sum_pad(0,0);
    for(size_t i=0; i<10; ++i){
      sum_pad += cs3[i];
      sum_pad += cs3[10+kTestSize+i];
    }
    REQUIRE(sum_pad == kComplexZero); // sum of padded cells is zero
  }

  SECTION("ComplexSignal generative constructor"){
    auto lambda = [](const long int x) {return std::complex<float>(6*x,-7);};
    ComplexSignal cs(lambda, kTestSize);
    for(size_t i=0; i<kTestSize; ++i){
      REQUIRE(cs[i] == lambda(i));
    }
  }

  SECTION("ComplexSignal is iterable, and can be used by IterableToString"){
    // make a STL iterable and create a string from it and fs2
    std::vector<std::complex<float> > v(arr, arr+kTestSize);
    std::string v_str = IterableToString(v);
    std::string cs2_str = IterableToString(cs2);
    // check both IterToString are equal
    REQUIRE(v_str.compare(cs2_str) == 0);
    // check that foreach loops work with FloatSignal, as well as STL dotprod:
    std::complex<float> dotprod = std::inner_product(cs2.begin(), cs2.end(),
                                                     cs2.begin(), kComplexZero);
    std::complex<float> check_dotprod = kComplexZero;
    for(auto& z : cs2){
      check_dotprod += z*z;
    }
    REQUIRE(dotprod == check_dotprod);
  }

  SECTION("ComplexSignal getters and setters"){
    // getSize
    REQUIRE(cs1.getSize() == kTestSize);
    REQUIRE(cs2.getSize() == kTestSize);
    REQUIRE(cs2.getSize() == cs4.getSize());
    REQUIRE(cs3.getSize() == kTestSize+20);
    // getData
    REQUIRE(cs1.getData() != arr); // constructor makes a copy of arr
    REQUIRE(cs2.getData() != cs1.getData()); // constructor makes a copy of arr
    REQUIRE(*(cs2.getData()+1) == cs2[1]);
    REQUIRE(cs2.getData() != cs4.getData());
  }


  SECTION("ComplexSignal conjugate method"){
    ComplexSignal cs_conj(arr, kTestSize);
    cs_conj.conjugate();
    for(size_t i=0; i<kTestSize; ++i){
      REQUIRE(cs_conj[i] == std::conj(cs2[i]));
    }
    cs_conj.conjugate();
    for(size_t i=0; i<kTestSize; ++i){
      REQUIRE(cs_conj[i] == cs2[i]);
    }
  }

  SECTION("ComplexSignal-to-constant compound assignment operators"){
    SECTION("+= and -= operators"){
      std::complex<float> kOffset(1000.1234, 1000.1234);
      cs2 += kOffset;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]+kOffset);
      }
      cs2 -= kOffset;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]);
      }
    }
    SECTION("*= operator"){
      std::complex<float> kOffset(1000, 1000);
      cs2 *= kOffset;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]*kOffset);
      }
    }
  }

  SECTION("ComplexSignal-to-ComplexSignal operators"){
    SECTION("+= and -+, with itself and with another:"){
      ComplexSignal other(arr, kTestSize);
      // this two should do nothing (since offset is out of bounds)
      cs2.addSignal(other, -44100*300);
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]);
      }
      cs2.addSignal(other, 44100*300);
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]);
      }
      // test adding and subtracting for offset=0
      cs2.addSignal(other);
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]+arr[i]);
      }
      cs2.subtractSignal(other);
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]);
      }
      // test the same but with the operators
      cs2 += other;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]+arr[i]);
      }
      cs2 -= other;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]);
      }
      // test the same but with itself
      cs2 += cs2;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]+arr[i]);
      }
      cs2 -= cs2;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == kComplexZero);
      }
      // test with some different offset
      cs2 *= 0;
      cs2.addSignal(other, -5);
      for(size_t i=0; i<kTestSize-5; ++i){
        REQUIRE(cs2[i] == arr[i+5]);
      }
    }
    SECTION("*=, brief test assuming most of += tests hold here"){
      ComplexSignal other(arr, kTestSize);
      // test "restart" cs2
      cs2 *= 0;
      cs2 += other;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]);
      }
      // multiply cs2 by other, no offset
      cs2 *= other;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(cs2[i] == arr[i]*arr[i]);
      }
      // multiply other by itself, no offset
      other *= other;
      for(size_t i=0; i<kTestSize; ++i){
        REQUIRE(other[i] == arr[i]*arr[i]);
      }
    }
  }
}
