// Copyright (C) 2017 Andres Fernandez (https://github.com/andres-fr)

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA

#ifndef HELPERS_H
#define HELPERS_H

// STL INCLUDES
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <deque>
#include <stdexcept>
#include <math.h>
#include <limits>
#include <algorithm>
#include <random>
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds



////////////////////////////////////////////////////////////////////////////////////////////////////
/// MATH
////////////////////////////////////////////////////////////////////////////////////////////////////

static const double DOUBLE_EPSILON = std::numeric_limits<double>::epsilon();


// This class is a simple wrapper to the random engine of the C++ STL
class RandGen{
private:
  std::random_device device_;  // only used once to initialise (seed) engine
  std::mt19937 generator_; // random-number engine used (Mersenne-Twister in this case)
public:
  explicit RandGen() : generator_(device_()){}
  // T can be any int-like type
  template<typename T>
  T unifInt(T low_included, T high_included){
    std::uniform_int_distribution<T> dist(low_included, high_included);
    return dist(generator_);
  }
  // T can be any float-like type
  template<typename T>
  T unifReal(T low_included, T high_included){
    std::uniform_real_distribution<T> dist(low_included, high_included);
    return dist(generator_);
  }

  // T can be any int-like type. Being A,B~unif(0,1), and C=max(A,B), The PDF of C is f(x)=2x,
  // that is, the histogram of N elements will approx a ramp from f(min)=0 to f(max)=2N/(max-min)
  // If rev=true, C=min(A,B), which will reverse the histogram: f(min)=2N/(max-min), f(max)=0
  template<typename T>
  T rampInt(T low_included, T high_included, bool rev=false){
    std::uniform_int_distribution<T> dist(low_included, high_included);
    T a = dist(generator_);
    T b = dist(generator_);
    return (rev)? std::min(a,b) : std::max(a,b);
  }

  // T can be any float-like type. See rampInt explanation
  template<typename T>
  T rampReal(T low_included, T high_included, bool rev=false){
    std::uniform_real_distribution<T> dist(low_included, high_included);
    T a = dist(generator_);
    T b = dist(generator_);
    return (rev)? std::min(a,b) : std::max(a,b);
  }

  // T can be any float-like type (int-like type not directly supported, cast down)
  template<typename T>
  T exp(T lambda){
    std::exponential_distribution<T> dist(lambda);
    return dist(generator_);
  }
  // T can be any float-like type (int-like type not directly supported, cast down)
  template<typename T>
  T normal(T mean, T stddev){
    std::normal_distribution<T> dist(mean, stddev);
    return dist(generator_);
  }
};

template<typename Iter>
static double Energy(Iter beg, Iter end){
  return std::inner_product(beg, end, beg, 0.0);
}

template<typename T>
static bool abs_compare(T a, T b){return (std::abs(a) < std::abs(b));}

size_t Pow2Ceil(const size_t x){return (x<=0)? 0 : pow(2, ceil(log2(x)));}

// for a given x between 0 and 1, returns a value between 0 and x, following an exponential curve
// (like a linear interp. but "bent down"). The steepness of the curve is determined by k.
double ExpInterpZeroOne(const double x, const double k){
  return (exp(k*x)-1) / (exp(k)-1);
}


// Given two real numbers a, b, and a fraction x in [0, 1], returns the linear interpolation
// between them. Example: LinInterp(5.0, 10.0, 0.5) returns 7.5
float LinInterp (const float first, const float second, const float x) {
  return first*(1.0f-x) + second*x;
}

size_t Factorial(long int x){
  size_t result = 1;
  for(; x>1; --x){result *= x;}
  return result;
}

size_t DoubleFactorial(long int x){
  size_t result = 1;
  for(; x>=2; x-=2){result *= x;}
  return result;
}

double FreqToMidi(const double freq){
  return 69 + 12*log2(freq/440);
}

double MidiToFreq(const double midi){
  return 440*pow(2, (midi-69)/12);
}

// GAMMA constants based in specfunc/gamma.c GSLv2.4 (Author G. Jungman)
#define PI 3.1415926535897932384626
#define TWO_PI 6.2831853071795864769252867665590057
#define E_NUMBER        2.71828182845904523536028747135
#define LogRootTwoPi_  0.9189385332046727418
#define InverseOfSqrtOfTwoPi 0.3989422804014326779399460599343
#define Exp7DivBy2Sqrt2Pi  218.746666493637437990776069044569169419001468
// coefficients for gamma=7, kmax=8  Lanczos method
static double lanczos_7_c[9] = {
  0.99999999999980993227684700473478,
  676.520368121885098567009190444019,
 -1259.13921672240287047156078755283,
  771.3234287776530788486528258894,
 -176.61502916214059906584551354,
  12.507343278686904814458936853,
 -0.13857109526572011689554707,
  9.984369578019570859563e-6,
  1.50563273514931155834e-7
};


// based in specfunc/gamma.c GSLv2.4 (Author G. Jungman)
// I'm aware of std::tgamma, but accessing the guts of Gamma allows chi2 optim.
static double Gamma(double x){
  x -= 1.0;
  double lanczos = lanczos_7_c[0];
  for(int k=1; k<=8; k++){lanczos += lanczos_7_c[k]/(x+k);}
  double term1 = (x+0.5)*log((x+7.5)/E_NUMBER);
  double log_gamma = term1 + LogRootTwoPi_ + log(lanczos) - 7.0;
  return std::exp(log_gamma);
}


// factorial implementation for integers (faster)
static double Gamma(size_t x){
  return Factorial(x-1);
}

// END GAMMA BASED ON GSL
////////////////////////////////////////////////////////////////////////////////////////////////////



// Fast implementation of Chi2 for real df integrating the Lanczos gamma approx
static double Chi2(double x, double df){
  double y = 0.5*df-1;
  double y_75 = y+7.5;
  // get lanczos
  double lanczos = lanczos_7_c[0];
  for(int k=1; k<=8; k++){lanczos += lanczos_7_c[k]/(y+k);}
  //
  double result = Exp7DivBy2Sqrt2Pi / lanczos;
  result *= pow(x*E_NUMBER/(2*y_75), y);
  result /= sqrt(y_75*exp(x-1));
  return result;
}

// // NOTE: this is about twice as fast as lanczos, but not worth it because for low df is wrong
// // // Chi2 function: for int k, Gamma(k/2) = sqrt(pi)*DoubleFactorial(k-2)/pow(2, 0.5*(k-1))
// // // This overload of Chi2 abuses this, yielding, for integer k and v:=0.5k:
// // // Chi2(x,k) = 1/sqrt(2pi) * exp(-0.5x) * pow(x, v-1) / DoubleFactorial(k-2)
// static double Chi2(double x, size_t df){
//   long int df2 = df-2;
//   return sqrt(exp(-x) * pow(x, df2)) * InverseOfSqrtOfTwoPi / DoubleFactorial(df2);
// }


////////////////////////////////////////////////////////////////////////////////////////////////////
/// STRING PROCESSING
////////////////////////////////////////////////////////////////////////////////////////////////////

// Given a text and a separator strings, runs over the text removing the separators and adding the
// resulting chunks into a vector, which is returned. Empty tokens are ignored
// Split("a|b c||d  e|||f   g||||h    i", "||") returns ["a|b c", "d  e", "|f   g", "h    i"]
std::vector<std::string> Split(const std::string &text, const std::string sep){
  const size_t kSepSize = sep.size();
  std::vector<std::string> tokens;
  std::size_t start = 0, end = 0;
  while ((end = text.find(sep, start)) != std::string::npos) {
    if (end != start){tokens.push_back(text.substr(start, end - start));}
    start = end + kSepSize;
  }
  if (end != start) {tokens.push_back(text.substr(start));}
  return tokens;
}




// given a string in the form <token1><separator><token2> ... <comment_marker><comments>...
// returns an iterator of strings with the found tokens
std::deque<std::string> ParseLine(std::string line,
                                  std::string separator=" ",
                                  std::string comment_marker="#"){
  // variables needed
  std::string::iterator it = line.begin();
  std::string::iterator end = line.end();
  const size_t kSepSize = separator.size();
  const size_t kCommentSize = comment_marker.size();
  // first remove the comments by (verbose but fast) pattern matching
  std::string::iterator comm_it = it+line.find_first_of(comment_marker[0]);
  if(end-comm_it>=kCommentSize &&
     std::equal(comm_it, comm_it+kCommentSize, comment_marker.begin(), comment_marker.end())){
    end = comm_it;
  }
  // Then find and remove separators. The rest are the tokens
  std::deque<std::string> line_tokens;
  std::string::iterator token_beg = it;
  for(std::string::iterator sep_beg=separator.begin(), sep_end=separator.end(); it<end; ++it){
    if(*it == separator[0] &&
       end-it >= kSepSize  &&
       std::equal(it, it+kSepSize, sep_beg, sep_end)){
      line_tokens.push_back(std::string(token_beg, it));
      it += kSepSize;
      token_beg = it;
    }
  } // finish loop for characters in line adding the last token found
  if(token_beg<end){line_tokens.push_back(std::string(token_beg, end));}
  return line_tokens;
}

// Given a container or its beginning and end iterables, converts the container to a string
// of the form {a, b, c} (like a basic version of Python's __str__).
template<typename T>
std::string IterableToString(T it, T end,
                             const std::string before="{",
                             const std::string after="}",
                             const std::string separator=", "){
  std::stringstream ss;
  ss << before;
  bool first = true;
  for (; it!=end; ++it){
    if (first){
      ss << *it;
      first = false;
    } else {
      ss << separator << *it;
    }
  } ss << after;
  return ss.str();
}
template <class C> // Overload to directly accept any Collection like vector<int>
std::string IterableToString(const C &c,
                             const std::string before="{",
                             const std::string after="}",
                             const std::string separator=", "){
  return IterableToString(c.begin(), c.end(), before, after, separator);
}
template <class T> // Overload to directly accept initializer_lists
std::string IterableToString(const std::initializer_list<T> c,
                             const std::string before="{",
                             const std::string after="}",
                             const std::string separator=", "){
  return IterableToString(c.begin(), c.end(), before, after, separator);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// TYPECHECK/ANTIBUGGING
////////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////////////////


// Given a container or its beginning and end iterables, checks wether all values contained in the
// iterable are equal and raises an exception if not. Usage example:
// CheckAllEqual({a.size_, b.size_, result.size_},
//                 "SpectralCorrelation: all sizes must be equal and are");
template<class I>
void CheckAllEqual(const I beg, const I end, const std::string &message){
  I it = beg;
  bool all_eq = true;
  auto last = (it==end)? end : std::prev(end);
  for(;it!=last; ++it){
    all_eq &= (*(it)==*(std::next(it)));
    if (!all_eq) {
      throw std::runtime_error(std::string("[ERROR] ") + message+" "+IterableToString(beg, end));
    }
  }
}
template <class C> // Overload to directly accept any Collection like vector<int>
void CheckAllEqual(const C &c, const std::string message){
  CheckAllEqual(c.begin(), c.end(), message);
}
template <class T> // Overload to directly accept initializer_lists
void CheckAllEqual(const std::initializer_list<T> c, const std::string message){
  CheckAllEqual(c.begin(), c.end(), message);
}

template<class I>
bool CompareIterables(const I beg1, const I end1, const I beg2, const I end2){
  // if sizes are not the same, return false
  if(std::distance(beg1,end1) != std::distance(beg1,end1)){
    return false;
  }
  // if any of the contents are not the same, return false
  for(I it1=beg1, it2=beg2; it1!=end1; ++it1, ++it2){
    if (*it1 != *it2) {
      return false;
    }
  } // otherwise return true
  return true;
}
template <class C> // Overload to directly accept any Collection like vector<int>
bool CompareIterables(const C &c1, const C &c2){
  return CompareIterables(c1.begin(), c1.end(), c2.begin(), c2.end());
}
template <class T> // Overload to directly accept initializer_lists
bool CompareIterables(const std::initializer_list<T> &c1, const std::initializer_list<T> &c2){
  return CompareIterables(c1.begin(), c1.end(), c2.begin(), c2.end());
}


// Raises an exception if complex_size!=(real_size/2+1), being "/" an integer division.
void CheckRealComplexRatio(const size_t real_size, const size_t complex_size,
                           const std::string func_name){
  if(complex_size!=(real_size/2+1)){
    throw std::runtime_error(std::string("[ERROR] ") + func_name +
                             ": size of ComplexSignal must equal size(FloatSignal)/2+1. " +
                             " Sizes were (float, complex): " +
                             IterableToString({real_size, complex_size}));
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

// Abstract function that performs a comparation between any 2 elements, and if the comparation
// returns a truthy value raises an exception with the given message.
template <class T, class Functor>
void CheckTwoElements(const T a, const T b, const Functor &binary_predicate,
                        const std::string message){
  if(binary_predicate(a,b)){
    throw std::runtime_error(std::string("[ERROR] ") + message + " " + IterableToString({a, b}));
  }
}


// Raises an exception with the given message if a>b.
void CheckLessEqual(const size_t a, const size_t b, const std::string message){
  CheckTwoElements(a, b, [](const size_t a, const size_t b){return a>b;},  message);
}


void CheckWithinRange(const size_t idx, const size_t min_allowed, const size_t max_allowed,
                      const std::string func_name){
  CheckTwoElements(idx, min_allowed, [](const size_t a, const size_t b){return a<b;},
                   func_name+" index cannot be smaller than minimum allowed!");
  CheckTwoElements(idx, max_allowed, [](const size_t a, const size_t b){return a>b;},
                   func_name+" index cannot be bigger than maximum allowed!");
}






////////////////////////////////////////////////////////////////////////////////////////////////////
/// MISC
////////////////////////////////////////////////////////////////////////////////////////////////////

class Timer{
private:
  std::chrono::high_resolution_clock::time_point timestamp_;
  public:
  explicit Timer(){reset();}
  void reset(){timestamp_ = std::chrono::high_resolution_clock::now();}
  double elapsed_ms(){
    std::chrono::high_resolution_clock::time_point x = std::chrono::high_resolution_clock::now();
    return((double)std::chrono::duration_cast<std::chrono::nanoseconds>(x-timestamp_).count())*1e-6;
  }
};



#endif
