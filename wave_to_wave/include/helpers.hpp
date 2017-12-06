#ifndef HELPERS_H
#define HELPERS_H

// STL INCLUDES
#include <sstream>
#include <string>
#include <stdexcept>
#include <math.h>



////////////////////////////////////////////////////////////////////////////////////////////////////
/// MATH
////////////////////////////////////////////////////////////////////////////////////////////////////

size_t Pow2Ceil(size_t x){return (x<=0)? 0 : pow(2, ceil(log2(x)));}

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

// GAMMA constants based in specfunc/gamma.c GSLv2.4 (Author G. Jungman)
#define E_NUMBER        2.71828182845904523536028747135
#define GSL_DBL_EPSILON        2.2204460492503131e-16
#define LogRootTwoPi_  0.9189385332046727418
#define InverseOfTwoTimesSqrtOfTwoPi 0.1994711402007163389699730299671909342379
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
  double lanczos = lanczos_7_c[0];
  x -= 1.0;
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

// Given v=0.5*k, chi2(x,k) = [x**(v-1) * exp(-v)] / [2**v * Gamma(v)] (wikipedia).
// Combining with the above approximation of Gamma, it can be reduced to the
// following formula (given L:=lanczos term, k:=degrees of freedom):
// Chi2(x, k) = 1/(2*sqrt(2*pi)*L) *(0.5x)**(v-1) *exp(6.5+0.5x) *(x+6.5)**(0.5-x)
static double Chi2(double x, double df){
  // avoid multiple calculations:
  double x_dec = x-1;
  double x_half = 0.5*x;
  double v = 0.5*df;
  // as in Gamma, compute Lanczos term:
  double lanczos = lanczos_7_c[0];
  for(int k=1; k<=8; k++){lanczos += lanczos_7_c[k]/(x_dec+k);}
  // multiply succesively the 4 terms to obtain the chi2 value:
  double result = InverseOfTwoTimesSqrtOfTwoPi/lanczos;
  result *= pow(x_half, v-1);
  result *= exp(x_half+6.5);
  result *= pow(x+6.5, 0.5-x);
  return result;
}

// The function Gamma(k/2) has faster, closed form expressions for integer k.
// This overload of Chi2 abuses this, yielding, for integer k:
// Chi2(x,k) = [x**(0.5k-1) * exp(-0.5x)] / [sqrt(2k) * DoubleFactorial(k-2)]
static double Chi2(double x, size_t df){
  double result = pow(x, 0.5*df-1) * exp(-0.5*df);
  result /= sqrt(2*df);
  return result/DoubleFactorial(df-2);
}



////////////////////////////////////////////////////////////////////////////////////////////////////
/// TYPECHECK/ANTIBUGGING
////////////////////////////////////////////////////////////////////////////////////////////////////

// Given a container or its beginning and end iterables, converts the container to a string
// of the form {a, b, c} (like a basic version of Python's __str__).
template<typename T>
std::string IterableToString(T it, T end){
  std::stringstream ss;
  ss << "{";
  bool first = true;
  for (; it!=end; ++it){
    if (first){
      ss << *it;
      first = false;
    } else {
      ss << ", " << *it;
    }
  } ss << "}";
  return ss.str();
}
template <class C> // Overload to directly accept any Collection like vector<int>
std::string IterableToString(const C &c){
  return IterableToString(c.begin(), c.end());
}
template <class T> // Overload to directly accept initializer_lists
std::string IterableToString(const std::initializer_list<T> c){
  return IterableToString(c.begin(), c.end());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

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



////////////////////////////////////////////////////////////////////////////////////////////////////

// Raises an exception if complex_size!=(real_size/2+1), being "/" an integer division.
void CheckRealComplexRatio(const size_t real_size, const size_t complex_size,
                           const std::string func_name="CheckRealComplexRatio");

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





#endif
