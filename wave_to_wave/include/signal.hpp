#ifndef SIGNAL_H
#define SIGNAL_H

// STL INCLUDES
#include <string.h>
#include <iostream>
#include <complex>
#include <cstdlib>
// SYSTEM-INSTALLED LIBRARIES
#include <fftw3.h>
// LOCAL INCLUDES
#include "helpers.hpp"



////////////////////////////////////////////////////////////////////////////////////////////////////

// The AudioSignal class is the central piece for any musical DSP. As a template class it can be
// instantiated with arbitrary types, although float and complex will suffice for most applications
// (and are already represented by the FloatSignal and ComplexSignal classes respectively).
// AudioSignal basically holds an array, provides basic getter/setter/print functionality to it,
// and (most notably) support for signal-to-constant and signal-to-signal arithmetic. Last but not
// least, AudioSignal implements the begin() and end() methods and is therefore iterable.
template <class T>
class AudioSignal {
protected:
  T* data_;
  size_t size_;
public:
  // CONSTRUCTORS AND DESTRUCTOR
  explicit AudioSignal(size_t size)
    : data_((T*)aligned_alloc(64, size*sizeof(T))),
      size_(size){
    std::fill(this->begin(), this->end(), 0);
  }
  explicit AudioSignal(T* data, size_t size)
    : AudioSignal(size){
    std::copy(data, data+size, data_);
  }
  explicit AudioSignal(T* data, size_t size, size_t pad_bef, size_t pad_aft)
    : AudioSignal(size+pad_bef+pad_aft){
    std::copy(data, data+size, data_+pad_bef);
  }
  ~AudioSignal(){free(data_);}
  // GETTERS/SETTERS/PRETTYPRINT
  size_t &getSize(){return size_;}
  const size_t &getSize() const{return size_;}
  T* getData(){return data_;}
  const T* getData() const{return data_;}
  void print(const std::string name="signal"){
    std::cout << std::endl;
    for(size_t i=0; i<size_; ++i){
      std::cout << name << "[" << i << "]  \t=\t" << data_[i] << std::endl;
    }
  }
  // ITERABLE INTERFACE
  T* begin(){return &data_[0];}
  T* end(){return &data_[size_];}
  const T* begin() const{return &data_[0];}
  const T* end() const{return &data_[size_];}
  // SIGNAL-TO-SIGNAL ARITHMETIC
  void addSignal(AudioSignal<T> &other, const int t=0){
    // if t>0, advance this.begin(). If t<0, advance signal.begin()
    T* this_it = begin() + std::max(t, 0);
    T* other_it = other.begin() - std::min(0, t);
    // loop length will equal the smallest size
    const size_t this_len = std::max<int>(std::distance(this_it, end()), 0);
    const size_t other_len = std::max<int>(std::distance(other_it, other.end()), 0);
    const size_t len = std::min(this_len, other_len);
    // loop expression as simple as possible
    for(size_t i=0; i<len; ++i, ++this_it, ++other_it){
      *this_it += *other_it;
    }
  }
  // OVERLOADED OPERATORS
  T &operator[](size_t idx){return data_[idx];}
  T &operator[](size_t idx) const {return data_[idx];}
  // signal-to-constant compound assignment operators
  void operator+=(const T x){for(size_t i=0; i<size_; ++i){data_[i] += x;}}
  void operator-=(const T x){for(size_t i=0; i<size_; ++i){data_[i] -= x;}}
  void operator*=(const T x){for(size_t i=0; i<size_; ++i){data_[i] *= x;}}
  // signal-to-signal compound assignment operators
  void operator+=(AudioSignal<T> &s){addSignal(s);}
};

////////////////////////////////////////////////////////////////////////////////////////////////////


// Wrapper class for AudioSignal<float32>
class FloatSignal : public AudioSignal<float>{
public:
  explicit FloatSignal(size_t size)
    : AudioSignal(size){}
  explicit FloatSignal(float* data, size_t size)
    : AudioSignal(data, size){}
  explicit FloatSignal(float* data, size_t size, size_t pad_bef, size_t pad_aft)
    : AudioSignal(data, size, pad_bef, pad_aft){}
};

// Wrapper class for AudioSignal<complex64>
class ComplexSignal : public AudioSignal<std::complex<float> >{
public:
  explicit ComplexSignal(size_t size)
    : AudioSignal(size){}
  explicit ComplexSignal(std::complex<float>* data, size_t size)
    : AudioSignal(data, size){}
  explicit ComplexSignal(std::complex<float>* data, size_t size, size_t pad_bef, size_t pad_aft)
    : AudioSignal(data, size, pad_bef, pad_aft){}
};



////////////////////////////////////////////////////////////////////////////////////////////////////

// This free function takes three complex signals a,b,c of the same size and computes the complex
// element-wise multiplication:   a+ib * c+id = ac+iad+ibc-bd = ac-bd + i(ad+bc)   The computation
// loop isn't sent to OMP because this function itself is already expected to be called by multiple
// threads, and it would actually slow down the process.
// It throws an exception if
void SpectralConvolution(const ComplexSignal &a, const ComplexSignal &b, ComplexSignal &result){
  const size_t kSize_a = a.getSize();
  const size_t kSize_b = b.getSize();
  const size_t kSize_result = result.getSize();
  CheckAllEqual({kSize_a, kSize_b, kSize_result},
                std::string("SpectralConvolution: all sizes must be equal and are"));
  for(size_t i=0; i<kSize_a; ++i){
    // a+ib * c+id = ac+iad+ibc-bd = ac-bd + i(ad+bc)
    result[i] = a[i]*b[i];
     // result[i][REAL] = a[i][REAL]*b[i][REAL] - a[i][IMAG]*b[i][IMAG];
     // result[i][IMAG] = a[i][IMAG]*b[i][REAL] + a[i][REAL]*b[i][IMAG];
  }
}

// This function behaves identically to SpectralConvolution, but computes c=a*conj(b) instead
// of c=a*b:         a * conj(b) = a+ib * c-id = ac-iad+ibc+bd = ac+bd + i(bc-ad)
void SpectralCorrelation(const ComplexSignal &a, const ComplexSignal &b, ComplexSignal &result){
  const size_t kSize_a = a.getSize();
  const size_t kSize_b = b.getSize();
  const size_t kSize_result = result.getSize();
  CheckAllEqual({kSize_a, kSize_b, kSize_result},
                  "SpectralCorrelation: all sizes must be equal and are");
  for(size_t i=0; i<kSize_a; ++i){
    // a * conj(b) = a+ib * c-id = ac-iad+ibc+bd = ac+bd + i(bc-ad)
    result[i] = a[i] * std::conj(b[i]);
    // result[i][REAL] = a[i][REAL]*b[i][REAL] + a[i][IMAG]*b[i][IMAG];
    // result[i][IMAG] = a[i][IMAG]*b[i][REAL] - a[i][REAL]*b[i][IMAG];

  }
}





#endif
