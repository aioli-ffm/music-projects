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

#define REAL 0
#define IMAG 1





// This is an abstract base class that provides some basic, type-independent functionality for
// any container that should behave as a signal. It is not intended to be instantiated directly.
template <class T>
class AudioSignal {
protected:
  T* data_;
  size_t size_;
  long int delay_;
public:
  // Given a size and a reference to an array, it fills the array with <SIZE> zeros.
  // Therefore, **IT DELETES THE CONTENTS OF THE ARRAY**. It is intended to be passed a newly
  // allocated array by the classes that inherit from AudioSignal, because it isn't an expensive
  // operation and avoids memory errors due to non-initialized values.
  explicit AudioSignal(size_t size)
    : data_((T*)aligned_alloc(64, size*sizeof(T))),
      size_(size),
      delay_(0){
    std::fill(this->begin(), this->end(), 0);
    // memset(data_, 0, sizeof(T)*size);
  }
  explicit AudioSignal(T* data, size_t size)
    : AudioSignal(size){
    // memcpy(data_, data, sizeof(T)*size);
    std::copy(data, data+size, data_);
  }
  explicit AudioSignal(T* data, size_t size, size_t pad_bef, size_t pad_aft)
    : AudioSignal(size+pad_bef+pad_aft){
    // memcpy(data_+pad_bef, data, sizeof(float)*size);
    std::copy(data, data+size, data_+pad_bef);
  }
  ~AudioSignal(){
    free(data_);
  }
  // getters
  size_t &getSize(){return size_;}
  const size_t &getSize() const{return size_;}
  T* getData(){return data_;}
  const T* getData() const{return data_;}
  // overloaded operators
  T &operator[](size_t idx){return data_[idx];}
  T &operator[](size_t idx) const {return data_[idx];}
  // compound assignment operators
  void operator+=(const T x){for(size_t i=0; i<size_; ++i){data_[i] += x;}}
  void operator-=(const T x){for(size_t i=0; i<size_; ++i){data_[i] -= x;}}
  void operator*=(const T x){for(size_t i=0; i<size_; ++i){data_[i] *= x;}}
  // make signals iterable
  T* begin(){return &data_[0];}
  T* end(){return &data_[size_];}
  const T* begin() const{return &data_[0];}
  const T* end() const{return &data_[size_];}
  // basic print function. It may be overriden if, for example, the type <T> is a struct.
  void print(const std::string name="signal"){
    std::cout << std::endl;
    for(size_t i=0; i<size_; ++i){
      std::cout << name << "[" << i << "]  \t=\t" << data_[i] << std::endl;
    }
  }
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
