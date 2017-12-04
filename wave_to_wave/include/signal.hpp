#ifndef SIGNAL_H
#define SIGNAL_H

// STL INCLUDES
#include <string.h>
#include <iostream>
#include <complex>
#include <cstdlib>

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

  // SIGNAL-TO-SIGNAL ARITHMETIC
  // This function is needed by each in-place signal-to-signal operation.
  // Given the required endpoints (by reference) of both signals, and the wanted "t" offset of the
  // "other" with respect to "this" (t=13 means that other[0] will be applied to this[13]),
  // this function adjust the endpoints to match the desired t, and returns the optimal loop size.
  // Example: see addSignal and mulSignal.
  const size_t _prepareSignalOpInPlace(T* &begin_this, T* &begin_other,
                                       const T* end_other, const int t) const {
    // if t>0, advance this.begin(). If t<0, advance other.begin()
    if(t>0){
      begin_this += t;
    } else if(t<0){
      begin_other -= t;
    }
    // return the minimal distance of both, based on the updated pointers (0 is the smallest)
    const int len_min = std::min(end()-begin_this, end_other-begin_other);
    return std::max(len_min, 0);
  }

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
  const T* begin() const{return &data_[0];}
  T* end(){return &data_[size_];}
  const T* end() const{return &data_[size_];}

  // Element-wise addition of two signals of the same type. Parameter t is the offset of the
  // "other" with respect to "this" (t=13 means that other[0] will be applied to this[13]),
  void addSignal(AudioSignal<T> &other, const int t=0){
    T* begin_this = begin();
    T* begin_other = other.begin();
    const T* end_other = other.end();
    const size_t lenn = _prepareSignalOpInPlace(begin_this, begin_other, end_other, t);
     for(size_t i=0; i<lenn; ++i, ++begin_this, ++begin_other){
      *begin_this += *begin_other;
    }
  }

  // Analogous to addSignal, performs element-wise subtraction
  void subtractSignal(AudioSignal<T> &other, const int t=0){
    T* begin_this = begin();
    T* begin_other = other.begin();
    const T* end_other = other.end();
    const size_t lenn = _prepareSignalOpInPlace(begin_this, begin_other, end_other, t);
    for(size_t i=0; i<lenn; ++i, ++begin_this, ++begin_other){
      *begin_this -= *begin_other;
    }
  }

  // Analogous to addSignal, performs element-wise multiplication
  void mulSignal(AudioSignal<T> &other, const int t=0){
    T* begin_this = begin();
    T* begin_other = other.begin();
    const T* end_other = other.end();
    const size_t lenn = _prepareSignalOpInPlace(begin_this, begin_other, end_other, t);
    for(size_t i=0; i<lenn; ++i, ++begin_this, ++begin_other){
      *begin_this *= *begin_other;
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
  void operator-=(AudioSignal<T> &s){subtractSignal(s);}
  void operator*=(AudioSignal<T> &s){mulSignal(s);}
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

// Wrapper class for AudioSignal<complex64> plus some extra functionality
class ComplexSignal : public AudioSignal<std::complex<float> >{
public:
  explicit ComplexSignal(size_t size)
    : AudioSignal(size){}
  explicit ComplexSignal(std::complex<float>* data, size_t size)
    : AudioSignal(data, size){}
  explicit ComplexSignal(std::complex<float>* data, size_t size, size_t pad_bef, size_t pad_aft)
    : AudioSignal(data, size, pad_bef, pad_aft){}
  // in-class extra functionality
  void conjugate(){
    float* data_flat = &reinterpret_cast<float(&)[2]>(data_[0])[0];
    for(size_t i=1, kFlatSize=2*size_; i<kFlatSize; i+=2){
      data_flat[i] *= -1;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// This free function takes three complex signals a,b,c of the same size and computes the complex
// element-wise multiplication:   a+ib * c+id = ac+iad+ibc-bd = ac-bd + i(ad+bc)   The computation
// loop isn't sent to OMP because this function itself is already expected to be called by multiple
// threads, and it would actually slow down the process.
// It expects all 3 signals to have equal size, throws an exception otherwise
void SpectralConvolution(const ComplexSignal &a, const ComplexSignal &b, ComplexSignal &result){
  const size_t kSize_a = a.getSize();
  const size_t kSize_b = b.getSize();
  const size_t kSize_result = result.getSize();
  CheckAllEqual({kSize_a, kSize_b, kSize_result},
                std::string("SpectralConvolution: all sizes must be equal and are"));
  for(size_t i=0; i<kSize_a; ++i){
    result[i] = a[i]*b[i];
  }
}

// Like SpectralConvolution, but compu
void SpectralCorrelation(const ComplexSignal &a, const ComplexSignal &b, ComplexSignal &result){
  const size_t kSize_a = a.getSize();
  const size_t kSize_b = b.getSize();
  const size_t kSize_result = result.getSize();
  CheckAllEqual({kSize_a, kSize_b, kSize_result},
                  "SpectralCorrelation: all sizes must be equal and are");
  for(size_t i=0; i<kSize_a; ++i){
    result[i] = a[i] * std::conj(b[i]);
  }
}



#endif
