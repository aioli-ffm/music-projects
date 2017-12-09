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

#ifndef SIGNAL_H
#define SIGNAL_H

// STL INCLUDES
#include <string.h>
#include <iostream>
#include <fstream>
#include <complex>
#include <cstdlib>
#include<sndfile.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// AUDIOSIGNAL
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
  // BASIC CONSTRUCTORS
  explicit AudioSignal() : data_(nullptr), size_(0){}
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
  // GENERATIVE CONSTRUCTOR
  // given a (long int -> T) function, instantiates the signal and fills it
  // with values between function(0) and function(size-1).
  explicit AudioSignal(std::function<T (const long int)> const&f, size_t size)
    : AudioSignal(size){
    for(size_t i=0; i<size; ++i){
      data_[i] = f(i);
    }
  }
  // DESTRUCTOR
  ~AudioSignal(){
    if(data_!=nullptr){free(data_);}
  }
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

  // make ASCII file with signal
  void toAscii(const std::string filename){
    std::ofstream stream(filename);
    for (int i=0; i <size_; ++i){stream << i << " " << data_[i] << std::endl;}
    stream.close();
  }

  /// SIGNAL ARITHMETIC ////////////////////////////////////////////////////////////////////////////

  // Element-wise addition of two signals of the same type. Parameter t is the offset of the
  // "other" with respect to "this" (t=13 means that other[0] will be applied to this[13]).
  // Note that all the "other" values that are out of the bounds of "this" will be ignored.
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
/// FLOATSIGNAL
////////////////////////////////////////////////////////////////////////////////////////////////////

class FloatSignal : public AudioSignal<float>{
public:
  explicit FloatSignal(size_t size)
    : AudioSignal(size){}
  explicit FloatSignal(float* data, size_t size)
    : AudioSignal(data, size){}
  explicit FloatSignal(float* data, size_t size, size_t pad_bef, size_t pad_aft)
    : AudioSignal(data, size, pad_bef, pad_aft){}
  explicit FloatSignal(std::function<float (const long int)> const&f, size_t size)
    : AudioSignal(f, size){}
  // from wav
  explicit FloatSignal(const std::string wav_path)
    :AudioSignal() {
    SF_INFO sf_info;
    sf_info.format = 0;
    SNDFILE* infile = sf_open(wav_path.c_str(), SFM_READ, &sf_info);
    if(infile == nullptr){
      throw std::invalid_argument("FloatSignal: Unable to open input stream: "+wav_path);
    } else { // if file opens...
      size_ = sf_info.frames;
      data_ = ((float*)aligned_alloc(64, size_*sizeof(float)));
      sf_read_float(infile, data_, size_);
      sf_close(infile);
    }
  }
  //
  void toWav(const std::string path_out, const size_t samplerate,
             const size_t format=SF_FORMAT_PCM_16){
    SF_INFO sf_info;
    // sf_info.frames = size_;
    sf_info.samplerate = samplerate;
    sf_info.channels = 1;
    sf_info.format = SF_FORMAT_WAV | format; // SF_FORMAT_FLOAT; //
    // declare and try to open outfile
    SNDFILE* outfile = sf_open(path_out.c_str(), SFM_WRITE, &sf_info);
    if(outfile == nullptr){ // if file doesn't open...
      throw std::invalid_argument("toWav: unable to open output stream "+path_out);
    } else{// if file opens...
      sf_write_float(outfile, &data_[0], size_);
      sf_write_sync(outfile);
      sf_close(outfile);
      std::cout  << "toWav: succesfully saved to "<< path_out << std::endl;
    }
  }
  void plot(const char* name="FloatSignal", const size_t samplerate=1,
            const size_t downsample_ratio=1, const float aspect_ratio=0.1f){
    // open persistent gnuplot window
    FILE* gnuplot_pipe = popen ("gnuplot -persistent", "w");
    // basic settings
    fprintf(gnuplot_pipe, "unset key\n"); // remove legend
    fprintf(gnuplot_pipe, "set lmargin at screen 0.06\n"); // margins and aspect ratio
    fprintf(gnuplot_pipe, "set rmargin at screen 0.995\n");
    fprintf(gnuplot_pipe, "set term wxt size 1000, %f\n", aspect_ratio*1800);
    fprintf(gnuplot_pipe, "set size ratio %f\n", aspect_ratio);
    fprintf(gnuplot_pipe, "set style line 1 lc rgb '#0011ff' lt 1 lw 1\n"); // linestyle and tics
    fprintf(gnuplot_pipe, "set ytics font ',5'\n");
    fprintf(gnuplot_pipe, "set xtics font ',5' rotate by 90 offset 0, -1.5\n");
    // fprintf(gnuplot_pipe, "set y2label 'amplitude'\n"); // labels, names
    // fprintf(gnuplot_pipe, "set xlabel 'time'\n");
    fprintf(gnuplot_pipe, "set arrow to %f,0 filled\n", ((float)size_)/samplerate);
    fprintf(gnuplot_pipe, "set title '%s'\n", name); // frame main title
    // fill it with data
    fprintf(gnuplot_pipe, "plot '-' with lines ls 1\n");
    for(size_t i=0; i<size_; ++i){
      if(i%downsample_ratio==0){
        fprintf(gnuplot_pipe, "%f %f\n", ((float)i)/samplerate, data_[i]);
      }
    }
    fprintf(gnuplot_pipe, "e\n");
    // refresh can probably be omitted
    fprintf(gnuplot_pipe, "refresh\n");
  }
};



////////////////////////////////////////////////////////////////////////////////////////////////////
/// COMPLEXSIGNAL
////////////////////////////////////////////////////////////////////////////////////////////////////

class ComplexSignal : public AudioSignal<std::complex<float> >{

  // see implementation for explanation of this friend functions
  friend void ComplexMul(const ComplexSignal &a, const ComplexSignal &b, ComplexSignal &result);
  friend void ComplexConjMul(const ComplexSignal &a, const ComplexSignal &b, ComplexSignal &result);

public:
  explicit ComplexSignal(size_t size)
    : AudioSignal(size){}
  explicit ComplexSignal(std::complex<float>* data, size_t size)
    : AudioSignal(data, size){}
  explicit ComplexSignal(std::complex<float>* data, size_t size, size_t pad_bef, size_t pad_aft)
    : AudioSignal(data, size, pad_bef, pad_aft){}
  explicit ComplexSignal(std::function< std::complex<float> (const long int) > const&f, size_t size)
    : AudioSignal(f, size){}
  // in-class extra functionality
  void conjugate(){
    float* data_flat = &reinterpret_cast<float(&)[2]>(data_[0])[0];
    for(size_t i=1, kFlatSize=2*size_; i<kFlatSize; i+=2){
      data_flat[i] *= -1;
    }
  }
  void plot(const char* name="ComplexSignal", const size_t samplerate=1,
            const size_t downsample_ratio=1){
    // open persistent gnuplot window
    FILE* gnuplot_pipe = popen ("gnuplot -persistent", "w");
    // basic settings
    fprintf(gnuplot_pipe, "unset key\n"); // remove legend
    fprintf(gnuplot_pipe, "set lmargin at screen 0.17\n"); // margins and aspect ratio
    fprintf(gnuplot_pipe, "set rmargin at screen 0.85\n");
    fprintf(gnuplot_pipe, "set term wxt size 1000, 500\n"); // term qt for static
    fprintf(gnuplot_pipe, "set size ratio 0.3\n");
    fprintf(gnuplot_pipe, "set style line 1 lc rgb '#0011ff' lt 1 lw 1\n"); // linestyle and tics
    fprintf(gnuplot_pipe, "set ytics font ',5'\n");
    fprintf(gnuplot_pipe, "set ztics font ',5'\n");
    fprintf(gnuplot_pipe, "set xtics font ',5' rotate by 90 offset 0, -1.5\n");
    // fprintf(gnuplot_pipe, "set y2label 'amplitude'\n"); // labels, names
    // fprintf(gnuplot_pipe, "set xlabel 'time'\n");
    // fprintf(gnuplot_pipe, "set label 'freq' at %f,0,0\n", 1.01*((float)size_)/samplerate);
    fprintf(gnuplot_pipe, "set arrow to %f,0,0 filled\n", ((float)size_)/samplerate);
    fprintf(gnuplot_pipe, "set title '%s'\n", name); // frame main title

    // fill it with data
    float* data_flat = &reinterpret_cast<float(&)[2]>(data_[0])[0];
    fprintf(gnuplot_pipe, "splot '-' with lines ls 1\n");
    for(size_t i=0, max=size_*2, sr=samplerate*2; i<max; i+=2){
      if(i%downsample_ratio==0){
        fprintf(gnuplot_pipe, "%f %f %f\n", ((float)i)/sr, data_flat[i], data_flat[i+1]);
      }
    }
    fprintf(gnuplot_pipe, "e\n");
    // refresh can probably be omitted
    fprintf(gnuplot_pipe, "refresh\n");
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// This friend function performs element-weise multiplication of a,b into c (c[i]=a[i]*b[i]) where
  // a,b,c are of type ComplexSignal and have the same length (outcome is undefined otherwise).
  // Performance: should SIMDize since signals are 64bit aligned (didn't check). The loop isn't send
  // to OMP to avoid thread overpopulation (since the overlap-add convolver already parallelizes).
void ComplexMul(const ComplexSignal &a, const ComplexSignal &b, ComplexSignal &result){
  for(size_t i=0, size=a.size_; i<size; ++i){
    result[i] = a[i]*b[i];
  }
}

// Like ComplexMul, but c[i] = a[i]*conj(b[i])
void ComplexConjMul(const ComplexSignal &a, const ComplexSignal &b, ComplexSignal &result){
  for(size_t i=0, size=a.size_; i<size; ++i){
    result[i] = a[i] * std::conj(b[i]);
  }
}

#endif
