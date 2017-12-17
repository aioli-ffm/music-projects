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

#ifndef CONVOLVER_H
#define CONVOLVER_H

// OPEN MP:
// comment this line to deactivate OpenMP for loop parallelizations, or if you want to debug
// memory management (valgrind reports OMP normal activity as error).
// the number is the minimum size that a 'for' loop needs to get sent to OMP (1=>always sent)
#define WITH_OPENMP_ABOVE 1

// STL INCLUDES
#include <string.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <map>
#include <initializer_list>
#include <iterator>
#include <algorithm>
#include <thread>
#include <memory>
// SYSTEM-INSTALLED LIBRARIES
#include <fftw3.h>
// LOCAL INCLUDES
#include "helpers.hpp"
#include "signal.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

// This function is a small script that calculates the FFT wisdom for all powers of two (since those
// are the only expected sizes to be used with the FFTs), and exports it to the given path. The
// wisdom is a brute-force search of the most efficient implementations for the FFTs: It takes a
// while to compute, but has to be done only once (per computer), and then it can be quickly loaded
// for faster FFT computation, as explained in the docs (http://www.fftw.org/#documentation).
// See also the docs for different flags. Note that using a wisdom file is optional.
void MakeAndExportFftwWisdom(const std::string path_out, const size_t min_2pow=0,
                        const size_t max_2pow=25, const unsigned flag=FFTW_PATIENT){
  std::vector<fftwf_plan> plans;
  for(size_t i=min_2pow; i<=max_2pow; ++i){
    size_t size = pow(2, i);
    FloatSignal fs(size);
    ComplexSignal cs(size/2+1);
    printf("creating forward and backward plans for size=2**%zu=%zu and flag %u...\n",i,size, flag);
    fftwf_plan fwd = fftwf_plan_dft_r2c_1d(fs.getSize(), fs.getData(),
                                           reinterpret_cast<fftwf_complex*>(&cs.getData()[0]),
                                           flag);
    fftwf_plan bwd = fftwf_plan_dft_c2r_1d(fs.getSize(),
                                           reinterpret_cast<fftwf_complex*>(&cs.getData()[0]),
                                           fs.getData(), flag | FFTW_PRESERVE_INPUT);
    plans.push_back(fwd);
    plans.push_back(bwd);
  }
  // after all plans have ben created, export wisdom and delete them
  std::cout << "MakeAndExportFftwWisdom: exporting wisdom to -->" << path_out;
  if(fftwf_export_wisdom_to_filename(path_out.c_str())){
    std::cout << "<-- was successful!" << std::endl;
  } else {
    std::cout << "<-- failed! ignoring..." << std::endl;
  }
  for(auto p : plans){
    fftwf_destroy_plan(p);
  }
}

// Given a path to a wisdom file generated with "MakeAndExportFftwWisdom", reads and loads it
// into FFTW to perform faster FFT computations. Using a wisdom file is optional.
void ImportFftwWisdom(const std::string path_in, const bool throw_exception_if_fail=true){
  int result = fftwf_import_wisdom_from_filename(path_in.c_str());
  if(result!=0){
    std::cout << "[ImportFftwWisdom] succesfully imported " << path_in << std::endl;
  } else{
    std::string message = "[ImportFftwWisdom] ";
    message += "couldn't import wisdom! is this a path to a valid wisdom file? -->"+path_in+"<--\n";
    if(throw_exception_if_fail){throw std::runtime_error(std::string("ERROR: ") + message);}
    else{std::cout << "WARNING: " << message;}
  }
}



////////////////////////////////////////////////////////////////////////////////////////////////////

struct FftTransformer {
  // THE TWO SIGNALS AND THE FFT PLANS BETWEEN THEM (GETTERS AREN'T NEEDED)
  FloatSignal* r;
  ComplexSignal* c;
  fftwf_plan forward_plan;
  fftwf_plan backward_plan;
  // THE CONSTRUCTOR GRABS THE REFERENCES TO THE SIGNALS, AND MAKES THE PLANS
  FftTransformer(FloatSignal* real, ComplexSignal* complex)
    : r(real), c(complex),
      forward_plan(fftwf_plan_dft_r2c_1d(r->getSize(), r->getData(),
                                         reinterpret_cast<fftwf_complex*>(&c->getData()[0]),
                                         FFTW_ESTIMATE)),
      backward_plan(fftwf_plan_dft_c2r_1d(r->getSize(),
                                          reinterpret_cast<fftwf_complex*>(&c->getData()[0]),
                                          r->getData(), FFTW_ESTIMATE | FFTW_PRESERVE_INPUT)){
    CheckRealComplexRatio(r->getSize(), c->getSize(), "FftTransformer");
  }
  // THE DESTRUCTOR TAKES CARE OF THE PLANS ONLY
  ~FftTransformer(){
    fftwf_destroy_plan(forward_plan);
    fftwf_destroy_plan(backward_plan);
  }
  // CONVENIENCE METHODS FOR EXECUTING THE FFT AND IFFT
  void forward(){fftwf_execute(forward_plan);}
  void backward(){fftwf_execute(backward_plan);}
};


////////////////////////////////////////////////////////////////////////////////////////////////////

//// This beauty-contest winning class does most of the dirty job implements the the overlap-save
// algorithm for doing spectral convolution and cross-correlation between a signal of length S and
// a patch of length P<=S (raises an exception if P>S). For that:
//
// 1. makes a copy of the patch (zero-padded at the end) with PP = max(kMinChunkSize, 2*Pow2Ceil(P))
// 2. pre-pads the signal with PP/2 zeros, and cuts it into overlapping chunks of length PP,
//    striding by V:=PP/2 steps. Note that the optimum would be V=PP-P+1, but this pipeline is
//    intended to be reused by all patches with same PP, which wouldn't be possible for such a
//    stride and would require re-instantiation of the whole pipeline (which is worse).
// 3. For each chunk creates a complex counterpart of size PP/2+1 and puts both to an FftTransformer
//    that takes care of the the FFT and IFFT plans and transformations (**NOTE**: the transforms
//    aren't implicitly done by the class, they have to be explicitly called as needed).
//
//// The convenience of this procedure is manifold:
// i.   The FFT for the signal chunks has to be performed only once (per padded_patch_size_).
// ii.  The FFT, IFFT and spectralConv operations can be parallelized, since they are independent.
// iii. Since FFT is O(n logn), this helps when patch is notably smaller than signal.
// iv.  When performing correlations among signals of different sizes, the FFT plans and signal
//      copies have to be generated only once per padded_patch_size_, which speeds up the creation
//      and saves memory.
//
//// Basic usage example:
//
// FloatSignal a([](long int x){return x+1;}, 44100);
// FloatSignal b([](long int x){return x+1;}, 4410);
// OverlapSaveConvolver xxx(a, b, true, true, 1);
// // cross-correlation pipeline
// xxx.forwardPatch();
// xxx.forwardSignal();
// xxx.spectralConv();
// xxx.backwardSignal();
// // extract result
// FloatSignal cc_placeholder(a.getSize()+b.getSize()-1);
// xxx.extractConvolvedTo(cc_placeholder);
//
//// Note that, to favour speed, the operations happen within float precision, which causes
// relatively small imprecisions: https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
// Also note that all the signal inputs are given by reference, and no specific action is needed
// to avoid memory leaking.
class OverlapSaveConvolver {
private:
  const size_t kMinChunkSize_;
  size_t signal_size_;
  size_t patch_size_;
  size_t conv_size_;
  size_t padded_patch_size_;
  size_t padded_size_half_;
  size_t padded_patch_size_complex_;
  //
  FftTransformer* patch_;
  std::vector<FftTransformer*> signal_vec_;
  //
public:
  explicit OverlapSaveConvolver(FloatSignal &signal, FloatSignal &patch,
                                const bool patch_reversed=true,  const bool normalize_patch=true,
                                const size_t min_chunk_size=2048)
    : kMinChunkSize_(min_chunk_size),
      signal_size_(signal.getSize()),
      patch_size_(patch.getSize()),
      padded_patch_size_(std::max(kMinChunkSize_, 2*Pow2Ceil(patch_size_))),
      padded_size_half_(padded_patch_size_/2),
      padded_patch_size_complex_(padded_size_half_+1){
    // length of signal can't be smaller than length of patch
    CheckLessEqual(patch_size_, signal_size_,
                   "OverlapSaveConvolver: len(signal) can't be smaller than len(patch)!");
    // copy and zero-pad patch: note that it is loaded in reverse order
    FloatSignal* padded_patch = new FloatSignal(padded_patch_size_);
    patch_ = new FftTransformer(padded_patch, new ComplexSignal(padded_patch_size_complex_));
    if(patch_reversed){std::reverse_copy(patch.begin(), patch.end(), padded_patch->begin());}
    else {std::copy(patch.begin(), patch.end(), padded_patch->begin());}
    if(normalize_patch){padded_patch->operator*=(1.0/padded_patch_size_);}
    // SIGNAL CHUNKING
    // copy and append the first, pre-padded signal chunk:
    float* it = signal.begin();
    float* end_it = signal.end();
    FloatSignal* sig = new FloatSignal(padded_patch_size_);
    std::copy(it, std::min(end_it, it+padded_size_half_), sig->begin()+padded_size_half_);
    signal_vec_.push_back(new FftTransformer(sig, new ComplexSignal(padded_patch_size_complex_)));
    // loop through the signal, adding further chunks
    for(; it<end_it; it+=padded_size_half_){
      FloatSignal* sig = new FloatSignal(padded_patch_size_);
      std::copy(it, std::min(end_it, it+padded_patch_size_), sig->begin());
      signal_vec_.push_back(new FftTransformer(sig, new ComplexSignal(padded_patch_size_complex_)));
    }
  }

  // GETTERS
  const size_t getPaddedSize() const {return padded_patch_size_;}
  const FftTransformer* getPatch() const {return patch_;}
  const std::vector<FftTransformer*>& getSignalVec() const {return signal_vec_;}

  // given a FloatSignal with length <= getPaddedSize (throws exception otherwise), rewrites the
  // patch (zero-padding the rest). If normalize_patch is true, the contents will be divided by
  // padded_patch_size (needed for the convolution). If fft_forward_after is true (default),
  // forwardPatch() is called after rewriting.
  void updatePatch(FloatSignal &new_patch, const bool reversed=true,
                   const bool normalize_patch=true,
                   const bool fft_forward_after=true){
    size_t new_size = new_patch.getSize();
    CheckLessEqual(new_size, padded_patch_size_,
                   "updatePatch(): len(new_patch) can't be smaller than getPaddedSize()!");
    patch_size_ = new_size;
    // copy new to old and zero-pad rest
    float* old_patch = patch_->r->getData();
    if(reversed){std::reverse_copy(new_patch.begin(), new_patch.end(), old_patch);}
    else {std::copy(new_patch.begin(), new_patch.end(), old_patch);}
    std::fill(old_patch+new_size, old_patch+padded_patch_size_, 0);
    // if flag is true, padded patch is divided by its length (FFTW does not normalize itself)
    if(normalize_patch){patch_->r->operator*=(1.0/padded_patch_size_);}
    //
    if(fft_forward_after){patch_->forward();}
  }

  void updateSignal(FloatSignal &signal, const bool fft_forward_after=true){
    // check length
    size_t sig_size = signal.getSize();
    CheckAllEqual({sig_size, signal_size_}, "updateSignal: length of signal can't change!");
    // SIGNAL CHUNKING
    // copy and append the first, pre-padded signal chunk:
    float* it = signal.begin();
    float* end_it = signal.end();
    FloatSignal* sig = signal_vec_[0]->r;
    float* sig_begin = sig->begin();
    std::fill(sig_begin, sig->end(), 0);
    std::copy(it, std::min(end_it, it+padded_size_half_), sig_begin+padded_size_half_);
    // loop for the rest. OPTIMIZE THIS!
    for(size_t i=1; it<end_it; it+=padded_size_half_, ++i){
      FloatSignal* sig = signal_vec_[i]->r;
      float* sig_begin = sig->begin();
      std::fill(sig_begin, sig->end(), 0);
      std::copy(it, std::min(end_it, it+padded_patch_size_), sig_begin);
    }

    if(fft_forward_after){forwardSignal();}
  }

  // calculates and writes the FFT of the patch->real into the patch->complex.
  void forwardPatch(){patch_->forward();}

  // calculates and writes the IFFT of the patch->complex into the patch->real.
  void backwardPatch(){patch_->backward();}

  // calculates and writes the FFT of every signal_vec[i]->real into the signal_vec[i]->complex.
  void forwardSignal(){
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for(size_t i=0; i<signal_vec_.size(); i++){
      signal_vec_[i]->forward();
    }
  }

  // calculates and writes the IFFT of every signal_vec[i]->complex into the signal_vec[i]->real.
  void backwardSignal(){
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for(size_t i=0; i<signal_vec_.size(); i++){
      signal_vec_[i]->backward();
    }
  }

  // This method performs destiny[i] = patch*signal_vec[i], for all i in the signal vector.
  // In other words, multiplies element-weise every the patch and signal spectra, which is a
  // necessary step for the convolution in the overlap-save algorithm.
  void spectralConv(){
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for(size_t i=0; i<signal_vec_.size(); i++){
      signal_vec_[i]->c->operator*=(*(patch_->c));
    }
  }

  // Writes cc_size=(signal_size_+patch_size_-1) elements to the passed FloatSignal, starting at
  // min_i. It throws an exception if len(sig)<(cc_size+min_i).
  void extractConvolvedTo(FloatSignal &sig, const size_t min_i=0){
    const size_t kConvSize = min_i+signal_size_+patch_size_-1;
    CheckLessEqual(sig.getSize(), kConvSize+min_i,
                   "extractSignalTo: given sig length can't be smaller than sig+patch+min_i-1!");
    // iterative variables
    float* sig_data = sig.begin();
    float* sig_loop_end = sig_data+kConvSize-padded_size_half_;
    float* sig_i = sig_data;
    float* fs_begin;
    size_t x = 0;
    // copy all chunks but the last
    for(; sig_i<sig_loop_end; ++x, sig_i+=padded_size_half_){
      fs_begin = signal_vec_[x]->r->begin()+padded_size_half_;
      std::copy(fs_begin, fs_begin+padded_size_half_, sig_i);
    }
    // copy last chunk
    long int remaining = sig_data+kConvSize-sig_i;
    if(remaining>0){
      fs_begin = signal_vec_[x]->r->begin()+padded_size_half_;
      std::copy(fs_begin, fs_begin+remaining, sig_i);
    }
}

  ~OverlapSaveConvolver(){
    if(patch_!=nullptr){
      delete patch_->r;
      delete patch_->c;
      delete patch_;
    }
    // clear vectors holding signals
    for(const FftTransformer* x : signal_vec_){
      delete x->r;
      delete x->c;
      delete x;
    }
    signal_vec_.clear();
  }
};



#endif
