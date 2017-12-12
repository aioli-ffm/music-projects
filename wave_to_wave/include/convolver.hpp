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
// SYSTEM-INSTALLED LIBRARIES
#include <fftw3.h>
// LOCAL INCLUDES
#include "helpers.hpp"
#include "signal.hpp"
#include "synth.hpp"


////////////////////////////////////////////////////////////////////////////////////////////////////

// This class is a simple wrapper for the memory management of the fftw plans. It is not expected
// to be used directly: rather, to be extended by specific plans, for instance, if working with
// real, 1D signals, only 1D complex<->real plans are needed.
class FftPlan{
private:
  fftwf_plan plan_;
public:
  explicit FftPlan(fftwf_plan p): plan_(p){}
  virtual ~FftPlan(){fftwf_destroy_plan(plan_);}
  void execute(){fftwf_execute(plan_);}
};

// This forward plan (1D, R->C) is adequate to process 1D floats (real).
class FftForwardPlan : public FftPlan{
public:
  // This constructor creates a real->complex plan that performs the FFT(real) and saves it into the
  // complex. Throws an exception if size(complex)!=size(real)/2+1, s explained in
  // http://www.fftw.org/#documentation
  explicit FftForwardPlan(FloatSignal &fs, ComplexSignal &cs)
    : FftPlan(fftwf_plan_dft_r2c_1d(fs.getSize(), fs.getData(),
                                    reinterpret_cast<fftwf_complex*>(&cs.getData()[0]),
                                    FFTW_ESTIMATE)){
    CheckRealComplexRatio(fs.getSize(), cs.getSize(), "FftForwardPlan");
  }
};

// This backward plan (1D, C->R) is adequate to process spectra of 1D floats (real).
class FftBackwardPlan : public FftPlan{
public:
  // This constructor creates a complex->real plan that performs the IFFT(complex) and saves it into
  // the real. Throws an exception if size(complex)!=size(real)/2+1, s explained in
  // http://www.fftw.org/#documentation
  explicit FftBackwardPlan(ComplexSignal &cs, FloatSignal &fs)
    : FftPlan(fftwf_plan_dft_c2r_1d(fs.getSize(),
                                    reinterpret_cast<fftwf_complex*>(&cs.getData()[0]),
                                    fs.getData(), FFTW_ESTIMATE)){
    CheckRealComplexRatio(fs.getSize(), cs.getSize(), "FftBackwardPlan");
  }
};



////////////////////////////////////////////////////////////////////////////////////////////////////

// This function is a small script that calculates the FFT wisdom for all powers of two (since those
// are the only expected sizes to be used with the FFTs), and exports it to the given path. The
// wisdom is a brute-force search of the most efficient implementations for the FFTs: It takes a
// while to compute, but has to be done only once (per computer), and then it can be quickly loaded
// for faster FFT computation, as explained in the docs (http://www.fftw.org/#documentation).
// See also the docs for different flags. Note that using a wisdom file is optional.
void MakeAndExportFftwWisdom(const std::string path_out, const size_t min_2pow=0,
                        const size_t max_2pow=25, const unsigned flag=FFTW_PATIENT){
  for(size_t i=min_2pow; i<=max_2pow; ++i){
    size_t size = pow(2, i);
    FloatSignal fs(size);
    ComplexSignal cs(size/2+1);
    printf("creating forward and backward plans for size=2**%zu=%zu and flag %u...\n",i,size, flag);
    FftForwardPlan fwd(fs, cs);
    FftBackwardPlan bwd(cs, fs);
  }
  std::cout << "MakeAndExportFftwWisdom: exporting wisdom to -->" << path_out;
  if(fftwf_export_wisdom_to_filename(path_out.c_str())){
    std::cout << "<-- was successful!" << std::endl;
  } else {
    std::cout << "<-- failed! ignoring..." << std::endl;
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
/// PERFORM CONVOLUTION/CORRELATION
////////////////////////////////////////////////////////////////////////////////////////////////////

// This class performs an efficient version of the spectral convolution/cross-correlation between
// two 1D float arrays, <SIGNAL> and <PATCH>, called overlap-save:
//http://www.comm.utoronto.ca/~dkundur/course_info/real-time-DSP/notes/8_Kundur_Overlap_Save_Add.pdf
// This algorithm requires that the length of <PATCH> is less or equal the length of <SIGNAL>,
// so an exception is thrown otherwise. The algorithm works as follows:
// given signal of length S and patch of length P, and being the conv (or xcorr) length U=S+P-1
//   1. pad the patch to X = 2*Pow2Ceil(P). FFTs with powers of 2 are the fastest.
//   2. cut the signal into chunks of size X, with an overlapping section of L=X-(P-1).
//      for that, pad the signal with (P-1) before, and with (X-U%L) after, to make it fit exactly.
//   3. Compute the forward FFT of the padded patch and of every chunk of the signal
//   4. Multiply the FFT of the padded patch with every signal chunk.
//      4a. If the operation is a convolution, perform a complex a*b multiplication
//      4b. If the operation is a cross-correlation, perform a complex a*conj(b) multiplication
//   5. Compute the inverse FFT of every result of step 4
//   6. Concatenate the resulting chunks, ignoring (P-1) samples per chunk
// Note that steps 3,4,5 may be parallelized with some significant gain in performance.
// In this class: X = result_chunksize, L = result_stride
class OverlapSaveConvolver {
private:
  const size_t kMinChunksize = 2048; // seems to be the fastest
  // grab input lengths
  size_t signal_size_;
  size_t patch_size_;
  size_t result_size_;
  // make padded copies of the inputs and get chunk measurements
  FloatSignal padded_patch_;
  size_t result_chunksize_;
  size_t result_chunksize_complex_;
  size_t result_stride_;
  ComplexSignal padded_patch_complex_;
  // padded copy of the signal
  FloatSignal padded_signal_;
  // the deconstructed signal
  std::vector<FloatSignal*> s_chunks_;
  std::vector<ComplexSignal*> s_chunks_complex_;
  // the corresponding chunks holding convs/xcorrs
  std::vector<FloatSignal*> result_chunks_;
  std::vector<ComplexSignal*> result_chunks_complex_;
  // the corresponding plans (plus the plan of the patch)
  std::vector<FftForwardPlan*> forward_plans_;
  std::vector<FftBackwardPlan*> backward_plans_;

  // Basic state management to prevent getters from being called prematurely.
  // Also to adapt the extractResult getter, since Conv and Xcorr padding behaves differently
  enum class State {kUninitialized, kConv, kXcorr};
  State _state_; // kUninitialized after instantiation, kConv/kXcorr after respective op.
  // This private method throws an exception if _state_ is kUninitialized, because that
  // means that some "getter" has ben called before any computation has been performed.
  void __check_last_executed_not_null(const std::string method_name){
    if(_state_ == State::kUninitialized){
      throw std::runtime_error(std::string("[ERROR] OverlapSaveConvolver.") + method_name +
                               "() can't be called before executeXcorr() or executeConv()!" +
                               " No meaningful data has been computed yet.");
    }
  }

  // This private method implements steps 3,4,5 of the algorithm. If the given flag is false,
  // it will perform a convolution (4a), and a cross-correlation (4b) otherwise.
  // Note the parallelization with OpenMP, which increases performance in supporting CPUs.
  void __execute(const bool cross_correlate){
    auto operation = (cross_correlate)? ComplexConjMul : ComplexMul;
    // do ffts
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for (size_t i =0; i<forward_plans_.size();i++){
      forward_plans_.at(i)->execute();
    }
    // multiply spectra
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for (size_t i =0; i<result_chunks_.size();i++){
      operation(*s_chunks_complex_.at(i), this->padded_patch_complex_,
                *result_chunks_complex_.at(i));
    }
    // do iffts
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for (size_t i =0; i<result_chunks_.size();i++){
      backward_plans_.at(i)->execute();
      *result_chunks_.at(i) *= (1.0f/result_chunksize_);
    }
  }

public:
  // The only constructor for the class, receives two signals and performs steps 1 and 2 of the
  // algorithm on them. The signals are passed by reference but the class works with padded copies
  // of them, so no care has to be taken regarding memory management.
  // The wisdomPath may be empty, or a path to a valid wisdom file.
  // Note that len(signal) can never be smaller than len(patch), or an exception is thrown.
  OverlapSaveConvolver(FloatSignal &signal, FloatSignal &patch, const std::string wisdomPath="")
    : signal_size_(signal.getSize()),
      patch_size_(patch.getSize()),
      result_size_(signal_size_+patch_size_-1),
      //
      padded_patch_(patch.getData(), patch_size_, 0,
                    std::max(kMinChunksize, 2*Pow2Ceil(patch_size_)-patch_size_)),
      result_chunksize_(padded_patch_.getSize()),
      result_chunksize_complex_(result_chunksize_/2+1),
      result_stride_(result_chunksize_-patch_size_+1),
      padded_patch_complex_(result_chunksize_complex_),
      //
      padded_signal_(signal.getData(),signal_size_,patch_size_-1,
                     result_chunksize_-(result_size_%result_stride_)),
      _state_(State::kUninitialized){
      // end of initializer list, now check that len(signal)>=len(patch)
    CheckLessEqual(patch_size_, signal_size_,
                         "OverlapSaveConvolver: len(signal) can't be smaller than len(patch)!");
    // and load the wisdom if required. If unsuccessful, no exception thrown, just print a warning.
    if(!wisdomPath.empty()){ImportFftwWisdom(wisdomPath, false);}
    // chunk the signal into strides of same size as padded patch
    // and make complex counterparts too, as well as the corresponding xcorr signals
    for(size_t i=0; i<=padded_signal_.getSize()-result_chunksize_; i+=result_stride_){
      s_chunks_.push_back(new FloatSignal(&padded_signal_[i], result_chunksize_));
      s_chunks_complex_.push_back(new ComplexSignal(result_chunksize_complex_));
      result_chunks_.push_back(new FloatSignal(result_chunksize_));
      result_chunks_complex_.push_back(new ComplexSignal(result_chunksize_complex_));
    }
    // make one forward plan per signal chunk, and one for the patch
    // Also backward plans for the xcorr chunks
    forward_plans_.push_back(new FftForwardPlan(padded_patch_, padded_patch_complex_));
    for (size_t i =0; i<s_chunks_.size();i++){
      forward_plans_.push_back(new FftForwardPlan(*s_chunks_.at(i), *s_chunks_complex_.at(i)));
      backward_plans_.push_back(new FftBackwardPlan(*result_chunks_complex_.at(i),
                                                     *result_chunks_.at(i)));
    }
  }

  //
  void executeConv(){
    __execute(false);
    _state_ = State::kConv;
  }
  void executeXcorr(){
    __execute(true);
    _state_ = State::kXcorr;
  }
  // getting info from the convolfer
  void printChunks(const std::string name="convolver"){
    __check_last_executed_not_null("printChunks");
    for (size_t i =0; i<result_chunks_.size();i++){
      std::cout << name << "_chunk_" << i << std::endl;

      result_chunks_.at(i)->print(name+"_chunk_"+std::to_string(i));
    }
  }

  // This method implements step 6 of the overlap-save algorithm. In convolution, the first (P-1)
  // samples of each chunk are discarded, in xcorr the last (P-1) ones. Therefore, depending on the
  // current _state_, the corresponding method is used. USAGE:
  // Every time it is called, this function returns a new FloatSignal instance of size
  // len(signal)+len(patch)-1. If the last operation performed was executeConv(), this function
  // will return the  convolution of signal and patch. If the last operation performed was
  // executeXcorr(), the result will contain the cross-correlation. If none of them was performed
  // at the moment of calling this function, an exception will be thrown.
  // The indexing will start with the most negative relation, and increase accordingly. Which means:
  //   given S:=len(signal), P:=len(patch), T:=S+P-1
  // for 0 <= i < T, result[i] will hold dot_product(patch, signal[i-(P-1) : i])
  //   where patch will be "reversed" if the convolution was performed. For example:
  // Signal :=        [1 2 3 4 5 6 7]    Patch = [1 1 1]
  // Result[0] =  [1 1 1]                        => 1*1         = 1  // FIRST ENTRY
  // Result[1] =    [1 1 1]                      => 1*1+1*2     = 3
  // Result[2] =      [1 1 1]                    => 1*1+1*2+1*3 = 8  // FIRST NON-NEG ENTRY AT P-1
  //   ...
  // Result[8] =                  [1 1 1]        => 1*7         = 7  // LAST ENTRY
  // Note that the returned signal object takes care of its own memory, so no management is needed.
  FloatSignal extractResult(){
    // make sure that an operation was called before
    __check_last_executed_not_null("extractResult");
    // set the offset for the corresponding operation (0 for xcorr).
    size_t discard_offset = 0;
    if(_state_==State::kConv){discard_offset = result_chunksize_ - result_stride_;}
    // instantiate new signal to be filled with the desired info
    FloatSignal result(result_size_);
    float* result_arr = result.getData(); // not const because of memcpy
    // fill!
    static size_t kNumChunks = result_chunks_.size();
    for (size_t i=0; i<kNumChunks;i++){
      float* xc_arr = result_chunks_.at(i)->getData();
      const size_t kBegin = i*result_stride_;
      // if the last chunk goes above result_size_, reduce copy size. else copy_size=result_stride_
      size_t copy_size = result_stride_;
      copy_size -= (kBegin+result_stride_>result_size_)? kBegin+result_stride_-result_size_ : 0;
      memcpy(result_arr+kBegin, xc_arr+discard_offset, sizeof(float)*copy_size);
    }
    return result;
  }
  ~OverlapSaveConvolver(){
    // clear vectors holding signals
    for (size_t i =0; i<s_chunks_.size();i++){
      delete (s_chunks_.at(i));
      delete (s_chunks_complex_.at(i));
      delete (result_chunks_.at(i));
      delete (result_chunks_complex_.at(i));
    }
    s_chunks_.clear();
    s_chunks_complex_.clear();
    result_chunks_.clear();
    result_chunks_complex_.clear();
    // clear vector holding forward FFT plans
    for (size_t i =0; i<forward_plans_.size();i++){
      delete (forward_plans_.at(i));
    }
    forward_plans_.clear();
    // clear vector holding backward FFT plans
    for (size_t i =0; i<backward_plans_.size();i++){
      delete (backward_plans_.at(i));
    }
    backward_plans_.clear();
  }
};














class CrossCorrelator {
private:
  const size_t kMinChunksize = 2048; // seems to be the fastest
  // grab input lengths
  size_t signal_size_;
  size_t patch_size_;
  size_t result_size_;
  // make padded copies of the inputs and get chunk measurements
  FloatSignal padded_patch_;
  size_t result_chunksize_;
  size_t result_chunksize_complex_;
  size_t result_stride_;
  ComplexSignal padded_patch_complex_;
  // padded copy of the signal
  FloatSignal padded_signal_;
  // the deconstructed signal
  std::vector<FloatSignal*> s_chunks_;
  std::vector<ComplexSignal*> s_chunks_complex_;
  // the corresponding chunks holding convs/xcorrs
  std::vector<FloatSignal*> result_chunks_;
  std::vector<ComplexSignal*> result_chunks_complex_;
  // the corresponding plans (plus the plan of the patch)
  std::vector<FftForwardPlan*> forward_plans_;
  std::vector<FftBackwardPlan*> backward_plans_;


public:
  CrossCorrelator(FloatSignal &signal, FloatSignal &patch, const std::string wisdomPath="")
    : signal_size_(signal.getSize()),
      patch_size_(patch.getSize()),
      result_size_(signal_size_+patch_size_-1),
      //
      padded_patch_(patch.getData(), patch_size_, 0,
                    std::max(kMinChunksize, 2*Pow2Ceil(patch_size_)-patch_size_)),
      result_chunksize_(padded_patch_.getSize()),
      result_chunksize_complex_(result_chunksize_/2+1), // THIS IS THE DICT KEY
      result_stride_(result_chunksize_-patch_size_+1), // THIS CAN BE PROBLEMATIC FOR THE VALUES, BUT MAYBE ISNT
      padded_patch_complex_(result_chunksize_complex_),
      //
      padded_signal_(signal.getData(),signal_size_,patch_size_-1, // THIS SHOULD BE CONSUMED FROM DICT!
                     result_chunksize_-(result_size_%result_stride_)){
      // end of initializer list, now check that len(signal)>=len(patch)
    CheckLessEqual(patch_size_, signal_size_,
                         "OverlapSaveConvolver: len(signal) can't be smaller than len(patch)!");
    // and load the wisdom if required. If unsuccessful, no exception thrown, just print a warning.
    if(!wisdomPath.empty()){ImportFftwWisdom(wisdomPath, false);}
    // chunk the signal into strides of same size as padded patch
    // and make complex counterparts too, as well as the corresponding xcorr signals
    for(size_t i=0; i<=padded_signal_.getSize()-result_chunksize_; i+=result_stride_){
      s_chunks_.push_back(new FloatSignal(&padded_signal_[i], result_chunksize_));
      s_chunks_complex_.push_back(new ComplexSignal(result_chunksize_complex_));
      result_chunks_.push_back(new FloatSignal(result_chunksize_));
      result_chunks_complex_.push_back(new ComplexSignal(result_chunksize_complex_));
    }
    // make one forward plan per signal chunk, and one for the patch
    // Also backward plans for the xcorr chunks
    forward_plans_.push_back(new FftForwardPlan(padded_patch_, padded_patch_complex_));
    for (size_t i =0; i<s_chunks_.size();i++){
      forward_plans_.push_back(new FftForwardPlan(*s_chunks_.at(i), *s_chunks_complex_.at(i)));
      backward_plans_.push_back(new FftBackwardPlan(*result_chunks_complex_.at(i),
                                                     *result_chunks_.at(i)));
    }
  }

    FloatSignal makeXcorr(){
    // do ffts
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for (size_t i =0; i<forward_plans_.size();i++){
      forward_plans_.at(i)->execute();
    }
    // multiply spectra
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for (size_t i =0; i<result_chunks_.size();i++){
      ComplexConjMul(*s_chunks_complex_.at(i), this->padded_patch_complex_,
                *result_chunks_complex_.at(i));
    }
    // do iffts
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for (size_t i =0; i<result_chunks_.size();i++){
      backward_plans_.at(i)->execute();
      *result_chunks_.at(i) *= (1.0f/result_chunksize_);
    }

    return extractResult();
  }

  FloatSignal extractResult(){
    // set the offset for the corresponding operation (0 for xcorr).
    // instantiate new signal to be filled with the desired info
    FloatSignal result(result_size_);
    float* result_arr = result.getData(); // not const because of memcpy
    // fill!
    static size_t kNumChunks = result_chunks_.size();
    for (size_t i=0; i<kNumChunks;i++){
      float* xc_arr = result_chunks_.at(i)->getData();
      const size_t kBegin = i*result_stride_;
      // if the last chunk goes above result_size_, reduce copy size. else copy_size=result_stride_
      size_t copy_size = result_stride_;
      copy_size -= (kBegin+result_stride_>result_size_)? kBegin+result_stride_-result_size_ : 0;
      memcpy(result_arr+kBegin, xc_arr, sizeof(float)*copy_size);
    }
    return result;
  }

    // getting info from the convolver
  void printChunks(const std::string name="convolver"){
    for (size_t i =0; i<result_chunks_.size();i++){
      std::cout << name << "_chunk_" << i << std::endl;
      result_chunks_.at(i)->print(name+"_chunk_"+std::to_string(i));
    }
  }

  ~CrossCorrelator(){
    // clear vectors holding signals
    for (size_t i =0; i<s_chunks_.size();i++){
      delete (s_chunks_.at(i));
      delete (s_chunks_complex_.at(i));
      delete (result_chunks_.at(i));
      delete (result_chunks_complex_.at(i));
    }
    s_chunks_.clear();
    s_chunks_complex_.clear();
    result_chunks_.clear();
    result_chunks_complex_.clear();
    // clear vector holding forward FFT plans
    for (size_t i =0; i<forward_plans_.size();i++){
      delete (forward_plans_.at(i));
    }
    forward_plans_.clear();
    // clear vector holding backward FFT plans
    for (size_t i =0; i<backward_plans_.size();i++){
      delete (backward_plans_.at(i));
    }
    backward_plans_.clear();
  }
};




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


//// This beauty-contest winning class does most of the dirty job required by the cross-correlation
// via the overlap-save algorithm. Given a signal of length S and a patch of length P<=S (raises
// exception if P>S):
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
// ii.  The FFT, IFFT and spectralConv operations can be parallelized as they are independent.
// iii. Since FFT is O(n logn), this slightly helps when patch is notably smaller than signal.
// iv.  The FFT plans and signal copies have to be generated only once (per padded_patch_size_)
//
//// To leverage this advantages, the pipeline can be used as follows:
// a. After instantiation, save it to a map<size_t, ConvolverPipeline> with the getPaddedSize()
//    as key, and check that map upon reusage in the next iterations.
// b. When reusing, just the patch has to be rewritten. For that, use the updatePatch() method
// c. If used in an iterative optimizer among multiple pipelines, it may be needed to update
//    the signal too. For that, use the updateSignal() method
// d  The FFTs and IFFTs(non-destructive) can be performed at will by calling forward() and
//    backward() on the FftTransformer structs returned by getPatch() and getSignalVec().
// d. The correlation operation is expected to be done in-place (but doesn't have to). Can be done
//    at will with the signals held in the FftTransformer structs.
//// As a final remark, note that all the signal inputs are given by reference, and the generated
// copies are managed by the destructor, so no action is needed to avoid memory leaking.
class ConvolverPipeline {
private:
  const size_t kMinChunkSize_;
  size_t signal_size_;
  size_t patch_size_;
  size_t padded_patch_size_;
  size_t padded_size_half_;
  size_t padded_patch_size_complex_;
  //
  FftTransformer* patch_;
  std::vector<FftTransformer*> signal_vec_;
  //
public:
  ConvolverPipeline(FloatSignal &signal, FloatSignal &patch, const bool patch_reversed=true,
                    const bool normalize_patch=true,
                    const size_t min_chunk_size=2048)
    : kMinChunkSize_(min_chunk_size),
      signal_size_(signal.getSize()),
      patch_size_(patch.getSize()),
      padded_patch_size_(std::max(kMinChunkSize_, 2*Pow2Ceil(patch_size_))),
      padded_size_half_(padded_patch_size_/2),
      padded_patch_size_complex_(padded_size_half_+1){
    // length of signal can't be smaller than length of patch
    CheckLessEqual(patch_size_, signal_size_,
                   "ConvolverPipeline: len(signal) can't be smaller than len(patch)!");
    // copy and zero-pad patch: note that it is loaded in reverse order
    FloatSignal* padded_patch = new FloatSignal(padded_patch_size_);
    patch_ = new FftTransformer(padded_patch, new ComplexSignal(padded_patch_size_complex_));
    if(patch_reversed){
      std::reverse_copy(patch.begin(), patch.end(), padded_patch->begin());
    } else {
      std::copy(patch.begin(), patch.end(), padded_patch->begin());
    }
    // if flag is true, padded patch is divided by its length (FFTW does not normalize itself)
    if(normalize_patch){padded_patch->operator*=(1.0/padded_patch_size_);}
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
  // patch (zero-padding the rest). If fft_forward_after is true (default), forwardPatch() is
  // called after.
  void updatePatch(FloatSignal &new_patch, const bool reversed=true,
                   const bool normalize_patch=true,
                   const bool fft_forward_after=true){
    size_t new_size = new_patch.getSize();
    CheckLessEqual(new_size, padded_patch_size_,
                   "updatePatch(): len(new_patch) can't be smaller than getPaddedSize()!");
    float* old_patch = patch_->r->getData();
    if(reversed){
      std::reverse_copy(new_patch.begin(), new_patch.end(), old_patch);
    } else {
      std::copy(new_patch.begin(), new_patch.end(), old_patch);
    }
    std::fill(old_patch+new_size, old_patch+padded_patch_size_, 0);
    // if flag is true, padded patch is divided by its length (FFTW does not normalize itself)
    if(normalize_patch){patch_->r->operator*=(1.0/padded_patch_size_);}
    //
    if(fft_forward_after){patch_->forward();}
  }
  // NOTE THAT THE FORWARD FFTs ARE NOT EXECUTED, THIS JUST UPDATES THE FLOATS
  void updateSignal(FloatSignal &signal, const bool fft_forward_after=true){
    // check length
    size_t sig_size = signal.getSize();
    CheckAllEqual({sig_size, signal_size_}, "updateSignal: length of signal can't change!");
    // copy the first, pre-padded signal chunk to the existing float signal
    float* it = signal.begin();
    float* end_it = signal.end();
    FloatSignal* sig = signal_vec_[0]->r;
    std::copy(it, std::min(end_it, it+padded_size_half_), sig->begin()+padded_size_half_);
    // loop through the signal, adding further chunks
    for(size_t i=1, end_i=signal_vec_.size(); i<end_i; ++i, it+=padded_size_half_){
      std::copy(it, std::min(end_it, it+padded_patch_size_), signal_vec_[i]->r->begin());
    }
    if(fft_forward_after){forwardSignal();}
  }
  //
  void forwardPatch(){patch_->forward();}
  void backwardPatch(){patch_->backward();}
  void forwardSignal(){
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for(size_t i=0; i<signal_vec_.size(); i++){
      signal_vec_[i]->forward();
    }
  }
  void backwardSignal(){
    // #ifdef WITH_OPENMP_ABOVE
    // #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    // #endif
    for(size_t i=0; i<signal_vec_.size(); i++){
      signal_vec_[i]->backward();
    }
  }

  // This method performs destiny[i] = patch*signal_vec[i], for all i in the signal vector.
  // In other words, multiplies element-weise every the patch and signal spectra, which is a
  // necessary step for the convolution in the overlap-save algorithm.
  void multiplyPatchWithSig(){
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for(size_t i=0; i<signal_vec_.size(); i++){
      signal_vec_[i]->c->operator*=(*(patch_->c));
    }
  }

  ~ConvolverPipeline(){
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


class Optimizer {
private:
  FloatSignal& original_;
  FloatSignal residual_;
  Chi2Server chi2_server_;
  std::vector<std::tuple<float, float, float> > sequence_;
  std::map<size_t, std::vector<FloatSignal> > originals_; // key is a power of 2 being chunk_size
  std::map<size_t, FloatSignal> materials_; // key is a power of 2 being chunk_size
  std::map<size_t, FftForwardPlan> forward_plans_; // key is a power of 2 being chunk_size
  std::map<size_t, FftBackwardPlan> backward_plans_; // key is a power of 2 being chunk_size
public:
  Optimizer(FloatSignal &original, size_t downsample_ratio=1)
    : original_(original), residual_(original.getData(), original.getSize()){
    // how to handle downsampling?
    // further housekeeping here.
  }

  void step(const size_t size, const double freq, const double env_ratio){
    // this method is to be called many times.

    // It should generate the blobs on the fly (using the server attribute),
    // copy them to the corresponding pow2 chunk (create one+plans if non existent),
    // also use/populate the corresponding original(keep in mind enough pre-padding has to be added)
    // execute the forward plans, multiplications and backward plans.
    // After that the most recent xcorr should be reparted in the iffted chunks, to be further used

    // then based on the current data in the iffted chunks, PLUS A GIVEN CRITERION (by default just
    // find one abs maximum per chunk), apply the current blob to the residual and addto the seq.

  }

  void optimize(size_t num_iters){
    // should call step num_iters and other main config. Should have a nice interface.
  }


  ~Optimizer(){
    // this fn should take care of deleting all the map and vector contents.
  }
};




#endif
