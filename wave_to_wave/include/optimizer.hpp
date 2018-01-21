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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H


// STL INCLUDES
#include <map>
#include <string>
// LOCAL INCLUDE
#include "w2w.hpp"



////////////////////////////////////////////////////////////////////////////////////////////////////
/// CRITERIA
////////////////////////////////////////////////////////////////////////////////////////////////////

// Given a signal, expected to be the cross-correlation between two other signals, returns a
// vector<position, value> of one single element with the values for the absolute maximum
std::vector<std::pair<long int, float> > SingleMaxCriterium(FloatSignal &fs){
  std::vector<std::pair<long int, float> > result;
  float* abs_max = std::max_element(fs.begin(), fs.end(), abs_compare<float>);
  result.push_back(std::pair<long int,float>(abs_max-fs.begin(), *abs_max));
  return result;
};

// Given a signal, expected to be the cross-correlation between a signal (longer) and a patch
// (shorter), returns a vector<position, value> that holds the absolute maximum, and all the
// local absolute maxima that finds before and after it,making sure that the distance between
// two elements of the vector is at least patch_sz, to a void interferences.
// The name reflects the intention of finding many non-colliding good values for a single
// correlation, which can greatly speed up the optimization process.
std::vector<std::pair<long int, float> > PopulateMaxCriterium(FloatSignal &fs,
                                                           size_t patch_sz,
                                                           float eps){
  std::vector<std::pair<long int, float> > result = SingleMaxCriterium(fs);
  float* fs_begin = fs.begin();
  float* fs_max = fs_begin+result.at(0).first;
  // std::cout << "maximum at: " << result.at(0).first << std::endl;
  float* end = fs.end();
  for(float* it=fs_max+patch_sz; it<end; it+=patch_sz){
    it = std::max_element(it, std::min(end, it+patch_sz), abs_compare<float>);
    if(std::abs(*it)>eps){
      result.push_back(std::pair<long int, float>(it-fs_begin, *it));
    }
  }
  const size_t kTwicePatchSize = (2*patch_sz)-1;
  for(float* it=fs_max-(patch_sz-1); it>fs_begin; it-=kTwicePatchSize){
    it = std::max_element(std::max(it-(patch_sz-1),fs_begin),it,abs_compare<float>);
    if(std::abs(*it)>eps){
      result.push_back(std::pair<long int, float>(it-fs_begin, *it));
    }
  }
  // std::cout << "vector:"<< std::endl;
  // for(const auto& x : result){
  //   std::cout << x.first << "   " << x.second << std::endl;
  // }
  return result;
};



////////////////////////////////////////////////////////////////////////////////////////////////////
/// OPTIMIZERS
////////////////////////////////////////////////////////////////////////////////////////////////////

// This class is the general implementation of the wav2wav algorithm. Given a signal to be
// optimized, it provides the "step" method, which, given a smaller signal, can be performed
// several times to build a reconstruction.
// It also provides methods for retrieving/exporting the reconstruction's current wave,
// sequence and energy.
class WavToWavOptimizer{
private:
  const std::string seq_separator_;
  const size_t kMinChunkSize_; // given at construction, unchanged
  FloatSignal& original_;      // given at construction, unchanged
  size_t original_size_;       // given at construction, unchanged
  FftTransformer*  residual_;  // copy of the original at construction, altered by step method
  std::vector<std::string> sequence_; // empty at construction, filled by step method
  std::map<std::pair<size_t, size_t>, OverlapSaveConvolver*> pipelines_; // machinery for performing cross-correlation
public:
  // CONSTRUCTOR AND DESTRUCTOR
  explicit WavToWavOptimizer(FloatSignal &original, const size_t min_chunk_size=2048,
                             const std::string seq_separator="||")
    : seq_separator_(seq_separator),
      kMinChunkSize_(min_chunk_size),
      original_(original),
      original_size_(original.getSize()),
      residual_(new FftTransformer(new FloatSignal(original.begin(), original_size_),
                                   new ComplexSignal(original_size_/2+1))){
    residual_->forward();
  }
  ~WavToWavOptimizer(){
    for (const auto& kv : pipelines_) {
      delete kv.second;
    }
    delete residual_->r;
    delete residual_->c;
    delete residual_;
  }

  // GETTERS
  std::string getSeqSeparator(){return seq_separator_;}
  FftTransformer* getResidual(){return residual_;}
  size_t getSeqLength(){return sequence_.size();}
  float getResidualEnergy(){
    FloatSignal* r = residual_->r;
    return std::inner_product(r->begin(), r->end(), r->begin(), 0.0f);
  }
  // Since the residual=original-reconstruction, it holds: reconstruction = original-residual.
  FloatSignal getReconstruction(){
    FloatSignal reconstruction(original_.begin(), original_size_);
    reconstruction -= *residual_->r;
    return reconstruction;
  }
  // Saves the current optimization sequence as an ASCII file with one
  void exportTxtSequence(const std::string path_out){
    std::ofstream out(path_out);
    if (out.is_open()){
      for(const std::string& s : sequence_){out << s << std::endl;}
      out.close();
    }
    else{
      throw std::invalid_argument("exportTxtSequence: unable to open output stream "+path_out);
    }
  }

  // OPTIMIZATION STEP:
  // Given:
  //  * a patch signal (expected to be shorter than the one given at construction) and its energy
  //  * a stride>=1 integer that, using the Signal.makeStride method will speed-up the analysis
  //    reducing its precision and resolution (only 1 every 'stride' samples regarded).
  //  * a criterium function of signature: signal->vec(long int, float), that will receive the
  //    NON-NORMALIZED cross-correlation between the strided original and the patch, and is
  // expected to return a vector of (position, intensity) tuples taken from that correlation
  //  * a string to be appended to the "position||intensity||" line that will be generated for each
  //    output of the criterium
  // This function:
  //  1. makes strided copies of the patch and the current residual
  //  2. performs the cross-correlation between the strided patch and current residual
  //  3. applies the given optimization criterium to the cross-correlation
  //  4. for every output of the criterium, subtracts the (non-strided) patch from the (non-strided)
  //     residual at the given position
  // with the given intensity and adds the corresponding "position||intensity||extra_info" element
  // to the optimization sequence vector. After that, the function returns the output vector of
  // the criterium performed for this step (can be ignored in most cases).
  std::vector<std::pair<long int, float> > step(FloatSignal& patch, const float patch_energy,
                                                const size_t stride,
                                                std::function<std::vector<std::pair<long int, float>
                                                             >(FloatSignal&)> opt_criterium,
                                                const std::string extra_info){

    //
    FloatSignal* s_patch = (FloatSignal*)patch.makeStrided(stride);
    FloatSignal* s_residual = (FloatSignal*)residual_->r->makeStrided(stride);
    const size_t kResidualStridedSize = s_residual->getSize();

    // adjust and update metadata before loading the pipeline
    const size_t kPatchSize = s_patch->getSize();
    const size_t kPaddedSize = std::max(kMinChunkSize_, 2*Pow2Ceil(kPatchSize));
    const float kNormFactor = patch_energy*kPaddedSize;
    // const bool kRepeats = last_pipeline_==kPaddedSize;
    // last_pipeline_ = kPaddedSize;
    // get (or create if didn't exist) the corresponding pipeline, and update the signal spectrum
    OverlapSaveConvolver* convolver = nullptr;
    std::pair<size_t, size_t> key = std::make_pair(kPaddedSize, kResidualStridedSize);
    auto it = pipelines_.find(key);
    if ( it != pipelines_.end()){ // THERE WAS A PRE-EXISTING PIPELINE, UPDATE IT
      convolver = it->second;
      convolver->updatePatch(*s_patch, true, false, false); // reverse, normalize, fft_after
      convolver->updateSignal(*s_residual, false);
    } else{ // THERE WASN'T A PREEXISTING PIPELINE: MAKE A NEW ONE
      convolver = new OverlapSaveConvolver(*s_residual, *s_patch, true, false, kPaddedSize);
      // pipelines_.insert(std::pair<size_t, OverlapSaveConvolver*>(key, convolver));
      pipelines_.insert(std::make_pair(key, convolver));
    }
    // at this point we have a pipeline with all the FloatSignals up-to-date. Calculate FFT of
    // both, perform spectral cross-correlation and calculate IFFT of correlation
    convolver->forwardPatch();
    convolver->forwardSignal();
    convolver->spectralConv();
    convolver->backwardSignal();
    // extract NON-NORMALIZED conv results as a newly constructed signal
    FloatSignal result(kResidualStridedSize+kPatchSize-1);
    convolver->extractConvolvedTo(result);
    delete s_patch;
    delete s_residual;
    // apply criterium to extract a list of <POSITION, XCORR_VALUE> changes
    std::vector<std::pair<long int, float> > changes = opt_criterium(result);
    // prepare parallel optimization of all elements in changes
    const size_t changes_size = changes.size();
    const size_t seq_size = sequence_.size();
    sequence_.reserve(changes_size);
    // // subtract the changes returned by the criterium to the residual signal
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for (size_t i=0; i<changes_size; ++i){
      auto elt = changes[i];
      long int position = (elt.first-kPatchSize+1)*stride;
      float factor = elt.second/kNormFactor;
      residual_->r->subtractMultipliedSignal(patch, factor, position);
      #pragma omp critical
      sequence_.push_back(std::to_string(position)+seq_separator_+std::to_string(factor)+
                          seq_separator_+extra_info);
    }
    return changes;
  }
};




// This class specializes the WavToWavOptimizer for the specific case in which the patches used
// to perform the "step" method are always outputs of the Chi2Synth.
class Chi2Optimizer : public WavToWavOptimizer {
private:
  Chi2Server chi2server_;
public:
  explicit Chi2Optimizer(FloatSignal &original, const size_t min_chunk_size=2048,
                          const std::string seq_separator="||")
    : WavToWavOptimizer(original, min_chunk_size, seq_separator){}

  // This method wraps the "step" method: instead of receiving a FloatSignal as a patch, generates
  // one with the "Chi2Synth" given its size in samples, freq in Hz and samplerate (samples/second).
  // The env_ratio is expected to be between 0 (exponential) and 1 (quasi-gaussian).
  std::vector<std::pair<long int, float> > chi2step(const size_t size, const double freq,
                                                    const size_t samplerate, const double env_ratio,
                                                    const size_t stride,
                                                    std::function<std::vector<std::pair<long int,
                                                       float> >(FloatSignal&)> opt_criterium,
                                                    const std::string extra_info){
    Chi2Synth patch(chi2server_, size, freq, samplerate, env_ratio, 1.5, 100);
    float patch_energy = Energy(patch.begin(), patch.end());
    return step(patch, patch_energy, stride, opt_criterium, extra_info);

  }
};





#endif
