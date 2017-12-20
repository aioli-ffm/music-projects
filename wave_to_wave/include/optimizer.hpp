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
// LOCAL INCLUDES
#include "helpers.hpp"
#include "signal.hpp"
#include "convolver.hpp"
#include "synth.hpp"


////////////////////////////////////////////////////////////////////////////////////////////////////

// FloatSignal* sig = new FloatSignal(padded_patch_size_);
// std::copy(it, std::min(end_it, it+padded_size_half_), sig->begin()+padded_size_half_);
// signal_vec_.push_back(new FftTransformer(sig, new ComplexSignal(padded_patch_size_complex_)));

// This class is a specialized and optimized version of the OverlapSaveConvolver, and is expected
// to give support to the WavToWavOptimizer with the following operations:
// 1. constructor(FloatSignal &sig, ComplexSignal &patch). Basically the same as the convolver, but
// automatically performs the forward FFTs after.
// 2.
class WavToWavOptimizer{
private:
  // size_t last_pipeline_ = 0;
  const size_t kMinChunkSize_;
  FloatSignal& original_;
  size_t original_size_;
  std::function<std::vector<std::pair<long int, float> >(FloatSignal&)> opt_criterium_;
  FftTransformer*  residual_;
  std::stringstream sequence_;
  std::map<size_t, OverlapSaveConvolver*> pipelines_;
  //
  // FftTransformer optimized_;
public:
  explicit WavToWavOptimizer(FloatSignal &original,
                             std::function<std::vector<std::pair<long int, float>
                             >(FloatSignal&)> opt_criterium,
                             const size_t min_chunk_size=2048)
    : kMinChunkSize_(min_chunk_size),
      original_(original),
      original_size_(original.getSize()),
      opt_criterium_(opt_criterium),
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

  std::vector<std::pair<long int, float> > step(FloatSignal& patch, const float patch_energy,
                                                const std::string seq_notes){
    // adjust and update metadata before loading the pipeline
    const size_t kPatchSize = patch.getSize();
    const size_t kPaddedSize = std::max(kMinChunkSize_, 2*Pow2Ceil(kPatchSize));
    const float kNormFactor = patch_energy*kPaddedSize;
    // const bool kRepeats = last_pipeline_==kPaddedSize;
    // last_pipeline_ = kPaddedSize;
    // get (or create if didn't exist) the corresponding pipeline, and update the signal spectrum
    OverlapSaveConvolver* convolver = nullptr;
    auto it = pipelines_.find(kPaddedSize);
    if ( it != pipelines_.end()){ // THERE WAS A PRE-EXISTING PIPELINE, UPDATE IT
      convolver = it->second;
      convolver->updatePatch(patch, true, false, false); // reverse, normalize, fft_after
      convolver->updateSignal(*residual_->r, false);
    } else{ // THERE WASN'T A PREEXISTING PIPELINE: MAKE A NEW ONE
      convolver = new OverlapSaveConvolver(*residual_->r, patch, true, false, kPaddedSize);
      pipelines_.insert(std::pair<size_t, OverlapSaveConvolver*>(kPaddedSize, convolver));
    }
    // at this point we have a pipeline with all the FloatSignals up-to-date. Calculate FFT of
    // both, perform spectral cross-correlation and calculate IFFT of correlation
    convolver->forwardPatch();
    convolver->forwardSignal();
    convolver->spectralConv();
    convolver->backwardSignal();
    // extract NON-NORMALIZED conv results as a newly constructed signal
    FloatSignal result(original_size_+kPatchSize-1);
    convolver->extractConvolvedTo(result);
    // apply criterium to extract a list of <POSITION, XCORR_VALUE> changes
    std::vector<std::pair<long int, float> > changes = opt_criterium_(result);
    // subtract the changes returned by the criterium to the residual signal
    for (const auto& elt : changes){
      long int position = elt.first;//-kPatchSize+1;
      float factor = elt.second/kNormFactor;
      // std::cout << position << "  <<pos,  factor>> " << factor << std::endl;
      residual_->r->subtractMultipliedSignal(patch, factor, position);
      sequence_ << position << " " << factor << " " << seq_notes << std::endl;
    }
    return changes;
  }

  FftTransformer* getResidual(){return residual_;}

  float getResidualEnergy(){
    FloatSignal* r = residual_->r;
    return std::inner_product(r->begin(), r->end(), r->begin(), 0.0f);
  }
  //
  void exportReconstruction(const std::string wav_export_path){
    FloatSignal out(original_.begin(), original_size_);
    out -= *residual_->r;
    out.toWav(wav_export_path, 22050);
  }
  //
  void exportTxtSequence(const std::string path_out){
    std::ofstream out(path_out);
    if (out.is_open()){
      out << sequence_.rdbuf();
      out.close();
    }
    else{
      throw std::invalid_argument("exportTxtSequence: unable to open output stream "+path_out);
    }
  }

};


#endif
