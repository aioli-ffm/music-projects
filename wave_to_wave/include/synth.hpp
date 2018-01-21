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

#ifndef SYNTH_H
#define SYNTH_H


// STL INCLUDES
#include<string>
#include<vector>
#include<map>
#include<list>
#include<complex>
#include<numeric>
#include<algorithm>
// LIB INCLUDES
#include<csound/csound.hpp>
// LOCAL INCLUDES
// #include "../include/helpers.hpp"
// #include "../include/signal.hpp"
#include "w2w.hpp"


////////////////////////////////////////////////////////////////////////////////////////////////////
/// EFFICIENT CHI2 GENERATOR
////////////////////////////////////////////////////////////////////////////////////////////////////

// This clas is intended to avoid duplicate, expensive computations of the same Chi2 curve.
// It features a hash table holding the already computed curves, and a single getter that returns
// the curve if existing, or generates, saves and returns it otherwise.
// Any interpolations, limitations in resolution, etc have to be managed by the consumer, or
// extending the class
class Chi2Server {
private:
  std::map<std::tuple<size_t, float, float>, FloatSignal*> chi2_map_; // (size, span, df)->signal
public:

  // The only consumer method of the server
  FloatSignal* get(const size_t size, const float span, const float deg_freedom){
    std::tuple<size_t, float, float> key(size, span, deg_freedom);
    std::map<std::tuple<size_t, float, float>, FloatSignal*>::iterator it = chi2_map_.find(key);
    if (it != chi2_map_.end()){ // if signal already exists, return it
      return it->second;
    } else {                    // if signal didn't exist...
      // ... generate a new signal ...
      FloatSignal* fs = new FloatSignal(size);
      float* fs_data = fs->getData();
      double delta = span/size;
      double x = DOUBLE_EPSILON;
      for(size_t i=0; i<size; ++i, x+=delta){
        fs_data[i] = Chi2(x, deg_freedom);
      }
      // ... save it to the map and return it
      chi2_map_.insert(std::make_pair(key, fs));
      return chi2_map_.find(key)->second;
    }
  }
  ~Chi2Server(){
    for(const auto& it : chi2_map_){delete it.second;}
  }
};



////////////////////////////////////////////////////////////////////////////////////////////////////
/// FAST "WAVELET" SYNTH
////////////////////////////////////////////////////////////////////////////////////////////////////

const double kInterpSteepness = 1.5; // see ExpInterp

class Chi2Synth : public FloatSignal {
private:
  Chi2Server& server_;
  double num_oscillations_;
  double sin_ratio_;
  double env_ratio_;
  double env_exp_ratio_;
  size_t env_ratio_resolution_;
  size_t chi_table_size_;
  float chi_span_;

public:
  Chi2Synth(Chi2Server &server,
            const size_t size,
            const double num_oscillations,
            const double env_ratio,
            const double env_exp_ratio=1.5,
            const size_t env_ratio_resolution=100, // num points allowed between env_ratio=0 and 1
            const size_t chi_table_size=1000,
            const float chi_span=30.0)
    : FloatSignal(size),
      server_(server),
      num_oscillations_(num_oscillations),
      sin_ratio_(num_oscillations_*TWO_PI/chi_span),
      env_ratio_(std::round(env_ratio*env_ratio_resolution)/env_ratio_resolution), // truncate
      env_exp_ratio_(env_exp_ratio),
      env_ratio_resolution_(env_ratio_resolution),
      chi_table_size_(chi_table_size),
      chi_span_(chi_span) {
    double df = ExpInterpZeroOne(env_ratio_, env_exp_ratio)*5+2;
    FloatSignal* chi_sig = server_.get(chi_table_size, chi_span, 2*df);
    float* chi_data = chi_sig->getData();
    double chi_idx=0;
    size_t chidxint = 0;
    double chi_idx_delta = (-1.0+chi_table_size)/size;
    double sin_delta = chi_span/size;
    double x = 0;
    double _; // unused (for the modf built-in function)
    // the size of the chi table usually doesn't match the size of this object. Therefore,
    // its values have to be interpolated in the following loop:
    for(size_t i=0; i<size; ++i, chi_idx+=chi_idx_delta, x+=sin_delta){
      chidxint = chi_idx;
      data_[i] = sin(x*sin_ratio_) * LinInterp(chi_data[chidxint],chi_data[chidxint+1], modf(chi_idx,&_));
    }
  }

  Chi2Synth(Chi2Server &server,
            const size_t size,
            const double freq,
            const size_t sample_rate,
            const double env_ratio,
            const double env_exp_ratio=1.5,
            const size_t env_ratio_resolution=500, // num points allowed between env_ratio=0 and 1
            const size_t chi_table_size=1000,
            const float chi_span=30.0)
    : Chi2Synth(server, size, (freq*size)/sample_rate, env_ratio, env_exp_ratio,
                env_ratio_resolution, chi_table_size, chi_span){}
};




void Test(const std::string outpath="/tmp/test.wav"){
  size_t samplerate = 44100;
  size_t secs =20;
  Chi2Server serv;
  FloatSignal seq(samplerate*secs);
  size_t num_notes = 10;
  for(double i=0, k=0; i<num_notes; ++i, k+=1.0/(num_notes-1)){
    Chi2Synth blob(serv, seq.getSize()/num_notes, 880.0, samplerate, k);
    seq.addSignal(blob, i*blob.getSize());
  }
  seq.plot("Linear evolution of sin(x)*Chi2(x, k) for k=2, ..., k=7", samplerate, 1, 0.4);
  seq.toWav(outpath, samplerate);
}


// int main(int argc, char **argv)
// {
//   Csound *cs = new Csound();
//   int result = cs->Compile(argc, argv);
//   if (result == 0) {
//     result = cs->Perform();
//   }
//   delete cs;
//   return (result >= 0 ? 0 : result);
// }




// // envelope management: we dont need thousands... come up with a fast way to deal with this:
// // probably generate 15 curves with 1000 points each and then select the curve by truncating
// // and the index also by
// template<class Iter>
// class Chi2Env {
// private:
//   std::vector<FloatSignal> envs;
// public:
//   Chi2Env(Iter k_beg, Iter k_end, const size_t num_points){
//     for(; k_beg!=k_end; ++k_beg){
//       // generate a sig with num_points and proper k, should be faded in out and balanced
//     }
//   }
// };


#endif
