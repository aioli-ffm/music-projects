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
#include<list>
#include<complex>
#include<numeric>
#include<algorithm>
// LIB INCLUDES
#include<csound/csound.hpp>
// LOCAL INCLUDES
#include "../include/helpers.hpp"
#include "../include/signal.hpp"


////////////////////////////////////////////////////////////////////////////////////////////////////

// Prototype of the synth used for analysis
class Chi2Synth : public FloatSignal {
private:
  const double kChi2Span{20.0};
  double num_oscillations_;
  double sin_ratio_;
  double env_ratio_;
  float gen(const double x){
    return 0;//std::sin(x*sin_ratio_);//Chi2(x, env_ratio_)*sin(x*sin_ratio_);
  }
public:
  Chi2Synth(const size_t size, const double num_oscillations, const double env_ratio)
    : FloatSignal(size),
      num_oscillations_(num_oscillations),
      sin_ratio_(num_oscillations*TWO_PI/kChi2Span),
      env_ratio_((env_ratio*0.15+0.1)*kChi2Span){
    size_t i=0;
    for(double x=GSL_DBL_EPSILON, delta=kChi2Span/size; i<size; ++i, x+=delta){
      data_[i] = gen(x);
    }
  }
  void test(const std::string outpath="/tmp/test.wav"){
    FloatSignal seq(44100*10);
    for(double i=0, freq=11; i<441000; i+=4410, freq+=1){
      Chi2Synth blob(4410*2, freq*2, 1);
      seq.addSignal(blob, i);
    }
    // seq.plot("hello", 44100, 0.2);
    seq.toWav(outpath, 44100);
  }
};

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
