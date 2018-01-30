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

// g++ -O3 -std=c++11 -Wall -Wextra wave_to_wave.cpp -Ithird_party -fopenmp -lfftw3f -lpython2.7 -o ./build/wave_to_wave && ./build/wave_to_wave -z -y wisdom -i 10g

// STL INCLUDES
#include <string>
#include <iostream>
// LOCAL THIRD-PARTY

// LOCAL INCLUDES
#include "include/w2w.hpp"


////////////////////////////////////////////////////////////////////////////////////////////////////
///
////////////////////////////////////////////////////////////////////////////////////////////////////
int main (int argc, char **argv){

  std::cout << "he wo"<< argc << argv[0] << std::endl;
  return 0;
}

// THe plan for the synth seems to be to include a name() method for dispatching that throws an exception by default and has to be overriden? how does the dispatcher find the name? what about the rest of the "string" stoi arguments?? think about this



////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO:

// for the optimization, seq dispatching, synth creation etc the app paradigm seems to be the
// best fit: allows static and dynamic info, little user overhead and easy debug (no macros). Do prototype

// If the app is OK: 1) convert seq into audio. 2) integrate seq with optimizer (new constructor?). 3) do a basic_usage example to get an idea of the API usability.


// that would be the end of the core API: helpers, signal, convolver, optimizer, sequencer, argparser. app.
// The "instruments" and "meta" APIs would build on top of it, and are the actual artistic work


////////////////////////////////////////////////////////////////////////////////////////////////////
// REPO MACHINERY:


// ARGPARSE?? IM NOT SURE IF THE API SHOULD TAKE CARE OF THIS.
// // Other 3rdparty dependencies
// // GFlags: DEFINE_bool, _int32, _int64, _uint64, _double, _string
// #include <gflags/gflags.h>
// // Allow Google Flags in Ubuntu 14
// #ifndef GFLAGS_GFLAGS_H_
//     namespace gflags = google;
// #endif

// NAMESPACING: CHECK THAT INCLUDE DEPENDENCIES ARE ACYCLIC! ASSOCIATE FOLDERS WITH NAMESPACES:
// w2w
//   -- core
//       -- helpers
//       -- signal
//       -- convolver
//       -- sequencer
//       -- app
//       -- argparser ?? should this be responsibility of the API?
//   -- opt
//       -- criteria
//       -- optimizer
//   -- synth
//       -- chi2
//       -- whatever
//
// namespace op
// {
//     template<typename TDatums,
//              typename TWorker = std::shared_ptr<Worker<std::shared_ptr<TDatums>>>,
//              typename TQueue = Queue<std::shared_ptr<TDatums>>>
//     class Wrapper
//     {
//     public:
// etc... then op::Wrapper

// Make an exception hierarchy to allow debugging of parser and api bugs (like a synth that doesn't implement the name() method).

// Revise all tests based on CompareIterables. Document and comment them...

// *  provide an interface to allow static typecheching of the newly created criteria and synths
// * comment, utest, valgrind, license, tidy up.


////////////////////////////////////////////////////////////////////////////////////////////////////
// OPTIMIZATIONS:
// given a stride, the stride_down function should be ~exp(1/stride), so that the level of compression and speedup remains but the artifacts (hopefully) go away. compute it once for the residual, and pick the beginning for the patch (to avoid desynch).

// the chi2 server should include a function that approximates the energy of the signal without having to compute it
// if the chi uses pure freqs, the opt.step could also be parallelized for different chi freqs.

// http://csoundjournal.com/issue17/gogins_composing_in_cpp.html
// sudo apt install libcsnd-dev libcsound64-dev
// explanation CSOUND API: http://write.flossmanuals.net/csound/a-the-csound-api/
// CSOUND 6 API: http://csound.com/docs/api/modules.html
// explanation csound plugins: http://write.flossmanuals.net/csound/extending-csound/
