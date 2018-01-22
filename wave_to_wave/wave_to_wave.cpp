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




int main (int argc, char **argv){

  std::cout << "he wo"<< argc << argv[0] << std::endl;
  return 0;
}


// TODO:

// sequence constructor and asString methods seem to work. Utest them. Finish the addToSignal method and integrate it into the optimizer (and utest it): it should: 1) host the signal object and use its functionality. 2) feature a new constructor, that does the same as the regular one, but also subtracts the seq from the residual.

// With that it should be possible to serialize the sequence better: easily resume optimizations, split them across sessions, devices...

// *  provide an interface to allow static typecheching of the newly created criteria
// * finish the input parser for the flags.
// * comment, utest, valgrind, license, tidy up.

// that would be the end of the core API: helpers, signals, convolver, optimizer, sequencer, inputparser.
// The "instruments" and "meta" APIs would build on top of it, and are the actual artistic work


// OPTIMIZATIONS:
// given a stride, the stride_down function should be ~exp(1/stride), so that the level of compression and speedup remains but the artifacts (hopefully) go away. compute it once for the residual, and pick the beginning for the patch (to avoid desynch).

// the chi2 server should include a function that approximates the energy of the signal without having to compute it
// if the chi uses pure freqs, the opt.step could also be parallelized for different chi freqs.

// http://csoundjournal.com/issue17/gogins_composing_in_cpp.html
// sudo apt install libcsnd-dev libcsound64-dev
// explanation CSOUND API: http://write.flossmanuals.net/csound/a-the-csound-api/
// CSOUND 6 API: http://csound.com/docs/api/modules.html
// explanation csound plugins: http://write.flossmanuals.net/csound/extending-csound/
