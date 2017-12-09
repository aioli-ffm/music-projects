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
// #include "third_party/matplotlibcpp.h"
#include <cxxopts.hpp>
// LOCAL INCLUDES
#include "include/signal.hpp"
#include "include/convolver.hpp"

int main(int argc,  char** argv){

  // DEFINE AND PARSE INPUT ARGUMENTS
  cxxopts::Options input_parser("WAVE TO WAVE", "Reconstructs a wave by combining others");
  //
  input_parser.add_options()
    ("z,debug-parser", "If true, shows all the flag values",
     cxxopts::value<bool>()->default_value("false"))
    ("h,help", "Show help to use the WAVE TO WAVE application",
     cxxopts::value<bool>()->default_value("false"))
    ("x,create-wisdom", "Create wisdom file to make FFTW run faster (do just once) and export to this path.",
     cxxopts::value<std::string>()->default_value(""))
    ("y,import-wisdom", "Path to a valid FFTW wisdom file to be loaded",
     cxxopts::value<std::string>()->default_value(""))
    ("i,num-iterations", "Number of iterations to be performed by the optimizer",
     cxxopts::value<size_t>()->default_value("10"));
  //
  cxxopts::ParseResult parsed_args = input_parser.parse(argc, argv);
  const bool kDebugParser = parsed_args["z"].as<bool>();
  const bool kHelp = parsed_args["h"].as<bool>();
  const std::string kCreateWisdom = parsed_args["x"].as<std::string>();
  const std::string kImportWisdom = parsed_args["y"].as<std::string>();
  const size_t kIterations = parsed_args["i"].as<size_t>();
  //
  if(kHelp){std::cout << "Here could be some helpful documentation." << std::endl;}
  if(kDebugParser){
    std::cout << "Input parser arguments:" << std::endl;
    std::cout << "z,debug-parser --> " << kDebugParser << std::endl;
    std::cout << "x,create-wisdom --> " << kCreateWisdom << std::endl;
    std::cout << "y,import-wisdom --> " << kImportWisdom << std::endl;
    std::cout << "i,num-iterations --> " << kIterations << std::endl;
  }
  //
  if(!kCreateWisdom.empty()){MakeAndExportFftwWisdom(kCreateWisdom, 0, 29);}



  return 0;
}

// TODO:

// at the end of convolver there is a sketch of the optimizer, finish it

// http://csoundjournal.com/issue17/gogins_composing_in_cpp.html
// sudo apt install libcsnd-dev libcsound64-dev
// explanation CSOUND API: http://write.flossmanuals.net/csound/a-the-csound-api/
// CSOUND 6 API: http://csound.com/docs/api/modules.html
// explanation csound plugins: http://write.flossmanuals.net/csound/extending-csound/

// put plotting deps in free funcs, not in signals: plot2D(x.beg(), y.end(), x.beg()) and plot3d...
// tidy up fft plan class tree (seems verbose).
// once design and implementation is "stable", utest convolver file (clean MAIN)
// put the parse into a separate file? clear parser questions.
// check valgrind... check imports
// write optimizer. support a lazy dict of {1024: FloatSignal(orig), 2046: orig}
//                  copies of the original, to be compatible with every patch.


// BUGS:
// plotting results of synth seems to be bugged (plots peaks of 1e32) when chitable size <200
