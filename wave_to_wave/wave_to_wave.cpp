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
#include <string.h>
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





  // create a test signal
  const size_t kSizeS = 44100*60/10;
  float* s_arr = new float[kSizeS]; for(size_t i=0; i<kSizeS; ++i){s_arr[i] = i+1;}
  FloatSignal s(s_arr, kSizeS);
  // s.plot("signal");

  // create several test patches:
  const size_t kSizeP1 =  44100*1/10;
  float* p1_arr = new float[kSizeP1]; for(size_t i=0; i<kSizeP1; ++i){p1_arr[i]=i+1;}
  FloatSignal p1(p1_arr, kSizeP1);
  // const size_t kSizeP2 = 3; // 44100*3/1;
  // float* p2_arr = new float[kSizeP2]; for(size_t i=0; i<kSizeP2; ++i){p2_arr[i]=i+1;}
  // FloatSignal p2(p2_arr, kSizeP2);
  // const size_t kSizeP3 = 4; // 44100*3/1;
  // float* p3_arr = new float[kSizeP3]; for(size_t i=0; i<kSizeP3; ++i){p3_arr[i]=i+1;}
  // FloatSignal p3(p3_arr, kSizeP3);

  // p.plot("patch");

  // Try some simple convolution
  OverlapSaveConvolver x1(s, p1, kImportWisdom);
  // OverlapSaveConvolver x2(s, p2);
  // OverlapSaveConvolver x3(s, p3);


  for(size_t i=0; i<kIterations; ++i){
    std::cout << "iter " << i << std::endl;
    x1.executeXcorr();
  }
  // x1.printChunks("xcorr");
  // x1.extractResult().print("xcorr");


  // clean memory and exit
  delete[] s_arr;
  delete[] p1_arr;
  // delete[] p2_arr;
  // delete[] p3_arr;


  return 0;
}
