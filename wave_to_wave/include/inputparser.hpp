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

#ifndef INPUTPARSER_H
#define INPUTPARSER_H

// THIRD-PARTY INCLUDES
#include <cxxopts.hpp>
// LOCAL INCLUDE
#include "w2w.hpp"

// // The intention behind this class is to make a parser that interacts fully with the API, but also
// // provides a good off-the-shelf functionality if called from command line. For that, a map like
// //   std::map<std::string, std::string> parsed_args = WavToWavInputParser(argc, argv).getMap();
// // should be callable from the API, but a main function with WavToWavInputParser(argc, argv).do()
// // should also perform an optimization... this involves advanced usage so is left for further dev.
// //
// class WavToWavInputParser {
// private:
//   //std::string help_ = "hello I'm a very helpful string!\n";
//   std::map<std::string, std::string> map_;
//   bool debug_parser_, help_;
//   std::string create_wisdom_, import_wisdom_;
//   size_t num_iterations_, stride_, verbosity;
// public:
//   explicit WavToWavInputParser(int argc, char** argv){
//     // make parser
//     cxxopts::Options input_parser("WAVE TO WAVE", "Reconstructs a wave by combining shorter ones");
//     // fill it with options
//     input_parser.add_options()
//       ("z,debug-parser", "If true, shows all the flag values",
//        cxxopts::value<bool>()->default_value("false"))
//       ("h,help", "Show help to use the WAVE TO WAVE application",
//        cxxopts::value<bool>()->default_value("false"))
//       //
//       ("x,create-wisdom", "Create wisdom file and export to this path.",
//        cxxopts::value<std::string>()->default_value(""))
//       ("y,import-wisdom", "Path to a valid FFTW wisdom file to be loaded",
//        cxxopts::value<std::string>()->default_value(""))
//       //
//       ("i,num-iterations", "Number of iterations to be performed by the optimizer",
//        cxxopts::value<size_t>()->default_value("100"))
//       ("r,stride", "The optimizer will regard only one every r samples",
//        cxxopts::value<size_t>()->default_value("10"))
//       ("v,verbosity", "The optimizer will output information every v iterations",
//        cxxopts::value<size_t>()->default_value("0"));
//       //
//       // s original
//       // l sequence
//       // m materials folder (if none chi2? what about freq-time-k distributions? use the API?)
//     // PARSE THE OPTIONS
//     cxxopts::ParseResult parsed_args = input_parser.parse(argc, argv);
//     const bool kDebugParser = parsed_args["z"].as<bool>();
//     const bool kHelp = parsed_args["h"].as<bool>();
//     const std::string kCreateWisdom = parsed_args["x"].as<std::string>();
//     const std::string kImportWisdom = parsed_args["y"].as<std::string>();
//     const size_t kIterations = parsed_args["i"].as<size_t>();
//   }
//   std::map<std::string, std::string> getMap(){return map_;}
// };


//  // if(kHelp){std::cout << "Here could be some helpful documentation." << std::endl;}
//  //  if(kExpectedPaddedSizeDebugParser){
//  //    std::cout << "Input parser arguments:" << std::endl;
//  //    std::cout << "z,debug-parser --> " << kDebugParser << std::endl;
//  //    std::cout << "x,create-wisdom --> " << kCreateWisdom << std::endl;
//  //    std::cout << "y,import-wisdom --> " << kImportWisdom << std::endl;
//  //    std::cout << "i,num-iterations --> " << kIterations << std::endl;
//  //  }
//  //  //
//  //  if(!kCreateWisdom.empty()){MakeAndExportFftwWisdom(kCreateWisdom, 0, 29);}
//  //  }



#endif
