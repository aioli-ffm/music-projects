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

#ifndef SEQUENCE_H
#define SEQUENCE_H


// // STL INCLUDES
// #include <map>
// #include <string>

// LOCAL INCLUDE
#include "w2w.hpp"



class Sequence {
private:
  std::vector<std::string> data_;
public:
  // CONSTRUCTORS
  explicit Sequence(){

  }
  explicit Sequence(const std::string kSeqPath){
    // PARSER? the DS should be then a vector of (int, float, string).
  }
  // to signal
  void addToSignal(FloatSignal &fs, const long int basic_delay, const float ){
    // do a for-each.The last element(a string should have a switch statement for known instruments)
  }
}

// sequence protocol by default refers to 1.position(samples), 2.scaling, 3. absolute wav paths
// but 3 can be decomposed into {instrument_name par1 par2 ...}


// it should have the addToFloatSignal(FS &fs, const long int basic_delay, const float basic_norm)
// that loops over the seq and adds the elements at the given delay+basic_delay and at
// given_norm*basic_norm.
// This can be used by the optimizer: if at construction a Sequence obj is passed, it will be
// subtracted right away from the residual.
// This can also be used by free functions to directly construct a wav from a sequence.

// This way we have solved the issues of
// 1. having a flexible, modularized protocol for signals that allow creation,editing,reconstruction
// 2. integration with the optimizer
// 3. allow interpreting as signalt itself, as graph or map-reduce to be used by further ML algos


#endif
