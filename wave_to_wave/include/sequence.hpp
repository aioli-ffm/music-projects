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
#include <vector>
#include <tuple>
#include <deque>
#include <string>
#include <sstream>

// LOCAL INCLUDE
#include "w2w.hpp"



////////////////////////////////////////////////////////////////////////////////////////////////////
///
////////////////////////////////////////////////////////////////////////////////////////////////////

class Sequence {
private:
  std::string separator_ = " ";
  std::string comment_marker_ = "#";
  std::vector<std::tuple<long int, float, std::deque<std::string> > > data_;
public:
  // CONSTRUCTORS
  explicit Sequence(){}
  explicit Sequence(const std::string kPathIn)
    : Sequence(){
    // check that the given path opens correctly
    std::ifstream in(kPathIn);
    if (!in.is_open()){
      throw std::invalid_argument("Sequence: unable to open input stream "+kPathIn);
    }
    for(std::string line; std::getline(in, line); ){
      extendFromString(line);
    }
    in.close();
  }

  void extendFromString(const std::string seq){
    //
    std::istringstream in(seq);
    const size_t kSepSize = separator_.size();
    const size_t kCommentSize = comment_marker_.size();
    // loop runs for every line in the stream
    for(std::string line; std::getline(in, line); ){
      std::deque<std::string> line_tokens = ParseLine(line, separator_, comment_marker_);
      if(line_tokens.size()>0){
        try {
          size_t idx = -1; // used by string->number conversors to tell where they stopped parsing
          const long int kDelay = std::stol(line_tokens.front(), &idx);
          CheckAllEqual({idx, line_tokens.front().size()}, "malformed delay!");
          line_tokens.pop_front();
          const float kNorm = std::stof(line_tokens.front(), &idx);
          CheckAllEqual({idx, line_tokens.front().size()}, "malformed norm factor!");
          line_tokens.pop_front();
          data_.push_back(std::make_tuple(kDelay, kNorm, line_tokens));
        } catch (...) {
          std::cout << "[Sequence]Ignored malformed line: " << line << std::endl;
        }
      }
    }
  }

  std::vector<std::tuple<long int, float, std::deque<std::string> > > getData(){return data_;}


  // comments NOT passed by reference (because it is destructed)
  // Given following seq: (123,0.123,{"1","2","3"}),(456,0.456,{"a"}),(789,0.789,{"asdf","fdsa"})
  // with separator=" " and comment_marker="#", calling seq.asString({{1,"hello"}, {3,"bye"}}) will
  // produce the following string:
  //   # hello
  //   123 0.123 1 2
  //   456 0.456 a
  //   # bye
  //   789 0.789 asdf fdsa
  std::string asString(std::deque<std::pair<size_t, std::string> > comments){
    std::sort(begin(comments), end(comments), // sort by size_t
              [](auto const &t1, auto const &t2) {return std::get<0>(t1) < std::get<0>(t2);});
    std::stringstream result;
    size_t i=1;
    for(auto const& line : data_){
      if(std::get<0>(comments[0])==i){
        result << comment_marker_ << " " << std::get<1>(comments[0]) << std::endl;
        comments.pop_front();
      }
      result << std::get<0>(line) << separator_ << std::get<1>(line);
      for(auto const& elt : std::get<2>(line)){
        result << separator_ << elt;
      }
      result << std::endl;
      i++;
    }
    return result.str();
  }
  std::string asString(){return asString({});} // overload for empty input


  // // parallelize this? #pragma omp critical?
  // void addToSignal(FloatSignal &fs, const long int basic_delay=0, const float basic_normfactor=1){
  //   for(auto const& elt : data_){
  //     const long int del = std::get<0>(elt)+basic_delay;
  //     const float norm = std::get<1>(elt)*basic_normfactor;
  //     FloatSignal* blob = nullptr;
  //     switch(std::get<1>(elt))
  //       {
  //       case 1:
  //         blob = new FloatSignal(100);
  //         break;
  //       }
  //     // once we have the blob, break if it wasn't recognized. else add it to fs
  //     if(blob==nullptr){
  //       std::cout << "addToSignal: ignoring line: xxx" << std::endl;
  //       break;
  //     }
  //     fs.addMultipliedSignal(blob, norm, del);
  //     delete blob;
  //   }
  // }

};

#endif
