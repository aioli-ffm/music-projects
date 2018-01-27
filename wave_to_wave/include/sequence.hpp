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
    //
    const size_t kSepSize = separator_.size();
    const size_t kCommentSize = comment_marker_.size();
    // loop runs for every line in the stream
    for(std::string line; std::getline(in, line); ){
      std::deque<std::string> line_tokens;
      const size_t kLineSize = line.size();
      if(kLineSize<1){continue;}
      std::string::iterator token_beg = line.begin();
      std::string::iterator it = line.begin();
      std::string::iterator end = line.end();
      std::string::iterator comm_beg = comment_marker_.begin();
      std::string::iterator comm_end = comment_marker_.end();
      std::string::iterator sep_beg = separator_.begin();
      std::string::iterator sep_end = separator_.end();
      // loop runs for every character in the line
      for(; it<end; ++it){
        // search for comment marker: if found, finish processing the line
        if(*it == comment_marker_[0] &&
           end-it >= kCommentSize   &&
           std::equal(it, it+kCommentSize, comm_beg, comm_end)){
          goto end_of_char_loop; // dijkstra? who?
        }
        // search for separator marker, and add new token if found
        if(*it == separator_[0] &&
           end-it >= kSepSize  &&
           std::equal(it, it+kSepSize, sep_beg, sep_end)){
          line_tokens.push_back(std::string(token_beg, it));
          it += kSepSize;
          token_beg = it;
        }
      } // finish loop for characters in line adding the last token found
      line_tokens.push_back(std::string(token_beg, end));
      // try to add the elements to the data_ vector without exceptions
      // If anything goes wrong, warn and ignore line
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
    end_of_char_loop:;
    } // finish loop for lines in file
    in.close(); // end of constructor
  }

  std::vector<std::tuple<long int, float, std::deque<std::string> > > getData(){
    return data_;
  }


  // comments NOT passed by reference (because it is destructed)
  // Given following contents: (123 0.123 1 2 3 4), (456 0.456 a b c d), (789 0.789 1a 2b 3c 4d)
  // calling asString({{1,"hello"}, {3,"bye"}}) with separator=" " and comment_marker="#"
  // will produce the following string:
  //   # hello
  //   123 0.123 1 2 3 4
  //   456 0.456 a b c d
  //   # bye
  //   789 0.789 1a 2b 3c 4d
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


  // to signal
  void addToSignal(FloatSignal &fs, const long int basic_delay, const float ){
    // // do a for-each.The last element(a string should have a switch statement for known instruments)
    // #ifdef WITH_OPENMP_ABOVE
    // #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    // #endif
    // for (size_t i=0; i<changes_size; ++i){
    //   auto elt = changes[i];
    //   long int position = (elt.first-kPatchSize+1)*stride;
    //   float factor = elt.second/kNormFactor;
    //   residual_->r->subtractMultipliedSignal(patch, factor, position);
    //   #pragma omp critical
    //   sequence_.push_back(std::to_string(position)+seq_separator_+std::to_string(factor)+
    //                       seq_separator_+extra_info);
    // }
  }
};


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
