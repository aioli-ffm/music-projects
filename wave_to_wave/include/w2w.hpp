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

#ifndef W2W_H
#define W2W_H

// OPEN MP:
// comment this line to deactivate OpenMP for loop parallelizations, or if
// you want to debug memory management (valgrind reports OMP normal activity).
// The number is the minimum size that a 'for' loop needs to get sent to OMP
// (1=>always sent)
#define WITH_OPENMP_ABOVE 1


#include "helpers.hpp"
#include "signal.hpp"
#include "convolver.hpp"
#include "synth.hpp"
#include "optimizer.hpp"
#include "sequence.hpp"


#endif



// #include <map>
// struct Dispatcher{
//   std::map<std::string, ??
// }

// class WavToWavApp{
// private:
//   typedef std::map<std::string, Synth*(*)()> dispatcher_map;
//   dispatcher_map dm_;
// public:
//   explicit WavToWavApp(){

//   }

//   template<typename SubSynth>
//   register()

// }


// template<typename SubSynth>
// Synth* instantiateSynth() {
//   return new SubSynth;
// }



// dispatcher_map dm;
// dm["Test"] = &instantiateSynth<TestSynth>;
// dm["Chi2"] = &instantiateSynth<Chi2Synth>;


// Synth* blob = map[Test]();



// // API definition:


// class Synth : public FloatSignal{
// private:
//   std::string name_;
// public:
//   explicit Synth(const std::string name, const std::function<float (const long int)>  &f,
//                  const size_t size)
//     : name_(name),
//       FloatSignal(f, size){}
// };


// // API user:

// class Test : public Synth{
// private:
//   float freq_;
// public:
//   explicit Test(float freq, size_t size)
//     : freq_(freq),
//       Synth("Test", generator, size){}
//   float generator(long int x){
//     return 0.01f*freq_;
//   }
// };



// // seq.txt:
// // 100 1.0 Test 440.0 22050

// // Sequence: loaded (100, 1.0, {"Test", "440.0", "22050"}). Now what to get Test(440.0, 22050)?

// // Tengo q enganhar al compilador, para q pille todas las clases q deriven de synth, y cree
// // un mapa tipo "name"->


// typedef std::map<std::string, Synth*(*)()> dispatcher_map;

// template<typename SubSynth>
// Synth* instantiateSynth() {
//   return new SubSynth;
// }



// dispatcher_map dm;
// dm["Test"] = &instantiateSynth<TestSynth>;
// dm["Chi2"] = &instantiateSynth<Chi2Synth>;


// Synth* blob = map[Test]();




// // https://stackoverflow.com/a/23690469
// template <typename T>
// class Singleton
// {
// public:
//    static Singleton<T>& getInstance() {
//        static Singleton<T> theInstance;
//        return theInstance;
//    }

// private:
//    Singleton() {}
//    Singleton(const Singleton<T>&);
//    Singleton<T>& operator=(const Singleton<T>&);
// };
