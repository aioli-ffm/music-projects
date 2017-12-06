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

#include "../include/signal.hpp"


////////////////////////////////////////////////////////////////////////////////////////////////////

// instances of Synth are FloatSignals, specifically thought to be controlled
// by an instrument. Therefore it implements a more specific interface
class Synth : public FloatSignal {
public:
  Synth() : FloatSignal(10){}
  void test(){
    int NUM_POINTS = 5;
    int NUM_COMMANDS = 2;
    std::string commandsForGnuplot[NUM_COMMANDS] = {"set title \"TITLEEEEE\"", "plot 'data.temp'"};
    double xvals[NUM_POINTS] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double yvals[NUM_POINTS] = {5.0 ,3.0, 1.0, 3.0, 5.0};
    FILE * temp = fopen("data.temp", "w");
    // Opens an interface that one can use to send commands as if they were typing
    // into the gnuplot command line.  "The -persistent" keeps the plot open even
    // after your C program terminates.
    FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");
    int i;
    for (i=0; i < NUM_POINTS; i++){
      //Write the data to a temporary file
      fprintf(temp, "%lf %lf \n", xvals[i], yvals[i]);
    }
    for (i=0; i < NUM_COMMANDS; i++){
      //Send commands to gnuplot one by one.
      fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i].c_str());
    }
  }
};

#endif
