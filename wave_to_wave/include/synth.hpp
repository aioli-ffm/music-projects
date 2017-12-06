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

#define GSL_DBL_EPSILON        2.2204460492503131e-16
#define LogRootTwoPi_  0.9189385332046727418

/* coefficients for gamma=7, kmax=8  Lanczos method */
static double lanczos_7_c[9] = {
  0.99999999999980993227684700473478,
  676.520368121885098567009190444019,
 -1259.13921672240287047156078755283,
  771.3234287776530788486528258894,
 -176.61502916214059906584551354,
  12.507343278686904814458936853,
 -0.13857109526572011689554707,
  9.984369578019570859563e-6,
  1.50563273514931155834e-7
};

/* Lanczos method for real x > 0;
 * gamma=7, truncated at 1/(z+8)
 * [J. SIAM Numer. Anal, Ser. B, 1 (1964) 86]
 */
static
int
lngamma_lanczos(double x, double &result)
{
  int k;
  double Ag;
  double term1, term2;

  x -= 1.0; /* Lanczos writes z! instead of Gamma(z) */

  Ag = lanczos_7_c[0];
  for(k=1; k<=8; k++) { Ag += lanczos_7_c[k]/(x+k); }

  /* (x+0.5)*log(x+7.5) - (x+7.5) + LogRootTwoPi_ + log(Ag(x)) */
  term1 = (x+0.5)*log((x+7.5)/M_E);
  term2 = LogRootTwoPi_ + log(Ag);
  result  = term1 + (term2 - 7.0);
  double err  = 2.0 * GSL_DBL_EPSILON * (fabs(term1) + fabs(term2) + 7.0);
  err += GSL_DBL_EPSILON * fabs(result);
  return 0;
}

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
