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

#endif
