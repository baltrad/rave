/* --------------------------------------------------------------------
Copyright (C) 2011 (your Organization)

This is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with HLHDF.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/

/** Description
 * @file
 * @author Firstname Lastname, Affiliation
 * @date YYYY-MM-DD
 */

#ifndef NAME_H
#define NAME_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "rave_io.h"
/* Add your RAVE APIs here */

/**
 * Description (example)
 * @param[in] obj - an object
 * @param[in] aname - a string
 * @param[in] dbl - a double value
 * @returns 1 on success or 0 if the attribute doesn't exist
 */
int FuncName(SomeObject* obj, const char* aname, double* dbl);


#endif
