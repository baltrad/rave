/* --------------------------------------------------------------------
Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI

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

/** Header file for reprojecting Cartesian data
 * @file
 * @author Daniel Michelson, SMHI
 * @date 2011-06-02
 */
#ifndef REPROJ_H
#define REPROJ_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "rave_io.h"
#include "cartesian.h"
#include "projectionregistry.h"
#include "arearegistry.h"
#include "projection.h"
#include "area.h"
#include "rave_object.h"
#include "rave_types.h"

/**
 * Returns a double attribute value from any polar object type.
 * Isn't used by reproj, but it's included anyway for tutorial purposes.
 * @param[in] obj - a polar volume, scan, or scan parameter
 * @param[in] aname - a string of the attribute to retrieve
 * @param[in] tmpd - the double value to retrieve
 * @returns 1 on success or 0 if the attribute doesn't exist
 */
int getDoubleAttribute(RaveCoreObject* obj, const char* aname, double* tmpd);

/**
 * Helper function for copying metadata from input to output objects
 * @param[in] source - input image
 * @param[in] dest - destination (output) image
 * @returns Nothing
 */
void CopyMetaData(Cartesian_t* source, Cartesian_t* dest);

/**
 * Function for reprojecting Cartesian data
 * @param[in] source - input image
 * @param[in] dest - string containing the identifier of the output Cartesian area definition
 * @returns Cartesian_t* object containing the re-projected data
 */
Cartesian_t* reproj(Cartesian_t* inobj, const char* areaid);

#endif
