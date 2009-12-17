/* --------------------------------------------------------------------
Copyright (C) 2009 Swedish Meteorological and Hydrological Institute, SMHI,

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/
/**
 * Wrapper around PROJ.4.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-20
 */
#ifndef PROJECTION_H
#define PROJECTION_H
#include "projects.h"
#include "rave_object.h"

/**
 * Defines a transformer
 */
typedef struct _Projection_t Projection_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType Projection_TYPE;

/**
 * Initializes a projection with the projection
 */
int Projection_init(Projection_t* projection, const char* id, const char* description, const char* definition);

/**
 * Returns the ID for this projection.
 * @param[in] projection - the projection
 * @return the ID for this projection
 */
const char* Projection_getID(Projection_t* projection);

/**
 * Returns the description for this projection.
 * @param[in] projection - the projection
 * @return the description for this projection
 */
const char* Projection_getDescription(Projection_t* projection);

/**
 * Returns the definition for this projection.
 * @param[in] projection - the projection
 * @return the definition for this projection
 */
const char* Projection_getDefinition(Projection_t* projection);

/**
 * Transforms the coordinates in this projection into the target projection.
 * @param[in] projection - this projection
 * @param[in] tgt - target projection
 * @param[in,out] x - coordinate
 * @param[in,out] y - coordinate
 * @param[in,out] z - coordinate (MAY BE NULL in some cases), see PROJ.4
 * @param[in] projection - the projection
 * @return 0 on failure, otherwise success
 */
int Projection_transform(Projection_t* projection, Projection_t* tgt, double* x, double* y, double* z);

/**
 * Translates surface coordinate into lon/lat.
 * @param[in] projection - the projection
 * @param[in]    x - the x coordinate
 * @param[in]    y - the y coordinate
 * @param[out] lon - the longitude (in radians)
 * @param[out] lat - the latitude  (in radians)
 * @return 0 on failure otherwise success
 */
int Projection_inv(Projection_t* projection, double x, double y, double* lon, double* lat);

/**
 * Translates lon/lat coordinate into a surface coordinate.
 * @param[in] projection - the projection
 * @param[in] lon - the longitude (in radians)
 * @param[in] lat - the latitude (in radians)
 * @param[out]  x - the x coordinate
 * @param[out]  y - the y coordinate
 * @return 0 on failure otherwise success
 */
int Projection_fwd(Projection_t* projection, double lon, double lat, double* x, double* y);

#endif
