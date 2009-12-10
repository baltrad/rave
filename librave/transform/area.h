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
 * Defines an area, the extent, projection, etc..
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#ifndef AREA_H
#define AREA_H
#include "rave_transform.h"
#include "projection.h"
#include "rave_object.h"

/**
 * Defines a Geographical Area
 */
typedef struct _Area_t Area_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType Area_TYPE;

/**
 * Sets the xsize
 * @param[in] area - the area
 * @param[in] xsize - the xsize
 */
void Area_setXSize(Area_t* area, long xsize);

/**
 * Returns the xsize
 * @param[in] cartesian - the cartesian product
 * @return the xsize
 */
long Area_getXSize(Area_t* area);

/**
 * Sets the ysize
 * @param[in] area - the area
 * @param[in] ysize - the ysize
 */
void Area_setYSize(Area_t* area, long ysize);

/**
 * Returns the ysize
 * @param[in] area - the area
 * @return the ysize
 */
long Area_getYSize(Area_t* area);

/**
 * Sets the xscale
 * @param[in] area - the area
 * @param[in] xscale - the xscale
 */
void Area_setXScale(Area_t* area, double xscale);

/**
 * Returns the xscale
 * @param[in] area - the area
 * @return the xscale
 */
double Area_getXScale(Area_t* area);

/**
 * Sets the yscale
 * @param[in] area - the area
 * @param[in] yscale - the yscale
 */
void Area_setYScale(Area_t* area, double yscale);

/**
 * Returns the yscale
 * @param[in] area - the area
 * @return the yscale
 */
double Area_getYScale(Area_t* area);

#endif /* AREA_H */
