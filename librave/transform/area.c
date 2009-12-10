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
#include "area.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>

/**
 * Represents the area
 */
struct _Area_t {
  RAVE_OBJECT_HEAD /** Always on top */

  // Where
  long xsize;
  long ysize;
  double xscale;
  double yscale;
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int Area_constructor(RaveCoreObject* obj)
{
  Area_t* result = (Area_t*)obj;
  result->xsize = 0;
  result->ysize = 0;
  result->xscale = 0.0L;
  result->yscale = 0.0L;
  return 1;
}

/**
 * Destroys the area
 * @param[in] obj - the the Area_t instance
 */
static void Area_destructor(RaveCoreObject* obj)
{
  Area_t* area = (Area_t*)obj;
  if (area != NULL) {
  }
}
/*@} End of Private functions */

/*@{ Interface functions */
void Area_setXSize(Area_t* area, long xsize)
{
  RAVE_ASSERT((area != NULL), "area was NULL");
  area->xsize = xsize;
}

long Area_getXSize(Area_t* area)
{
  RAVE_ASSERT((area != NULL), "area was NULL");
  return area->xsize;
}

void Area_setYSize(Area_t* area, long ysize)
{
  RAVE_ASSERT((area != NULL), "area was NULL");
  area->ysize = ysize;
}

long Area_getYSize(Area_t* area)
{
  RAVE_ASSERT((area != NULL), "area was NULL");
  return area->ysize;
}

void Area_setXScale(Area_t* area, double xscale)
{
  RAVE_ASSERT((area != NULL), "area was NULL");
  area->xscale = xscale;
}

double Area_getXScale(Area_t* area)
{
  RAVE_ASSERT((area != NULL), "area was NULL");
  return area->xscale;
}

void Area_setYScale(Area_t* area, double yscale)
{
  RAVE_ASSERT((area != NULL), "area was NULL");
  area->yscale = yscale;
}

double Area_getYScale(Area_t* area)
{
  RAVE_ASSERT((area != NULL), "area was NULL");
  return area->yscale;
}
/*@} End of Interface functions */

RaveCoreObjectType Area_TYPE = {
    "Area",
    sizeof(Area_t),
    Area_constructor,
    Area_destructor
};
