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

  char* id;

  // Where
  long xsize;
  long ysize;
  double xscale;
  double yscale;

  double llX;
  double llY;
  double urX;
  double urY;

  Projection_t* projection;
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int Area_constructor(RaveCoreObject* obj)
{
  Area_t* result = (Area_t*)obj;
  result->id = NULL;
  result->xsize = 0;
  result->ysize = 0;
  result->xscale = 0.0L;
  result->yscale = 0.0L;
  result->llX = 0.0L;
  result->llY = 0.0L;
  result->urX = 0.0L;
  result->urY = 0.0L;
  result->projection = NULL;
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
    RAVE_FREE(area->id);
    RAVE_OBJECT_RELEASE(area->projection);
  }
}
/*@} End of Private functions */

/*@{ Interface functions */
void Area_setID(Area_t* area, const char* id)
{
  RAVE_ASSERT((area != NULL), "area was NULL");
  RAVE_FREE(area->id);
  if (id != NULL) {
    area->id = RAVE_STRDUP(id);
    if (area->id == NULL) {
      RAVE_CRITICAL0("Failure when copying id");
    }
  }
}

const char* Area_getID(Area_t* area)
{
  RAVE_ASSERT((area != NULL), "area was NULL");
  return (const char*)area->id;
}

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

void Area_setExtent(Area_t* area, double llX, double llY, double urX, double urY)
{
  RAVE_ASSERT((area != NULL), "area was NULL");
  area->llX = llX;
  area->llY = llY;
  area->urX = urX;
  area->urY = urY;
}

void Area_getExtent(Area_t* area, double* llX, double* llY, double* urX, double* urY)
{
  RAVE_ASSERT((area != NULL), "area was NULL");
  if (llX != NULL) {
    *llX = area->llX;
  }
  if (llY != NULL) {
    *llY = area->llY;
  }
  if (urX != NULL) {
    *urX = area->urX;
  }
  if (urY != NULL) {
    *urY = area->urY;
  }
}

void Area_setProjection(Area_t* area, Projection_t* projection)
{
  RAVE_ASSERT((area != NULL), "area was NULL");
  RAVE_OBJECT_RELEASE(area->projection);
  area->projection = RAVE_OBJECT_COPY(projection);
}

Projection_t* Area_getProjection(Area_t* area)
{
  RAVE_ASSERT((area != NULL), "area was NULL");
  return RAVE_OBJECT_COPY(area->projection);
}
/*@} End of Interface functions */

RaveCoreObjectType Area_TYPE = {
    "Area",
    sizeof(Area_t),
    Area_constructor,
    Area_destructor
};
