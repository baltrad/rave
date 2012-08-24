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
 * Defines the functions available when working with vertical profiles
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2012-08-24
 */
#include "vertical_profile.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>
#include <math.h>
#include "rave_object.h"
#include "rave_datetime.h"
#include "rave_transform.h"
#include "rave_utilities.h"
/**
 * Represents one vertical profile
 */
struct _VerticalProfile_t {
  RAVE_OBJECT_HEAD /** Always on top */
  // Date/Time
  RaveDateTime_t* datetime;     /**< the date, time instance */
  char* source;    /**< the source string */
  RaveField_t* ff; /**< Mean horizontal wind velocity (m/s) */
  double lon; /**< the longitude in radians */
  double lat; /**< the latitude in radians */
  double height; /**< the height of the centre of the antenna */
  long levels; /**< the number of points in the profile */
  double interval; /**< Vertical distance (m) between height intervals, or 0.0 if variable */
  double minheight; /**< Minimum height in meters above mean sea level */
  double maxheight; /**< Maximum height in meters above mean sea level */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int VerticalProfile_constructor(RaveCoreObject* obj)
{
  VerticalProfile_t* self = (VerticalProfile_t*)obj;
  self->lon = 0.0;
  self->lat = 0.0;
  self->height = 0.0;
  self->levels = 0;
  self->interval = 0.0;
  self->minheight = 0.0;
  self->maxheight = 0.0;
  self->datetime = NULL;
  self->ff = NULL;
  self->source = NULL;
  self->datetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  if (self->datetime == NULL) {
    goto error;
  }
  return 1;
error:
  RAVE_OBJECT_RELEASE(self->datetime);
  return 0;
}

static int VerticalProfile_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  VerticalProfile_t* self = (VerticalProfile_t*)obj;
  VerticalProfile_t* src = (VerticalProfile_t*)srcobj;
  self->lon = src->lon;
  self->lat = src->lat;
  self->height = src->height;
  self->levels = src->levels;
  self->interval = src->interval;
  self->minheight = src->minheight;
  self->maxheight = src->maxheight;
  self->datetime = NULL;
  self->source = NULL;
  self->datetime = RAVE_OBJECT_CLONE(src->datetime);
  self->ff = RAVE_OBJECT_CLONE(src->ff);
  if (self->datetime == NULL || (src->ff != NULL && self->ff == NULL)) {
    goto error;
  }
  if (!VerticalProfile_setSource(self, VerticalProfile_getSource(src))) {
    goto error;
  }

  return 1;
error:
  RAVE_OBJECT_RELEASE(self->datetime);
  RAVE_OBJECT_RELEASE(self->ff);
  RAVE_FREE(self->source);
  return 0;
}

/**
 * Destructor.
 */
static void VerticalProfile_destructor(RaveCoreObject* obj)
{
  VerticalProfile_t* self = (VerticalProfile_t*)obj;
  RAVE_OBJECT_RELEASE(self->datetime);
  RAVE_OBJECT_RELEASE(self->ff);
  RAVE_FREE(self->source);
}
/*@} End of Private functions */

/*@{ Interface functions */
int VerticalProfile_setDate(VerticalProfile_t* self, const char* value)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_setDate(self->datetime, value);
}

const char* VerticalProfile_getDate(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_getDate(self->datetime);
}

int VerticalProfile_setTime(VerticalProfile_t* self, const char* value)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_setTime(self->datetime, value);
}

const char* VerticalProfile_getTime(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_getTime(self->datetime);
}

int VerticalProfile_setSource(VerticalProfile_t* self, const char* value)
{
  char* tmp = NULL;
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (value != NULL) {
    tmp = RAVE_STRDUP(value);
    if (tmp != NULL) {
      RAVE_FREE(self->source);
      self->source = tmp;
      tmp = NULL;
      result = 1;
    }
  } else {
    RAVE_FREE(self->source);
    result = 1;
  }
  return result;
}

const char* VerticalProfile_getSource(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->source;
}

void VerticalProfile_setLongitude(VerticalProfile_t* self, double lon)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->lon = lon;
}

double VerticalProfile_getLongitude(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->lon;
}

void VerticalProfile_setLatitude(VerticalProfile_t* self, double lat)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->lat = lat;
}

double VerticalProfile_getLatitude(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->lat;
}

void VerticalProfile_setHeight(VerticalProfile_t* self, double h)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->height = h;
}

double VerticalProfile_getHeight(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->height;
}

int VerticalProfile_setLevels(VerticalProfile_t* self, long l)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->levels = l;
  return 1;
}

long VerticalProfile_getLevels(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->levels;
}

void VerticalProfile_setInterval(VerticalProfile_t* self, double i)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->interval = i;
}

double VerticalProfile_getInterval(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->interval;
}

void VerticalProfile_setMinheight(VerticalProfile_t* self, double h)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->minheight = h;
}

double VerticalProfile_getMinheight(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->minheight;
}

void VerticalProfile_setMaxheight(VerticalProfile_t* self, double h)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->maxheight = h;
}

double VerticalProfile_getMaxheight(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->maxheight;
}

RaveField_t* VerticalProfile_getFF(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RAVE_OBJECT_COPY(self->ff);
}

int VerticalProfile_setFF(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_OBJECT_RELEASE(self->ff);
  self->ff = RAVE_OBJECT_COPY(ff);
  return 1;
}


/*@} End of Interface functions */

RaveCoreObjectType VerticalProfile_TYPE = {
    "VerticalProfile",
    sizeof(VerticalProfile_t),
    VerticalProfile_constructor,
    VerticalProfile_destructor,
    VerticalProfile_copyconstructor
};
