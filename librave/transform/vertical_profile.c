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
#include "raveobject_hashtable.h"

/**
 * Represents one vertical profile
 */
struct _VerticalProfile_t {
  RAVE_OBJECT_HEAD /** Always on top */
  // Date/Time
  RaveDateTime_t* datetime;     /**< the date, time instance */
  char* source;    /**< the source string */
  RaveObjectHashTable_t* attrs; /**< attributes */
  RaveObjectHashTable_t* fields; /**< the fields */
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
  self->fields = NULL;
  self->source = NULL;
  self->attrs = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  self->datetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  self->fields = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  if (self->datetime == NULL || self->fields == NULL || self->attrs == NULL) {
    goto error;
  }
  return 1;
error:
  RAVE_OBJECT_RELEASE(self->datetime);
  RAVE_OBJECT_RELEASE(self->fields);
  RAVE_OBJECT_RELEASE(self->attrs);
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
  self->attrs = RAVE_OBJECT_CLONE(src->attrs);
  self->datetime = RAVE_OBJECT_CLONE(src->datetime);
  self->fields = RAVE_OBJECT_CLONE(src->fields);
  if (self->datetime == NULL || src->fields == NULL || src->attrs == NULL) {
    goto error;
  }
  if (!VerticalProfile_setSource(self, VerticalProfile_getSource(src))) {
    goto error;
  }

  return 1;
error:
  RAVE_OBJECT_RELEASE(self->datetime);
  RAVE_OBJECT_RELEASE(self->fields);
  RAVE_OBJECT_RELEASE(self->attrs);
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
  RAVE_OBJECT_RELEASE(self->fields);
  RAVE_OBJECT_RELEASE(self->attrs);
  RAVE_FREE(self->source);
}

static int VerticalProfileInternal_addField(VerticalProfile_t* self, RaveField_t* field, const char* quantity)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  RaveAttribute_t* attr = RaveAttributeHelp_createString("what/quantity", quantity);
  if (attr == NULL || !RaveField_addAttribute(field, attr)) {
    RAVE_ERROR0("Failed to add what/quantity attribute to field");
    goto done;
  }
  RAVE_OBJECT_RELEASE(attr);
  result = VerticalProfile_addField(self, field);
done:
  return result;
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

int VerticalProfile_addAttribute(VerticalProfile_t* self, RaveAttribute_t* attribute)
{
  const char* name = NULL;
  char* aname = NULL;
  char* gname = NULL;
  int result = 0;

  RAVE_ASSERT((attribute != NULL), "attribute == NULL");
  RAVE_ASSERT((self != NULL), "self == NULL");

  name = RaveAttribute_getName(attribute);
  if (name != NULL) {
    if (!RaveAttributeHelp_extractGroupAndName(name, &gname, &aname)) {
      RAVE_ERROR1("Failed to extract group and name from %s", name);
      goto done;
    }
    if ((strcasecmp("how", gname)==0) &&
         strchr(aname, '/') == NULL) {
      result = RaveObjectHashTable_put(self->attrs, name, (RaveCoreObject*)attribute);
    } else {
      RAVE_DEBUG1("Trying to add attribute: %s but only valid attributes are how/...", name);
    }
  }

done:
  RAVE_FREE(aname);
  RAVE_FREE(gname);
  return result;
}

RaveAttribute_t* VerticalProfile_getAttribute(VerticalProfile_t* self, const char* name)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (name == NULL) {
    RAVE_ERROR0("Trying to get an attribute with NULL name");
    return NULL;
  }
  return (RaveAttribute_t*)RaveObjectHashTable_get(self->attrs, name);
}

RaveList_t* VerticalProfile_getAttributeNames(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveObjectHashTable_keys(self->attrs);
}

RaveObjectList_t* VerticalProfile_getAttributeValues(VerticalProfile_t* self)
{
  RaveObjectList_t* result = NULL;
  RaveObjectList_t* tableattrs = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  tableattrs = RaveObjectHashTable_values(self->attrs);
  if (tableattrs == NULL) {
    goto error;
  }
  result = RAVE_OBJECT_CLONE(tableattrs);
  if (result == NULL) {
    goto error;
  }

  RAVE_OBJECT_RELEASE(tableattrs);
  return result;
error:
  RAVE_OBJECT_RELEASE(result);
  RAVE_OBJECT_RELEASE(tableattrs);
  return NULL;
}

RaveField_t* VerticalProfile_getFF(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfile_getField(self, "ff");
}

int VerticalProfile_setFF(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfileInternal_addField(self, ff, "ff");
}

RaveField_t* VerticalProfile_getFFDev(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfile_getField(self, "ff_dev");
}

int VerticalProfile_setFFDev(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfileInternal_addField(self, ff, "ff_dev");
}

RaveField_t* VerticalProfile_getW(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfile_getField(self, "w");
}

int VerticalProfile_setW(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfileInternal_addField(self, ff, "w");
}

RaveField_t* VerticalProfile_getWDev(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfile_getField(self, "w_dev");
}

int VerticalProfile_setWDev(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfileInternal_addField(self, ff, "w_dev");
}

RaveField_t* VerticalProfile_getDD(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfile_getField(self, "dd");
}

int VerticalProfile_setDD(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfileInternal_addField(self, ff, "dd");
}

RaveField_t* VerticalProfile_getDDDev(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfile_getField(self, "dd_dev");
}

int VerticalProfile_setDDDev(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfileInternal_addField(self, ff, "dd_dev");
}

RaveField_t* VerticalProfile_getDiv(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfile_getField(self, "div");
}

int VerticalProfile_setDiv(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfileInternal_addField(self, ff, "div");
}

RaveField_t* VerticalProfile_getDivDev(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfile_getField(self, "div_dev");
}

int VerticalProfile_setDivDev(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfileInternal_addField(self, ff, "div_dev");
}

RaveField_t* VerticalProfile_getDef(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfile_getField(self, "def");
}

int VerticalProfile_setDef(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfileInternal_addField(self, ff, "def");
}

RaveField_t* VerticalProfile_getDefDev(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfile_getField(self, "def_dev");
}

int VerticalProfile_setDefDev(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfileInternal_addField(self, ff, "def_dev");
}

RaveField_t* VerticalProfile_getAD(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfile_getField(self, "ad");
}

int VerticalProfile_setAD(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfileInternal_addField(self, ff, "ad");
}

RaveField_t* VerticalProfile_getADDev(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfile_getField(self, "ad_dev");
}

int VerticalProfile_setADDev(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfileInternal_addField(self, ff, "ad_dev");
}

RaveField_t* VerticalProfile_getDBZ(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfile_getField(self, "dbz");
}

int VerticalProfile_setDBZ(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfileInternal_addField(self, ff, "dbz");
}

RaveField_t* VerticalProfile_getDBZDev(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfile_getField(self, "dbz_dev");
}

int VerticalProfile_setDBZDev(VerticalProfile_t* self, RaveField_t* ff)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return VerticalProfileInternal_addField(self, ff, "dbz_dev");
}


RaveObjectList_t* VerticalProfile_getFields(VerticalProfile_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveObjectHashTable_values(self->fields);
}

int VerticalProfile_addField(VerticalProfile_t* self, RaveField_t* field)
{
  int result = 0;
  RaveAttribute_t* attr = NULL;
  char* str = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  attr = RaveField_getAttribute(field, "what/quantity");
  if (attr == NULL || !RaveAttribute_getString(attr, &str)) {
    RAVE_ERROR0("Field to not have any what/quantity attribute");
    goto done;
  }
  if (str == NULL) {
    RAVE_ERROR0("Field to not have any what/quantity attribute value");
    goto done;
  }
  if (RaveField_getXsize(field) != 1) {
    RAVE_ERROR0("Field must have xsize == 1");
    goto done;
  }

  if (RaveObjectHashTable_size(self->fields) != 0 || self->levels > 0) {
    if (self->levels != RaveField_getYsize(field)) {
      RAVE_ERROR0("Fields ysize must be same as levels");
      goto done;
    }
  }
  if (strcmp("ff", str) == 0 ||
      strcmp("ff_dev", str) == 0 ||
      strcmp("w", str) == 0 ||
      strcmp("w_dev", str) == 0 ||
      strcmp("dd", str) == 0 ||
      strcmp("dd_dev", str) == 0 ||
      strcmp("div", str) == 0 ||
      strcmp("div_dev", str) == 0 ||
      strcmp("def", str) == 0 ||
      strcmp("def_dev", str) == 0 ||
      strcmp("ad", str) == 0 ||
      strcmp("ad_dev", str) == 0 ||
      strcmp("dbz", str) == 0 ||
      strcmp("dbz_dev", str) == 0) {
    result = RaveObjectHashTable_put(self->fields, str, (RaveCoreObject*)field);
    if (result) {
      self->levels = RaveField_getYsize(field);
    }
  } else {
    RAVE_ERROR1("Fields what/quantity is of unknown value: %s", str);
  }

done:
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

RaveField_t* VerticalProfile_getField(VerticalProfile_t* self, const char* quantity)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (RaveField_t*)RaveObjectHashTable_get(self->fields, quantity);
}

/*@} End of Interface functions */

RaveCoreObjectType VerticalProfile_TYPE = {
    "VerticalProfile",
    sizeof(VerticalProfile_t),
    VerticalProfile_constructor,
    VerticalProfile_destructor,
    VerticalProfile_copyconstructor
};
