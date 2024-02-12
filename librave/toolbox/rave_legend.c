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
 * Used for managing attributes and handle different versions.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2022-03-30
 */
#include "rave_attribute.h"
#include "rave_legend.h"
#include "rave_object.h"
#include "raveobject_list.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveobject_list.h"
#include <string.h>
#include <stdio.h>

/**
 * Represents one scan in a volume.
 */
struct _RaveLegend_t {
  RAVE_OBJECT_HEAD     /** Always on top */
  RaveObjectList_t* entries; /**< the entries */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int RaveLegend_constructor(RaveCoreObject* obj)
{
  RaveLegend_t* attr = (RaveLegend_t*)obj;
  attr->entries = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (attr->entries == NULL) {
    return 0;
  }
  return 1;
}

static int RaveLegend_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RaveLegend_t* this = (RaveLegend_t*)obj;
  RaveLegend_t* src = (RaveLegend_t*)srcobj;
  this->entries = RAVE_OBJECT_CLONE(src->entries);
  if (this->entries == NULL) {
    return 0;
  }
  return 1;
}

/**
 * Destructor.
 */
static void RaveLegend_destructor(RaveCoreObject* obj)
{
  RaveLegend_t* this = (RaveLegend_t*)obj;
  RAVE_OBJECT_RELEASE(this->entries);
}
/*@} End of Private functions */

/*@{ Interface functions */
int RaveLegend_addValue(RaveLegend_t* self, const char* key, const char* value)
{
  RaveAttribute_t* attribute = NULL;
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (key == NULL || value == NULL) {
    RAVE_ERROR0("Trying to set key or value to NULL in legend");
    return 0;
  }
  attribute = RaveAttributeHelp_createString(key, value);
  if (attribute == NULL) {
    goto done;
  }
  if (!RaveObjectList_add(self->entries, (RaveCoreObject*)attribute)) {
    goto done;
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(attribute);
  return result;
}

int RaveLegend_size(RaveLegend_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveObjectList_size(self->entries);
}

int RaveLegend_exists(RaveLegend_t* self, const char* key)
{
  int result = 0;
  int nlen = 0, i = 0;
  RaveAttribute_t* attribute = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (key == NULL) {
    RAVE_ERROR0("Trying to check for existance with NULL key");
    return 0;
  }

  nlen = RaveObjectList_size(self->entries);
  for (i = 0; result == 0 && i < nlen; i++) {
    attribute = (RaveAttribute_t*) RaveObjectList_get(self->entries, i);
    if (strcmp(key, RaveAttribute_getName(attribute)) == 0) {
      result = 1;
    }
    RAVE_OBJECT_RELEASE(attribute);
  }

  return result;
}

const char* RaveLegend_getValue(RaveLegend_t* self, const char* key)
{
  char* result = NULL;
  int nlen = 0, i = 0;
  RaveAttribute_t* attribute = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (key == NULL) {
    RAVE_ERROR0("Trying to get value using NULL key");
    return 0;
  }

  nlen = RaveObjectList_size(self->entries);
  for (i = 0; result == NULL && i < nlen; i++) {
    attribute = (RaveAttribute_t*) RaveObjectList_get(self->entries, i);
    if (strcmp(key, RaveAttribute_getName(attribute)) == 0) {
      RaveAttribute_getString(attribute, &result);
    }
    RAVE_OBJECT_RELEASE(attribute);
  }

  return (const char*)result;
}

const char* RaveLegend_getValueAt(RaveLegend_t* self, int index)
{
  char* result = NULL;
  RaveAttribute_t* attribute = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (index < 0 || index >= RaveObjectList_size(self->entries)) {
    RAVE_ERROR0("Trying to get value ouside size");
    return 0;
  }

  attribute = (RaveAttribute_t*)RaveObjectList_get(self->entries, index);
  RaveAttribute_getString(attribute, &result);
  RAVE_OBJECT_RELEASE(attribute);

  return (const char*)result;
}

const char* RaveLegend_getNameAt(RaveLegend_t* self, int index)
{
  char* result = NULL;
  RaveAttribute_t* attribute = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (index < 0 || index >= RaveObjectList_size(self->entries)) {
    RAVE_ERROR0("Trying to get name ouside size");
    return 0;
  }

  attribute = (RaveAttribute_t*)RaveObjectList_get(self->entries, index);
  result = (char*)RaveAttribute_getName(attribute);
  RAVE_OBJECT_RELEASE(attribute);

  return (const char*)result;
}

int RaveLegend_clear(RaveLegend_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RaveObjectList_clear(self->entries);
  return 1;
}

int RaveLegend_remove(RaveLegend_t* self, const char* key)
{
  int nlen = 0, i = 0, index = -1;
  RaveAttribute_t* attribute = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (key == NULL) {
    RAVE_ERROR0("Trying to remove value using NULL key");
    return 0;
  }

  nlen = RaveObjectList_size(self->entries);
  for (i = 0; index == -1 && i < nlen; i++) {
    attribute = (RaveAttribute_t*) RaveObjectList_get(self->entries, i);
    if (strcmp(key, RaveAttribute_getName(attribute)) == 0) {
      index = i;
    }
    RAVE_OBJECT_RELEASE(attribute);
  }

  return RaveLegend_removeAt(self, index);
}

int RaveLegend_removeAt(RaveLegend_t* self, int index)
{
  RaveCoreObject* obj = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (index < 0 || index >= RaveObjectList_size(self->entries)) {
    RAVE_ERROR0("Trying to remove object ouside size");
    return 0;
  }
  obj = RaveObjectList_remove(self->entries, index);
  RAVE_OBJECT_RELEASE(obj);
  return 1;
}

int RaveLegend_maxKeyLength(RaveLegend_t* self)
{
  int nlen = 0, i = 0, result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  nlen = RaveObjectList_size(self->entries);
  for (i = 0; i < nlen; i++) {
    RaveAttribute_t* attribute = (RaveAttribute_t*)RaveObjectList_get(self->entries, i);
    int alen = strlen(RaveAttribute_getName(attribute));
    if (alen > result) {
      result = alen;
    }
    RAVE_OBJECT_RELEASE(attribute);
  }
  return result;
}

int RaveLegend_maxValueLength(RaveLegend_t* self)
{
  int nlen = 0, i = 0, result = 0;
  char* value = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  nlen = RaveObjectList_size(self->entries);
  for (i = 0; i < nlen; i++) {
    RaveAttribute_t* attribute = (RaveAttribute_t*)RaveObjectList_get(self->entries, i);
    RaveAttribute_getString(attribute, &value);
    int alen = strlen(value);
    if (alen > result) {
      result = alen;
    }
    RAVE_OBJECT_RELEASE(attribute);
  }
  return result;
}

/*@} End of Interface functions */

RaveCoreObjectType RaveLegend_TYPE = {
  "RaveLegend",
  sizeof(RaveLegend_t),
  RaveLegend_constructor,
  RaveLegend_destructor,
  RaveLegend_copyconstructor
};
