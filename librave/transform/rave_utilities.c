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
 * Contains various utility functions that makes life easier when working
 * with the rave framework.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-06-10
 */
#include "rave_utilities.h"
#include "rave_object.h"
#include "rave_debug.h"
#include <string.h>

int RaveUtilities_addLongAttributeToList(RaveObjectList_t* l, const char* name, long value)
{
  int result = 0;
  RaveAttribute_t* attr = NULL;
  RAVE_ASSERT((l != NULL), "l == NULL");
  attr = RaveAttributeHelp_createLong(name, value);
  if (attr != NULL) {
    if (RaveObjectList_add(l, (RaveCoreObject*)attr)) {
      result = 1;
    }
  }
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

int RaveUtilities_addDoubleAttributeToList(RaveObjectList_t* l, const char* name, double value)
{
  int result = 0;
  RaveAttribute_t* attr = NULL;
  RAVE_ASSERT((l != NULL), "l == NULL");
  attr = RaveAttributeHelp_createDouble(name, value);
  if (attr != NULL) {
    if (RaveObjectList_add(l, (RaveCoreObject*)attr)) {
      result = 1;
    }
  }
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

int RaveUtilities_addStringAttributeToList(RaveObjectList_t* l, const char* name, const char* value)
{
  int result = 0;
  RaveAttribute_t* attr = NULL;
  RAVE_ASSERT((l != NULL), "l == NULL");
  attr = RaveAttributeHelp_createString(name, value);
  if (attr != NULL) {
    if (RaveObjectList_add(l, (RaveCoreObject*)attr)) {
      result = 1;
    }
  }
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

int RaveUtilities_replaceLongAttributeInList(RaveObjectList_t* l, const char* name, long value)
{
  int result = 0;
  RaveAttribute_t* attr = NULL;
  int n = 0;
  int i = 0;
  RAVE_ASSERT((l != NULL), "l == NULL");
  RAVE_ASSERT((name != NULL), "name == NULL");

  n = RaveObjectList_size(l);

  for (i = 0; result == 0 && i < n; i++) {
    attr = (RaveAttribute_t*)RaveObjectList_get(l, i);
    if (attr != NULL && RaveAttribute_getName(attr) != NULL && strcmp(name, RaveAttribute_getName(attr)) == 0) {
      RaveAttribute_setLong(attr, value);
      result = 1;
    }
    RAVE_OBJECT_RELEASE(attr);
  }

  if (result == 0) {
    result = RaveUtilities_addLongAttributeToList(l, name, value);
  }

  return result;

}

int RaveUtilities_replaceDoubleAttributeInList(RaveObjectList_t* l, const char* name, double value)
{
  int result = 0;
  RaveAttribute_t* attr = NULL;
  int n = 0;
  int i = 0;
  RAVE_ASSERT((l != NULL), "l == NULL");
  RAVE_ASSERT((name != NULL), "name == NULL");

  n = RaveObjectList_size(l);

  for (i = 0; result == 0 && i < n; i++) {
    attr = (RaveAttribute_t*)RaveObjectList_get(l, i);
    if (attr != NULL && RaveAttribute_getName(attr) != NULL && strcmp(name, RaveAttribute_getName(attr)) == 0) {
      RaveAttribute_setDouble(attr, value);
      result = 1;
    }
    RAVE_OBJECT_RELEASE(attr);
  }

  if (result == 0) {
    result = RaveUtilities_addDoubleAttributeToList(l, name, value);
  }

  return result;
}

int RaveUtilities_replaceStringAttributeInList(RaveObjectList_t* l, const char* name, const char* value)
{
  int result = 0;
  RaveAttribute_t* attr = NULL;
  int n = 0;
  int i = 0;
  int found = 0;
  RAVE_ASSERT((l != NULL), "l == NULL");
  RAVE_ASSERT((name != NULL), "name == NULL");

  n = RaveObjectList_size(l);

  for (i = 0; found == 0 && i < n; i++) {
    attr = (RaveAttribute_t*)RaveObjectList_get(l, i);
    if (attr != NULL && RaveAttribute_getName(attr) != NULL && strcmp(name, RaveAttribute_getName(attr)) == 0) {
      result = RaveAttribute_setString(attr, value);
      found = 1;
    }
    RAVE_OBJECT_RELEASE(attr);
  }

  if (found == 0) {
    result = RaveUtilities_addStringAttributeToList(l, name, value);
  }

  return result;
}

int RaveUtilities_getRaveAttributeDoubleFromHash(RaveObjectHashTable_t* h, const char* name, double* v)
{
  RaveAttribute_t* attr = NULL;
  int result = 0;

  RAVE_ASSERT((h != NULL), "h == NULL");
  RAVE_ASSERT((v != NULL), "v == NULL");

  attr = (RaveAttribute_t*)RaveObjectHashTable_get(h, name);
  if (attr != NULL) {
    result = RaveAttribute_getDouble(attr, v);
  }

  RAVE_OBJECT_RELEASE(attr);
  return result;
}
