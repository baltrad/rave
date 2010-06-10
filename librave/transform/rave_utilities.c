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

