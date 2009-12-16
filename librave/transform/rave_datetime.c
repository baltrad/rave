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
 * Object for managing date and time.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-16
 */
#include "rave_datetime.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>

/**
 * Represents a date time instance
 */
struct _RaveDateTime_t {
  RAVE_OBJECT_HEAD /** Always on top */
  char date[9];    /**< the date string, format is YYYYMMDD */
  char time[7];    /**< the time string, format is HHmmss */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int RaveDateTime_constructor(RaveCoreObject* obj)
{
  RaveDateTime_t* dt = (RaveDateTime_t*)obj;
  strcpy(dt->date,"");
  strcpy(dt->time,"");
  return 1;
}

/**
 * Copy constructor.
 */
static int RaveDateTime_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RaveDateTime_t* dt = (RaveDateTime_t*)obj;
  RaveDateTime_t* srcdt = (RaveDateTime_t*)srcobj;
  strcpy(dt->date, srcdt->date);
  strcpy(dt->time, srcdt->time);
  return 1;
}

/**
 * Destructor.
 */
static void RaveDateTime_destructor(RaveCoreObject* obj)
{
  //RaveDateTime_t* scan = (RaveDateTime_t*)obj;
}

/**
 * Verifies that the string only contains digits.
 * @param[in] value - the null terminated string
 * @returns 1 if the string only contains digits, otherwise 0
 */
static int RaveDateTimeInternal_isDigits(const char* value)
{
  int result = 0;
  if (value != NULL) {
    int len = strlen(value);
    int i = 0;
    result = 1;
    for (i = 0; result == 1 && i < len; i++) {
      if (value[i] < 0x30 || value[i] > 0x39) {
        result = 0;
      }
    }
  }
  return result;
}
/*@} End of Private functions */

/*@{ Interface functions */
int RaveDateTime_setTime(RaveDateTime_t* dt, const char* value)
{
  int result = 0;
  RAVE_ASSERT((dt != NULL), "dt was NULL");
  if (value == NULL) {
    strcpy(dt->time, "");
  } else {
    if (strlen(value) == 6 && RaveDateTimeInternal_isDigits(value)) {
      strcpy(dt->time, value);
      result = 1;
    }
  }
  return result;
}

const char* RaveDateTime_getTime(RaveDateTime_t* dt)
{
  RAVE_ASSERT((dt != NULL), "dt was NULL");
  if (strcmp(dt->time, "") == 0) {
    return NULL;
  }
  return (const char*)dt->time;
}

int RaveDateTime_setDate(RaveDateTime_t* dt, const char* value)
{
  int result = 0;
  RAVE_ASSERT((dt != NULL), "dt was NULL");
  if (value == NULL) {
    strcpy(dt->date, "");
  } else {
    if (strlen(value) == 8 && RaveDateTimeInternal_isDigits(value)) {
      strcpy(dt->date, value);
      result = 1;
    }
  }
  return result;
}

const char* RaveDateTime_getDate(RaveDateTime_t* dt)
{
  RAVE_ASSERT((dt != NULL), "dt was NULL");
  if (strcmp(dt->date, "") == 0) {
    return NULL;
  }
  return (const char*)dt->date;
}


/*@} End of Interface functions */
RaveCoreObjectType RaveDateTime_TYPE = {
    "RaveDateTime",
    sizeof(RaveDateTime_t),
    RaveDateTime_constructor,
    RaveDateTime_destructor,
    RaveDateTime_copyconstructor
};
