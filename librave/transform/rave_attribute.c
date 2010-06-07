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
 * Defines the functions available when working with polar scans
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-15
 */
#include "rave_attribute.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>
#include "rave_object.h"
#include "rave_datetime.h"
#include "rave_transform.h"
#include "raveobject_hashtable.h"

/**
 * This is the default parameter value that should be used when working
 * with scans.
 */
#define DEFAULT_PARAMETER_NAME "DBZH"

/**
 * Represents one scan in a volume.
 */
struct _RaveAttribute_t {
  RAVE_OBJECT_HEAD /** Always on top */
  char* name;    /**< the source string */
  RaveAttribute_Format format;  /**< the attribute format */
  char* sdata;    /**< the string value */
  long ldata;       /**< the long value */
  double ddata;     /**< the double value */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int RaveAttribute_constructor(RaveCoreObject* obj)
{
  RaveAttribute_t* attr = (RaveAttribute_t*)obj;
  attr->name = NULL;
  attr->format = RaveAttribute_Format_Undefined;
  attr->sdata = NULL;
  attr->ldata = 0;
  attr->ddata = 0.0;
  return 1;
}

static int RaveAttribute_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RaveAttribute_t* this = (RaveAttribute_t*)obj;
  RaveAttribute_t* src = (RaveAttribute_t*)srcobj;
  this->name = NULL;
  this->sdata = NULL;
  this->ldata = 0;
  this->ddata = 0.0;
  this->format = RaveAttribute_Format_Undefined;

  if (!RaveAttribute_setName(this, RaveAttribute_getName(src))) {
    goto error;
  }
  this->format = src->format;
  if (this->format == RaveAttribute_Format_Long) {
    this->ldata = src->ldata;
  } else if (this->format == RaveAttribute_Format_Double) {
    this->ddata = src->ddata;
  } else if (this->format == RaveAttribute_Format_String) {
    if (!RaveAttribute_setString(this, src->sdata)) {
      goto error;
    }
  }

  return 1;
error:
  RAVE_FREE(this->name);
  RAVE_FREE(this->sdata);
  return 0;
}

/**
 * Destructor.
 */
static void RaveAttribute_destructor(RaveCoreObject* obj)
{
  RaveAttribute_t* attr = (RaveAttribute_t*)obj;
  RAVE_FREE(attr->name);
  RAVE_FREE(attr->sdata);
}

/*@} End of Private functions */

/*@{ Interface functions */
int RaveAttribute_setName(RaveAttribute_t* attr, const char* name)
{
  RAVE_ASSERT((attr != NULL), "attr == NULL");
  RAVE_FREE(attr->name);
  if (name != NULL) {
    attr->name = RAVE_STRDUP(name);
    if (attr->name == NULL) {
      RAVE_CRITICAL0("Failure when copying name");
      return 0;
    }
  }
  return 1;
}

const char* RaveAttribute_getName(RaveAttribute_t* attr)
{
  RAVE_ASSERT((attr != NULL), "attr == NULL");
  return (const char*)attr->name;
}

RaveAttribute_Format RaveAttribute_getFormat(RaveAttribute_t* attr)
{
  RAVE_ASSERT((attr != NULL), "attr == NULL");
  return attr->format;
}

void RaveAttribute_setLong(RaveAttribute_t* attr, long value)
{
  RAVE_ASSERT((attr != NULL), "attr == NULL");
  RAVE_FREE(attr->sdata);
  attr->ldata = value;
  attr->format = RaveAttribute_Format_Long;
}

void RaveAttribute_setDouble(RaveAttribute_t* attr, double value)
{
  RAVE_ASSERT((attr != NULL), "attr == NULL");
  RAVE_FREE(attr->sdata);
  attr->ddata = value;
  attr->format = RaveAttribute_Format_Double;
}

int RaveAttribute_setString(RaveAttribute_t* attr, char* value)
{
  RAVE_ASSERT((attr != NULL), "attr == NULL");
  char* tdata = NULL;
  if (value != NULL) {
    tdata = RAVE_STRDUP(value);
    if (tdata == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for string");
      goto error;
    }
  }
  RAVE_FREE(attr->sdata);
  if (tdata != NULL) {
    attr->sdata = tdata;
  }
  attr->format = RaveAttribute_Format_String;
  return 1;
error:
  return 0;
}

int RaveAttribute_getLong(RaveAttribute_t* attr, long* value)
{
  int result = 0;
  RAVE_ASSERT((attr != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  if (attr->format == RaveAttribute_Format_Long) {
    *value = attr->ldata;
    result = 1;
  }
  return result;
}

int RaveAttribute_getDouble(RaveAttribute_t* attr, double* value)
{
  int result = 0;
  RAVE_ASSERT((attr != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  if (attr->format == RaveAttribute_Format_Double) {
    *value = attr->ddata;
    result = 1;
  }
  return result;
}

int RaveAttribute_getString(RaveAttribute_t* attr, char** value)
{
  int result = 0;
  RAVE_ASSERT((attr != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  if (attr->format == RaveAttribute_Format_String) {
    *value = attr->sdata;
    result = 1;
  }
  return result;
}

int RaveAttributeHelp_extractGroupAndName(
  const char* attrname, char** group, char** name)
{
  char *n1 = NULL, *n2 = NULL, *p = NULL;
  int slen = 0, n1len = 0, n2len = 0;
  int result = 0;
  RAVE_ASSERT((attrname != NULL), "attrname == NULL");

  slen = strlen(attrname);
  p = strchr(attrname, '/');
  if (p != NULL) {
    n1len = (p - attrname);
    n2len = (slen - (n1len+1));
    if (n1len <= 0 || n2len <= 0) {
      RAVE_ERROR0("attrname is not in format <group>/<name> where group is how,what or where");
      goto done;
    }
    n1 = RAVE_MALLOC((n1len + 1)*sizeof(char));
    n2 = RAVE_MALLOC((n2len + 1)*sizeof(char));
    if (n1 == NULL || n2 == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for n1 or n2");
      goto done;
    }
    strncpy(n1, attrname, n1len);
    n1[n1len] = '\0';
    strncpy(n2, attrname+(n1len+1), n2len);
    n2[n2len] = '\0';
    *group = n1;
    *name = n2;
    n1 = NULL; /* release responsibility for memory */
    n2 = NULL; /* release responsibility for memory */
  } else {
    RAVE_ERROR0("attrname is not in format <group>/<name> where group is how,what or where\n");
    goto done;
  }

  result = 1;
done:
  RAVE_FREE(n1);
  RAVE_FREE(n2);
  return result;
}
/*@} End of Interface functions */

RaveCoreObjectType RaveAttribute_TYPE = {
  "RaveAttribute",
  sizeof(RaveAttribute_t),
  RaveAttribute_constructor,
  RaveAttribute_destructor,
  RaveAttribute_copyconstructor
};
