/* --------------------------------------------------------------------
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Data value abstraction
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-02-14
 */
#include "rave_value.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>
#include "rave_object.h"
#include "rave_data2d.h"
#include <errno.h>
#include <stdio.h>

/**
 * Represents one scan in a volume.
 */
struct _RaveValue_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveValue_Type type;  /**< the value type */
  char* stringValue;    /**< the string value */
  long longValue;      /**< the long value */
  double doubleValue;  /**< the double value */
  long* longArray;     /**< the long array */
  double* doubleArray; /**< the double array */
  char** stringArray;  /**< the string array */
  int arraylen;        /**< length of arrays */
};

/*@{ Private functions */
static void RaveValueInternal_freeStringArray(char*** sdata, int len)
{
  if (sdata != NULL) {
    if (*sdata != NULL) {
      int i;
      for (i = 0; i < len; i++) {
        RAVE_FREE((*sdata)[i]);
      }
      RAVE_FREE(*sdata);
      *sdata = NULL;
    }
  }
}

/**
 * Constructor.
 */
static int RaveValue_constructor(RaveCoreObject* obj)
{
  RaveValue_t* this = (RaveValue_t*)obj;
  this->type = RaveValue_Type_Undefined;
  this->stringValue = NULL;
  this->longValue = 0;
  this->doubleValue = 0.0;
  this->longArray = NULL;
  this->doubleArray = NULL;
  this->stringArray = NULL;
  this->arraylen = 0;
  return 1;
}

static int RaveValue_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RaveValue_t* this = (RaveValue_t*)obj;
  RaveValue_t* src = (RaveValue_t*)srcobj;
  this->type = RaveValue_Type_Undefined;
  this->longValue = src->longValue;
  this->doubleValue = src->doubleValue;
  this->arraylen = src->arraylen;
  this->stringValue = NULL;
  this->longArray = NULL;
  this->doubleArray = NULL;
  this->stringArray = NULL;

  if (src->stringValue != NULL && !RaveValue_setString(this, (const char*)src->stringValue)) {
    RAVE_ERROR0("Failed to clone string");
    goto fail;
  }

  if (src->stringArray != NULL && !RaveValue_setStringArray(this, (const char**)src->stringArray, src->arraylen)) {
    RAVE_ERROR0("Failed to clone string array");
    goto fail;
  }

  if (src->longArray != NULL && !RaveValue_setLongArray(this, src->longArray, src->arraylen)) {
    RAVE_ERROR0("Failed to clone longArray");
    goto fail;
  }

  if (src->doubleArray != NULL && !RaveValue_setDoubleArray(this, src->doubleArray, src->arraylen)) {
    RAVE_ERROR0("Failed to clone doubleArray");
    goto fail;
  }

  return 1;
fail:
  RAVE_FREE(this->stringValue);
  RAVE_FREE(this->longArray);
  RAVE_FREE(this->doubleArray);
  RaveValueInternal_freeStringArray(&this->stringArray, this->arraylen);
  return 0;
}

/**
 * Destructor.
 */
static void RaveValue_destructor(RaveCoreObject* obj)
{
  RaveValue_t* this = (RaveValue_t*)obj;
  RAVE_FREE(this->stringValue);
  RAVE_FREE(this->longArray);
  RAVE_FREE(this->doubleArray);
  
}

/*@} End of Private functions */

/*@{ Interface functions */
RaveValue_Type RaveValue_type(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->type;
}

void RaveValue_reset(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->type = RaveValue_Type_Undefined;
  RAVE_FREE(self->stringValue);
  self->longValue = 0;
  self->doubleValue = 0.0;
  RAVE_FREE(self->longArray);
  RAVE_FREE(self->doubleArray);
  RaveValueInternal_freeStringArray(&self->stringArray, self->arraylen);  
  self->arraylen = 0;

}


RaveValue_t* RaveValue_createString(const char* value)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);
  if (result != NULL) {
    if (!RaveValue_setString(result, value)) {
      RAVE_OBJECT_RELEASE(result);
    }
  }
  return result;
}

int RaveValue_setString(RaveValue_t* self, const char* value)
{
  char* tdata = NULL;
  RAVE_ASSERT((self != NULL), "attr == NULL");

  if (value != NULL) {
    tdata = RAVE_STRDUP(value);
    if (tdata == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for string");
      goto fail;
    }
  }
  RaveValue_reset(self);

  if (tdata != NULL) {
    self->stringValue = tdata;
  }
  self->type = RaveValue_Type_String;
  return 1;
fail:
  return 0;
}

int RaveValue_getString(RaveValue_t* self, char** value)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  if (self->type == RaveValue_Type_String) {
    *value = self->stringValue;
    result = 1;
  }
  return result;
}

const char* RaveValue_toString(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "attr == NULL");
  return (const char*)self->stringValue;
}

RaveValue_t* RaveValue_createLong(long value)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);

  if (result != NULL) {
    RaveValue_setLong(result, value);
  }

  return result;
}

void RaveValue_setLong(RaveValue_t* self, long value)
{
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RaveValue_reset(self);
  self->longValue = value;
  self->type = RaveValue_Type_Long;
}

int RaveValue_getLong(RaveValue_t* self, long* value)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  if (self->type == RaveValue_Type_Long) {
    *value = self->longValue;
    result = 1;
  }
  return result;
}

long RaveValue_toLong(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "attr == NULL");
  return self->longValue;
}

RaveValue_t* RaveValue_createDouble(double value)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);

  if (result != NULL) {
    RaveValue_setDouble(result, value);
  }

  return result;

}

void RaveValue_setDouble(RaveValue_t* self, double value)
{
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RaveValue_reset(self);
  self->doubleValue = value;
  self->type = RaveValue_Type_Double;

}

int RaveValue_getDouble(RaveValue_t* self, double* value)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  if (self->type == RaveValue_Type_Double) {
    *value = self->doubleValue;
    result = 1;
  }
  return result;
}

long RaveValue_toDouble(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "attr == NULL");
  return self->doubleValue;
}

RaveValue_t* RaveValue_createStringArray(const char** value, int len)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);
  if (result != NULL) {
    if (!RaveValue_setStringArray(result, value, len)) {
      RAVE_OBJECT_RELEASE(result);
    }
  }
  return result;
}

int RaveValue_setStringArray(RaveValue_t* self, const char** value, int len)
{
  char** sdata = NULL;
  RAVE_ASSERT((self != NULL), "attr == NULL");

  if (value != NULL) {
    int i = 0;
    sdata = RAVE_MALLOC(sizeof(char*) * len);
    if (sdata == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for string array");
      goto error;
    }
    memcpy(sdata, value, sizeof(char*) * len);
    for (i = 0; i < len; i++) {
      sdata[i] = RAVE_STRDUP(value[i]);
      if (sdata[i] == NULL) {
        RAVE_ERROR0("Failed to duplicate string");
        RaveValueInternal_freeStringArray(&sdata, len);
        goto error;
      }
    }
  }
  RaveValue_reset(self);
  self->stringArray = sdata;
  self->arraylen = len;
  
  self->type = RaveValue_Type_StringArray;
  return 1;
error:
  return 0;  
}

int RaveValue_getStringArray(RaveValue_t* self, char*** value, int* len)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  RAVE_ASSERT((len != NULL), "len == NULL");
  if (self->type == RaveValue_Type_StringArray) {
    *value = self->stringArray;
    *len = self->arraylen;
    result = 1;
  }
  return result;  
}

RaveValue_t* RaveValue_createLongArray(long* value, int len)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);
  if (result != NULL) {
    if (!RaveValue_setLongArray(result, value, len)) {
      RAVE_OBJECT_RELEASE(result);
    }
  }
  return result;
}

int RaveValue_setLongArray(RaveValue_t* self, long* value, int len)
{
  long* ldata = NULL;
  RAVE_ASSERT((self != NULL), "attr == NULL");

  if (value != NULL) {
    ldata = RAVE_MALLOC(sizeof(long) * len);
    if (ldata == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for long array");
      goto error;
    }
    memcpy(ldata, value, sizeof(long) * len);
  }
  RaveValue_reset(self);
  self->longArray = ldata;
  self->arraylen = len;
  
  self->type = RaveValue_Type_LongArray;
  return 1;
error:
  return 0;
}

int RaveValue_getLongArray(RaveValue_t* self, long** value, int* len)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  if (self->type == RaveValue_Type_LongArray) {
    *value = self->longArray;
    *len = self->arraylen;    
    result = 1;
  }
  return result;
}

RaveValue_t* RaveValue_createDoubleArray(double* value, int len)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);
  if (result != NULL) {
    if (!RaveValue_setDoubleArray(result, value, len)) {
      RAVE_OBJECT_RELEASE(result);
    }
  }
  return result;
}

int RaveValue_setDoubleArray(RaveValue_t* self, double* value, int len)
{
  double* ddata = NULL;
  RAVE_ASSERT((self != NULL), "attr == NULL");

  if (value != NULL) {
    ddata = RAVE_MALLOC(sizeof(double) * len);
    if (ddata == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for double array");
      goto error;
    }
    memcpy(ddata, value, sizeof(double) * len);
  }
  RaveValue_reset(self);
  self->doubleArray = ddata;
  self->arraylen = len;
  
  self->type = RaveValue_Type_DoubleArray;
  return 1;
error:
  return 0;
}

int RaveValue_getDoubleArray(RaveValue_t* self, double** value, int* len)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  if (self->type == RaveValue_Type_DoubleArray) {
    *value = self->doubleArray;
    *len = self->arraylen;    
    result = 1;
  }
  return result;
}

/*@} End of Interface functions */

RaveCoreObjectType RaveValue_TYPE = {
  "RaveValue",
  sizeof(RaveValue_t),
  RaveValue_constructor,
  RaveValue_destructor,
  RaveValue_copyconstructor
};
