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
 * Represents a 2-dimensional data array.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-17
 */
#include "rave_data2d.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include <limits.h>
#include <float.h>
#include <string.h>

/**
 * Represents a date time instance
 */
struct _RaveData2D_t {
  RAVE_OBJECT_HEAD /** Always on top */
  long xsize;        /**< xsize */
  long ysize;        /**< ysize */
  RaveDataType type; /**< data type */
  void* data;        /**< data ptr */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int RaveData2D_constructor(RaveCoreObject* obj)
{
  RaveData2D_t* data = (RaveData2D_t*)obj;
  data->xsize = 0;
  data->ysize = 0;
  data->type = RaveDataType_UNDEFINED;
  data->data = NULL;
  return 1;
}

/**
 * Copy constructor.
 */
static int RaveData2D_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RaveData2D_t* data = (RaveData2D_t*)obj;
  RaveData2D_t* srcdata = (RaveData2D_t*)srcobj;
  data->xsize = 0;
  data->ysize = 0;
  data->type = RaveDataType_UNDEFINED;
  data->data = NULL;
  return RaveData2D_setData(data, srcdata->xsize, srcdata->ysize, srcdata->data, srcdata->type);
}

/**
 * Destructor.
 */
static void RaveData2D_destructor(RaveCoreObject* obj)
{
  RaveData2D_t* data = (RaveData2D_t*)obj;
  if (data != NULL) {
    RAVE_FREE(data->data);
  }
}

/*@} End of Private functions */

/*@{ Interface functions */
long RaveData2D_getXsize(RaveData2D_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->xsize;
}

long RaveData2D_getYsize(RaveData2D_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->ysize;

}

RaveDataType RaveData2D_getType(RaveData2D_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->type;

}

void* RaveData2D_getData(RaveData2D_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->data;
}

int RaveData2D_setData(RaveData2D_t* self, long xsize, long ysize, void* data, RaveDataType type)
{
  int result = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  result = RaveData2D_createData(self, xsize, ysize, type);
  if (result == 1 && data != NULL) {
    long sz = 0;
    long nbytes = 0;
    sz = get_ravetype_size(type);
    nbytes = xsize*ysize*sz;
    memcpy(self->data, data, nbytes);
  }

  return result;
}

int RaveData2D_createData(RaveData2D_t* self, long xsize, long ysize, RaveDataType type)
{
  long sz = 0;
  long nbytes = 0;
  void* ptr = NULL;
  int result = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (type <= RaveDataType_UNDEFINED || type >= RaveDataType_LAST) {
    RAVE_ERROR1("RaveData2D does not support the data type %d", type);
    return 0;
  }

  sz = get_ravetype_size(type);
  nbytes = xsize*ysize*sz;
  ptr = RAVE_MALLOC(nbytes);

  if (ptr == NULL) {
    RAVE_CRITICAL1("Failed to allocate memory (%d bytes)", (int)nbytes);
    goto fail;
  }
  memset(ptr, 0, nbytes);
  RAVE_FREE(self->data);
  self->data = ptr;
  self->xsize = xsize;
  self->ysize = ysize;
  self->type = type;
  result = 1;
fail:
  return result;
}

int RaveData2D_setValue(RaveData2D_t* self, long x, long y, double v)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->data == NULL) {
    RAVE_ERROR0("Atempting to set value when there is no data array");
    return 0;
  }
  if (x >= 0 && x < self->xsize && y >= 0 && y < self->ysize) {
    result = RaveData2D_setValueUnchecked(self, x, y, v);
  }
  return result;
}

int RaveData2D_setValueUnchecked(RaveData2D_t* self, long x, long y, double v)
{
  int result = 1;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->data == NULL) {
    RAVE_ERROR0("Atempting to set value when there is no data array");
    return 0;
  }

  switch (self->type) {
  case RaveDataType_CHAR: {
    char *a = (char *) self->data;
    int c = mytrunc(v);

    if (c < -128)
      c = -128;
    if (c > 127)
      c = 127;
    a[y * self->xsize + x] = c;
    break;
  }
  case RaveDataType_UCHAR: {
    unsigned char *a = (unsigned char *) self->data;
    unsigned char c;

    if (v < 0) /* Oops: Not allowed!*/
      v = 0;
    if (v > 255)
      v = 255;
    c = mytrunc(v);
    a[y * self->xsize + x] = c;
    break;
  }
  case RaveDataType_SHORT: {
    short *a = (short *) self->data;
    int c = mytrunc(v);
    if (c < SHRT_MIN) /* Oops: Not allowed!*/
      c = SHRT_MIN;
    if (c > SHRT_MAX)
      c = SHRT_MAX;
    a[y * self->xsize + x] = c;
    break;
  }
  case RaveDataType_INT: {
    int *a = (int *) self->data;
    int c;
    if (v > INT_MAX)
      v = INT_MAX;
    if (v < INT_MIN)
      v = INT_MIN;
    c = mytrunc(v);
    a[y * self->xsize + x] = c;
    break;
  }
  case RaveDataType_LONG: {
    long *a = (long *) self->data;
    long c;
    if (v > LONG_MAX)
      v = LONG_MAX;
    if (v < LONG_MIN)
      v = LONG_MIN;
    c = v; /* Should work on 64bit boxes after above preparations. */
    a[y * self->xsize + x] = c;
    break;
  }

  case RaveDataType_FLOAT: {
    float *a = (float *) self->data;
    if (v > FLT_MAX)
      v = FLT_MAX;
    if (v < FLT_MIN)
      v = FLT_MIN;
    a[y * self->xsize + x] = v;
    break;
  }
  case RaveDataType_DOUBLE: {
    double *a = (double *) self->data;
    a[y * self->xsize + x] = v;
    break;
  }
  default:
    RAVE_WARNING1("RaveData2D_setValue: Unsupported type: '%d'\n", self->type);
    result = 0;
  }

  return result;
}

int RaveData2D_getValue(RaveData2D_t* self, long x, long y, double* v)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->data == NULL) {
    RAVE_ERROR0("Atempting to get value when there is no data array");
    return 0;
  }

  if (v == NULL) {
    RAVE_ERROR0("Atempting to get a value without providing a value pointer");
    return 0;
  }

  if (x >= 0 && x < self->xsize && y >= 0 && y < self->ysize) {
    result = RaveData2D_getValueUnchecked(self, x, y, v);
  }

  return result;
}

int RaveData2D_getValueUnchecked(RaveData2D_t* self, long x, long y, double* v)
{
  int result = 1;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->data == NULL) {
    RAVE_ERROR0("Atempting to get value when there is no data array");
    return 0;
  }

  if (v == NULL) {
    RAVE_ERROR0("Atempting to get a value without providing a value pointer");
    return 0;
  }

  switch (self->type) {
  case RaveDataType_CHAR: {
    char *a = (char *) self->data;
    *v = a[y * self->xsize + x];
    break;
  }
  case RaveDataType_UCHAR: {
    unsigned char *a = (unsigned char *) self->data;
    *v = a[y * self->xsize + x];
    break;
  }
  case RaveDataType_SHORT: {
    short *a = (short *) self->data;
    *v = a[y * self->xsize + x];
    break;
  }
  case RaveDataType_INT: {
    int *a = (int *) self->data;
    *v = a[y * self->xsize + x];
    break;
  }
  case RaveDataType_LONG: {
    long *a = (long *) self->data;
    *v = a[y * self->xsize + x];
    break;
  }
  case RaveDataType_FLOAT: {
    float *a = (float *) self->data;
    *v = a[y * self->xsize + x];
    break;
  }
  case RaveDataType_DOUBLE: {
    double *a = (double *) self->data;
    *v = a[y * self->xsize + x];
    break;
  }
  default:
    RAVE_WARNING1("RaveData2D_getValue: Unsupported type: '%d'\n", self->type);
    result = 0;
  }

  return result;
}

int RaveData2D_hasData(RaveData2D_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");

  if (self->data != NULL && self->xsize > 0 && self->ysize > 0) {
    return 1;
  }

  return 0;
}

RaveData2D_t* RaveData2D_createObject(long xsize, long ysize, RaveDataType type)
{
  RaveData2D_t* result = NULL;
  result = RAVE_OBJECT_NEW(&RaveData2D_TYPE);
  if (result != NULL) {
    if (!RaveData2D_createData(result, xsize, ysize, type)) {
      RAVE_OBJECT_RELEASE(result);
    }
  }
  return result;
}

/*@} End of Interface functions */
RaveCoreObjectType RaveData2D_TYPE = {
    "RaveData2D",
    sizeof(RaveData2D_t),
    RaveData2D_constructor,
    RaveData2D_destructor,
    RaveData2D_copyconstructor
};
