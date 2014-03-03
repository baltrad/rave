/* --------------------------------------------------------------------
Copyright (C) 2014 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Implementation of the QI-total algorithm
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2014-02-27
 */
#include "rave_qitotal.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include <string.h>
#include <stdio.h>
/**
 * Represents the QI total generator
 */
struct _RaveQITotal_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveDataType dtype; /**< data type to be used for the result */
  double offset; /**< offset to use for the result */
  double gain; /**< gain to use for the result */

};

/*@{ Private functions */
/**
 * Constructor
 */
static int RaveQITotal_constructor(RaveCoreObject* obj)
{
  RaveQITotal_t* self = (RaveQITotal_t*)obj;
  self->dtype = RaveDataType_DOUBLE;
  self->gain = 1.0;
  self->offset = 0.0;
  return 1;
}

/**
 * Copy constructor
 */
static int RaveQITotal_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RaveQITotal_t* self = (RaveQITotal_t*)obj;
  RaveQITotal_t* src = (RaveQITotal_t*)obj;
  self->dtype = src->dtype;
  self->gain = src->gain;
  self->offset = src->offset;
  return 1;
}

/**
 * Destructor
 */
static void RaveQITotal_destructor(RaveCoreObject* obj)
{
  /*RaveQITotal_t* self = (RaveQITotal_t*)obj;*/
}

static int RaveQITotalInternal_checkFieldsInList(RaveObjectList_t* fields)
{
  int nlen = 0, i = 0, result = 0;
  RaveCoreObject* ob = NULL;
  nlen = RaveObjectList_size(fields);
  for (i = 0; i < nlen; i++) {
    ob = RaveObjectList_get(fields, i);
    if (!RAVE_OBJECT_CHECK_TYPE(ob, &RaveField_TYPE)) {
      RAVE_ERROR0("Can only process RaveFieldCore objects");
      goto done;
    }
    RAVE_OBJECT_RELEASE(ob);
  }
  result = 1;
done:
  RAVE_OBJECT_RELEASE(ob);
  return result;
}

static int RaveQITotalInternal_checkFieldConsistency(RaveObjectList_t* fields, long* oxsize, long* oysize)
{
  int nlen = 0, i = 0, result = 0;
  long xsize = 0, ysize = 0;
  RaveField_t* field = NULL;

  nlen = RaveObjectList_size(fields);
  if (nlen == 0) {
    RAVE_ERROR0("No fields provided");
    goto done;
  }

  if (!RaveQITotalInternal_checkFieldsInList(fields)) {
    goto done;
  }

  field = (RaveField_t*)RaveObjectList_get(fields, 0);
  xsize = RaveField_getXsize(field);
  ysize = RaveField_getYsize(field);
  RAVE_OBJECT_RELEASE(field);

  for (i = 1; i < nlen; i++) {
    field = (RaveField_t*)RaveObjectList_get(fields, i);
    if (RaveField_getXsize(field) != xsize ||
        RaveField_getYsize(field) != ysize) {
      RAVE_ERROR0("Fields are not of the same dimension");
      goto done;
    }
    RAVE_OBJECT_RELEASE(field);
  }
  *oxsize = xsize;
  *oysize = ysize;

  result = 1;
done:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

static void RaveQITotalInternal_getOffsetGain(RaveField_t* field, double* offset, double* gain)
{
  RaveAttribute_t* attr = NULL;

  *offset = 0.0;
  *gain = 1.0;
  attr = RaveField_getAttribute(field, "what/offset");
  if (attr != NULL) {
    if (RaveAttribute_getFormat(attr) == RaveAttribute_Format_Double) {
      RaveAttribute_getDouble(attr, offset);
    } else if (RaveAttribute_getFormat(attr) == RaveAttribute_Format_Long) {
      long v = 0;
      RaveAttribute_getLong(attr, &v);
      *offset = (double)v;
    }
  }
  RAVE_OBJECT_RELEASE(attr);

  attr = RaveField_getAttribute(field, "what/gain");
  if (attr != NULL) {
    if (RaveAttribute_getFormat(attr) == RaveAttribute_Format_Double) {
      RaveAttribute_getDouble(attr, gain);
    } else if (RaveAttribute_getFormat(attr) == RaveAttribute_Format_Long) {
      long v = 0;
      RaveAttribute_getLong(attr, &v);
      *gain = (double)v;
    }
  }
  RAVE_OBJECT_RELEASE(attr);

}

static int QITotalInternal_addDoubleAttribute(RaveField_t* field, const char* attrname, double v)
{
  int result = 0;
  RaveAttribute_t* attr = RaveAttributeHelp_createDouble(attrname, v);
  if (!attr) {
    RAVE_CRITICAL1("Failed to create attribute %s", attrname);
    goto done;
  }
  if (!RaveField_addAttribute(field, attr)) {
    goto done;
  }
  result = 1;
done:
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

/*@} End of Private functions */

/*@{ Interface functions */
void RaveQITotal_setDatatype(RaveQITotal_t* self, RaveDataType dtype)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->dtype = dtype;
}

RaveDataType RaveQITotal_getDatatype(RaveQITotal_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->dtype;
}

int RaveQITotal_setGain(RaveQITotal_t* self, double gain)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (gain == 0.0) {
    RAVE_ERROR0("Can not set gain to 0.0");
    return 0;
  }
  self->gain = gain;
  return 1;
}

double RaveQITotal_getGain(RaveQITotal_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->gain;
}

void RaveQITotal_setOffset(RaveQITotal_t* self, double offset)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->offset = offset;
}

double RaveQITotal_getOffset(RaveQITotal_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->offset;
}

RaveField_t* RaveQITotal_multiplicative(RaveQITotal_t* self, RaveObjectList_t* fields)
{
  int nlen = 0, i = 0;
  long xsize = 0, ysize = 0, x = 0, y = 0;
  double offset = 0.0, gain = 0.0;
  RaveField_t* result = NULL;
  RaveField_t* qifield = NULL;
  RaveField_t* qifield_conv = NULL;
  RaveField_t* field = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (!RaveQITotalInternal_checkFieldConsistency(fields, &xsize, &ysize)) {
    RAVE_ERROR0("Fields are not consistant in dimensions");
    goto done;
  }

  qifield = RAVE_OBJECT_NEW(&RaveField_TYPE);
  if (!qifield || !RaveField_createData(qifield, xsize, ysize, RaveDataType_DOUBLE)) {
    RAVE_CRITICAL0("Memory allocation error");
    goto done;
  }

  qifield_conv = RAVE_OBJECT_NEW(&RaveField_TYPE);
  if (!qifield_conv || !RaveField_createData(qifield_conv, xsize, ysize, self->dtype)) {
    RAVE_CRITICAL0("Memory allocation error");
    goto done;
  }

  if (!QITotalInternal_addDoubleAttribute(qifield_conv, "what/gain", self->gain) ||
      !QITotalInternal_addDoubleAttribute(qifield_conv, "what/offset", self->offset)) {
    goto done;
  }

  nlen = RaveObjectList_size(fields);
  field = (RaveField_t*)RaveObjectList_get(fields, 0);
  RaveQITotalInternal_getOffsetGain(field, &offset, &gain);

  for (x = 0; x < xsize; x++) {
    for (y = 0; y < ysize; y++) {
      double v = 0.0;
      RaveField_getValue(field, x, y, &v);
      RaveField_setValue(qifield, x, y, v * gain + offset);
    }
  }

  RAVE_OBJECT_RELEASE(field);

  for (i = 1; i < nlen; i++) {
    field = (RaveField_t*)RaveObjectList_get(fields, i);
    RaveQITotalInternal_getOffsetGain(field, &offset, &gain);
    for (x = 0; x < xsize; x++) {
      for (y = 0; y < ysize; y++) {
        double v = 0.0;
        double qivalue = 0.0;
        RaveField_getValue(qifield, x, y, &qivalue);
        RaveField_getValue(field, x, y, &v);
        RaveField_setValue(qifield, x, y, (v * gain + offset) * qivalue);
      }
    }
    RAVE_OBJECT_RELEASE(field);
  }

  for (x = 0; x < xsize; x++) {
    for (y = 0; y < ysize; y++) {
      double v = 0.0;
      RaveField_getValue(qifield, x, y, &v);
      RaveField_setValue(qifield_conv, x, y, (v - self->offset)/self->gain);
    }
  }


  result = RAVE_OBJECT_COPY(qifield_conv);
done:
  RAVE_OBJECT_RELEASE(qifield_conv);
  RAVE_OBJECT_RELEASE(qifield);
  RAVE_OBJECT_RELEASE(field);
  return result;
}

RaveField_t* RaveQITotal_additive(RaveQITotal_t* self, RaveObjectList_t* fields)
{
  int nlen = 0, i = 0;
  long xsize = 0, ysize = 0, x = 0, y = 0;
  double offset = 0.0, gain = 0.0;
  RaveField_t* result = NULL;
  RaveField_t* qifield = NULL;
  RaveField_t* qifield_conv = NULL;
  RaveField_t* field = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (!RaveQITotalInternal_checkFieldConsistency(fields, &xsize, &ysize)) {
    RAVE_ERROR0("Fields are not consistant in dimensions");
    goto done;
  }

  qifield = RAVE_OBJECT_NEW(&RaveField_TYPE);
  if (!qifield || !RaveField_createData(qifield, xsize, ysize, RaveDataType_DOUBLE)) {
    RAVE_CRITICAL0("Memory allocation error");
    goto done;
  }

  qifield_conv = RAVE_OBJECT_NEW(&RaveField_TYPE);
  if (!qifield_conv || !RaveField_createData(qifield_conv, xsize, ysize, self->dtype)) {
    RAVE_CRITICAL0("Memory allocation error");
    goto done;
  }

  if (!QITotalInternal_addDoubleAttribute(qifield_conv, "what/gain", self->gain) ||
      !QITotalInternal_addDoubleAttribute(qifield_conv, "what/offset", self->offset)) {
    goto done;
  }

  nlen = RaveObjectList_size(fields);
  field = (RaveField_t*)RaveObjectList_get(fields, 0);
  RaveQITotalInternal_getOffsetGain(field, &offset, &gain);

  for (x = 0; x < xsize; x++) {
    for (y = 0; y < ysize; y++) {
      double v = 0.0;
      RaveField_getValue(field, x, y, &v);
      RaveField_setValue(qifield, x, y, v * gain + offset);
    }
  }

  RAVE_OBJECT_RELEASE(field);

  for (i = 1; i < nlen; i++) {
    field = (RaveField_t*)RaveObjectList_get(fields, i);
    RaveQITotalInternal_getOffsetGain(field, &offset, &gain);
    for (x = 0; x < xsize; x++) {
      for (y = 0; y < ysize; y++) {
        double v = 0.0;
        double qivalue = 0.0;
        RaveField_getValue(qifield, x, y, &qivalue);
        RaveField_getValue(field, x, y, &v);
        RaveField_setValue(qifield, x, y, (v * gain + offset) + qivalue);
      }
    }
    RAVE_OBJECT_RELEASE(field);
  }

  for (x = 0; x < xsize; x++) {
    for (y = 0; y < ysize; y++) {
      double v = 0.0;
      RaveField_getValue(qifield, x, y, &v);
      RaveField_setValue(qifield_conv, x, y, ((v/(double)nlen) - self->offset)/self->gain);
    }
  }


  result = RAVE_OBJECT_COPY(qifield_conv);
done:
  RAVE_OBJECT_RELEASE(qifield_conv);
  RAVE_OBJECT_RELEASE(qifield);
  RAVE_OBJECT_RELEASE(field);
  return result;
}

#define RQIT_MIN(a,b) (a > b)?b:a

RaveField_t* RaveQITotal_minimum(RaveQITotal_t* self, RaveObjectList_t* fields)
{
  int nlen = 0, i = 0;
  long xsize = 0, ysize = 0, x = 0, y = 0;
  double offset = 0.0, gain = 0.0;
  RaveField_t* result = NULL;
  RaveField_t* qifield = NULL;
  RaveField_t* qifield_conv = NULL;
  RaveField_t* field = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (!RaveQITotalInternal_checkFieldConsistency(fields, &xsize, &ysize)) {
    RAVE_ERROR0("Fields are not consistant in dimensions");
    goto done;
  }

  qifield = RAVE_OBJECT_NEW(&RaveField_TYPE);
  if (!qifield || !RaveField_createData(qifield, xsize, ysize, RaveDataType_DOUBLE)) {
    RAVE_CRITICAL0("Memory allocation error");
    goto done;
  }

  qifield_conv = RAVE_OBJECT_NEW(&RaveField_TYPE);
  if (!qifield_conv || !RaveField_createData(qifield_conv, xsize, ysize, self->dtype)) {
    RAVE_CRITICAL0("Memory allocation error");
    goto done;
  }

  if (!QITotalInternal_addDoubleAttribute(qifield_conv, "what/gain", self->gain) ||
      !QITotalInternal_addDoubleAttribute(qifield_conv, "what/offset", self->offset)) {
    goto done;
  }

  nlen = RaveObjectList_size(fields);
  field = (RaveField_t*)RaveObjectList_get(fields, 0);
  RaveQITotalInternal_getOffsetGain(field, &offset, &gain);

  for (x = 0; x < xsize; x++) {
    for (y = 0; y < ysize; y++) {
      double v = 0.0;
      RaveField_getValue(field, x, y, &v);
      RaveField_setValue(qifield, x, y, v * gain + offset);
    }
  }

  RAVE_OBJECT_RELEASE(field);

  for (i = 1; i < nlen; i++) {
    field = (RaveField_t*)RaveObjectList_get(fields, i);
    RaveQITotalInternal_getOffsetGain(field, &offset, &gain);
    for (x = 0; x < xsize; x++) {
      for (y = 0; y < ysize; y++) {
        double v = 0.0;
        double qivalue = 0.0;
        RaveField_getValue(qifield, x, y, &qivalue);
        RaveField_getValue(field, x, y, &v);
        RaveField_setValue(qifield, x, y, RQIT_MIN(v * gain + offset, qivalue));
      }
    }
    RAVE_OBJECT_RELEASE(field);
  }

  for (x = 0; x < xsize; x++) {
    for (y = 0; y < ysize; y++) {
      double v = 0.0;
      RaveField_getValue(qifield, x, y, &v);
      RaveField_setValue(qifield_conv, x, y, (v - self->offset)/self->gain);
    }
  }


  result = RAVE_OBJECT_COPY(qifield_conv);
done:
  RAVE_OBJECT_RELEASE(qifield_conv);
  RAVE_OBJECT_RELEASE(qifield);
  RAVE_OBJECT_RELEASE(field);
  return result;
}
/*@} End of Interface functions */

RaveCoreObjectType RaveQITotal_TYPE = {
    "RaveQITotal",
    sizeof(RaveQITotal_t),
    RaveQITotal_constructor,
    RaveQITotal_destructor,
    RaveQITotal_copyconstructor
};
