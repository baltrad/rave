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
#include "polarscanparam.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>
#include "rave_object.h"
#include "rave_datetime.h"
#include "rave_transform.h"
#include "rave_data2d.h"
#include "raveobject_hashtable.h"
#include "rave_utilities.h"

/**
 * Represents one param in a scan
 */
struct _PolarScanParam_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveData2D_t* data; /**< data ptr */
  char* quantity;    /**< what does this data represent */
  double gain;       /**< gain when scaling */
  double offset;     /**< offset when scaling */
  double nodata;     /**< nodata */
  double undetect;   /**< undetect */
  RaveObjectHashTable_t* attrs; /**< attributes */
  RaveObjectList_t* qualityfields; /**< quality fields */
};

/*@{ Private functions */

/**
 * Constructor.
 */
static int PolarScanParam_constructor(RaveCoreObject* obj)
{
  PolarScanParam_t* this = (PolarScanParam_t*)obj;
  this->data = RAVE_OBJECT_NEW(&RaveData2D_TYPE);
  this->attrs = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  this->qualityfields = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  this->quantity = NULL;
  this->gain = 0.0L;
  this->offset = 0.0L;
  this->nodata = 0.0L;
  this->undetect = 0.0L;
  if (this->data == NULL || this->attrs == NULL || this->qualityfields == NULL) {
    goto error;
  }
  return 1;
error:
  RAVE_OBJECT_RELEASE(this->data);
  RAVE_OBJECT_RELEASE(this->attrs);
  RAVE_OBJECT_RELEASE(this->qualityfields);

  return 0;
}

static int PolarScanParam_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  PolarScanParam_t* this = (PolarScanParam_t*)obj;
  PolarScanParam_t* src = (PolarScanParam_t*)srcobj;
  this->data = RAVE_OBJECT_CLONE(src->data);
  this->attrs = RAVE_OBJECT_CLONE(src->attrs);
  this->qualityfields = RAVE_OBJECT_CLONE(src->qualityfields);
  this->quantity = NULL;

  if (this->data == NULL || this->attrs == NULL || this->qualityfields == NULL) {
    goto error;
  }
  if (!PolarScanParam_setQuantity(this, PolarScanParam_getQuantity(src))) {
    goto error;
  }

  this->gain = src->gain;
  this->offset = src->offset;
  this->nodata = src->nodata;
  this->undetect = src->undetect;
  return 1;
error:
  RAVE_OBJECT_RELEASE(this->data);
  RAVE_OBJECT_RELEASE(this->attrs);
  RAVE_OBJECT_RELEASE(this->qualityfields);
  RAVE_FREE(this->quantity);
  return 0;
}

/**
 * Destructor.
 */
static void PolarScanParam_destructor(RaveCoreObject* obj)
{
  PolarScanParam_t* this = (PolarScanParam_t*)obj;
  RAVE_OBJECT_RELEASE(this->data);
  RAVE_OBJECT_RELEASE(this->attrs);
  RAVE_OBJECT_RELEASE(this->qualityfields);
  RAVE_FREE(this->quantity);
}

/*@} End of Private functions */

/*@{ Interface functions */
int PolarScanParam_setQuantity(PolarScanParam_t* scanparam, const char* quantity)
{
  int result = 0;
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  if (quantity != NULL) {
    char* tmp = RAVE_STRDUP(quantity);
    if (tmp != NULL) {
      RAVE_FREE(scanparam->quantity);
      scanparam->quantity = tmp;
      result = 1;
    }
  } else {
    RAVE_FREE(scanparam->quantity);
    result = 1;
  }
  return result;
}

const char* PolarScanParam_getQuantity(PolarScanParam_t* scanparam)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  return (const char*)scanparam->quantity;
}

void PolarScanParam_setGain(PolarScanParam_t* scanparam, double gain)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  scanparam->gain = gain;
}

double PolarScanParam_getGain(PolarScanParam_t* scanparam)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  return scanparam->gain;
}

void PolarScanParam_setOffset(PolarScanParam_t* scanparam, double offset)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  scanparam->offset = offset;
}

double PolarScanParam_getOffset(PolarScanParam_t* scanparam)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  return scanparam->offset;
}

void PolarScanParam_setNodata(PolarScanParam_t* scanparam, double nodata)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  scanparam->nodata = nodata;
}

double PolarScanParam_getNodata(PolarScanParam_t* scanparam)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  return scanparam->nodata;
}

void PolarScanParam_setUndetect(PolarScanParam_t* scanparam, double undetect)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  scanparam->undetect = undetect;
}

double PolarScanParam_getUndetect(PolarScanParam_t* scanparam)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  return scanparam->undetect;
}

int PolarScanParam_setData(PolarScanParam_t* scanparam, long nbins, long nrays, void* data, RaveDataType type)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  return RaveData2D_setData(scanparam->data, nbins, nrays, data, type);
}

int PolarScanParam_createData(PolarScanParam_t* scanparam, long nbins, long nrays, RaveDataType type)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  return RaveData2D_createData(scanparam->data, nbins, nrays, type);
}

void* PolarScanParam_getData(PolarScanParam_t* scanparam)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  return RaveData2D_getData(scanparam->data);
}

long PolarScanParam_getNbins(PolarScanParam_t* scanparam)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  return RaveData2D_getXsize(scanparam->data);
}

long PolarScanParam_getNrays(PolarScanParam_t* scanparam)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  return RaveData2D_getYsize(scanparam->data);
}

RaveDataType PolarScanParam_getDataType(PolarScanParam_t* scanparam)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  return RaveData2D_getType(scanparam->data);
}

RaveValueType PolarScanParam_getValue(PolarScanParam_t* scanparam, int bin, int ray, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  double value = 0.0;

  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");

  value = scanparam->nodata;

  if (RaveData2D_getValue(scanparam->data, bin, ray, &value)) {
    result = RaveValueType_DATA;
    if (value == scanparam->nodata) {
      result = RaveValueType_NODATA;
    } else if (value == scanparam->undetect) {
      result = RaveValueType_UNDETECT;
    }
  }

  if (v != NULL) {
    *v = value;
  }

  return result;
}

RaveValueType PolarScanParam_getConvertedValue(PolarScanParam_t* scanparam, int bin, int ray, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  if (v != NULL) {
    result =  PolarScanParam_getValue(scanparam, bin, ray, v);
    if (result == RaveValueType_DATA) {
      *v = scanparam->offset + (*v) * scanparam->gain;
    }
  }
  return result;
}

int PolarScanParam_setValue(PolarScanParam_t* scanparam, int bin, int ray, double v)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  return RaveData2D_setValue(scanparam->data, bin, ray, v);
}

int PolarScanParam_addAttribute(PolarScanParam_t* scanparam,
  RaveAttribute_t* attribute)
{
  const char* name = NULL;
  char* aname = NULL;
  char* gname = NULL;
  int result = 0;
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  RAVE_ASSERT((attribute != NULL), "attribute == NULL");

  name = RaveAttribute_getName(attribute);
  if (name != NULL) {
    /*
     * what/gain
     * what/offset
     * what/nodata
     * what/undetect
     * what/quantity
     */
    if (strcasecmp("what/gain", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract what/gain as a double");
      }
      PolarScanParam_setGain(scanparam, value);
    } else if (strcasecmp("what/offset", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract what/offset as a double");
      }
      PolarScanParam_setOffset(scanparam, value);
    } else if (strcasecmp("what/nodata", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract what/nodata as a double");
      }
      PolarScanParam_setNodata(scanparam, value);
    } else if (strcasecmp("what/undetect", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract what/undetect as a double");
      }
      PolarScanParam_setUndetect(scanparam, value);
    } else if (strcasecmp("what/quantity", name)==0) {
      char* value = NULL;
      if(!RaveAttribute_getString(attribute, &value)) {
        RAVE_ERROR0("Failed to extract what/quantity as a string");
        goto done;
      }
      if (!(result = PolarScanParam_setQuantity(scanparam, value))) {
        RAVE_ERROR1("Failed to add %s attribute", name);
        goto done;
      }
    } else {
      if (!RaveAttributeHelp_extractGroupAndName(name, &gname, &aname)) {
        RAVE_ERROR1("Failed to extract group and name from %s", name);
        goto done;
      }
      if ((strcasecmp("how", gname)==0 ||
          strcasecmp("what", gname)==0 ||
          strcasecmp("where", gname)==0) &&
        strchr(aname, '/') == NULL) {
        result = RaveObjectHashTable_put(scanparam->attrs, name, (RaveCoreObject*)attribute);
      }
    }
  }

done:
  RAVE_FREE(aname);
  RAVE_FREE(gname);
  return result;
}

RaveAttribute_t* PolarScanParam_getAttribute(PolarScanParam_t* scanparam,
  const char* name)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  if (name == NULL) {
    RAVE_ERROR0("Trying to get an attribute with NULL name");
    return NULL;
  }
  return (RaveAttribute_t*)RaveObjectHashTable_get(scanparam->attrs, name);
}

RaveList_t* PolarScanParam_getAttributeNames(PolarScanParam_t* scanparam)
{
  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  return RaveObjectHashTable_keys(scanparam->attrs);
}

RaveObjectList_t* PolarScanParam_getAttributeValues(PolarScanParam_t* scanparam)
{
  RaveObjectList_t* result = NULL;
  RaveObjectList_t* tableattrs = NULL;

  RAVE_ASSERT((scanparam != NULL), "scanparam == NULL");
  tableattrs = RaveObjectHashTable_values(scanparam->attrs);
  if (tableattrs == NULL) {
    goto error;
  }
  result = RAVE_OBJECT_CLONE(tableattrs);
  if (result == NULL) {
    goto error;
  }

  /*
   * what/gain
   * what/offset
   * what/nodata
   * what/undetect
   * what/quantity
   */
  if (!RaveUtilities_addDoubleAttributeToList(result, "what/gain", PolarScanParam_getGain(scanparam)) ||
      !RaveUtilities_addDoubleAttributeToList(result, "what/offset", PolarScanParam_getOffset(scanparam)) ||
      !RaveUtilities_addDoubleAttributeToList(result, "what/nodata", PolarScanParam_getNodata(scanparam)) ||
      !RaveUtilities_addDoubleAttributeToList(result, "what/undetect", PolarScanParam_getUndetect(scanparam)) ||
      !RaveUtilities_addStringAttributeToList(result, "what/quantity", PolarScanParam_getQuantity(scanparam))) {
    goto error;
  }

  RAVE_OBJECT_RELEASE(tableattrs);
  return result;
error:
  RAVE_OBJECT_RELEASE(result);
  RAVE_OBJECT_RELEASE(tableattrs);
  return NULL;
}

int PolarScanParam_addQualityField(PolarScanParam_t* param, RaveField_t* field)
{
  RAVE_ASSERT((param != NULL), "param == NULL");
  return RaveObjectList_add(param->qualityfields, (RaveCoreObject*)field);
}

RaveField_t* PolarScanParam_getQualityField(PolarScanParam_t* param, int index)
{
  RAVE_ASSERT((param != NULL), "param == NULL");
  return (RaveField_t*)RaveObjectList_get(param->qualityfields, index);
}

int PolarScanParam_getNumberOfQualityFields(PolarScanParam_t* param)
{
  RAVE_ASSERT((param != NULL), "param == NULL");
  return RaveObjectList_size(param->qualityfields);
}

void PolarScanParam_removeQualityField(PolarScanParam_t* param, int index)
{
  RaveField_t* field = NULL;
  RAVE_ASSERT((param != NULL), "param == NULL");
  field = (RaveField_t*)RaveObjectList_remove(param->qualityfields, index);
  RAVE_OBJECT_RELEASE(field);
}

/*@} End of Interface functions */

RaveCoreObjectType PolarScanParam_TYPE = {
    "PolarScanParam",
    sizeof(PolarScanParam_t),
    PolarScanParam_constructor,
    PolarScanParam_destructor,
    PolarScanParam_copyconstructor
};
