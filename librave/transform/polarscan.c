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
#include "polarscan.h"
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
 * This is the default parameter value that should be used when working
 * with scans.
 */
#define DEFAULT_PARAMETER_NAME "DBZH"

/**
 * Represents one scan in a volume.
 */
struct _PolarScan_t {
  RAVE_OBJECT_HEAD /** Always on top */

  char* source;    /**< the source string */

  long nbins;      /**< number of bins */
  long nrays;      /**< number of rays */

  // Where
  double elangle;    /**< elevation of scan */
  double rscale;     /**< scale */
  double rstart;     /**< start of ray */
  long a1gate;       /**< something */

  // How
  double beamwidth;  /**< beam width */

  // Date/Time
  RaveDateTime_t* datetime;     /**< the date, time instance */

  // Navigator
  PolarNavigator_t* navigator; /**< a navigator for calculating polar navigation */

  // Projection wrapper
  Projection_t* projection; /**< projection for this scan */

  // Keeps all parameters
  RaveObjectHashTable_t* parameters;

  // Manages the default parameter
  char* paramname;
  PolarScanParam_t* param;

  RaveObjectHashTable_t* attrs; /**< attributes */

  RaveObjectList_t* qualityfields; /**< quality fields */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int PolarScan_constructor(RaveCoreObject* obj)
{
  PolarScan_t* scan = (PolarScan_t*)obj;
  scan->nbins = 0;
  scan->nrays = 0;
  scan->datetime = NULL;
  scan->source = NULL;
  scan->elangle = 0.0;
  scan->rscale = 0.0;
  scan->rstart = 0.0;
  scan->a1gate = 0;
  scan->beamwidth = 0.0;
  scan->navigator = NULL;
  scan->projection = NULL;
  scan->parameters = NULL;
  scan->paramname = NULL;
  scan->param = NULL;
  scan->datetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  scan->parameters = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  scan->projection = RAVE_OBJECT_NEW(&Projection_TYPE);
  scan->attrs = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  scan->qualityfields = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);

  if (scan->projection != NULL) {
    if(!Projection_init(scan->projection, "lonlat", "lonlat", "+proj=latlong +ellps=WGS84 +datum=WGS84")) {
      goto error;
    }
  }
  scan->navigator = RAVE_OBJECT_NEW(&PolarNavigator_TYPE);
  if (scan->datetime == NULL || scan->projection == NULL ||
      scan->navigator == NULL || scan->parameters == NULL ||
      scan->attrs == NULL || scan->qualityfields == NULL) {
    goto error;
  }
  if (!PolarScan_setDefaultParameter(scan, DEFAULT_PARAMETER_NAME)) {
    goto error;
  }
  return 1;
error:
  RAVE_OBJECT_RELEASE(scan->datetime);
  RAVE_OBJECT_RELEASE(scan->projection);
  RAVE_OBJECT_RELEASE(scan->navigator);
  RAVE_OBJECT_RELEASE(scan->parameters);
  RAVE_OBJECT_RELEASE(scan->attrs);
  RAVE_OBJECT_RELEASE(scan->qualityfields);
  RAVE_FREE(scan->paramname);
  RAVE_OBJECT_RELEASE(scan->param);
  return 0;
}

static int PolarScan_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  PolarScan_t* this = (PolarScan_t*)obj;
  PolarScan_t* src = (PolarScan_t*)srcobj;
  this->datetime = NULL;
  this->elangle = 0.0;
  this->rscale = 0.0;
  this->rstart = 0.0;
  this->a1gate = 0;
  this->beamwidth = 0.0;
  this->navigator = NULL;
  this->projection = NULL;
  this->paramname = NULL;
  this->param = NULL;

  this->source = NULL;
  this->datetime = RAVE_OBJECT_CLONE(src->datetime);
  this->projection = RAVE_OBJECT_CLONE(src->projection);
  this->navigator = RAVE_OBJECT_CLONE(src->navigator);
  this->parameters = RAVE_OBJECT_CLONE(src->parameters);
  this->attrs = RAVE_OBJECT_CLONE(src->attrs);
  this->qualityfields = RAVE_OBJECT_CLONE(src->qualityfields);
  if (this->datetime == NULL || this->projection == NULL ||
      this->navigator == NULL || this->parameters == NULL ||
      this->attrs == NULL || this->qualityfields == NULL) {
    goto error;
  }
  if (!PolarScan_setSource(this, PolarScan_getSource(src))) {
    goto error;
  }
  if (!PolarScan_setDefaultParameter(this, PolarScan_getDefaultParameter(src))) {
    goto error;
  }
  if (this->paramname != NULL && src->param != NULL) {
    this->param = (PolarScanParam_t*)RaveObjectHashTable_get(this->parameters, this->paramname);
    if (this->param == NULL) {
      goto error;
    }
  }
  return 1;
error:
  RAVE_FREE(this->source);
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_OBJECT_RELEASE(this->projection);
  RAVE_OBJECT_RELEASE(this->navigator);
  RAVE_OBJECT_RELEASE(this->parameters);
  RAVE_FREE(this->paramname);
  RAVE_OBJECT_RELEASE(this->param);
  RAVE_OBJECT_RELEASE(this->attrs);
  RAVE_OBJECT_RELEASE(this->qualityfields);
  return 0;
}

/**
 * Destructor.
 */
static void PolarScan_destructor(RaveCoreObject* obj)
{
  PolarScan_t* scan = (PolarScan_t*)obj;
  RAVE_FREE(scan->source);
  RAVE_OBJECT_RELEASE(scan->datetime);
  RAVE_OBJECT_RELEASE(scan->navigator);
  RAVE_OBJECT_RELEASE(scan->projection);
  RAVE_OBJECT_RELEASE(scan->parameters);
  RAVE_OBJECT_RELEASE(scan->param);
  RAVE_FREE(scan->paramname);
  RAVE_OBJECT_RELEASE(scan->attrs);
  RAVE_OBJECT_RELEASE(scan->qualityfields);
}

/*@} End of Private functions */

/*@{ Interface functions */
void PolarScan_setNavigator(PolarScan_t* scan, PolarNavigator_t* navigator)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  RAVE_ASSERT((navigator != NULL), "navigator was NULL");
  RAVE_OBJECT_RELEASE(scan->navigator);
  scan->navigator = RAVE_OBJECT_COPY(navigator);
}

PolarNavigator_t* PolarScan_getNavigator(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return RAVE_OBJECT_COPY(scan->navigator);
}

void PolarScan_setProjection(PolarScan_t* scan, Projection_t* projection)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  RAVE_OBJECT_RELEASE(scan->projection);
  scan->projection = RAVE_OBJECT_COPY(projection);
}

Projection_t* PolarScan_getProjection(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return RAVE_OBJECT_COPY(scan->projection);
}

int PolarScan_setTime(PolarScan_t* scan, const char* value)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return RaveDateTime_setTime(scan->datetime, value);
}

const char* PolarScan_getTime(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return RaveDateTime_getTime(scan->datetime);
}

int PolarScan_setDate(PolarScan_t* scan, const char* value)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return RaveDateTime_setDate(scan->datetime, value);
}

const char* PolarScan_getDate(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return RaveDateTime_getDate(scan->datetime);
}

int PolarScan_setSource(PolarScan_t* scan, const char* value)
{
  char* tmp = NULL;
  int result = 0;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  if (value != NULL) {
    tmp = RAVE_STRDUP(value);
    if (tmp != NULL) {
      RAVE_FREE(scan->source);
      scan->source = tmp;
      tmp = NULL;
      result = 1;
    }
  } else {
    RAVE_FREE(scan->source);
    result = 1;
  }
  return result;
}

const char* PolarScan_getSource(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return (const char*)scan->source;
}


void PolarScan_setLongitude(PolarScan_t* scan, double lon)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  PolarNavigator_setLon0(scan->navigator, lon);
}

double PolarScan_getLongitude(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return PolarNavigator_getLon0(scan->navigator);
}

void PolarScan_setLatitude(PolarScan_t* scan, double lat)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  PolarNavigator_setLat0(scan->navigator, lat);
}

double PolarScan_getLatitude(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return PolarNavigator_getLat0(scan->navigator);
}

void PolarScan_setHeight(PolarScan_t* scan, double height)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  PolarNavigator_setAlt0(scan->navigator, height);
}

double PolarScan_getHeight(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return PolarNavigator_getAlt0(scan->navigator);
}

double PolarScan_getDistance(PolarScan_t* scan, double lon, double lat)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return PolarNavigator_getDistance(scan->navigator, lat, lon);
}

void PolarScan_setElangle(PolarScan_t* scan, double elangle)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  scan->elangle = elangle;
}

double PolarScan_getElangle(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return scan->elangle;
}

long PolarScan_getNbins(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return scan->nbins;
}

void PolarScan_setRscale(PolarScan_t* scan, double rscale)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  scan->rscale = rscale;
}

double PolarScan_getRscale(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return scan->rscale;
}

long PolarScan_getNrays(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return scan->nrays;
}

void PolarScan_setRstart(PolarScan_t* scan, double rstart)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  scan->rstart = rstart;
}

double PolarScan_getRstart(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return scan->rstart;
}

RaveDataType PolarScan_getDataType(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  if (scan->param == NULL) {
    return RaveDataType_UNDEFINED;
  }
  return PolarScanParam_getDataType(scan->param);
}

void PolarScan_setA1gate(PolarScan_t* scan, long a1gate)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  scan->a1gate = a1gate;
}

long PolarScan_getA1gate(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return scan->a1gate;
}

void PolarScan_setBeamWidth(PolarScan_t* scan, long beamwidth)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  scan->beamwidth = beamwidth;
}

double PolarScan_getBeamWidth(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return scan->beamwidth;
}

int PolarScan_setDefaultParameter(PolarScan_t* scan, const char* quantity)
{
  int result = 0;
  char* tmp = NULL;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  if (quantity == NULL) {
    return 0;
  }
  tmp = RAVE_STRDUP(quantity);
  if (tmp != NULL) {
    RAVE_FREE(scan->paramname);
    scan->paramname = tmp;
    RAVE_OBJECT_RELEASE(scan->param);
    scan->param = (PolarScanParam_t*)RaveObjectHashTable_get(scan->parameters, quantity);
    result = 1;
  }
  return result;
}

const char* PolarScan_getDefaultParameter(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return (const char*)scan->paramname;
}

int PolarScan_addParameter(PolarScan_t* scan, PolarScanParam_t* parameter)
{
  const char* quantity;
  int result = 0;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  if (parameter == NULL) {
    RAVE_WARNING0("Passing in NULL as parameter");
    return 0;
  }
  quantity = PolarScanParam_getQuantity(parameter);
  if (quantity == NULL) {
    RAVE_WARNING0("No quantity in parameter, can not handle");
    return 0;
  }
  if (RaveObjectHashTable_size(scan->parameters)<=0) {
    scan->nrays = PolarScanParam_getNrays(parameter);
    scan->nbins = PolarScanParam_getNbins(parameter);
  } else {
    if (scan->nrays != PolarScanParam_getNrays(parameter) ||
        scan->nbins != PolarScanParam_getNbins(parameter)) {
      RAVE_WARNING0("Different number of rays/bins for various parameters are not allowed");
      return 0;
    }
  }
  result = RaveObjectHashTable_put(scan->parameters, quantity, (RaveCoreObject*)parameter);

  if (result == 1 && strcmp(quantity, scan->paramname)==0) {
    RAVE_OBJECT_RELEASE(scan->param);
    scan->param = RAVE_OBJECT_COPY(parameter);
  }

  return result;
}

PolarScanParam_t* PolarScan_removeParameter(PolarScan_t* scan, const char* quantity)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return (PolarScanParam_t*)RaveObjectHashTable_remove(scan->parameters, quantity);
}

PolarScanParam_t* PolarScan_getParameter(PolarScan_t* scan, const char* quantity)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return (PolarScanParam_t*)RaveObjectHashTable_get(scan->parameters, quantity);
}

int PolarScan_hasParameter(PolarScan_t* scan, const char* quantity)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return RaveObjectHashTable_exists(scan->parameters, quantity);
}

RaveList_t* PolarScan_getParameterNames(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return RaveObjectHashTable_keys(scan->parameters);
}

int PolarScan_addQualityField(PolarScan_t* scan, RaveField_t* field)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return RaveObjectList_add(scan->qualityfields, (RaveCoreObject*)field);
}

RaveField_t* PolarScan_getQualityField(PolarScan_t* scan, int index)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return (RaveField_t*)RaveObjectList_get(scan->qualityfields, index);
}

int PolarScan_getNumberOfQualityFields(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return RaveObjectList_size(scan->qualityfields);
}

void PolarScan_removeQualityField(PolarScan_t* scan, int index)
{
  RaveField_t* field = NULL;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  field = (RaveField_t*)RaveObjectList_remove(scan->qualityfields, index);
  RAVE_OBJECT_RELEASE(field);
}

int PolarScan_getRangeIndex(PolarScan_t* scan, double r)
{
  int result = -1;
  double range = 0.0L;

  RAVE_ASSERT((scan != NULL), "scan was NULL");

  if (scan->nbins <= 0 || scan->rscale <= 0.0) {
    RAVE_WARNING0("Can not calculate range index");
    return -1;
  }

  range = r - scan->rstart*1000.0;

  if (range >= 0.0) {
    result = (int)floor(range/scan->rscale);
  }

  if (result >= scan->nbins || result < 0) {
    result = -1;
  }
  return result;
}

int PolarScan_getAzimuthIndex(PolarScan_t* scan, double a)
{
  int result = -1;
  double azOffset = 0.0L;
  RAVE_ASSERT((scan != NULL), "scan was NULL");

  if (scan->nrays <= 0) {
    RAVE_WARNING0("Can not calculate azimuth index");
    return -1;
  }

  azOffset = 2*M_PI/scan->nrays;
  result = (int)rint(a/azOffset);
  if (result >= scan->nrays) {
    result -= scan->nrays;
  } else if (result < 0) {
    result += scan->nrays;
  }
  return result;
}

int PolarScan_setValue(PolarScan_t* scan, int bin, int ray, double v)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  if (scan->param == NULL) {
    return 0;
  }
  return PolarScanParam_setValue(scan->param, bin, ray, v);
}

int PolarScan_setParameterValue(PolarScan_t* scan, const char* quantity, int bin, int ray, double v)
{
  PolarScanParam_t* param = NULL;
  int result = 0;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  RAVE_ASSERT((quantity != NULL), "quantity == NULL");
  param = (PolarScanParam_t*)RaveObjectHashTable_get(scan->parameters, quantity);
  if (param != NULL) {
    result = PolarScanParam_setValue(param, bin, ray, v);
  }
  RAVE_OBJECT_RELEASE(param);
  return result;
}

RaveValueType PolarScan_getValue(PolarScan_t* scan, int bin, int ray, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  double value = 0.0;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  if (scan->param == NULL) {
    return RaveValueType_UNDEFINED;
  }
  value = PolarScanParam_getNodata(scan->param);
  result = PolarScanParam_getValue(scan->param, bin, ray, &value);
  if (v != NULL) {
    *v = value;
  }

  return result;
}

RaveValueType PolarScan_getParameterValue(PolarScan_t* scan, const char* quantity, int bin, int ray, double* v)
{
  PolarScanParam_t* param = NULL;
  RaveValueType result = 0;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  RAVE_ASSERT((quantity != NULL), "quantity == NULL");
  param = (PolarScanParam_t*)RaveObjectHashTable_get(scan->parameters, quantity);
  if (param != NULL) {
    result = PolarScanParam_getValue(param, bin, ray, v);
  }
  RAVE_OBJECT_RELEASE(param);
  return result;
}

RaveValueType PolarScan_getConvertedValue(PolarScan_t* scan, int bin, int ray, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  if (scan->param == NULL) {
    return RaveValueType_UNDEFINED;
  }

  if (v != NULL) {
    result =  PolarScan_getValue(scan, bin, ray, v);
    if (result == RaveValueType_DATA) {
      *v = PolarScanParam_getOffset(scan->param) + (*v) * PolarScanParam_getGain(scan->param);
    }
  }
  return result;
}

RaveValueType PolarScan_getConvertedParameterValue(PolarScan_t* scan, const char* quantity, int bin, int ray, double* v)
{
  PolarScanParam_t* param = NULL;
  RaveValueType result = RaveValueType_UNDEFINED;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  RAVE_ASSERT((quantity != NULL), "quantity == NULL");

  param = (PolarScanParam_t*)RaveObjectHashTable_get(scan->parameters, quantity);
  if (param != NULL) {
    result = PolarScanParam_getConvertedValue(param, bin, ray, v);
  }
  RAVE_OBJECT_RELEASE(param);
  return result;
}

int PolarScan_getIndexFromAzimuthAndRange(PolarScan_t* scan, double a, double r, int* ray, int* bin)
{
  int result = 0;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  RAVE_ASSERT((bin != NULL), "bin == NULL");
  RAVE_ASSERT((ray != NULL), "ray == NULL");
  *ray = PolarScan_getAzimuthIndex(scan, a);
  if (*ray < 0) {
    goto done;
  }
  *bin = PolarScan_getRangeIndex(scan, r);
  if (*bin < 0) {
    goto done;
  }
  result = 1;
done:
  return result;
}

int PolarScan_getAzimuthAndRangeFromIndex(PolarScan_t* scan, int bin, int ray, double* a, double* r)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  RAVE_ASSERT((a != NULL), "a == NULL");
  RAVE_ASSERT((r != NULL), "r == NULL");
  *r = bin * scan->rscale;
  *a = (2*M_PI/scan->nrays)*ray;
  return 1;
}

RaveValueType PolarScan_getValueAtAzimuthAndRange(PolarScan_t* scan, double a, double r, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  int ai = 0, ri = 0;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  RAVE_ASSERT((v != NULL), "v == NULL");
  if (scan->param == NULL) {
    return RaveValueType_UNDEFINED;
  }
  if (v != NULL) {
    *v = PolarScanParam_getNodata(scan->param);
  }
  if (!PolarScan_getIndexFromAzimuthAndRange(scan, a, r, &ai, &ri)) {
    goto done;
  }

  result = PolarScan_getValue(scan, ri, ai, v);
done:
  return result;
}

RaveValueType PolarScan_getParameterValueAtAzimuthAndRange(PolarScan_t* scan, const char* quantity, double a, double r, double* v)
{
  RaveValueType result = RaveValueType_UNDEFINED;
  int ai = 0, ri = 0;
  PolarScanParam_t* param = NULL;

  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_ASSERT((v != NULL), "v was NULL");
  param = PolarScan_getParameter(scan, quantity);
  if (param != NULL) {
    result = RaveValueType_NODATA;
    *v = PolarScanParam_getNodata(param);
    if (!PolarScan_getIndexFromAzimuthAndRange(scan, a, r, &ai, &ri)) {
      goto done;
    }
    result = PolarScanParam_getValue(param, ri, ai, v);
  }
done:
  RAVE_OBJECT_RELEASE(param);
  return result;
}

RaveValueType PolarScan_getNearest(PolarScan_t* scan, double lon, double lat, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  double d = 0.0L, a = 0.0L, r = 0.0L, h = 0.0L;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_ASSERT((v != NULL), "v was NULL");
  if (scan->param == NULL) {
    return RaveValueType_UNDEFINED;
  }

  PolarNavigator_llToDa(scan->navigator, lat, lon, &d, &a);
  PolarNavigator_deToRh(scan->navigator, d, scan->elangle, &r, &h);

  result = PolarScan_getValueAtAzimuthAndRange(scan, a, r, v);

  return result;
}

RaveValueType PolarScan_getNearestParameterValue(PolarScan_t* scan, const char* quantity, double lon, double lat, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  double d = 0.0L, a = 0.0L, r = 0.0L, h = 0.0L;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_ASSERT((v != NULL), "v was NULL");
  if (scan->param == NULL) {
    return RaveValueType_UNDEFINED;
  }

  PolarNavigator_llToDa(scan->navigator, lat, lon, &d, &a);
  PolarNavigator_deToRh(scan->navigator, d, scan->elangle, &r, &h);

  result = PolarScan_getParameterValueAtAzimuthAndRange(scan, quantity, a, r, v);

  return result;
}

int PolarScan_getNearestIndex(PolarScan_t* scan, double lon, double lat, int* bin, int* ray)
{
  double d = 0.0L, a = 0.0L, r = 0.0L, h = 0.0L;
  int result = 0;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  RAVE_ASSERT((bin != NULL), "bin == NULL");
  RAVE_ASSERT((ray != NULL), "ray == NULL");

  PolarNavigator_llToDa(scan->navigator, lat, lon, &d, &a);
  PolarNavigator_deToRh(scan->navigator, d, scan->elangle, &r, &h);

  result = PolarScan_getIndexFromAzimuthAndRange(scan, a, r, ray, bin);

  return result;
}

int PolarScan_getLonLatFromIndex(PolarScan_t* scan, int bin, int ray, double* lon, double* lat)
{
  int result = 0;
  double d = 0.0, h = 0.0, a = 0.0, r = 0.0;

  if (!PolarScan_getAzimuthAndRangeFromIndex(scan, bin, ray, &a, &r)) {
    goto done;
  }
  PolarNavigator_reToDh(scan->navigator, r, scan->elangle, &d, &h);
  PolarNavigator_daToLl(scan->navigator, d, a, lat, lon);

  result = 1;
done:
  return result;
}

int PolarScan_isTransformable(PolarScan_t* scan)
{
  int result = 0;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  if (scan->projection != NULL &&
      scan->navigator != NULL &&
      scan->rscale > 0.0) {
    result = 1;
  }
  return result;
}

int PolarScan_addAttribute(PolarScan_t* scan, RaveAttribute_t* attribute)
{
  const char* name = NULL;
  char* aname = NULL;
  char* gname = NULL;
  int result = 0;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  RAVE_ASSERT((attribute != NULL), "attribute == NULL");

  name = RaveAttribute_getName(attribute);
  if (name != NULL) {
    /*
     * where/elangle
     * where/a1gate
     * where/rscale
     * where/rstart
     * what/startdate
     * what/starttime
     */
    if (strcasecmp("where/elangle", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract where/elangle as a double");
      }
      PolarScan_setElangle(scan, value * M_PI/180.0);
    } else if (strcasecmp("where/a1gate", name)==0) {
      long value = 0;
      if (!(result = RaveAttribute_getLong(attribute, &value))) {
        RAVE_ERROR0("Failed to extract where/a1gate as a long");
      }
      PolarScan_setA1gate(scan, value);
    } else if (strcasecmp("where/rscale", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract where/rscale as a double");
      }
      PolarScan_setRscale(scan, value);
    } else if (strcasecmp("where/rstart", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract where/rstart as a double");
      }
      PolarScan_setRstart(scan, value);
    } else if (strcasecmp("what/startdate", name)==0) {
      char* value = NULL;
      if (!RaveAttribute_getString(attribute, &value)) {
        RAVE_ERROR0("Failed to extract where/startdate as a string");
        goto done;
      }
      result = PolarScan_setDate(scan, value);
    } else if (strcasecmp("what/starttime", name)==0) {
      char* value = NULL;
      if (!RaveAttribute_getString(attribute, &value)) {
        RAVE_ERROR0("Failed to extract where/starttime as a string");
        goto done;
      }
      result = PolarScan_setTime(scan, value);
    } else {
      if (!RaveAttributeHelp_extractGroupAndName(name, &gname, &aname)) {
        RAVE_ERROR1("Failed to extract group and name from %s", name);
        goto done;
      }
      if ((strcasecmp("how", gname)==0 ||
           strcasecmp("what", gname)==0 ||
           strcasecmp("where", gname)==0) &&
        strchr(aname, '/') == NULL) {
        result = RaveObjectHashTable_put(scan->attrs, name, (RaveCoreObject*)attribute);
      }
    }
  }

done:
  RAVE_FREE(aname);
  RAVE_FREE(gname);
  return result;
}

RaveAttribute_t* PolarScan_getAttribute(PolarScan_t* scan, const char* name)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  if (name == NULL) {
    RAVE_ERROR0("Trying to get an attribute with NULL name");
    return NULL;
  }
  return (RaveAttribute_t*)RaveObjectHashTable_get(scan->attrs, name);
}

RaveList_t* PolarScan_getAttributeNames(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return RaveObjectHashTable_keys(scan->attrs);
}

RaveObjectList_t* PolarScan_getAttributeValues(PolarScan_t* scan, Rave_ObjectType otype, int rootattributes)
{
  RaveObjectList_t* result = NULL;
  RaveObjectList_t* tableattrs = NULL;

  RAVE_ASSERT((scan != NULL), "scan == NULL");
  tableattrs = RaveObjectHashTable_values(scan->attrs);
  if (tableattrs == NULL) {
    goto error;
  }
  result = RAVE_OBJECT_CLONE(tableattrs);
  if (result == NULL) {
    goto error;
  }

  if (otype == Rave_ObjectType_SCAN) {
    if (rootattributes == 1) {
      if (!RaveUtilities_replaceDoubleAttributeInList(result, "how/beamwidth", PolarScan_getBeamWidth(scan)) ||
          !RaveUtilities_replaceStringAttributeInList(result, "what/date", PolarScan_getDate(scan)) ||
          !RaveUtilities_replaceStringAttributeInList(result, "what/time", PolarScan_getTime(scan)) ||
          !RaveUtilities_replaceStringAttributeInList(result, "what/source", PolarScan_getSource(scan)) ||
          !RaveUtilities_replaceDoubleAttributeInList(result, "where/height", PolarScan_getHeight(scan)) ||
          !RaveUtilities_replaceDoubleAttributeInList(result, "where/lat", PolarScan_getLatitude(scan)*180.0/M_PI) ||
          !RaveUtilities_replaceDoubleAttributeInList(result, "where/lon", PolarScan_getLongitude(scan)*180.0/M_PI)) {
        goto error;
      }
      RaveUtilities_removeAttributeFromList(result, "what/enddate");
      RaveUtilities_removeAttributeFromList(result, "what/endtime");
    } else {
      if (!RaveUtilities_replaceStringAttributeInList(result, "what/product", "SCAN") ||
          !RaveUtilities_replaceLongAttributeInList(result, "where/a1gate", PolarScan_getA1gate(scan)) ||
          !RaveUtilities_replaceDoubleAttributeInList(result, "where/elangle", PolarScan_getElangle(scan)*180.0/M_PI) ||
          !RaveUtilities_replaceLongAttributeInList(result, "where/nbins", PolarScan_getNbins(scan)) ||
          !RaveUtilities_replaceLongAttributeInList(result, "where/nrays", PolarScan_getNrays(scan)) ||
          !RaveUtilities_replaceDoubleAttributeInList(result, "where/rscale", PolarScan_getRscale(scan)) ||
          !RaveUtilities_replaceDoubleAttributeInList(result, "where/rstart", PolarScan_getRstart(scan)) ||
          !RaveUtilities_replaceStringAttributeInList(result, "what/startdate", PolarScan_getDate(scan)) ||
          !RaveUtilities_replaceStringAttributeInList(result, "what/starttime", PolarScan_getTime(scan))) {
        goto error;
      }
    }
  } else {
    if (!RaveUtilities_replaceDoubleAttributeInList(result, "where/elangle", PolarScan_getElangle(scan)*180.0/M_PI) ||
        !RaveUtilities_replaceLongAttributeInList(result, "where/a1gate", PolarScan_getA1gate(scan)) ||
        !RaveUtilities_replaceDoubleAttributeInList(result, "where/rscale", PolarScan_getRscale(scan)) ||
        !RaveUtilities_replaceDoubleAttributeInList(result, "where/rstart", PolarScan_getRstart(scan))) {
      goto error;
    }
    if (!RaveObjectHashTable_exists(scan->attrs, "what/startdate") &&
        !RaveUtilities_addStringAttributeToList(result, "what/startdate", PolarScan_getDate(scan))) {
      goto error;
    }
    if (!RaveObjectHashTable_exists(scan->attrs, "what/starttime") &&
        !RaveUtilities_addStringAttributeToList(result, "what/starttime", PolarScan_getTime(scan))) {
      goto error;
    }

    if (!RaveObjectHashTable_exists(scan->attrs, "what/enddate") &&
        !RaveUtilities_addStringAttributeToList(result, "what/enddate", PolarScan_getDate(scan))) {
      goto error;
    }
    if (!RaveObjectHashTable_exists(scan->attrs, "what/endtime") &&
        !RaveUtilities_addStringAttributeToList(result, "what/endtime", PolarScan_getTime(scan))) {
      goto error;
    }
  }

  RAVE_OBJECT_RELEASE(tableattrs);
  return result;
error:
  RAVE_OBJECT_RELEASE(result);
  RAVE_OBJECT_RELEASE(tableattrs);
  return NULL;

}

int PolarScan_isValid(PolarScan_t* scan, Rave_ObjectType otype)
{
  int result = 1;
  RAVE_ASSERT((scan != NULL), "scan == NULL");

  if (otype == Rave_ObjectType_PVOL) {
    if (PolarScan_getTime(scan) == NULL ||
        PolarScan_getDate(scan) == NULL ||
        !RaveObjectHashTable_exists(scan->attrs, "what/enddate") ||
        !RaveObjectHashTable_exists(scan->attrs, "what/endtime")) {
      RAVE_INFO0("Missing start/end date/time information");
      goto done;
    }
    if (PolarScan_getNbins(scan) <= 0 ||
        PolarScan_getNrays(scan) <= 0) {
      RAVE_INFO0("Missing size information");
      goto done;
    }
    if (RaveObjectHashTable_size(scan->parameters) <= 0) {
      RAVE_INFO0("Must at least contain one parameter");
      goto done;
    }
  } else if (otype == Rave_ObjectType_SCAN) {
    if (PolarScan_getTime(scan) == NULL ||
        PolarScan_getDate(scan) == NULL ||
        PolarScan_getSource(scan) == NULL) {
      RAVE_INFO0("date, time and source must be specified");
      goto done;
    }
    if (PolarScan_getNbins(scan) <= 0 ||
        PolarScan_getNrays(scan) <= 0) {
      RAVE_INFO0("Missing size information");
      goto done;
    }
    if (RaveObjectHashTable_size(scan->parameters) <= 0) {
      RAVE_INFO0("Must at least contain one parameter");
      goto done;
    }
  } else {
    RAVE_ERROR0("Only valid types for isValid are PVOL and SCAN");
    goto done;
  }

  result = 1;
done:
  return result;
}

/*@} End of Interface functions */

RaveCoreObjectType PolarScan_TYPE = {
    "PolarScan",
    sizeof(PolarScan_t),
    PolarScan_constructor,
    PolarScan_destructor,
    PolarScan_copyconstructor
};
