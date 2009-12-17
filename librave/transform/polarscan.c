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

/**
 * Represents one scan in a volume.
 */
struct _PolarScan_t {
  RAVE_OBJECT_HEAD /** Always on top */

  char* source;    /**< the source string */

  // Where
  double elangle;    /**< elevation of scan */
  double rscale;     /**< scale */
  double rstart;     /**< start of ray */
  long a1gate;       /**< something */

  // How
  double beamwidth;  /**< beam width */

  // What
  char* quantity;    /**< what does this data represent */
  double gain;       /**< gain when scaling */
  double offset;     /**< offset when scaling */
  double nodata;     /**< nodata */
  double undetect;   /**< undetect */

  // Data
  RaveData2D_t* data; /**< data ptr */

  // Date/Time
  RaveDateTime_t* datetime;     /**< the date, time instance */

  // Navigator
  PolarNavigator_t* navigator; /**< a navigator for calculating polar navigation */

  // Projection wrapper
  Projection_t* projection; /**< projection for this scan */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int PolarScan_constructor(RaveCoreObject* obj)
{
  PolarScan_t* scan = (PolarScan_t*)obj;
  scan->datetime = NULL;
  scan->source = NULL;
  scan->elangle = 0.0;
  scan->rscale = 0.0;
  scan->rstart = 0.0;
  scan->a1gate = 0;
  scan->beamwidth = 0.0;
  scan->quantity = NULL;
  scan->gain = 0.0;
  scan->offset = 0.0;
  scan->nodata = 0.0;
  scan->undetect = 0.0;
  scan->data = NULL;
  scan->navigator = NULL;
  scan->projection = NULL;

  scan->datetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);

  scan->projection = RAVE_OBJECT_NEW(&Projection_TYPE);
  if (scan->projection != NULL) {
    if(!Projection_init(scan->projection, "lonlat", "lonlat", "+proj=latlong +ellps=WGS84 +datum=WGS84")) {
      goto error;
    }
  }
  scan->navigator = RAVE_OBJECT_NEW(&PolarNavigator_TYPE);
  scan->data = RAVE_OBJECT_NEW(&RaveData2D_TYPE);

  if (scan->datetime == NULL || scan->projection == NULL ||
      scan->navigator == NULL || scan->data == NULL) {
    goto error;
  }
  return 1;
error:
  RAVE_OBJECT_RELEASE(scan->datetime);
  RAVE_OBJECT_RELEASE(scan->projection);
  RAVE_OBJECT_RELEASE(scan->navigator);
  RAVE_OBJECT_RELEASE(scan->data);
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
  this->gain = 0.0;
  this->offset = 0.0;
  this->nodata = 0.0;
  this->undetect = 0.0;
  this->data = NULL;
  this->navigator = NULL;
  this->projection = NULL;

  this->source = NULL;
  this->quantity = NULL;

  this->datetime = RAVE_OBJECT_CLONE(src->datetime);
  this->projection = RAVE_OBJECT_CLONE(src->projection);
  this->navigator = RAVE_OBJECT_CLONE(src->navigator);
  this->data = RAVE_OBJECT_CLONE(src->data);

  if (this->datetime == NULL || this->projection == NULL ||
      this->navigator == NULL || this->data == NULL) {
    goto error;
  }
  if (!PolarScan_setSource(this, PolarScan_getSource(src))) {
    goto error;
  }
  if (!PolarScan_setQuantity(this, PolarScan_getQuantity(src))) {
    goto error;
  }
  return 1;
error:
  RAVE_FREE(this->source);
  RAVE_FREE(this->quantity);
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_OBJECT_RELEASE(this->projection);
  RAVE_OBJECT_RELEASE(this->navigator);
  RAVE_OBJECT_RELEASE(this->data);
  return 0;
}

/**
 * Destructor.
 */
static void PolarScan_destructor(RaveCoreObject* obj)
{
  PolarScan_t* scan = (PolarScan_t*)obj;
  RAVE_FREE(scan->source);
  RAVE_FREE(scan->quantity);
  RAVE_OBJECT_RELEASE(scan->datetime);
  RAVE_OBJECT_RELEASE(scan->data);
  RAVE_OBJECT_RELEASE(scan->navigator);
  RAVE_OBJECT_RELEASE(scan->projection);
}

/*@} End of Private functions */

/*@{ Interface functions */
void PolarScan_setNavigator(PolarScan_t* scan, PolarNavigator_t* navigator)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_ASSERT((navigator != NULL), "navigator was NULL");
  RAVE_OBJECT_RELEASE(scan->navigator);
  scan->navigator = RAVE_OBJECT_COPY(navigator);
}

PolarNavigator_t* PolarScan_getNavigator(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return RAVE_OBJECT_COPY(scan->navigator);
}

void PolarScan_setProjection(PolarScan_t* scan, Projection_t* projection)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_OBJECT_RELEASE(scan->projection);
  scan->projection = RAVE_OBJECT_COPY(projection);
}

Projection_t* PolarScan_getProjection(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return RAVE_OBJECT_COPY(scan->projection);
}

int PolarScan_setTime(PolarScan_t* scan, const char* value)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return RaveDateTime_setTime(scan->datetime, value);
}

const char* PolarScan_getTime(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return RaveDateTime_getTime(scan->datetime);
}

int PolarScan_setDate(PolarScan_t* scan, const char* value)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return RaveDateTime_setDate(scan->datetime, value);
}

const char* PolarScan_getDate(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return RaveDateTime_getDate(scan->datetime);
}

int PolarScan_setSource(PolarScan_t* scan, const char* value)
{
  char* tmp = NULL;
  int result = 0;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
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
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return (const char*)scan->source;
}


void PolarScan_setLongitude(PolarScan_t* scan, double lon)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  PolarNavigator_setLon0(scan->navigator, lon);
}

double PolarScan_getLongitude(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return PolarNavigator_getLon0(scan->navigator);
}

void PolarScan_setLatitude(PolarScan_t* scan, double lat)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  PolarNavigator_setLat0(scan->navigator, lat);
}

double PolarScan_getLatitude(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return PolarNavigator_getLat0(scan->navigator);
}

void PolarScan_setHeight(PolarScan_t* scan, double height)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  PolarNavigator_setAlt0(scan->navigator, height);
}

double PolarScan_getHeight(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return PolarNavigator_getAlt0(scan->navigator);
}

void PolarScan_setElangle(PolarScan_t* scan, double elangle)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->elangle = elangle;
}

double PolarScan_getElangle(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->elangle;
}

long PolarScan_getNbins(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return RaveData2D_getXsize(scan->data);
}

void PolarScan_setRscale(PolarScan_t* scan, double rscale)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->rscale = rscale;
}

double PolarScan_getRscale(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->rscale;
}

long PolarScan_getNrays(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return RaveData2D_getYsize(scan->data);
}

void PolarScan_setRstart(PolarScan_t* scan, double rstart)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->rstart = rstart;
}

double PolarScan_getRstart(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->rstart;
}

RaveDataType PolarScan_getDataType(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return RaveData2D_getType(scan->data);
}

void PolarScan_setA1gate(PolarScan_t* scan, long a1gate)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->a1gate = a1gate;
}

long PolarScan_getA1gate(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->a1gate;
}

void PolarScan_setBeamWidth(PolarScan_t* scan, long beamwidth)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->beamwidth = beamwidth;
}

double PolarScan_getBeamWidth(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->beamwidth;
}

int PolarScan_setQuantity(PolarScan_t* scan, const char* quantity)
{
  int result = 0;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  if (quantity != NULL) {
    char* tmp = RAVE_STRDUP(quantity);
    if (tmp != NULL) {
      RAVE_FREE(scan->quantity);
      scan->quantity = tmp;
      result = 1;
    }
  } else {
    RAVE_FREE(scan->quantity);
    result = 1;
  }
  return result;
}

const char* PolarScan_getQuantity(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return (const char*)scan->quantity;
}

void PolarScan_setGain(PolarScan_t* scan, double gain)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->gain = gain;
}

double PolarScan_getGain(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->gain;
}

void PolarScan_setOffset(PolarScan_t* scan, double offset)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->offset = offset;
}

double PolarScan_getOffset(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->offset;
}

void PolarScan_setNodata(PolarScan_t* scan, double nodata)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->nodata = nodata;
}

double PolarScan_getNodata(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->nodata;
}

void PolarScan_setUndetect(PolarScan_t* scan, double undetect)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->undetect = undetect;
}

double PolarScan_getUndetect(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->undetect;
}

int PolarScan_setData(PolarScan_t* scan, long nbins, long nrays, void* data, RaveDataType type)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return RaveData2D_setData(scan->data, nbins, nrays, data, type);
}

int PolarScan_createData(PolarScan_t* scan, long nbins, long nrays, RaveDataType type)
{
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  return RaveData2D_createData(scan->data, nbins, nrays, type);
}


void* PolarScan_getData(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return RaveData2D_getData(scan->data);
}

int PolarScan_getRangeIndex(PolarScan_t* scan, double r)
{
  int result = -1;
  double range = 0.0L;
  long nbins = 0;

  RAVE_ASSERT((scan != NULL), "scan was NULL");

  nbins = RaveData2D_getXsize(scan->data);

  if (nbins <= 0 || scan->rscale <= 0.0) {
    RAVE_WARNING0("Can not calculate range index");
    return -1;
  }

  range = r - scan->rstart*1000.0;

  if (range >= 0.0) {
    result = (int)floor(range/scan->rscale);
  }

  if (result >= nbins || result < 0) {
    result = -1;
  }
  return result;
}

int PolarScan_getAzimuthIndex(PolarScan_t* scan, double a)
{
  int result = -1;
  double azOffset = 0.0L;
  long nrays = 0;
  RAVE_ASSERT((scan != NULL), "scan was NULL");

  nrays = RaveData2D_getYsize(scan->data);
  if (nrays <= 0) {
    RAVE_WARNING0("Can not calculate azimuth index");
    return -1;
  }

  azOffset = 2*M_PI/nrays;
  result = (int)rint(a/azOffset);
  if (result >= nrays) {
    result -= nrays;
  } else if (result < 0) {
    result += nrays;
  }
  return result;
}

RaveValueType PolarScan_getValue(PolarScan_t* scan, int bin, int ray, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  double value = 0.0;
  RAVE_ASSERT((scan != NULL), "scan == NULL");

  value = scan->nodata;

  if (RaveData2D_getValue(scan->data, bin, ray, &value)) {
    result = RaveValueType_DATA;
    if (value == scan->nodata) {
      result = RaveValueType_NODATA;
    } else if (value == scan->undetect) {
      result = RaveValueType_UNDETECT;
    }
  }

  if (v != NULL) {
    *v = value;
  }

  return result;
}

RaveValueType PolarScan_getConvertedValue(PolarScan_t* scan, int bin, int ray, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  if (v != NULL) {
    result =  PolarScan_getValue(scan, bin, ray, v);
    if (result == RaveValueType_DATA) {
      *v = scan->offset + (*v) * scan->gain;
    }
  }
  return result;
}

RaveValueType PolarScan_getValueAtAzimuthAndRange(PolarScan_t* scan, double a, double r, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  int ai = 0, ri = 0;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_ASSERT((v != NULL), "v was NULL");
  *v = scan->nodata;
  ai = PolarScan_getAzimuthIndex(scan, a);
  if (ai < 0) {
    goto done;
  }
  ri = PolarScan_getRangeIndex(scan, r);
  if (ri < 0) {
    goto done;
  }

  result = PolarScan_getValue(scan, ri, ai, v);
done:
  return result;
}

RaveValueType PolarScan_getNearest(PolarScan_t* scan, double lon, double lat, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  double d = 0.0L, a = 0.0L, r = 0.0L, h = 0.0L;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_ASSERT((v != NULL), "v was NULL");
  *v = scan->nodata;

  PolarNavigator_llToDa(scan->navigator, lat, lon, &d, &a);
  PolarNavigator_deToRh(scan->navigator, d, scan->elangle, &r, &h);

  result = PolarScan_getValueAtAzimuthAndRange(scan, a, r, v);

  return result;
}

int PolarScan_isTransformable(PolarScan_t* scan)
{
  int result = 0;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  if (RaveData2D_hasData(scan->data) &&
      scan->projection != NULL &&
      scan->navigator != NULL &&
      scan->rscale > 0.0) {
    result = 1;
  }
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
