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

/**
 * Represents one scan in a volume.
 */
struct _PolarScan_t {
  RAVE_OBJECT_HEAD /** Always on top */

  // Date/Time
  RaveDateTime_t* datetime;     /**< the date, time instance */

  char* source;    /**< the source string */

  // Where
  double elangle;    /**< elevation of scan */
  long nbins;        /**< number of bins */
  double rscale;     /**< scale */
  RaveDataType type; /**< data type */
  long nrays;        /**< number of rays / scan */
  double rstart;     /**< start of ray */
  long a1gate;       /**< something */

  // How
  double beamwidth;  /**< beam width */

  // What
  char quantity[64]; /**< what does this data represent */
  double gain;       /**< gain when scaling */
  double offset;     /**< offset when scaling */
  double nodata;     /**< nodata */
  double undetect;   /**< undetect */

  // Data
  void* data;        /**< data ptr */

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
  scan->nbins = 0;
  scan->rscale = 0.0;
  scan->type = RaveDataType_UNDEFINED;
  scan->nrays = 0;
  scan->rstart = 0.0;
  scan->a1gate = 0;
  scan->beamwidth = 0.0;
  strcpy(scan->quantity, "");
  scan->gain = 0.0;
  scan->offset = 0.0;
  scan->nodata = 0.0;
  scan->undetect = 0.0;
  scan->data = NULL;
  scan->navigator = NULL;
  scan->projection = NULL;

  scan->datetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  if (scan->datetime == NULL) {
    goto error;
  }

  scan->projection = RAVE_OBJECT_NEW(&Projection_TYPE);
  if (scan->projection != NULL) {
    if(!Projection_init(scan->projection, "lonlat", "lonlat", "+proj=latlong +ellps=WGS84 +datum=WGS84")) {
      goto error;
    }
  } else {
    goto error;
  }
  scan->navigator = RAVE_OBJECT_NEW(&PolarNavigator_TYPE);
  if (scan->navigator == NULL) {
    goto error;
  }

  return 1;
error:
  RAVE_OBJECT_RELEASE(scan->datetime);
  RAVE_OBJECT_RELEASE(scan->projection);
  RAVE_OBJECT_RELEASE(scan->navigator);
  return 0;
}

/**
 * Destructor.
 */
static void PolarScan_destructor(RaveCoreObject* obj)
{
  PolarScan_t* scan = (PolarScan_t*)obj;
  RAVE_OBJECT_RELEASE(scan->datetime);
  RAVE_FREE(scan->source);
  RAVE_FREE(scan->data);
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

void PolarScan_setNbins(PolarScan_t* scan, long nbins)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->nbins = nbins;
}

long PolarScan_getNbins(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->nbins;
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

void PolarScan_setNrays(PolarScan_t* scan, long nrays)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->nrays = nrays;
}

long PolarScan_getNrays(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->nrays;
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

int PolarScan_setDataType(PolarScan_t* scan, RaveDataType type)
{
  int result = 0;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  if (type >= RaveDataType_UNDEFINED && type < RaveDataType_LAST) {
    scan->type = type;
    result = 1;
  }
  return result;
}

RaveDataType PolarScan_getDataType(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->type;
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

void PolarScan_setQuantity(PolarScan_t* scan, const char* quantity)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  if (quantity != NULL) {
    strcpy(scan->quantity, quantity);
  }
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
  long sz = 0;
  long nbytes = 0;
  void* ptr = NULL;
  int result = 0;

  RAVE_ASSERT((scan != NULL), "scan was NULL");

  sz = get_ravetype_size(type);
  nbytes = nbins*nrays*sz;
  ptr = RAVE_MALLOC(nbytes);

  if (ptr == NULL) {
    RAVE_CRITICAL1("Failed to allocate memory (%d bytes)", (int)nbytes);
    goto fail;
  }
  memcpy(ptr, data, nbytes);
  RAVE_FREE(scan->data);
  scan->data = ptr;
  PolarScan_setNbins(scan, nbins);
  PolarScan_setNrays(scan, nrays);
  PolarScan_setDataType(scan, type);
  result = 1;
fail:
  return result;
}

void* PolarScan_getData(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->data;
}

int PolarScan_getRangeIndex(PolarScan_t* scan, double r)
{
  int result = -1;
  double range = 0.0L;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_ASSERT((scan->nbins > 0), "nbins must be > 0");
  RAVE_ASSERT((scan->rscale > 0.0), "rscale must be > 0.0");

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
  RAVE_ASSERT((scan->nrays > 0), "nrays must be > 0");

  azOffset = 2*M_PI/scan->nrays;
  result = (int)rint(a/azOffset);
  if (result >= scan->nrays) {
    result -= scan->nrays;
  } else if (result < 0) {
    result += scan->nrays;
  }
  return result;
}

RaveValueType PolarScan_getValueAtIndex(PolarScan_t* scan, int ray, int bin, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_ASSERT((v != NULL), "v was NULL");
  *v = scan->nodata;
  if (ray >= 0 && ray < scan->nrays && bin >= 0 && bin < scan->nbins) {
    result = RaveValueType_DATA;
    *v = get_array_item_2d(scan->data, bin, ray, scan->type, scan->nbins);
    if (*v == scan->nodata) {
      result = RaveValueType_NODATA;
    } else if (*v == scan->undetect) {
      result = RaveValueType_UNDETECT;
    }
  }
  return result;
}

RaveValueType PolarScan_getConvertedValueAtIndex(PolarScan_t* scan, int ray, int bin, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_ASSERT((v != NULL), "v was NULL");
  result =  PolarScan_getValueAtIndex(scan, ray, bin, v);
  if (result == RaveValueType_DATA) {
    *v = scan->offset + (*v) * scan->gain;
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

  result = PolarScan_getValueAtIndex(scan, ai, ri, v);
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
  if (scan->projection != NULL &&
      scan->navigator != NULL &&
      scan->nrays > 0 &&
      scan->nbins > 0 &&
      scan->rscale > 0.0 &&
      scan->data != NULL) {
    result = 1;
  }
  return result;
}
/*@} End of Interface functions */

RaveCoreObjectType PolarScan_TYPE = {
    "PolarScan",
    sizeof(PolarScan_t),
    PolarScan_constructor,
    PolarScan_destructor
};
