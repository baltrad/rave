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
 * Defines the functions available when working with polar volumes
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-15
 */
#include "polarvolume.h"
#include "polarnav.h"
#include "raveobject_list.h"
#include "rave_datetime.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveobject_hashtable.h"
#include "rave_utilities.h"
#include <string.h>

/**
 * This is the default parameter value that should be used when working
 * with scans.
 */
#define DEFAULT_PARAMETER_NAME "DBZH"

/**
 * Represents a volume
 */
struct _PolarVolume_t {
  RAVE_OBJECT_HEAD /** Always on top */
  Projection_t* projection; /**< projection for this volume */
  PolarNavigator_t* navigator; /**< a polar navigator */
  RaveObjectList_t* scans;  /**< the list of scans */
  RaveDateTime_t* datetime; /**< the date / time */
  RaveObjectHashTable_t* attrs; /**< the attributes */
  char* source;             /**< the source string */
  char* paramname;          /**< the default parameter */
  double beamwidth;         /**< the beamwidth, default bw is 1.0 * M_PI/180.0 */
};

/*@{ Private functions */
/**
 * Constructor
 */
static int PolarVolume_constructor(RaveCoreObject* obj)
{
  PolarVolume_t* this = (PolarVolume_t*)obj;
  this->projection = NULL;
  this->navigator = NULL;
  this->scans = NULL;
  this->datetime = NULL;
  this->source = NULL;
  this->paramname = NULL;
  this->beamwidth = 1.0 * M_PI/180.0;
  this->attrs = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  this->datetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);

  // Always initialize to default projection for lon/lat calculations
  this->projection = RAVE_OBJECT_NEW(&Projection_TYPE);
  if (this->projection != NULL) {
    if(!Projection_init(this->projection, "lonlat", "lonlat", "+proj=latlong +ellps=WGS84 +datum=WGS84")) {
      goto error;
    }
  }
  this->navigator = RAVE_OBJECT_NEW(&PolarNavigator_TYPE);
  this->scans = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);

  if (this->datetime == NULL || this->projection == NULL ||
      this->scans == NULL || this->navigator == NULL || this->attrs == NULL) {
    goto error;
  }

  if (!PolarVolume_setDefaultParameter(this, DEFAULT_PARAMETER_NAME)) {
    goto error;
  }

  return 1;
error:
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_OBJECT_RELEASE(this->projection);
  RAVE_OBJECT_RELEASE(this->navigator);
  RAVE_OBJECT_RELEASE(this->scans);
  RAVE_OBJECT_RELEASE(this->attrs);
  RAVE_FREE(this->source);
  RAVE_FREE(this->paramname);
  return 0;
}

/**
 * Copy constructor
 */
static int PolarVolume_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  PolarVolume_t* this = (PolarVolume_t*)obj;
  PolarVolume_t* src = (PolarVolume_t*)srcobj;

  this->projection = RAVE_OBJECT_CLONE(src->projection);
  this->navigator = RAVE_OBJECT_CLONE(src->navigator);
  this->scans = RAVE_OBJECT_CLONE(src->scans); // the list only contains scans and they are cloneable
  this->datetime = RAVE_OBJECT_CLONE(src->datetime);
  this->attrs = RAVE_OBJECT_CLONE(src->attrs);
  this->source = NULL;
  this->paramname = NULL;
  this->beamwidth = src->beamwidth;

  if (this->datetime == NULL || this->projection == NULL ||
      this->scans == NULL || this->navigator == NULL || this->attrs == NULL) {
    goto error;
  }

  if (!PolarVolume_setSource(this, src->source)) {
    goto error;
  }
  if (!PolarVolume_setDefaultParameter(this, src->paramname)) {
    goto error;
  }

  return 1;
error:
  RAVE_FREE(this->source);
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_OBJECT_RELEASE(this->projection);
  RAVE_OBJECT_RELEASE(this->navigator);
  RAVE_OBJECT_RELEASE(this->scans);
  RAVE_OBJECT_RELEASE(this->attrs);
  RAVE_FREE(this->source);
  RAVE_FREE(this->paramname);
  return 0;
}

/**
 * Destructor
 */
static void PolarVolume_destructor(RaveCoreObject* obj)
{
  PolarVolume_t* volume = (PolarVolume_t*)obj;
  RAVE_OBJECT_RELEASE(volume->datetime);
  RAVE_OBJECT_RELEASE(volume->projection);
  RAVE_OBJECT_RELEASE(volume->navigator);
  RAVE_OBJECT_RELEASE(volume->scans);
  RAVE_OBJECT_RELEASE(volume->attrs);
  RAVE_FREE(volume->source);
  RAVE_FREE(volume->paramname);
}

/**
 * Returns the elevation angle for the specified scan index.
 * @param
 */
static double PolarVolumeInternal_getElangle(PolarVolume_t* pvol, int index)
{
  PolarScan_t* scan = NULL;
  double elangle = 0.0L;
  scan = (PolarScan_t*)RaveObjectList_get(pvol->scans, index);
  if (scan != NULL) {
    elangle = PolarScan_getElangle(scan);
  } else {
    RAVE_CRITICAL1("Could not fetch scan for index = %d\n", index);
  }

  RAVE_OBJECT_RELEASE(scan);
  return elangle;
}

/**
 * Used to sort the scans by elevation in ascending order
 * @param[in] a - scan a (will be casted to *(PolarScan_t**))
 * @param[in] b - scan b (will be casted to *(PolarScan_t**))
 * @return -1 if a.elangle < b.elangle, 1 if a.elangle > b.elangle and 0 otherwise
 */
static int ascendingElevationSort(const void* a, const void* b)
{
  PolarScan_t* scanA = *(PolarScan_t**)a;
  PolarScan_t* scanB = *(PolarScan_t**)b;
  double angleA = PolarScan_getElangle(scanA);
  double angleB = PolarScan_getElangle(scanB);
  if (angleA < angleB) {
    return -1;
  } else if (angleA > angleB) {
    return 1;
  }
  return 0;
}

/**
 * Used to sort the scans by elevation in descending order
 * @param[in] a - scan a (will be casted to *(PolarScan_t**))
 * @param[in] b - scan b (will be casted to *(PolarScan_t**))
 * @return -1 if a.elangle > b.elangle, 1 if a.elangle < b.elangle and 0 otherwise
 */
static int descendingElevationSort(const void* a, const void* b)
{
  PolarScan_t* scanA = *(PolarScan_t**)a;
  PolarScan_t* scanB = *(PolarScan_t**)b;
  double angleA = PolarScan_getElangle(scanA);
  double angleB = PolarScan_getElangle(scanB);
  if (angleA > angleB) {
    return -1;
  } else if (angleA < angleB) {
    return 1;
  }
  return 0;
}

/*@} End of Private functions */

/*@{ Interface functions */
int PolarVolume_setTime(PolarVolume_t* pvol, const char* value)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return RaveDateTime_setTime(pvol->datetime, value);
}

const char* PolarVolume_getTime(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return RaveDateTime_getTime(pvol->datetime);
}

int PolarVolume_setDate(PolarVolume_t* pvol, const char* value)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return RaveDateTime_setDate(pvol->datetime, value);
}

const char* PolarVolume_getDate(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return RaveDateTime_getDate(pvol->datetime);
}

int PolarVolume_setSource(PolarVolume_t* pvol, const char* value)
{
  char* tmp = NULL;
  int result = 0;
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  if (value != NULL) {
    tmp = RAVE_STRDUP(value);
    if (tmp != NULL) {
      RAVE_FREE(pvol->source);
      pvol->source = tmp;
      tmp = NULL;
      result = 1;
    }
  } else {
    RAVE_FREE(pvol->source);
    result = 1;
  }
  return result;
}

const char* PolarVolume_getSource(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return (const char*)pvol->source;
}

void PolarVolume_setLongitude(PolarVolume_t* pvol, double lon)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  PolarNavigator_setLon0(pvol->navigator, lon);
}

double PolarVolume_getLongitude(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return PolarNavigator_getLon0(pvol->navigator);
}

void PolarVolume_setLatitude(PolarVolume_t* pvol, double lat)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  PolarNavigator_setLat0(pvol->navigator, lat);
}

double PolarVolume_getLatitude(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return PolarNavigator_getLat0(pvol->navigator);
}

void PolarVolume_setHeight(PolarVolume_t* pvol, double height)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  PolarNavigator_setAlt0(pvol->navigator, height);
}

double PolarVolume_getHeight(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return PolarNavigator_getAlt0(pvol->navigator);
}

void PolarVolume_setBeamwidth(PolarVolume_t* pvol, double bw)
{
  int i = 0, nlen = 0;
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  pvol->beamwidth = bw;

  nlen = RaveObjectList_size(pvol->scans);
  for (i = 0; i < nlen; i++) {
    PolarScan_t* scan = (PolarScan_t*)RaveObjectList_get(pvol->scans, i);
    PolarScanInternal_setPolarVolumeBeamwidth(scan, bw);
    RAVE_OBJECT_RELEASE(scan);
  }
}

double PolarVolume_getBeamwidth(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return pvol->beamwidth;
}

double PolarVolume_getDistance(PolarVolume_t* pvol, double lon, double lat)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return PolarNavigator_getDistance(pvol->navigator, lat, lon);
}

double PolarVolume_getMaxDistance(PolarVolume_t* pvol)
{
  int nrscans = 0;
  int i = 0;
  double maxdistance = 0.0;
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  nrscans = PolarVolume_getNumberOfScans(pvol);
  for (i = 0; i < nrscans; i++) {
    PolarScan_t* scan = PolarVolume_getScan(pvol, i);
    double dist = PolarScan_getMaxDistance(scan);
    if (dist > maxdistance) {
      maxdistance = dist;
    }
    RAVE_OBJECT_RELEASE(scan);
  }
  return maxdistance;
}

PolarScan_t* PolarVolume_getScanWithMaxDistance(PolarVolume_t* pvol)
{
  int nrscans = 0;
  int i = 0;
  double maxdistance = 0.0;
  PolarScan_t* result = NULL;

  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  nrscans = PolarVolume_getNumberOfScans(pvol);
  for (i = 0; i < nrscans; i++) {
    PolarScan_t* scan = PolarVolume_getScan(pvol, i);
    double dist = PolarScan_getMaxDistance(scan);
    if (dist > maxdistance) {
      maxdistance = dist;
      RAVE_OBJECT_RELEASE(result);
      result = RAVE_OBJECT_COPY(scan);
    }
    RAVE_OBJECT_RELEASE(scan);
  }
  return result;

}

void PolarVolume_setProjection(PolarVolume_t* pvol, Projection_t* projection)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  RAVE_OBJECT_RELEASE(pvol->projection);
  pvol->projection = NULL;
  if (projection != NULL) {
    int index = 0;
    int nrScans = RaveObjectList_size(pvol->scans);
    pvol->projection = RAVE_OBJECT_COPY(projection);
    for (index = 0; index < nrScans; index++) {
      PolarScan_t* scan = (PolarScan_t*)RaveObjectList_get(pvol->scans, index);
      PolarScan_setProjection(scan, projection);
      RAVE_OBJECT_RELEASE(scan);
    }
  }
}

Projection_t* PolarVolume_getProjection(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  if (pvol->projection != NULL) {
    return RAVE_OBJECT_COPY(pvol->projection);
  }
  return NULL;
}

int PolarVolume_addScan(PolarVolume_t* pvol, PolarScan_t* scan)
{
  int result = 0;
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  if (RaveObjectList_add(pvol->scans, (RaveCoreObject*)scan)) {
    PolarScan_setNavigator(scan, pvol->navigator);
    PolarScan_setProjection(scan, pvol->projection);
    PolarScan_setDefaultParameter(scan, pvol->paramname);
    if (PolarScan_getSource(scan) == NULL) {
      if (!PolarScan_setSource(scan, PolarVolume_getSource(pvol))) {
        goto done;
      }
    }
    if (PolarScanInternal_isPolarVolumeBeamwidth(scan) == -1) { /* if default beamwidth */
      PolarScanInternal_setPolarVolumeBeamwidth(scan, pvol->beamwidth);
    }

    if (PolarScan_getTime(scan) == NULL || PolarScan_getDate(scan) == NULL) {
      if (!PolarScan_setTime(scan, PolarVolume_getTime(pvol)) ||
          !PolarScan_setDate(scan, PolarVolume_getDate(pvol))) {
        goto done;
      }
    }

    result = 1;
  }
done:
  return result;
}

PolarScan_t* PolarVolume_getScan(PolarVolume_t* pvol, int index)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return (PolarScan_t*)RaveObjectList_get(pvol->scans, index);
}

int PolarVolume_getNumberOfScans(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return RaveObjectList_size(pvol->scans);
}

PolarScan_t* PolarVolume_getScanClosestToElevation(PolarVolume_t* pvol, double e, int inside)
{
  double se = 0.0L, eld = 0.0L;
  int ei = 0;
  int i = 0;
  int nrScans = 0;
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");

  nrScans = RaveObjectList_size(pvol->scans);

  if (inside) {
    if ((e < PolarVolumeInternal_getElangle(pvol, 0)) ||
        (e > PolarVolumeInternal_getElangle(pvol, nrScans-1))) {
      return NULL;
    }
  }

  se = PolarVolumeInternal_getElangle(pvol, 0);
  ei = 0;
  eld = fabs(e - se);
  for (i = 1; i < nrScans; i++) {
    double elev = PolarVolumeInternal_getElangle(pvol, i);
    double elevd = fabs(e - elev);
    if (elevd < eld) {
      se = elev;
      eld = elevd;
      ei = i;
    } else {
      break;
    }
  }
  return (PolarScan_t*)RaveObjectList_get(pvol->scans, ei);
}

RaveValueType PolarVolume_getNearest(PolarVolume_t* pvol, double lon, double lat, double height, int insidee, double* v)
{
  double d = 0.0L, a = 0.0L, r = 0.0L, e = 0.0L;
  RaveValueType result = RaveValueType_NODATA;
  PolarScan_t* scan = NULL;
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  RAVE_ASSERT((v != NULL), "v == NULL");
  *v = 0.0;

  PolarNavigator_llToDa(pvol->navigator, lat, lon, &d, &a);
  PolarNavigator_dhToRe(pvol->navigator, d, height, &r, &e);

  scan = PolarVolume_getScanClosestToElevation(pvol, e, insidee);
  if (scan != NULL) {
    //@todo: Eventually use the actual elevation and calculate proper range instead.
    // Now we have the elevation angle, fetch value by providing azimuth and range.
    result = PolarScan_getValueAtAzimuthAndRange(scan, a, r, v);
  }

  RAVE_OBJECT_RELEASE(scan);

  return result;
}

RaveValueType PolarVolume_getNearestParameterValue(PolarVolume_t* pvol, const char* quantity, double lon, double lat, double height, int insidee, double* v)
{
  double d = 0.0L, a = 0.0L, r = 0.0L, e = 0.0L;
  RaveValueType result = RaveValueType_NODATA;
  PolarScan_t* scan = NULL;

  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  RAVE_ASSERT((quantity != NULL), "quantity == NULL");
  RAVE_ASSERT((v != NULL), "v == NULL");
  *v = 0.0;

  PolarNavigator_llToDa(pvol->navigator, lat, lon, &d, &a);
  PolarNavigator_dhToRe(pvol->navigator, d, height, &r, &e);

  // Find relevant elevation
  scan = PolarVolume_getScanClosestToElevation(pvol, e, insidee);
  if (scan != NULL) {
    result = PolarScan_getParameterValueAtAzimuthAndRange(scan, quantity, a, r, v);
  }

  RAVE_OBJECT_RELEASE(scan);

  return result;
}

RaveValueType PolarVolume_getNearestConvertedParameterValue(PolarVolume_t* pvol, const char* quantity, double lon, double lat, double height, int insidee, double* v)
{
  double d = 0.0L, a = 0.0L, r = 0.0L, e = 0.0L;
  RaveValueType result = RaveValueType_NODATA;
  PolarScan_t* scan = NULL;

  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  RAVE_ASSERT((quantity != NULL), "quantity == NULL");
  RAVE_ASSERT((v != NULL), "v == NULL");
  *v = 0.0;

  PolarNavigator_llToDa(pvol->navigator, lat, lon, &d, &a);
  PolarNavigator_dhToRe(pvol->navigator, d, height, &r, &e);

  // Find relevant elevation
  scan = PolarVolume_getScanClosestToElevation(pvol, e, insidee);
  if (scan != NULL) {
    result = PolarScan_getConvertedParameterValueAtAzimuthAndRange(scan, quantity, a, r, v);
  }

  RAVE_OBJECT_RELEASE(scan);

  return result;
}

int PolarVolume_setDefaultParameter(PolarVolume_t* pvol, const char* quantity)
{
  int result = 0;
  char* tmp = NULL;
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  RAVE_ASSERT((quantity != NULL), "quantity == NULL");
  tmp = RAVE_STRDUP(quantity);
  if (tmp != NULL) {
    int i = 0;
    int nlen = RaveObjectList_size(pvol->scans);
    result = 1; /* Asume everything is ok and let the scans default parameter decide the result */
    RAVE_FREE(pvol->paramname);
    pvol->paramname = tmp;
    for (i = 0; result == 1 && i < nlen; i++) {
      PolarScan_t* scan = (PolarScan_t*)RaveObjectList_get(pvol->scans, i);
      if (scan != NULL) {
        result = PolarScan_setDefaultParameter(scan, quantity);
      }
      RAVE_OBJECT_RELEASE(scan);
    }
  }
  return result;
}

const char* PolarVolume_getDefaultParameter(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return (const char*)pvol->paramname;
}

void PolarVolume_sortByElevations(PolarVolume_t* pvol, int ascending)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");

  if (ascending == 1) {
    RaveObjectList_sort(pvol->scans, ascendingElevationSort);
  } else {
    RaveObjectList_sort(pvol->scans, descendingElevationSort);
  }
}

int PolarVolume_isAscendingScans(PolarVolume_t* pvol)
{
  int result = 1;
  int index = 0;
  double lastelev = 0.0L;
  int nrScans = 0;
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  nrScans = RaveObjectList_size(pvol->scans);
  if (nrScans > 0) {
    lastelev = PolarVolumeInternal_getElangle(pvol, 0);
    for (index = 1; result == 1 && index < nrScans; index++) {
      double nextelev = PolarVolumeInternal_getElangle(pvol, index);
      if (nextelev < lastelev) {
        result = 0;
      }
      lastelev = nextelev;
    }
  }
  return result;
}

int PolarVolume_isTransformable(PolarVolume_t* pvol)
{
  int result = 0;
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  // Verify that the volume at least got one scan and that the scans
  // are sorted in ascending order.
  if (RaveObjectList_size(pvol->scans) > 0 && PolarVolume_isAscendingScans(pvol)) {
    result = 1;
  }
  return result;
}

int PolarVolume_addAttribute(PolarVolume_t* pvol,
  RaveAttribute_t* attribute)
{
  const char* name = NULL;
  char* aname = NULL;
  char* gname = NULL;
  int result = 0;
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  RAVE_ASSERT((attribute != NULL), "attribute == NULL");

  name = RaveAttribute_getName(attribute);
  if (name != NULL) {
    if (!RaveAttributeHelp_extractGroupAndName(name, &gname, &aname)) {
      RAVE_ERROR1("Failed to extract group and name from %s", name);
      goto done;
    }
    if ((strcasecmp("how", gname)==0 ||
         strcasecmp("what", gname)==0 ||
         strcasecmp("where", gname)==0) &&
         strchr(aname, '/') == NULL) {
      result = RaveObjectHashTable_put(pvol->attrs, name, (RaveCoreObject*)attribute);
    }
  }

done:
  RAVE_FREE(aname);
  RAVE_FREE(gname);
  return result;
}

RaveAttribute_t* PolarVolume_getAttribute(PolarVolume_t* pvol,
  const char* name)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  if (name == NULL) {
    RAVE_ERROR0("Trying to get an attribute with NULL name");
    return NULL;
  }
  return (RaveAttribute_t*)RaveObjectHashTable_get(pvol->attrs, name);
}

RaveList_t* PolarVolume_getAttributeNames(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return RaveObjectHashTable_keys(pvol->attrs);
}

int PolarVolume_hasAttribute(PolarVolume_t* pvol, const char* name)
{
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  return RaveObjectHashTable_exists(pvol->attrs, name);
}

RaveObjectList_t* PolarVolume_getAttributeValues(PolarVolume_t* pvol)
{
  RaveObjectList_t* result = NULL;
  RaveObjectList_t* tableattrs = NULL;

  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  tableattrs = RaveObjectHashTable_values(pvol->attrs);
  if (tableattrs == NULL) {
    goto error;
  }
  result = RAVE_OBJECT_CLONE(tableattrs);
  if (result == NULL) {
    goto error;
  }

  RAVE_OBJECT_RELEASE(tableattrs);
  return result;
error:
  RAVE_OBJECT_RELEASE(result);
  RAVE_OBJECT_RELEASE(tableattrs);
  return NULL;
}

int PolarVolume_isValid(PolarVolume_t* pvol)
{
  int result = 0;
  int nscans = 0;
  int i = 0;
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");

  if (PolarVolume_getDate(pvol) == NULL ||
      PolarVolume_getTime(pvol) == NULL ||
      PolarVolume_getSource(pvol) == NULL) {
    RAVE_INFO0("date, time and source must be specified");
    goto done;
  }

  if ((nscans = RaveObjectList_size(pvol->scans)) <= 0) {
    RAVE_INFO0("Must have at least one scan");
    goto done;
  }

  result = 1;

  for (i = 0; result == 1 && i < nscans; i++) {
    PolarScan_t* scan = PolarVolume_getScan(pvol, i);
    result = PolarScan_isValid(scan, Rave_ObjectType_PVOL);
    RAVE_OBJECT_RELEASE(scan);
  }

done:
  return result;
}

/*@} End of Interface functions */
RaveCoreObjectType PolarVolume_TYPE = {
    "PolarVolume",
    sizeof(PolarVolume_t),
    PolarVolume_constructor,
    PolarVolume_destructor,
    PolarVolume_copyconstructor
};
