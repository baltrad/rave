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
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>

/**
 * Represents a volume
 */
struct _PolarVolume_t {
  long pv_refCount; /**< ref counter */

  Projection_t* projection; /**< projection for this volume */
  PolarNavigator_t* navigator; /**< a polar navigator */

  int nrAllocatedScans; /**< Number of scans that the volume currently can hold */
  int nrScans; /**< The number of scans that this volume is defined by */

  int debug; /**< debugging flag */

  PolarScan_t** scans; /**< the scans that this volume is defined by */
};

/*@{ Private functions */
/**
 * Ensures that the volume can at least manage one more scan
 * @returns 0 on failure, otherwise it was a success
 */
static int PolarVolume_ensureScanCapacity(PolarVolume_t* pvol)
{
  int result = 0;
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  if (pvol->scans == NULL) {
    pvol->scans = RAVE_MALLOC(32 * sizeof(PolarScan_t*));
    if (pvol->scans == NULL) {
      RAVE_ERROR0("Failed to allocate scans");
      goto fail;
    }
    memset(pvol->scans, 0, 32*sizeof(PolarScan_t*));
    pvol->nrAllocatedScans = 32;
  } else {
    if (pvol->nrScans == pvol->nrAllocatedScans) {
      int i = 0;
      int newAllocSize = pvol->nrAllocatedScans + 32;
      PolarScan_t** newscans = RAVE_REALLOC(pvol->scans, newAllocSize*sizeof(PolarScan_t*));
      if (newscans == NULL) {
        RAVE_ERROR0("Failed to reallocate scans");
        goto fail;
      }
      for (i = pvol->nrAllocatedScans; i < newAllocSize; i++) {
        newscans[i] = NULL;
      }
      pvol->scans = newscans;
      pvol->nrAllocatedScans = newAllocSize;
    }
  }
  result = 1;
fail:
  return result;
}

/**
 * Destroys the polar volume.
 * @param[in] volume - the volume to destroy
 */
static void PolarVolume_destroy(PolarVolume_t* volume)
{
  if (volume != NULL) {
    Projection_release(volume->projection);
    PolarNavigator_release(volume->navigator);
    if (volume->scans != NULL) {
      int i = 0;
      for (i = 0; i < volume->nrScans; i++) {
        PolarScan_release(volume->scans[i]);
        volume->scans[i] = NULL;
      }
      RAVE_FREE(volume->scans);
    }
    RAVE_FREE(volume);
  }
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
PolarVolume_t* PolarVolume_new(void)
{
  PolarVolume_t* result = NULL;
  result = RAVE_MALLOC(sizeof(PolarVolume_t));
  if (result != NULL) {
    result->pv_refCount = 1;

    result->nrAllocatedScans = 0;
    result->nrScans = 0;
    result->scans = NULL;
    result->projection = NULL;
    result->navigator = NULL;
    result->debug = 0;

    if (!PolarVolume_ensureScanCapacity(result)) {
      PolarVolume_destroy(result);
      result = NULL;
      goto done;
    }

    // Always initialize to default projection for lon/lat calculations
    result->projection = Projection_new("lonlat", "lonlat", "+proj=latlong +ellps=WGS84 +datum=WGS84");
    if (result->projection == NULL) {
      PolarVolume_destroy(result);
      result = NULL;
      goto done;
    }

    // And a navigator as well
    result->navigator = PolarNavigator_new();
    if (result->navigator == NULL) {
      PolarVolume_destroy(result);
      result = NULL;
      goto done;
    }
  }
done:
  return result;
}

/**
 * Releases the responsibility for the volume, it is not certain that
 * it will be deleted though if there still are references existing
 * to this scan.
 * @param[in] pvol - the polar volume
 */
void PolarVolume_release(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  pvol->pv_refCount--;
  if (pvol->pv_refCount <= 0) {
    PolarVolume_destroy(pvol);
  }
}

/**
 * Copies the reference to this instance by increasing a
 * reference counter.
 * @param[in] pvol - the polar volume to be copied
 * @return a pointer to the volume
 */
PolarVolume_t* PolarVolume_copy(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  pvol->pv_refCount++;
  return pvol;
}

void PolarVolume_setLongitude(PolarVolume_t* pvol, double lon)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  PolarNavigator_setLon0(pvol->navigator, lon);
}

double PolarVolume_getLongitude(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  return PolarNavigator_getLon0(pvol->navigator);
}

void PolarVolume_setLatitude(PolarVolume_t* pvol, double lat)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  PolarNavigator_setLat0(pvol->navigator, lat);
}

double PolarVolume_getLatitude(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  return PolarNavigator_getLat0(pvol->navigator);
}

void PolarVolume_setHeight(PolarVolume_t* pvol, double height)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  PolarNavigator_setAlt0(pvol->navigator, height);
}

double PolarVolume_getHeight(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  return PolarNavigator_getAlt0(pvol->navigator);
}

void PolarVolume_setProjection(PolarVolume_t* pvol, Projection_t* projection)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  Projection_release(pvol->projection);
  pvol->projection = NULL;
  if (projection != NULL) {
    pvol->projection = Projection_copy(projection);
  }
}

Projection_t* PolarVolume_getProjection(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  if (pvol->projection != NULL) {
    return Projection_copy(pvol->projection);
  }
  return NULL;
}

int PolarVolume_addScan(PolarVolume_t* pvol, PolarScan_t* scan)
{
  int result = 0;
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  if (PolarVolume_ensureScanCapacity(pvol)) {
    PolarScan_setNavigator(scan, pvol->navigator);
    pvol->scans[pvol->nrScans++] = PolarScan_copy(scan);
    result = 1;
  }
  return result;
}

PolarScan_t* PolarVolume_getScan(PolarVolume_t* pvol, int index)
{
  PolarScan_t* scan = NULL;
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  if (index >= 0 && index < pvol->nrScans) {
    scan = PolarScan_copy(pvol->scans[index]);
  }
  return scan;
}

int PolarVolume_getNumberOfScans(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  return pvol->nrScans;
}

PolarScan_t* PolarVolume_getScanNearestElevation(PolarVolume_t* pvol, double e)
{
  double se = 0.0L, eld = 0.0L;
  int ei = 0;
  int i = 0;
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  RAVE_ASSERT((index != NULL), "index was NULL");

  se = PolarScan_getElangle(pvol->scans[0]);
  ei = 0;
  eld = fabs(e - se);
  for (i = 1; i < pvol->nrScans; i++) {
    double elev = PolarScan_getElangle(pvol->scans[i]);
    double elevd = fabs(e - elev);
    if (elevd < eld) {
      se = elev;
      eld = elevd;
      ei = i;
    } else {
      break;
    }
  }
  return PolarScan_copy(pvol->scans[ei]);
}

RaveValueType PolarVolume_getNearest(PolarVolume_t* pvol, double lon, double lat, double height, double* v)
{
  double d = 0.0L, a = 0.0L, r = 0.0L, e = 0.0L;
  RaveValueType result = RaveValueType_NODATA;
  PolarScan_t* scan = NULL;
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  RAVE_ASSERT((v != NULL), "v was NULL");
  *v = 0.0;

  PolarNavigator_llToDa(pvol->navigator, lat, lon, &d, &a);
  PolarNavigator_dhToRe(pvol->navigator, d, height, &r, &e);

  // Find relevant elevation
  scan = PolarVolume_getScanNearestElevation(pvol, e);

  //@todo: Eventually use the actual elevation and calculate proper range instead.

  // Now we have the elevation angle, fetch value by providing azimuth and range.
  result = PolarScan_getValueAtAzimuthAndRange(scan, a, r, v);

  PolarScan_release(scan);

  return result;
}

RaveValueType PolarVolume_getNearestForElevation(PolarVolume_t* pvol, double lon, double lat, int index, double* v)
{
  double d = 0.0L, a = 0.0L, r = 0.0L, e = 0.0L, height = 0.0L;
  RaveValueType result = RaveValueType_NODATA;
  PolarScan_t* scan = NULL;

  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  RAVE_ASSERT((v != NULL), "v was NULL");
  *v = 0.0;

  if ((scan = PolarVolume_getScan(pvol, index)) == NULL) {
    RAVE_ERROR2("Attempting to get value for elevation index %d but there are only %d elevations available", index, pvol->nrScans);
    return result;
  }

  e = PolarScan_getElangle(scan);

  PolarNavigator_llToDa(pvol->navigator, lat, lon, &d, &a);
  PolarNavigator_deToRh(pvol->navigator, d, e, &r, &height);

  result = PolarScan_getValueAtAzimuthAndRange(scan, a, r, v);

  PolarScan_release(scan);

  return result;
}

void PolarVolume_sortByElevations(PolarVolume_t* pvol, int ascending)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");

  if (ascending == 1) {
    qsort(pvol->scans, pvol->nrScans, sizeof(PolarScan_t*), ascendingElevationSort);
  } else {
    qsort(pvol->scans, pvol->nrScans, sizeof(PolarScan_t*), descendingElevationSort);
  }
}

int PolarVolume_isAscendingScans(PolarVolume_t* pvol)
{
  int result = 1;
  int index = 0;
  double lastelev = 0.0L;
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  if (pvol->nrScans > 0) {
    lastelev = PolarScan_getElangle(pvol->scans[0]);
    for (index = 1; result == 1 && index < pvol->nrScans; index++) {
      double nextelev = PolarScan_getElangle(pvol->scans[index]);
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
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  // Verify that the volume at least got one scan and that the scans
  // are sorted in ascending order.
  if (pvol->nrScans > 0 && PolarVolume_isAscendingScans(pvol)) {
    result = 1;
  }
  return result;
}

void PolarVolume_setDebug(PolarVolume_t* pvol, int enable)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  pvol->debug = enable;
}

/*@} End of Interface functions */
