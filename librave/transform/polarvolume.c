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
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>

/**
 * Represents a volume
 */
struct _PolarVolume_t {
  long pv_refCount; /**< ref counter */

  double lon; /**< longitude of the radar that this volume originated from */
  double lat; /**< latitude of the radar that this volume originated from */
  double height; /**< altitude of the radar that this volume originated from */

  Projection_t* projection; /**< projection for this volume */

  int nrAllocatedScans; /**< Number of scans that the volume currently can hold */
  int nrScans; /**< The number of scans that this volume is defined by */


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

    result->lon = 0.0;
    result->lat = 0.0;
    result->height = 0.0;

    result->nrAllocatedScans = 0;
    result->nrScans = 0;
    result->scans = NULL;
    result->projection = NULL;

    if (!PolarVolume_ensureScanCapacity(result)) {
      PolarVolume_destroy(result);
      result = NULL;
    }

    // Always initialize to default projection for lon/lat calculations
    result->projection = Projection_new("lonlat", "lonlat", "+proj=latlong +ellps=WGS84 +datum=WGS84");
    if (result->projection == NULL) {
      PolarVolume_destroy(result);
      result = NULL;
    }
  }
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
  pvol->lon = lon;
}

double PolarVolume_getLongitude(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  return pvol->lon;
}

void PolarVolume_setLatitude(PolarVolume_t* pvol, double lat)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  pvol->lat = lat;
}

double PolarVolume_getLatitude(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  return pvol->lat;
}

void PolarVolume_setHeight(PolarVolume_t* pvol, double height)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  pvol->height = height;
}

double PolarVolume_getHeight(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  return pvol->height;
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
    pvol->scans[pvol->nrScans++] = PolarScan_copy(scan);
    result = 1;
  }
  return result;
}

int PolarVolume_getScan(PolarVolume_t* pvol, int index, PolarScan_t** scan)
{
  int result = 0;
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  if (index >= 0 && index < pvol->nrScans) {
    *scan = PolarScan_copy(pvol->scans[index]);
    result = 1;
  }
  return result;
}

int PolarVolume_getNumberOfScans(PolarVolume_t* pvol)
{
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  return pvol->nrScans;
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

/*@} End of Interface functions */
