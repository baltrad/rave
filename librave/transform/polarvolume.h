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
 * @date 2009-10-14
 */
#ifndef POLARVOLUME_H
#define POLARVOLUME_H
#include "polarscan.h"
#include "projection.h"

/**
 * Defines a Polar Volume
 */
typedef struct _PolarVolume_t PolarVolume_t;

/**
 * Allocates a new instance of PolarVolume_t
 * @returns a new instance or NULL on failure
 */
PolarVolume_t* PolarVolume_new(void);

/**
 * Releases the responsibility for the volume, it is not certain that
 * it will be deleted though if there still are references existing
 * to this volume.
 * @param[in] pvol - the instance to be released
 */
void PolarVolume_release(PolarVolume_t* pvol);

/**
 * Copies the reference to this instance by increasing a
 * reference counter.
 * @param[in] pvol - the volume to be copied
 * @return a pointer to the volume
 */
PolarVolume_t* PolarVolume_copy(PolarVolume_t* pvol);

/**
 * Sets the longitude for the volume
 * @param[in] pvol - the polar volume
 * @param[in] double - the longitude
 */
void PolarVolume_setLongitude(PolarVolume_t* pvol, double lon);

/**
 * Returns the longitude for the volume
 * @param[in] pvol - the polar volume
 * @returns the longitude
 */
double PolarVolume_getLongitude(PolarVolume_t* pvol);

/**
 * Sets the latitude for the volume
 * @param[in] pvol - the polar volume
 * @param[in] double - the longitude
 */
void PolarVolume_setLatitude(PolarVolume_t* pvol, double lat);

/**
 * Returns the latitude for the volume
 * @param[in] pvol - the polar volume
 * @returns the longitude
 */
double PolarVolume_getLatitude(PolarVolume_t* pvol);

/**
 * Sets the height for the volume
 * @param[in] pvol - the polar volume
 * @param[in] double - the longitude
 */
void PolarVolume_setHeight(PolarVolume_t* pvol, double height);

/**
 * Returns the latitude for the volume
 * @param[in] pvol - the polar volume
 * @returns the longitude
 */
double PolarVolume_getHeight(PolarVolume_t* pvol);

/**
 * Sets the projection for this polar volume.
 * @param[in] pvol - the polar volume
 * @param[in] projection - the projection
 */
void PolarVolume_setProjection(PolarVolume_t* pvol, Projection_t* projection);

/**
 * Returns a copy of the projection that is used for this polar volume.
 * I.e. remember to release it.
 * @param[in] pvol - the polar volume
 * @returns a projection (or NULL if none is set)
 */
Projection_t* PolarVolume_getProjection(PolarVolume_t* pvol);

/**
 * Adds a scan to the volume.
 * @param[in] pvol - the volume
 * @param[in] scan - the scan
 * ®return 0 on failure, otherwise success
 */
int PolarVolume_addScan(PolarVolume_t* pvol, PolarScan_t* scan);

/**
 * Returns the scan at given index.
 * @param[in] pvol - the volume
 * @param[in] index - the index
 * @param[out] scan - the scan at specified index
 * @returns 0 on failure, otherwise it was a success
 */
int PolarVolume_getScan(PolarVolume_t* pvol, int index, PolarScan_t** scan);

/**
 * Returns the number of scans.
 * @param[in] pvol - the volume
 * @returns -1 on failure, otherwise a value >= 0
 */
int PolarVolume_getNumberOfScans(PolarVolume_t* pvol);

/**
 * Returns the index of the scan that got an elevation that is closest
 * to the provided elevation.
 * @param[in] pvol - the polar volume
 * @param[in] e - the elevation (in radians)
 * @param[out] index - the found scan index.
 */
void PolarVolume_getNearestElevation(PolarVolume_t* pvol, double e, int* index);

/**
 * Fetches the value nearest to the specified position.
 * @param[in] pvol - the polar volume
 * @param[in] lon  - the longitude (in radians)
 * @param[in] lat  - the latitude (in radians)
 * @param[in] height - the height
 * @param[out] v - the value
 * @return what type of value that has been set in v.
 */
RaveValueType PolarVolume_getNearest(PolarVolume_t* pvol, double lon, double lat, double height, double* v);

/**
 * Calculates the value at the specified lon/lat/height.
 * @param[in] pvol - the volume
 * @param[in] lon - the longitude
 * @param[in] lat - the latitude
 * @param[in] height - the height
 * @param[in] param - the transformation parameters
 */
//double PolarVolume_getWeightedValue(PolarVolume_t* pvol, double lon,
//  double lat, double height, TransformParam* param);

/**
 * Arranges the scans in either ascending or descending elevation.
 * @param[in] pvol - the volume
 * @param[in] ascending - if 1, ascending sort will be done, otherwise descending
 */
void PolarVolume_sortByElevations(PolarVolume_t* pvol, int ascending);

/**
 * Enables/Disables calculation debugging for a polar volume
 * @param[in] pvol - the volume
 * @param[in] enable - 0 to disable (default) 1 to activate
 */
void PolarVolume_setDebug(PolarVolume_t* pvol, int enable);

#endif
