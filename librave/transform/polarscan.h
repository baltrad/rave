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
#ifndef POLARSCAN_H
#define POLARSCAN_H
#include "rave_transform.h"
#include "polarnav.h"
#include "projection.h"
#include "rave_object.h"

/**
 * Defines a Polar Scan
 */
typedef struct _PolarScan_t PolarScan_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType PolarScan_TYPE;

/**
 * Sets a navigator for the polar scan, this is preferrable to use
 * when this scan is included in a volume since the settings will
 * be identical for all scans included in the volume. Otherwise,
 * if the scan is managed separately, use longitude/latitude and height
 * instead.
 * @param[in] scan - the scan
 * @param[in] navigator - the polar navigator (MAY NOT BE NULL)
 */
void PolarScan_setNavigator(PolarScan_t* scan, PolarNavigator_t* navigator);

/**
 * Returns the navigator that is used for this scan.
 * @param[in] scan - the scan
 * @returns the polar navigator
 */
PolarNavigator_t* PolarScan_getNavigator(PolarScan_t* scan);

/**
 * Sets the projection to be used for this scan.
 * @param[in] scan - the scan
 * @param[in] projection - the projection (MAY NOT BE NULL)
 */
void PolarScan_setProjection(PolarScan_t* scan, Projection_t* projection);

/**
 * Returns the current projection for this scan.
 * @param[in] scan - the scan
 * @returns the projection used within this scan.
 */
Projection_t* PolarScan_getProjection(PolarScan_t* scan);

/**
 * Sets the longitude for the volume
 * @param[in] pvol - the polar volume
 * @param[in] double - the longitude
 */
void PolarScan_setLongitude(PolarScan_t* scan, double lon);

/**
 * Returns the longitude for the volume
 * @param[in] pvol - the polar volume
 * @returns the longitude
 */
double PolarScan_getLongitude(PolarScan_t* scan);

/**
 * Sets the latitude for the volume
 * @param[in] pvol - the polar volume
 * @param[in] double - the longitude
 */
void PolarScan_setLatitude(PolarScan_t* scan, double lat);

/**
 * Returns the latitude for the volume
 * @param[in] pvol - the polar volume
 * @returns the longitude
 */
double PolarScan_getLatitude(PolarScan_t* scan);

/**
 * Sets the height for the volume
 * @param[in] pvol - the polar volume
 * @param[in] double - the longitude
 */
void PolarScan_setHeight(PolarScan_t* scan, double height);

/**
 * Returns the latitude for the volume
 * @param[in] pvol - the polar volume
 * @returns the longitude
 */
double PolarScan_getHeight(PolarScan_t* scan);

/**
 * Sets the elevation angle for the scan
 * @param[in] scan - the scan
 * @param[in] elangle - the elevation angle
 */
void PolarScan_setElangle(PolarScan_t* scan, double elangle);

/**
 * Returns the elevation angle for the scan
 * @param[in] scan - the scan
 * @return the elevation angle
 */
double PolarScan_getElangle(PolarScan_t* scan);

/**
 * Sets the number of bins for the scan
 * @param[in] scan - the scan
 * @param[in] nbins - the number of bins
 */
void PolarScan_setNbins(PolarScan_t* scan, long nbins);

/**
 * Returns the number of bins
 * @param[in] scan - the scan
 * @return the number of bins
 */
long PolarScan_getNbins(PolarScan_t* scan);

/**
 * Sets the range scale for the scan
 * @param[in] scan - the scan
 * @param[in] elangle - the elevation angle
 */
void PolarScan_setRscale(PolarScan_t* scan, double rscale);

/**
 * Returns the range scale for the scan
 * @param[in] scan - the scan
 * @return the elevation angle
 */
double PolarScan_getRscale(PolarScan_t* scan);

/**
 * Sets the number of rays/scan
 * @param[in] scan - the scan
 * @param[in] nrays - the number of rays
 */
void PolarScan_setNrays(PolarScan_t* scan, long nrays);

/**
 * Returns the number of rays/scan
 * @param[in] scan - the scan
 * @return the number of rays
 */
long PolarScan_getNrays(PolarScan_t* scan);

/**
 * Sets the ray start for the scan
 * @param[in] scan - the scan
 * @param[in] rstart - the start position of the ray
 */
void PolarScan_setRstart(PolarScan_t* scan, double rstart);

/**
 * Returns the ray start for the scan
 * @param[in] scan - the scan
 * @return the ray start position
 */
double PolarScan_getRstart(PolarScan_t* scan);

/**
 * Sets the data type of the data that is worked with
 * @param[in] scan - the scan
 * @param[in] type - the data type
 * @return 0 if type is not known, otherwise the type was set
 */
int PolarScan_setDataType(PolarScan_t* scan, RaveDataType type);

/**
 * Returns the data type
 * @param[in] scan - the scan
 * @return the data type
 */
RaveDataType PolarScan_getDataType(PolarScan_t* scan);

/**
 * Sets the a1gate
 * @param[in] scan - the scan
 * @param[in] a1gate - a1gate
 */
void PolarScan_setA1gate(PolarScan_t* scan, long a1gate);

/**
 * Returns the a1gate
 * @param[in] scan - the scan
 * @return the a1gate
 */
long PolarScan_getA1gate(PolarScan_t* scan);


/**
 * Sets the beamwidth. If not set, it will default to 360/nrays.
 * @param[in] scan - the polar scan
 * @param[in] beamwidth - the beam width
 */
void PolarScan_setBeamWidth(PolarScan_t* scan, long beamwidth);

/**
 * Returns the beamwidth. If no beamwidth has been explicitly set, the returned
 * value will be 360/nrays or 0.0 if no nrays has been set either.
 * @param[in] scan - the polar scan
 * @return the beam width
 */
double PolarScan_getBeamWidth(PolarScan_t* scan);

/**
 * Sets the quantity
 * @param[in] scan - the scan
 * @param[in] quantity - the quantity, e.g. DBZH
 */
void PolarScan_setQuantity(PolarScan_t* scan, const char* quantity);

/**
 * Returns the quantity
 * @param[in] scan - the scan
 * @return the quantity
 */
const char* PolarScan_getQuantity(PolarScan_t* scan);

/**
 * Sets the gain
 * @param[in] scan - the scan
 * @param[in] gain - the gain
 */
void PolarScan_setGain(PolarScan_t* scan, double gain);

/**
 * Returns the gain
 * @param[in] scan - the scan
 * @return the gain
 */
double PolarScan_getGain(PolarScan_t* scan);

/**
 * Sets the offset
 * @param[in] scan - the scan
 * @param[in] offset - the offset
 */
void PolarScan_setOffset(PolarScan_t* scan, double offset);

/**
 * Returns the offset
 * @param[in] scan - the scan
 * @return the offset
 */
double PolarScan_getOffset(PolarScan_t* scan);

/**
 * Sets the nodata
 * @param[in] scan - the scan
 * @param[in] nodata - the nodata
 */
void PolarScan_setNodata(PolarScan_t* scan, double nodata);

/**
 * Returns the nodata
 * @param[in] scan - the scan
 * @return the nodata
 */
double PolarScan_getNodata(PolarScan_t* scan);

/**
 * Sets the undetect
 * @param[in] scan - the scan
 * @param[in] undetect - the undetect
 */
void PolarScan_setUndetect(PolarScan_t* scan, double undetect);

/**
 * Returns the undetect
 * @param[in] scan - the scan
 * @return the undetect
 */
double PolarScan_getUndetect(PolarScan_t* scan);

/**
 * Sets the data
 * @param[in] scan  - the scan
 * @param[in] nbins - number of bins
 * @param[in] nrays - number of rays
 * @param[in] data  - the data
 * @param[in] type  - the data type
 */
int PolarScan_setData(PolarScan_t* scan, long nbins, long nrays, void* data, RaveDataType type);

/**
 * Returns a pointer to the internal data storage.
 * @param[in] scan - the scan
 * @return the internal data pointer (NOTE! Do not release this pointer)
 */
void* PolarScan_getData(PolarScan_t* scan);

/**
 * Returns the range index for the specified range (in meters).
 * @param[in] scan - the scan
 * @param[in] r - the range
 * @return -1 on failure, otherwise a index between 0 and nbins
 */
int PolarScan_getRangeIndex(PolarScan_t* scan, double r);

/**
 * Returns the azimuth index for the specified azimuth.
 * @param[in] scan - the scan
 * @param[in] a - the azimuth (in radians)
 * @return -1 on failure, otherwise a index between 0 and nrays.
 */
int PolarScan_getAzimuthIndex(PolarScan_t* scan, double a);

/**
 * Returns the value at the specified azimuth and range index.
 * @param[in] scan - the scan
 * @param[in] ai - the azimuth index
 * @param[in] ri - the range index
 * @param[out] v - the data at the specified index
 * @return the type of data
 */
RaveValueType PolarScan_getValueAtIndex(PolarScan_t* scan, int ai, int ri, double* v);

/**
 * Gets the value at the provided azimuth and range.
 * @param[in] scan - the scan
 * @param[in] a - the azimuth (in radians)
 * @param[in] r - the range (in meters)
 * @param[out] v - the value
 * @return a rave value type
 */
RaveValueType PolarScan_getValueAtAzimuthAndRange(PolarScan_t* scan, double a, double r, double* v);

/**
 * Returns the nearest value to the specified longitude, latitude.
 * @param[in] scan - the scan
 * @param[in] lon  - the longitude (in radians)
 * @param[in] lat  - the latitude  (in radians)
 * @returns a rave value type
 */
RaveValueType PolarScan_getNearest(PolarScan_t* scan, double lon, double lat, double* v);

/**
 * Verifies that all preconditions are met in order to perform
 * a transformation.
 * @param[in] scan - the polar scan
 * @returns 1 if the polar scan is ready, otherwise 0.
 */
int PolarScan_isTransformable(PolarScan_t* scan);

#ifdef KALLE
/**
 * Function for keeping track of some sort of metadata that should be
 * associated with this scan.
 * @param[in] scan - the scan
 * @param[in] ptr - a void pointer that should be stored
 */
void PolarScan_setVoidPtr(PolarScan_t* scan, void* ptr);

/**
 * Returns the void data.
 * @param[in] scan - the scan
 * @return the void data
 */
void* PolarScan_getVoidPtr(PolarScan_t* scan);
#endif

/**
 * Enables/Disables calculation debugging for a polar scan
 * @param[in] scan - the scan
 * @param[in] enable - 0 to disable (default) 1 to activate
 */
void PolarScan_setDebug(PolarScan_t* scan, int enable);

#endif
