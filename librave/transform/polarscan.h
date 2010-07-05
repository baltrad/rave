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
 * Defines the functions available when working with polar scans.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-15
 */
#ifndef POLARSCAN_H
#define POLARSCAN_H
#include "polarnav.h"
#include "projection.h"
#include "polarscanparam.h"
#include "rave_object.h"
#include "rave_types.h"
#include "rave_list.h"
#include "raveobject_list.h"
#include "rave_field.h"

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
 * Sets the nominal time.
 * @param[in] scan - self
 * @param[in] value - the time in the format HHmmss
 * @returns 1 on success, otherwise 0
 */
int PolarScan_setTime(PolarScan_t* scan, const char* value);

/**
 * Returns the nominal time.
 * @param[in] scan - self
 * @returns the nominal time (or NULL if there is none)
 */
const char* PolarScan_getTime(PolarScan_t* scan);

/**
 * Sets the nominal date.
 * @param[in] scan - self
 * @param[in] value - the date in the format YYYYMMDD
 * @returns 1 on success, otherwise 0
 */
int PolarScan_setDate(PolarScan_t* scan, const char* value);

/**
 * Returns the nominal date.
 * @param[in] scan - self
 * @returns the nominal time (or NULL if there is none)
 */
const char* PolarScan_getDate(PolarScan_t* scan);

/**
 * Sets the source.
 * @param[in] scan - self
 * @param[in] value - the source
 * @returns 1 on success, otherwise 0
 */
int PolarScan_setSource(PolarScan_t* scan, const char* value);

/**
 * Returns the source.
 * @param[in] scan - self
 * @returns the source or NULL if there is none
 */
const char* PolarScan_getSource(PolarScan_t* scan);

/**
 * Sets the longitude
 * @param[in] scan - self
 * @param[in] lon - the longitude
 */
void PolarScan_setLongitude(PolarScan_t* scan, double lon);

/**
 * Returns the longitude
 * @param[in] scan - self
 * @returns the longitude
 */
double PolarScan_getLongitude(PolarScan_t* scan);

/**
 * Sets the latitude
 * @param[in] scan - self
 * @param[in] lat - the latitude
 */
void PolarScan_setLatitude(PolarScan_t* scan, double lat);

/**
 * Returns the latitude
 * @param[in] scan - self
 * @returns the latitude
 */
double PolarScan_getLatitude(PolarScan_t* scan);

/**
 * Sets the height
 * @param[in] scan - self
 * @param[in] height - the height
 */
void PolarScan_setHeight(PolarScan_t* scan, double height);

/**
 * Returns the height
 * @param[in] scan - self
 * @returns the height
 */
double PolarScan_getHeight(PolarScan_t* scan);

/**
 * Sets the elevation angle for the scan
 * @param[in] scan - self
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
 * Returns the number of bins
 * @param[in] scan - the scan
 * @return the number of bins
 */
long PolarScan_getNbins(PolarScan_t* scan);

/**
 * Sets the range scale for the scan
 * @param[in] scan - the scan
 * @param[in] rscale - the scale of the range bin
 */
void PolarScan_setRscale(PolarScan_t* scan, double rscale);

/**
 * Returns the range bin scale for the scan
 * @param[in] scan - the scan
 * @return the scale of the range bin
 */
double PolarScan_getRscale(PolarScan_t* scan);

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
 * Sets the default parameter for this scan. I.e. all operations
 * that retrieves/sets values that does not contain a parameter name
 * as well will use the default parameter. Note, there is nothing
 * verifying if the parameter actually exists so if you are uncertain
 * use \ref #hasParameter first.
 * @param[in] scan - self
 * @param[in] quantity - the parameter
 * @returns 1 on success otherwise 0
 */
int PolarScan_setDefaultParameter(PolarScan_t* scan, const char* quantity);

/**
 * Returns the currently specified default parameter name.
 * @param[in] scan - self
 * @returns the default parameter name
 */
const char* PolarScan_getDefaultParameter(PolarScan_t* scan);

/**
 * Adds a parameter to the polar scan. Note, if there already exists
 * a parameter with the same quantity, that parameter will be replaced
 * by this. Also, several consistency checks will be performed to ensure
 * that dimensions and similar are the same for all parameters that
 * are added.
 * @param[in] scan - self
 * @param[in] parameter - the parameter
 * @returns 1 on success, otherwise 0
 */
int PolarScan_addParameter(PolarScan_t* scan, PolarScanParam_t* parameter);

/**
 * Removes (and returns) the parameter that is specified by the quantity.
 * Note, since the parameter returned is inc-refed, remember to release it.
 * @param[in] scan - self
 * @param[in] key - the quantity name
 * @returns NULL if nothing found or the parameter if it exists.
 */
PolarScanParam_t* PolarScan_removeParameter(PolarScan_t* scan, const char* quantity);

/**
 * Returns the parameter that is specified by the quantity.
 * Note, since the parameter returned is inc-refed, remember to release it.
 * @param[in] scan - self
 * @param[in] key - the quantity name
 * @returns NULL if nothing found or the parameter if it exists.
 */
PolarScanParam_t* PolarScan_getParameter(PolarScan_t* scan, const char* quantity);

/**
 * Returns if the scan contains the specified parameter or not.
 * @param[in] scan - self
 * @param[in] quantity - the quantity name
 * @returns 1 if the parameter exists, otherwise 0
 */
int PolarScan_hasParameter(PolarScan_t* scan, const char* quantity);

/**
 * Returns this scans parameter names.
 * @param[in] scan - self
 * @returns this scans contained parameters. NULL on failure. Use \ref #RaveList_freeAndDestroy to destroy
 */
RaveList_t* PolarScan_getParameterNames(PolarScan_t* scan);

/**
 * Adds a quality field to this scan.
 * @param[in] scan - self
 * @param[in] field - the field to add
 * @returns 1 on success otherwise 0
 */
int PolarScan_addQualityField(PolarScan_t* scan, RaveField_t* field);

/**
 * Returns the quality field at the specified location.
 * @param[in] scan - self
 * @param[in] index - the index
 * @returns the quality field if found, otherwise NULL
 */
RaveField_t* PolarScan_getQualityField(PolarScan_t* scan, int index);

/**
 * Returns the number of quality fields
 * @param[in] scan - self
 * @returns the number of quality fields
 */
int PolarScan_getNumberOfQualityFields(PolarScan_t* scan);

/**
 * Removes the quality field at the specified location
 * @param[in] scan - self
 * @param[in] index - the index
 */
void PolarScan_removeQualityField(PolarScan_t* scan, int index);

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
 * Returns the value at the specified index.
 * @param[in] scan - the scan
 * @param[in] bin - the bin index
 * @param[in] ray - the ray index
 * @param[out] v - the data at the specified index
 * @return the type of data
 */
RaveValueType PolarScan_getValue(PolarScan_t* scan, int bin, int ray, double* v);

/**
 * Returns the parameter value at the specified index.
 * @param[in] scan - self
 * @param[in] quantity - the parameter (MAY NOT be NULL)
 * @param[in] bin - the bin index
 * @param[in] ray - the ray index
 * @param[out] v - the found value
 */
RaveValueType PolarScan_getParameterValue(PolarScan_t* scan, const char* quantity, int bin, int ray, double* v);

/**
 * Returns the linear converted value at the specified index. That is,
 * offset + gain * value;
 * @param[in] scan - the scan
 * @param[in] bin - the bin index
 * @param[in] ray - the ray index
 * @param[out] v - the data at the specified index
 * @return the type of data
 */
RaveValueType PolarScan_getConvertedValue(PolarScan_t* scan, int bin, int ray, double* v);

/**
 * Returns the linear converted parameter value at the specified index. That is,
 * offset + gain * value;
 * @param[in] scan - the scan
 * @param[in] quantity - the parameter (MAY NOT BE NULL)
 * @param[in] bin - the bin index
 * @param[in] ray - the ray index
 * @param[out] v - the data at the specified index
 * @return the type of data
 */
RaveValueType PolarScan_getConvertedParameterValue(PolarScan_t* scan, const char* quantity, int bin, int ray, double* v);

/**
 * Returns the bin and ray index from a specified azimuth and range.
 * @param[in] scan - self (MAY NOT BE NULL)
 * @param[in] a - the azimuth (in radians)
 * @param[in] r - the range (in meters)
 * @param[out] ray - the ray index (MAY NOT BE NULL)
 * @param[out] bin - the bin index (MAY NOT BE NULL)
 * @returns 1 on success, otherwise 0 and in that case, bin and ray can not be relied on.
 */
int PolarScan_getIndexFromAzimuthAndRange(PolarScan_t* scan, double a, double r, int* ray, int* bin);

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
 * Returns the parameter value at the specified azimuth and range
 * @param[in] scan - self
 * @param[in] quantity - the parameter name
 * @param[in] a - the azimuth (in radians)
 * @param[in] r - the range (in meters)
 * @param[out] v - the value
 * @returns a rave value type (if scan does not contain specified parameter, RaveValueType_UNDEFINED will be returned).
 */
RaveValueType PolarScan_getParameterValueAtAzimuthAndRange(PolarScan_t* scan, const char* quantity, double a, double r, double* v);

/**
 * Returns the nearest value to the specified longitude, latitude.
 * @param[in] scan - the scan
 * @param[in] lon  - the longitude (in radians)
 * @param[in] lat  - the latitude  (in radians)
 * @param[out] v - the found value
 * @returns a rave value type
 */
RaveValueType PolarScan_getNearest(PolarScan_t* scan, double lon, double lat, double* v);

/**
 * Returns the nearest index to the specified long/lat pair.
 * @param[in] scan - self
 * @param[in] lon - the longitude (in radians)
 * @param[in] lat - the latitude (in radians)
 * @param[out] bin - the bin index (MAY NOT BE NULL)
 * @param[out] ray - the ray index (MAY NOT BE NULL)
 * @returns 0 if either bin and/or ray is outside boundaries, otherwise 1
 */
int PolarScan_getNearestIndex(PolarScan_t* scan, double lon, double lat, int* bin, int* ray);

/**
 * Verifies that all preconditions are met in order to perform
 * a transformation.
 * @param[in] scan - the polar scan
 * @returns 1 if the polar scan is ready, otherwise 0.
 */
int PolarScan_isTransformable(PolarScan_t* scan);

/**
 * Adds a rave attribute to the scan. If attribute maps to the
 * member attributes it will be used to set the specific member
 * instead. E.g. where/elangle (in degrees) will be accessible
 * from PolarScan_getElangle() in radians.
 * @param[in] scan - self
 * @param[in] attribute - the attribute
 * @return 1 on success otherwise 0
 */
int PolarScan_addAttribute(PolarScan_t* scan, RaveAttribute_t* attribute);

/**
 * Returns the rave attribute that is named accordingly.
 * @param[in] scan - self
 * @param[in] name - the name of the attribute
 * @returns the attribute if found otherwise NULL
 */
RaveAttribute_t* PolarScan_getAttribute(PolarScan_t* scan, const char* name);

/**
 * Returns a list of attribute names. Release with \@ref #RaveList_freeAndDestroy.
 * @param[in] scanparam - self
 * @returns a list of attribute names
 */
RaveList_t* PolarScan_getAttributeNames(PolarScan_t* scan);

/**
 * Returns a list of attribute values that should be stored for this scan. Corresponding
 * members will also be added as attribute values. E.g. elangle will be stored
 * as a double with name where/elangle in degrees. Since a polar scan must contain different
 * data depending on if it belongs to a volume or not, two more arguments is used.
 * otype is if this scan belongs to a volume or if it is a stand alone scan. rootattributes is used
 * if otype is scan, in that case, rootattributes defines if it is the root attributes that should
 * be returned or if it is the dataset specific ones.
 * @param[in] scan - self
 * @param[in] otype - what type of attributes that should be returned
 * @param[in] rootattributes - if it is the root attributes or not.
 * @returns a list of RaveAttributes.
 */
RaveObjectList_t* PolarScan_getAttributeValues(PolarScan_t* scan, Rave_ObjectType otype, int rootattributes);

/**
 * Validates the scan can be seen to be valid regarding storage.
 * @param[in] scan - self
 * @param[in] otype - the object type this scan should be accounted for
 * @returns 1 if valid, otherwise 0
 */
int PolarScan_isValid(PolarScan_t* scan, Rave_ObjectType otype);

#endif
