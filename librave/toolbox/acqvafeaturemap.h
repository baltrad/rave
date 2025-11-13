/* --------------------------------------------------------------------
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Feature map used when working with acqva. 
 *
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-11-11
 */
#ifndef ACQVA_FEATURE_MAP_H
#define ACQVA_FEATURE_MAP_H

#include "rave_object.h"
#include "raveobject_list.h"
#include "rave_datetime.h"
#include "rave_types.h"

/**
 * Defines a feature map
 */
typedef struct _AcqvaFeatureMap_t AcqvaFeatureMap_t;

/**
 * Defines a feature map elevation. This is mostly used internally by the feature map to group
 * different elevation angles.
 */
typedef struct _AcqvaFeatureMapElevation_t AcqvaFeatureMapElevation_t;

/**
 * Defines a feature map field
 */
typedef struct _AcqvaFeatureMapField_t AcqvaFeatureMapField_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType AcqvaFeatureMap_TYPE;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType AcqvaFeatureMapElevation_TYPE;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType AcqvaFeatureMapField_TYPE;

/**
 * Loads a ACQVA feature map
 * @param[in] filename - the filename
 * @return the feature map or NULL on failure
 */
AcqvaFeatureMap_t* AcqvaFeatureMap_load(const char* filename);

/**
 * Saves the ACQVA feature map
 * @param[in] self - self
 * @param[in] filename - the name to use when saving the feature map
 * @returns 1 on success otherwise 0
 */
int AcqvaFeatureMap_save(AcqvaFeatureMap_t* self, const char* filename);

/**
 * Sets the nod for this feature map.
 * @param[in] self - self
 * @param[in] nod - the nod
 * @returns 1 on success otherwise 0
 */
int AcqvaFeatureMap_setNod(AcqvaFeatureMap_t* self, const char* source);

/**
 * Returns the nod for this feature map
 * @param[in] self - self
 * @returns the nod
 */
const char* AcqvaFeatureMap_getNod(AcqvaFeatureMap_t* self);

/**
 * The longitude for this source
 * @param[in] self - self
 * @param[in] lon - the longitude (in radians)
 */
void AcqvaFeatureMap_setLongitude(AcqvaFeatureMap_t* self, double lon);

/**
 * The longitude for this source
 * @param[in] self - self
 * @return the longitude in radians
 */
double AcqvaFeatureMap_getLongitude(AcqvaFeatureMap_t* self);

/**
 * The latitude for this source
 * @param[in] self - self
 * @param[in] lat - the latitude (in radians)
 */
void AcqvaFeatureMap_setLatitude(AcqvaFeatureMap_t* self, double lat);

/**
 * The latitude for this source
 * @param[in] self - self
 * @return the latitude in radians
 */
double AcqvaFeatureMap_getLatitude(AcqvaFeatureMap_t* self);

/**
 * The height for this source
 * @param[in] self - self
 * @param[in] height - the altitude above sea level in meters
 */
void AcqvaFeatureMap_setHeight(AcqvaFeatureMap_t* self, double height);

/**
 * The height for this source
 * @param[in] self - self
 * @return the altitude above sea level in meters
 */
double AcqvaFeatureMap_getHeight(AcqvaFeatureMap_t* self);

/**
 * Sets the start date of the period this map represents.
 * @param[in] self - self
 * @param[in] date - the date (YYYYmmdd)
 * @return 1 on success otherwise 0
 */
int AcqvaFeatureMap_setStartdate(AcqvaFeatureMap_t* self, const char* date);

/**
 * Returns the start date of the period this map represents.
 * @param[in] self - self
 * @return the date (YYYYmmdd)
 */
const char* AcqvaFeatureMap_getStartdate(AcqvaFeatureMap_t* self);

/**
 * Sets the end date of the period this map represents.
 * @param[in] self - self
 * @param[in] date - the date (YYYYmmdd)
 * @return 1 on success otherwise 0
 */
int AcqvaFeatureMap_setEnddate(AcqvaFeatureMap_t* self, const char* date);

/**
 * Returns the start date of the period this map represents.
 * @param[in] self - self
 * @return the date (YYYYmmdd)
 */
const char* AcqvaFeatureMap_getEnddate(AcqvaFeatureMap_t* self);

/**
 * Creates a field in the feature map with wanted elevation angle and the geometry (nrays, nbins) and the field data is set to 1. 
 * @param[in] self - self
 * @param[in] nbins - number of bins
 * @param[in] nrays - number of rays
 * @param[in] type - the datatype
 * @param[in] elangle - the elevation angle of the field
 * ®return the field
 */
AcqvaFeatureMapField_t* AcqvaFeatureMap_createField(AcqvaFeatureMap_t* self, long nbins, long nrays, RaveDataType type, double elangle);

/**
 * Adds a field (scan) to the feature map. The field must contain the elevation angle and the geometry (nrays, nbins). Offset
 * is assumed to be 0, gain = 1
 * @param[in] self - self
 * @param[in] field - the field
 * ®return 0 on failure, otherwise success
 */
int AcqvaFeatureMap_addField(AcqvaFeatureMap_t* self, AcqvaFeatureMapField_t* field);

/**
 * Creates an elevation group in the feature map unless it already exists (same elangle) which will
 * instead return that elevation group.
 * @param[in] self - self
 * @param[in] elangle - the elevation angle in radians.
 */
AcqvaFeatureMapElevation_t* AcqvaFeatureMap_createElevation(AcqvaFeatureMap_t* self, double elangle);

/**
 * Adds an elevation group in the feature map unless it already exists. If it already exists an error will be returned.
 * @param[in] self - self
 * @param[in] elevation - the elevation group
 */
int AcqvaFeatureMap_addElevation(AcqvaFeatureMap_t* self, AcqvaFeatureMapElevation_t* elevation);

/**
 * Returns the number of elevation groups in the feature map.
 * @param[in] self - self
 * @returns the number of elevation groups
 */
int AcqvaFeatureMap_getNumberOfElevations(AcqvaFeatureMap_t* self);

/**
 * Returns the elevation group at given index.
 * @param[in] self - self
 * @param[in] index - the index
 * @returns the elevation group
 */
AcqvaFeatureMapElevation_t* AcqvaFeatureMap_getElevation(AcqvaFeatureMap_t* self, int index);

/**
 * Removes the elevation group at specified index
 * @param[in] self - self
 * @param[in] index - the index
 */
void AcqvaFeatureMap_removeElevation(AcqvaFeatureMap_t* self, int index);

/**
 * Locates a matching field that is very close to elangle < 1e-4 with wanted nrays & nbins.
 * @param[in] self - self
 * @param[in] nrays - number of rays
 * @param[in] nbins - number of bins
 * @param[in] elangle - the elevation angle in radians
 * @return the rave field
 */
AcqvaFeatureMapField_t* AcqvaFeatureMap_findField(AcqvaFeatureMap_t* self, long nbins, long nrays, double elangle);

/**
 * Locates a matching field that is very close to elangle < 1e-4. 
 * @param[in] self - self
 * @param[in] elangle - the elevation angle in radians
 * @return the feature map elevation group
 */
AcqvaFeatureMapElevation_t* AcqvaFeatureMap_findElevation(AcqvaFeatureMap_t* self, double elangle);

/**
 * Sets the elevation angle for this group
 * @param[in] self - self
 * @param[in] elangle - the elevation angle of this group (in radians)
 * @return 1 if the elevation angle is possible to set
 */
int AcqvaFeatureMapElevation_setElangle(AcqvaFeatureMapElevation_t* self, double elangle);

/**
 * Gets the elevation angle for this group
 * @param[in] self - self
 * @return the elevation angle of this group (in radians)
 */
double AcqvaFeatureMapElevation_getElangle(AcqvaFeatureMapElevation_t* self);

/**
 * Adds a field to the elevation group. The field must have same elevation angle as the group and the nbins and nrays must not already exist.
 * @param[in] self - the elevation
 * @param[in] field - the field. nbins/nrays must already exist in the group
 * @return 1 if successful, otherwise 0
 */
int AcqvaFeatureMapElevation_add(AcqvaFeatureMapElevation_t* self, AcqvaFeatureMapField_t* field);

/**
 * Returns number of fields in this elevation group
 * @param[in] self - the elevation
 * @return number of fields in group
 */
int AcqvaFeatureMapElevation_size(AcqvaFeatureMapElevation_t* self);

/**
 * Returns field at index
 * @param[in] self - self
 * @param[in] index - index
 * @return the featuremap field or NULL
*/
AcqvaFeatureMapField_t* AcqvaFeatureMapElevation_get(AcqvaFeatureMapElevation_t* self, int index);

/**
 * Returns field at index
 * @param[in] self - self
 * @param[in] index - index
 * @return the featuremap field or NULL
*/
void AcqvaFeatureMapElevation_remove(AcqvaFeatureMapElevation_t* self, int index);

/**
 * Locates a field that matches the nbins and nrays
 * @param[in] self - self
 * @param[in] nrays - number of rays
 * @param[in] nbins - number of bins
 * @return the featuremap field
*/
AcqvaFeatureMapField_t* AcqvaFeatureMapElevation_find(AcqvaFeatureMapElevation_t* self, long nbins, long nrays);

/**
 * Locates a field that matches the nbins and nrays and returns true or false depending on if it exists
 * @param[in] self - self
 * @param[in] nrays - number of rays
 * @param[in] nbins - number of bins
 * @return true or false
*/
int AcqvaFeatureMapElevation_has(AcqvaFeatureMapElevation_t* self, long nbins, long nrays);

/**
 * Sets the elevation angle for this field
 * @param[in] self - self
 * @param[in] elangle - the elevation angle of this group (in radians)
 * @return 1 if the elevation angle is possible to set
 */
int AcqvaFeatureMapField_setElangle(AcqvaFeatureMapField_t* self, double elangle);

/**
 * Gets the elevation angle for this field
 * @param[in] self - self
 * @return the elevation angle of this group (in radians)
 */
double AcqvaFeatureMapField_getElangle(AcqvaFeatureMapField_t* self);

/**
 * @param[in] self - self
 * @return number of bins
 */
long AcqvaFeatureMapField_getNbins(AcqvaFeatureMapField_t* self);

/**
 * @param[in] self - self
 * @return number of rays
 */
long AcqvaFeatureMapField_getNrays(AcqvaFeatureMapField_t* self);

/**
 * @param[in] self - self
 * @return data type
 */
RaveDataType AcqvaFeatureMapField_getDatatype(AcqvaFeatureMapField_t* self);

/**
 * Creates a data field with specified geometry and type and will be initialized to 0
 * @param[in] self - self
 * @param[in] nbins - number of bins
 * @param[in] nrays - number of rays
 * @param[in] type - data type
 * @return 1 on success otherwise 0
 */
int AcqvaFeatureMapField_createData(AcqvaFeatureMapField_t* self, long nbins, long nrays, RaveDataType type);

/**
 * Sets a data field with specified data, geometry and type
 */
int AcqvaFeatureMapField_setData(AcqvaFeatureMapField_t* self, long nbins, long nrays, void* data, RaveDataType type);

/**
 * Returns the data
 */
void* AcqvaFeatureMapField_getData(AcqvaFeatureMapField_t* self);

/**
 * Fills the complete array with wanted value.
 * @param[in] self - self
 * @param[in] value - the value to use for all cells
 * @return 1 on success otherwise 0
 */
int AcqvaFeatureMapField_fill(AcqvaFeatureMapField_t* self, double value);

/**
 * Sets a value in the data field.
 * @param[in] self - self
 * @param[in] bin - bin index
 * @param[in] ray - ray index
 * @param[in] v - the value to set
 * @return 1 on success otherwise 0
 */
int AcqvaFeatureMapField_setValue(AcqvaFeatureMapField_t* self, int bin, int ray, double v);

/**
 * Gets a value from the data field.
 * @param[in] self - self
 * @param[in] bin - bin index
 * @param[in] ray - ray index
 * @param[in,out] v - the value to retrieve (must be != NULL)
 * @return 1 on success otherwise 0
 */
int AcqvaFeatureMapField_getValue(AcqvaFeatureMapField_t* self, int bin, int ray, double* v);

/**
 * Creates a feature map field with wanted dimensions, type and elangle.
 */
AcqvaFeatureMapField_t* AcqvaFeatureMapField_createField(long nbins, long nrays, RaveDataType type, double elangle);

#endif