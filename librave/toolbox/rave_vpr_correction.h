/* --------------------------------------------------------------------
Copyright (C) 2012 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Implementation of the vpr correction functionality
 * @file
 * @author Lars Norin Original code - (Swedish Meteorological and Hydrological Institute, SMHI)
 * @author Anders Henja Adapted to rave fwk (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2013-11-19
 */
#ifndef RAVE_VPR_CORRECTION_H
#define RAVE_VPR_CORRECTION_H
#include "rave_object.h"
#include "polarvolume.h"

/**
 * Defines an algorithm for vpr correction
 */
typedef struct _RaveVprCorrection_t RaveVprCorrection_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RaveVprCorrection_TYPE;

/**
 * Sets the min limit for when the reflectivity should be seen as stratiform. All values
 * about this limit will be defined to be stratiform
 * @param[in] self - self
 * @param[in] limit - the min reflectivity for stratiform rain
 */
void RaveVprCorrection_setMinZLimitStratiformCloud(RaveVprCorrection_t* self, double limit);

/**
 * Returns the min limit for when the reflectivity should be seen as stratiform. All values
 * about this limit will be defined to be stratiform
 * @param[in] self - self
 * @returns - the min reflectivity for stratiform rain
 */
double RaveVprCorrection_getMinZLimitStratiformCloud(RaveVprCorrection_t* self);

/**
 * Sets the min reflectivity to be used when generating the vpr profile
 * @param[in] self - self
 * @param[in] limit - the min reflectivity
 */
void RaveVprCorrection_setMinReflectivity(RaveVprCorrection_t* self, double limit);

/**
 * Returns the min reflectivity to be used when generating the vpr profile
 * @param[in] self - self
 * @return the min reflectivity
 */
double RaveVprCorrection_getMinReflectivity(RaveVprCorrection_t* self);

/**
 * Sets the height limit for what reflectivities should be used
 * @param[in] self - self
 * @param[in] limit - the max height for when reflectivities should be used in the identification
 * of stratiform and convective rain
 */
void RaveVprCorrection_setHeightLimit(RaveVprCorrection_t* self, double limit);

/**
 * Returns the height limit for what reflectivities should be used in the identification of stratiform and convective rain
 * @param[in] self - self
 * @returns the height limit
 */
double RaveVprCorrection_getHeightLimit(RaveVprCorrection_t* self);

/**
 * Sets the height of the individual profile bins. The resolution of the reflectivity profile will be heightLimit / profileHeight.
 * @param[in] self - self
 * @param[in] height - the bin height
 * @return 0 if trying to set a height = 0.0. Otherwise 1
 */
int RaveVprCorrection_setProfileHeight(RaveVprCorrection_t* self, double height);

/**
 * Gets the height of the individual profile bins.
 * @param[in] self - self
 * @return the bin height
 */
double RaveVprCorrection_getProfileHeight(RaveVprCorrection_t* self);

/**
 * Sets the min distance limit for what reflectivities should be used
 * @param[in] self - self
 * @param[in] limit - the min distance for when reflectivities should be used in the identification
 * of stratiform and convective rain
 */
void RaveVprCorrection_setMinDistance(RaveVprCorrection_t* self, double limit);

/**
 * Returns the min distance limit for what reflectivities should be used in the identification of stratiform and convective rain
 * @param[in] self - self
 * @returns the min distance limit
 */
double RaveVprCorrection_getMinDistance(RaveVprCorrection_t* self);

/**
 * Sets the max distance limit for what reflectivities should be used
 * @param[in] self - self
 * @param[in] limit - the max distance for when reflectivities should be used in the identification
 * of stratiform and convective rain
 */
void RaveVprCorrection_setMaxDistance(RaveVprCorrection_t* self, double limit);

/**
 * Returns the max distance limit for what reflectivities should be used in the identification of stratiform and convective rain
 * @param[in] self - self
 * @returns the max distance limit
 */
double RaveVprCorrection_getMaxDistance(RaveVprCorrection_t* self);

/**
 * Separates stratiform and convective rain
 * @param[in] self - self
 * @param[in] pvol - the polar volume
 * @return 1 on success otherwise 0
 */
PolarVolume_t* RaveVprCorrection_separateSC(RaveVprCorrection_t* self, PolarVolume_t* pvol);

/**
 * Returns the mean reflectivity array from the sc settings.
 * @param[in] self - self
 * @param[in] pvol - the polar volume from where the reflectivity array should be extracted
 * @param[in,out] nElements - the size of the returned array
 * @return an array containing the mean reflectivities for each profile height bin
 */
double* RaveVprCorrection_getReflectivityArray(RaveVprCorrection_t* self, PolarVolume_t* pvol, int* nElements);

#endif /* RAVE_VPR_CORRECTION_H */
