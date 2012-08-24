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
 * Defines the functions available when working with vertical profiles.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2012-08-24
 */
#ifndef VERTICALPROFILE_H
#define VERTICALPROFILE_H
#include "rave_object.h"
#include "rave_types.h"
#include "rave_field.h"

/**
 * Defines a Vertical Profile
 */
typedef struct _VerticalProfile_t VerticalProfile_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType VerticalProfile_TYPE;

/**
 * Sets the nominal time.
 * @param[in] self - self
 * @param[in] value - the time in the format HHmmss
 * @returns 1 on success, otherwise 0
 */
int VerticalProfile_setTime(VerticalProfile_t* self, const char* value);

/**
 * Returns the nominal time.
 * @param[in] self - self
 * @returns the nominal time (or NULL if there is none)
 */
const char* VerticalProfile_getTime(VerticalProfile_t* self);

/**
 * Sets the nominal date.
 * @param[in] self - self
 * @param[in] value - the date in the format YYYYMMDD
 * @returns 1 on success, otherwise 0
 */
int VerticalProfile_setDate(VerticalProfile_t* self, const char* value);

/**
 * Returns the nominal date.
 * @param[in] scan - self
 * @returns the nominal time (or NULL if there is none)
 */
const char* VerticalProfile_getDate(VerticalProfile_t* self);

/**
 * Sets the source.
 * @param[in] self - self
 * @param[in] value - the source
 * @returns 1 on success, otherwise 0
 */
int VerticalProfile_setSource(VerticalProfile_t* self, const char* value);

/**
 * Returns the source.
 * @param[in] self - self
 * @returns the source or NULL if there is none
 */
const char* VerticalProfile_getSource(VerticalProfile_t* self);

/**
 * Sets the longitude
 * @param[in] self - self
 * @param[in] lon - the longitude (in radians)
 */
void VerticalProfile_setLongitude(VerticalProfile_t* self, double lon);

/**
 * Returns the longitude
 * @param[in] self - self
 * @return the longitude in radians
 */
double VerticalProfile_getLongitude(VerticalProfile_t* self);

/**
 * Sets the latitude
 * @param[in] self - self
 * @param[in] lat - the latitude (in radians)
 */
void VerticalProfile_setLatitude(VerticalProfile_t* self, double lat);

/**
 * Returns the latitude
 * @param[in] self - self
 * @return the latitude in radians
 */
double VerticalProfile_getLatitude(VerticalProfile_t* self);

/**
 * Sets the height of the centre of the antenna
 * @param[in] self - self
 * @param[in] height - the height (in meters)
 */
void VerticalProfile_setHeight(VerticalProfile_t* self, double h);

/**
 * Returns the height of the centre of the antenna
 * @param[in] self - self
 * @return the height
 */
double VerticalProfile_getHeight(VerticalProfile_t* self);

/**
 * Sets the number of levels in the profile
 * @param[in] self - self
 * @param[in] levels - the number of levels
 * @return 1 on success or 0 on failure
 */
int VerticalProfile_setLevels(VerticalProfile_t* self, long l);

/**
 * Returns the number of levels in the profile
 * @param[in] self - self
 * @return the number of levels
 */
long VerticalProfile_getLevels(VerticalProfile_t* self);

/**
 * Sets the vertical distance (m) between height intervals, or 0.0 if variable
 * @param[in] self - self
 * @param[in] i - the interval (in meters)
 */
void VerticalProfile_setInterval(VerticalProfile_t* self, double i);

/**
 * Returns the vertical distance (m) between height intervals, or 0.0 if variable
 * @param[in] self - self
 * @return the interval
 */
double VerticalProfile_getInterval(VerticalProfile_t* self);

/**
 * Sets the minimum height in meters above mean sea level
 * @param[in] self - self
 * @param[in] h - the height (in meters)
 */
void VerticalProfile_setMinheight(VerticalProfile_t* self, double h);

/**
 * Returns the minimum height in meters above mean sea level
 * @param[in] self - self
 * @return the interval
 */
double VerticalProfile_getMinheight(VerticalProfile_t* self);

/**
 * Sets the maximum height in meters above mean sea level
 * @param[in] self - self
 * @param[in] h - the height (in meters)
 */
void VerticalProfile_setMaxheight(VerticalProfile_t* self, double h);

/**
 * Returns the maximum height in meters above mean sea level
 * @param[in] self - self
 * @return the interval
 */
double VerticalProfile_getMaxheight(VerticalProfile_t* self);

/**
 * Returns the Mean horizontal wind velocity (m/s)
 * @param[in] self - self
 * @return the mean horizontal wind velocity
 */
RaveField_t* VerticalProfile_getFF(VerticalProfile_t* self);

/**
 * Sets the Mean horizontal wind velocity (m/s)
 * @param[in] self - self
 * @param[in] ff - ff (must be a 1 dimensional field with same dim as the other members).
 * @return 1 on success otherwise 0
 */
int VerticalProfile_setFF(VerticalProfile_t* self, RaveField_t* ff);

#endif
