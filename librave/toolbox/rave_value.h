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
 * A value object that can represent standard types like int, double, .. It can also contain arrays, lists and hashtables
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-02-14
 */
#ifndef RAVE_VALUE_H
#define RAVE_VALUE_H
#include "rave_object.h"
#include <stdarg.h>

typedef enum RaveValue_Type {
  RaveValue_Type_Undefined = -1,  /**< Undefined */
  RaveValue_Type_String = 0,      /**< String */
  RaveValue_Type_Long = 1,        /**< Long */
  RaveValue_Type_Double = 2,      /**< Double */
  RaveValue_Type_StringArray = 3, /**< Simple 1-dimensional array of strings */
  RaveValue_Type_LongArray = 4,   /**< Simple 1-dimensional array of longs */
  RaveValue_Type_DoubleArray = 5, /**< Simple 1-dimensional array of doubles */
  RaveValue_Type_Data2D = 6,      /**< 2D array */
  RaveValue_Type_Hashtable = 7    /**< Hash table */
} RaveValue_Type;

/**
 * Defines a rave value
 */
typedef struct _RaveValue_t RaveValue_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RaveValue_TYPE;

/**
 * Returns the format for this attribute
 * @param[in] self - self
 * @returns the format
 */
RaveValue_Type RaveValue_type(RaveValue_t* self);

/**
 * Resets the value object
 * @param[in] self - self
 */
void RaveValue_reset(RaveValue_t* self);

/**
 * Creates a string rave value.
 * @param[in] value - the string
 * @returns the value on success otherwise NULL
 */
RaveValue_t* RaveValue_createString(const char* value);

/**
 * Sets a string value in self
 * @param[in] self - self
 * @param[in] value - the value
 * @return 1 on success otherwise 0
 */
int RaveValue_setString(RaveValue_t* self, const char* value);

/**
 * Returns the value as a string.
 * @param[in] self - self
 * @param[out] value - the internal 0-terminated string, DO NOT RELEASE memory
 * @returns 1 on success or 0 if format of the data not is a string
 */
int RaveAttribute_getString(RaveValue_t* self, char** value);

/**
 * Returns the string value. NOTE! It up to user to ensure that value actually is a string otherwise behavior will be undefined. 
 * @param[in] self - self
 * @return the string value or NULL
 */
const char* RaveValue_toString(RaveValue_t* self);

/**
 * Creates a long rave value.
 * @param[in] value - the long
 * @returns the value on success otherwise NULL
 */
RaveValue_t* RaveValue_createLong(long value);

/**
 * Sets the value as a long.
 * @param[in] self - self
 * @param[in] value - the value
 */
void RaveValue_setLong(RaveValue_t* self, long value);

/**
 * Returns the value as a long.
 * @param[in] self - self
 * @param[out] value - the long value
 * @returns 1 on success or 0 if format of the data not is a long
 */
int RaveValue_getLong(RaveValue_t* self, long* value);

/**
 * Returns the long value. NOTE! It up to user to ensure that value actually is a long otherwise behavior will be undefined. 
 * @param[in] self - self
 * @return the long value or 0
 */
long RaveValue_toLong(RaveValue_t* self);


/**
 * Creates a double rave value.
 * @param[in] value - the double
 * @returns the value on success otherwise NULL
 */
RaveValue_t* RaveValue_createDouble(double value);

/**
 * Sets the value as a double.
 * @param[in] self - self
 * @param[in] value - the value
 */
void RaveValue_setDouble(RaveValue_t* self, double value);

/**
 * Returns the value as a double.
 * @param[in] self - self
 * @param[out] value - the double value
 * @returns 1 on success or 0 if format of the data not is a double
 */
int RaveValue_getDouble(RaveValue_t* self, double* value);

/**
 * Returns the double value. NOTE! It up to user to ensure that value actually is a double otherwise behavior will be undefined. 
 * @param[in] self - self
 * @return the double value or -
 */
double RaveValue_toDouble(RaveValue_t* self);

/**
 * Creates a string array rave value.
 * @param[in] value - the string array
 * @param[in] len - the length of the array
 * @returns the value on success otherwise NULL
 */
RaveValue_t* RaveValue_createStringArray(const char** value, int len);

/**
 * Sets the value as a simple 1-dimensional double array.
 * @param[in] self - self
 * @param[in] value - the value
 * @param[in] len - the number of doubles in the array
 * @returns 1 on success otherwise 0
 */
int RaveValue_setStringArray(RaveValue_t* self, const char** value, int len);

/**
 * Returns the value as a string array.
 * @param[in] self - self
 * @param[out] value - the internal string array, DO NOT RELEASE memory
 * @param[out] len - the number of values in the array
 * @returns 1 on success or 0 if format of the data not is a double array
 */
int RaveValue_getStringArray(RaveValue_t* self, char*** value, int* len);

/**
 * Creates a long array rave value.
 * @param[in] value - the long array
 * @param[in] len - the length of the array
 * @returns the value on success otherwise NULL
 */
RaveValue_t* RaveValue_createLongArray(long* value, int len);

/**
 * Sets the value as a simple 1-dimensional long array.
 * @param[in] self - self
 * @param[in] value - the value
 * @param[in] len - the number of longs in the array
 * @returns 1 on success otherwise 0
 */
int RaveValue_setLongArray(RaveValue_t* self, long* value, int len);

/**
 * Returns the value as a long array.
 * @param[in] self - self
 * @param[out] value - the internal long array, DO NOT RELEASE memory
 * @param[out] len - the number of values in the array
 * @returns 1 on success or 0 if format of the data not is a long array
 */
int RaveValue_getLongArray(RaveValue_t* self, long** value, int* len);

/**
 * Creates a double array rave value.
 * @param[in] value - the double array
 * @param[in] len - the length of the array
 * @returns the value on success otherwise NULL
 */
RaveValue_t* RaveValue_createDoubleArray(double* value, int len);

/**
 * Sets the value as a simple 1-dimensional double array.
 * @param[in] self - self
 * @param[in] value - the value
 * @param[in] len - the number of doubles in the array
 * @returns 1 on success otherwise 0
 */
int RaveValue_setDoubleArray(RaveValue_t* self, double* value, int len);

/**
 * Returns the value as a double array.
 * @param[in] self - self
 * @param[out] value - the internal double array, DO NOT RELEASE memory
 * @param[out] len - the number of values in the array
 * @returns 1 on success or 0 if format of the data not is a double array
 */
int RaveValue_getDoubleArray(RaveValue_t* self, double** value, int* len);

#endif /* RAVE_VALUE_H */

