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
 * Used for keeping track on attributes.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-06-03
 */
#ifndef RAVE_ATTRIBUTE_H
#define RAVE_ATTRIBUTE_H
#include "rave_object.h"

typedef enum RaveAttribute_Format {
  RaveAttribute_Format_Undefined = -1, /**< Undefined */
  RaveAttribute_Format_String = 0,     /**< String */
  RaveAttribute_Format_Long = 1,       /**< Long */
  RaveAttribute_Format_Double = 2      /**< Double */
} RaveAttribute_Format;

/**
 * Defines a Geographical Area
 */
typedef struct _RaveAttribute_t RaveAttribute_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RaveAttribute_TYPE;

/**
 * Sets the name of this attribute
 * @param[in] attr - self
 * @param[in] name - the name this attribute should have
 * @returns 1 on success otherwise 0
 */
int RaveAttribute_setName(RaveAttribute_t* attr, const char* name);

/**
 * Returns the name of this attribute
 * @param[in] attr - self
 * @returns the name of this attribute
 */
const char* RaveAttribute_getName(RaveAttribute_t* attr);

/**
 * Returns the format for this attribute
 * @param[in] attr - self
 * @returns the format
 */
RaveAttribute_Format RaveAttribute_getFormat(RaveAttribute_t* attr);

/**
 * Sets the value as a long.
 * @param[in] attr - self
 * @param[in] value - the value
 */
void RaveAttribute_setLong(RaveAttribute_t* attr, long value);

/**
 * Sets the value as a double.
 * @param[in] attr - self
 * @param[in] value - the value
 */
void RaveAttribute_setDouble(RaveAttribute_t* attr, double value);

/**
 * Sets the value as a string
 * @param[in] attr - self
 * @param[in] value - the value
 * @return 1 on success otherwise 0
 */
int RaveAttribute_setString(RaveAttribute_t* attr, char* value);

/**
 * Returns the value as a long.
 * @param[in] attr - self
 * @param[out] value - the long value
 * @returns 1 on success or 0 if format of the data not is a long
 */
int RaveAttribute_getLong(RaveAttribute_t* attr, long* value);

/**
 * Returns the value as a double.
 * @param[in] attr - self
 * @param[out] value - the double value
 * @returns 1 on success or 0 if format of the data not is a double
 */
int RaveAttribute_getDouble(RaveAttribute_t* attr, double* value);

/**
 * Returns the value as a string.
 * @param[in] attr - self
 * @param[out] value - the internal 0-terminated string, DO NOT RELEASE memory
 * @returns 1 on success or 0 if format of the data not is a string
 */
int RaveAttribute_getString(RaveAttribute_t* attr, char** value);

/**
 * Helper function for extracting the group and name part from a
 * string with the format <group>/<name>.
 * @param[in] attrname - the string that should get group and name extracted
 * @param[out] group   - the group name (allocated memory so free it)
 * @param[out] name    - the attr name (allocated memory so free it)
 * @returns 1 on success otherwise 0
 */
int RaveAttributeHelp_extractGroupAndName(
  const char* attrname, char** group, char** name);

#endif /* RAVE_ATTRIBUTE_H */

