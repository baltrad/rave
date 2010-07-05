/* --------------------------------------------------------------------
Copyright (C) 2009-2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Generic field that only provides a 2-dim data field and a number of dynamic
 * attributes. This object supports cloning.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-07-05
 */
#ifndef RAVE_FIELD_H
#define RAVE_FIELD_H
#include "rave_object.h"
#include "rave_types.h"
#include "rave_attribute.h"
#include "rave_list.h"
#include "raveobject_list.h"
#include "rave_data2d.h"

/**
 * Defines a Rave field
 */
typedef struct _RaveField_t RaveField_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RaveField_TYPE;

/**
 * Sets the data in the rave field.
 * @param[in] field - self
 * @param[in] xsize - the xsize
 * @param[in] ysize - the ysize
 * @param[in] data - the data
 * @param[in] type - the data type
 * @returns 1 on success otherwise 0
 */
int RaveField_setData(RaveField_t* field, long xsize, long ysize, void* data, RaveDataType type);

/**
 * Creates a empty data field
 * @param[in] field - self
 * @param[in] xsize - the xsize
 * @param[in] ysize - the ysize
 * @param[in] type - the data type
 * @returns 1 on success otherwise 0
 */
int RaveField_createData(RaveField_t* field, long xsize, long ysize, RaveDataType type);

/**
 * Returns a pointer to the internal data storage.
 * @param[in] field - self
 * @return the internal data pointer (NOTE! Do not release this pointer)
 */
void* RaveField_getData(RaveField_t* field);

/**
 * Returns the value at the specified index.
 * @param[in] field - self
 * @param[in] x - the x-pos / bin index
 * @param[in] y - the y-pos / ray index
 * @param[out] v - the data at the specified index
 * @return the type of data
 */
RaveValueType RaveField_getValue(RaveField_t* field, long x, long y, double* v);

/**
 * Sets the value at specified position
 * @param[in] field - self
 * @param[in] x - x coordinate
 * @param[in] y - y coordinate
 * @param[in] value - the value to be set at specified coordinate
 */
int RaveField_setValue(RaveField_t* field, long x, long y, double value);

/**
 * Returns the xsize / number of bins
 * @param[in] field - self
 * @return the xsize / number of bins
 */
long RaveField_getXsize(RaveField_t* field);

/**
 * Returns the ysize / number of rays
 * @param[in] field - self
 * @return the ysize / number of rays
 */
long RaveField_getYsize(RaveField_t* field);

/**
 * Returns the data type
 * @param[in] field - self
 * @return the data type
 */
RaveDataType RaveField_getDataType(RaveField_t* field);

/**
 * Adds a rave attribute to the parameter.
 * @param[in] field - self
 * @param[in] attribute - the attribute
 * @return 1 on success otherwise 0
 */
int RaveField_addAttribute(RaveField_t* field,  RaveAttribute_t* attribute);

/**
 * Returns the rave attribute that is named accordingly.
 * @param[in] field - self
 * @param[in] name - the name of the attribute
 * @returns the attribute if found otherwise NULL
 */
RaveAttribute_t* RaveField_getAttribute(RaveField_t* field, const char* name);

/**
 * Returns a list of attribute names. Release with \@ref #RaveList_freeAndDestroy.
 * @param[in] field - self
 * @returns a list of attribute names
 */
RaveList_t* RaveField_getAttributeNames(RaveField_t* field);

/**
 * Returns a list of attribute values that should be stored for this parameter.
 * @param[in] field - self
 * @returns a list of RaveAttributes.
 */
RaveObjectList_t* RaveField_getAttributeValues(RaveField_t* field);

#endif /* RAVE_FIELD_H */
