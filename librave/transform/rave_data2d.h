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
 * Represents a 2-dimensional data array.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-17
 */
#ifndef RAVE_DATA2D_H
#define RAVE_DATA2D_H
#include "rave_object.h"
#include "rave_types.h"

/**
 * Defines a Rave 2-dimensional data array
 */
typedef struct _RaveData2D_t RaveData2D_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RaveData2D_TYPE;

/**
 * Returns the xsize
 * @param[in] self - self
 * @return the xsize
 */
long RaveData2D_getXsize(RaveData2D_t* self);

/**
 * Returns the ysize
 * @param[in] self - self
 * @return the ysize
 */
long RaveData2D_getYsize(RaveData2D_t* self);

/**
 * Returns the data type
 * @param[in] self - self
 * @return the data type
 */
RaveDataType RaveData2D_getType(RaveData2D_t* self);

/**
 * Returns a pointer to the internal data storage.
 * @param[in] self - self
 * @return the internal data pointer (NOTE! Do not release this pointer)
 */
void* RaveData2D_getData(RaveData2D_t* self);

/**
 * Sets the data.
 * @param[in] self  - self
 * @param[in] xsize - x-size
 * @param[in] ysize - y-size
 * @param[in] data  - the data
 * @param[in] type  - the data type
 * @returns 1 on success, otherwise 0
 */
int RaveData2D_setData(RaveData2D_t* self, long xsize, long ysize, void* data, RaveDataType type);

/**
 * Creates a data field with the specified size and type.
 * @param[in] self  - self
 * @param[in] xsize - x-size
 * @param[in] ysize - y-size
 * @param[in] type  - the data type
 * @returns 1 on success, otherwise 0
 */
int RaveData2D_createData(RaveData2D_t* self, long xsize, long ysize, RaveDataType type);

/**
 * Sets the value at the specified coordinates.
 * @param[in] self - self
 * @param[in] x - the x-position
 * @param[in] y - the y-position
 * @param[in] v - the value to set
 * @return 1 if the value could be set, otherwise 0
 */
int RaveData2D_setValue(RaveData2D_t* self, long x, long y, double v);

/**
 * Same as RaveData2D_setValue but there is no boundary checking performed.
 * I.e. unless you know what you are doing you might be accessing unreserved memory.
 * @param[in] self - self
 * @param[in] x - the x-position
 * @param[in] y - the y-position
 * @param[in] v - the value to set
 * @return 1 if the value could be set, otherwise 0
 */
int RaveData2D_setValueUnchecked(RaveData2D_t* self, long x, long y, double v);

/**
 * Returns the value at the specified x and y position. If coordinates is outside
 * the boundaries, v will be left as is so initialize it before calling this function.
 * @param[in] self - self
 * @param[in] x - the x index
 * @param[in] y - the y index
 * @param[out] v - the data at the specified index
 * @return 1 if the value could be extracted and returned, otherwise 0
 */
int RaveData2D_getValue(RaveData2D_t* self, long x, long y, double* v);

/**
 * Same as RaveData2D_getValue but there is no boundary checking performed.
 * I.e. unless you know what you are doing you might be accessing unreserved memory.
 * @param[in] self - self
 * @param[in] x - the x index
 * @param[in] y - the y index
 * @param[out] v - the data at the specified index
 * @return 1 if the value could be extracted and returned, otherwise 0
 */
int RaveData2D_getValueUnchecked(RaveData2D_t* self, long x, long y, double* v);

/**
 * Returns if this object contains data and a xsize and ysize > 0.
 * @param[in] self - self
 * @returns 1 if this object contains data, otherwise 0.
 */
int RaveData2D_hasData(RaveData2D_t* self);

#endif /* RAVE_DATA2D_H */
