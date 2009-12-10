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
 * Defines the functions available when working with cartesian products
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-16
 */
#ifndef CARTESIAN_H
#define CARTESIAN_H
#include "rave_transform.h"
#include "projection.h"
#include "rave_object.h"

/**
 * Defines a Cartesian product
 */
typedef struct _Cartesian_t Cartesian_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType Cartesian_TYPE;

/**
 * Sets the xsize
 * @param[in] cartesian - the cartesian product
 * @param[in] xsize - the xsize
 */
void Cartesian_setXSize(Cartesian_t* cartesian, long xsize);

/**
 * Returns the xsize
 * @param[in] cartesian - the cartesian product
 * @return the xsize
 */
long Cartesian_getXSize(Cartesian_t* cartesian);

/**
 * Sets the ysize
 * @param[in] cartesian - the cartesian product
 * @param[in] ysize - the ysize
 */
void Cartesian_setYSize(Cartesian_t* cartesian, long ysize);

/**
 * Returns the ysize
 * @param[in] cartesian - the cartesian product
 * @return the ysize
 */
long Cartesian_getYSize(Cartesian_t* cartesian);

/**
 * Sets the area extent for this cartesian product.
 * @param[in] cartesian - the cartesian product
 * @param[in] llX - lower left X position
 * @param[in] llY - lower left Y position
 * @param[in] urX - upper right X position
 * @param[in] urY - upper right Y position
 */
void Cartesian_setAreaExtent(Cartesian_t* cartesian, double llX, double llY, double urX, double urY);

/**
 * Gets the area extent for this cartesian product.
 * @param[in] cartesian - the cartesian product
 * @param[out] llX - lower left X position (may be NULL)
 * @param[out] llY - lower left Y position (may be NULL)
 * @param[out] urX - upper right X position (may be NULL)
 * @param[out] urY - upper right Y position (may be NULL)
 */
void Cartesian_getAreaExtent(Cartesian_t* cartesian, double* llX, double* llY, double* urX, double* urY);

/**
 * Sets the xscale
 * @param[in] cartesian - the cartesian product
 * @param[in] xscale - the xscale
 */
void Cartesian_setXScale(Cartesian_t* cartesian, double xscale);

/**
 * Returns the xscale
 * @param[in] cartesian - the cartesian product
 * @return the xscale
 */
double Cartesian_getXScale(Cartesian_t* cartesian);

/**
 * Sets the yscale
 * @param[in] cartesian - the cartesian product
 * @param[in] yscale - the yscale
 */
void Cartesian_setYScale(Cartesian_t* cartesian, double yscale);

/**
 * Returns the yscale
 * @param[in] cartesian - the cartesian product
 * @return the yscale
 */
double Cartesian_getYScale(Cartesian_t* cartesian);

/**
 * Returns the location within the area as identified by a x-position.
 * Evaluated as: upperLeft.x + xscale * x
 * @param[in] cartesian - the cartesian product
 * @param[in] x - the x position in the area definition
 * @returns the x location
 */
double Cartesian_getLocationX(Cartesian_t* cartesian, long x);

/**
 * Returns the location within the area as identified by a y-position.
 * Evaluated as: upperLeft.y - yscale * y
 * @param[in] cartesian - the cartesian product
 * @param[in] y - the y position in the area definition
 * @returns the y location
 */
double Cartesian_getLocationY(Cartesian_t* cartesian, long y);

/**
 * Sets the data type of the data that is worked with
 * @param[in] cartesian - the cartesian product
 * @param[in] type - the data type
 * @return 0 if type is not known, otherwise the type was set
 */
int Cartesian_setDataType(Cartesian_t* cartesian, RaveDataType type);

/**
 * Returns the data type
 * @param[in] cartesian - the cartesian product
 * @return the data type
 */
RaveDataType Cartesian_getDataType(Cartesian_t* cartesian);

/**
 * Sets the quantity
 * @param[in] cartesian - the cartesian product
 * @param[in] quantity - the quantity, e.g. DBZH
 */
void Cartesian_setQuantity(Cartesian_t* cartesian, const char* quantity);

/**
 * Returns the quantity
 * @param[in] cartesian - the cartesian product
 * @return the quantity
 */
const char* Cartesian_getQuantity(Cartesian_t* cartesian);

/**
 * Sets the gain
 * @param[in] cartesian - the cartesian product
 * @param[in] gain - the gain
 */
void Cartesian_setGain(Cartesian_t* cartesian, double gain);

/**
 * Returns the gain
 * @param[in] cartesian - the cartesian product
 * @return the gain
 */
double Cartesian_getGain(Cartesian_t* cartesian);

/**
 * Sets the offset
 * @param[in] cartesian - the cartesian product
 * @param[in] offset - the offset
 */
void Cartesian_setOffset(Cartesian_t* cartesian, double offset);

/**
 * Returns the offset
 * @param[in] cartesian - the cartesian product
 * @return the offset
 */
double Cartesian_getOffset(Cartesian_t* cartesian);

/**
 * Sets the nodata
 * @param[in] cartesian - the cartesian product
 * @param[in] nodata - the nodata
 */
void Cartesian_setNodata(Cartesian_t* cartesian, double nodata);

/**
 * Returns the nodata
 * @param[in] cartesian - the cartesian product
 * @return the nodata
 */
double Cartesian_getNodata(Cartesian_t* cartesian);

/**
 * Sets the undetect
 * @param[in] cartesian - the cartesian product
 * @param[in] undetect - the undetect
 */
void Cartesian_setUndetect(Cartesian_t* cartesian, double undetect);

/**
 * Returns the undetect
 * @param[in] cartesian - the cartesian product
 * @return the undetect
 */
double Cartesian_getUndetect(Cartesian_t* cartesian);

/**
 * Sets the projection that defines this cartesian product.
 * @param[in] cartesian - the cartesian product
 * @param[in] projection - the projection
 */
void Cartesian_setProjection(Cartesian_t* cartesian, Projection_t* projection);

/**
 * Returns a copy of the projection that is used for this cartesian product.
 * I.e. remember to release it.
 * @param[in] cartesian - the cartesian product
 * @returns a projection (or NULL if none is set)
 */
Projection_t* Cartesian_getProjection(Cartesian_t* cartesian);

/**
 * Sets the data
 * @param[in] cartesian  - the cartesian product
 * @param[in] xsize - x-size
 * @param[in] ysize - y-size
 * @param[in] data  - the data
 * @param[in] type  - the data type
 */
int Cartesian_setData(Cartesian_t* cartesian, long xsize, long ysize, void* data, RaveDataType type);

/**
 * Returns a pointer to the internal data storage.
 * @param[in] cartesian - the cartesian product
 * @return the internal data pointer (NOTE! Do not release this pointer)
 */
void* Cartesian_getData(Cartesian_t* cartesian);

/**
 * Sets the value at the specified coordinates.
 * @param[in] cartesian - the cartesian product
 * @param[in] x - the x-position
 * @param[in] y - the y-position
 * @param[in] v - the value to set
 * @return 1 on success, otherwise 0
 */
int Cartesian_setValue(Cartesian_t* cartesian, long x, long y, double v);

/**
 * Returns the value at the specified x and y position.
 * @param[in] cartesian - the cartesian product
 * @param[in] x - the x index
 * @param[in] y - the y index
 * @param[out] v - the data at the specified index
 * @return the type of data
 */
RaveValueType Cartesian_getValue(Cartesian_t* cartesian, long x, long y, double* v);

/**
 * Verifies that all preconditions are met in order to perform
 * a transformation.
 * @param[in] cartesian - the cartesian product
 * @returns 1 if the cartesian product is ready, otherwise 0.
 */
int Cartesian_isTransformable(Cartesian_t* cartesian);

/**
 * Enables/disables debugging.
 * @param[in] debug - 0 means no debugging (default) otherwise debugging is enabled.
 */
void Cartesian_setDebug(Cartesian_t* cartesian, int debug);
#endif
