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

/**
 * Defines a Polar Scan
 */
typedef struct _Cartesian_t Cartesian_t;

/**
 * Creates a new cartesian instance
 * @return a new instance or NULL on failure
 */
Cartesian_t* Cartesian_new(void);

/**
 * Releases the responsibility for the scan, it is not certain that
 * it will be deleted though if there still are references existing
 * to this scan.
 * @param[in] cartesian - the cartesian product
 */
void Cartesian_release(Cartesian_t* cartesian);

/**
 * Copies the reference to this instance by increasing a
 * reference counter.
 * @param[in] cartesian - the cartesian product to be copied
 * @return a pointer to the cartesian product
 */
Cartesian_t* Cartesian_copy(Cartesian_t* cartesian);

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
 * Sets the data
 * @param[in] cartesian  - the cartesian product
 * @param[in] xsize - x-size
 * @param[in] ysize - y-size
 * @param[in] data  - the data
 * @param[in] type  - the data type
 */
int Cartesian_setData(Cartesian_t* cartesian, long xsize, long ysize, void* data, RaveDataType type);

/**
 * Function for keeping track of some sort of metadata that should be
 * associated with this scan.
 * @param[in] cartesian - the cartesian product
 * @param[in] ptr - a void pointer that should be stored
 */
void Cartesian_setVoidPtr(Cartesian_t* cartesian, void* ptr);

/**
 * Returns the void data.
 * @param[in] cartesian - the cartesian product
 * @return the void data
 */
void* Cartesian_getVoidPtr(Cartesian_t* cartesian);
#endif
