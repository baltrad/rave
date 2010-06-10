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
 * Defines the functions available when working with one parameter in a polar scan.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-01-22
 */
#ifndef POLARSCANPARAM_H
#define POLARSCANPARAM_H
#include "polarnav.h"
#include "projection.h"
#include "rave_object.h"
#include "rave_types.h"
#include "rave_attribute.h"
#include "rave_list.h"
#include "raveobject_list.h"

/**
 * Defines a Polar Scan Parameter
 */
typedef struct _PolarScanParam_t PolarScanParam_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType PolarScanParam_TYPE;

/**
 * Sets the quantity
 * @param[in] scanparam - self
 * @param[in] quantity - the quantity, e.g. DBZH
 * @returns 1 on success, otherwise 0
 */
int PolarScanParam_setQuantity(PolarScanParam_t* scanparam, const char* quantity);

/**
 * Returns the quantity
 * @param[in] scanparam - self
 * @return the quantity or NULL if not set
 */
const char* PolarScanParam_getQuantity(PolarScanParam_t* scanparam);

/**
 * Sets the gain
 * @param[in] scanparam - self
 * @param[in] gain - the gain
 */
void PolarScanParam_setGain(PolarScanParam_t* scanparam, double gain);

/**
 * Returns the gain
 * @param[in] scanparam - self
 * @return the gain
 */
double PolarScanParam_getGain(PolarScanParam_t* scanparam);

/**
 * Sets the offset
 * @param[in] scanparam - self
 * @param[in] offset - the offset
 */
void PolarScanParam_setOffset(PolarScanParam_t* scanparam, double offset);

/**
 * Returns the offset
 * @param[in] scanparam - self
 * @return the offset
 */
double PolarScanParam_getOffset(PolarScanParam_t* scanparam);

/**
 * Sets the nodata
 * @param[in] scanparam - self
 * @param[in] nodata - the nodata
 */
void PolarScanParam_setNodata(PolarScanParam_t* scanparam, double nodata);

/**
 * Returns the nodata
 * @param[in] scanparam - self
 * @return the nodata
 */
double PolarScanParam_getNodata(PolarScanParam_t* scanparam);

/**
 * Sets the undetect
 * @param[in] scanparam - self
 * @param[in] undetect - the undetect
 */
void PolarScanParam_setUndetect(PolarScanParam_t* scanparam, double undetect);

/**
 * Returns the undetect
 * @param[in] scanparam - self
 * @return the undetect
 */
double PolarScanParam_getUndetect(PolarScanParam_t* scanparam);

/**
 * Sets the data
 * @param[in] scanparam  - self
 * @param[in] nbins - number of bins
 * @param[in] nrays - number of rays
 * @param[in] data  - the data
 * @param[in] type  - the data type
 */
int PolarScanParam_setData(PolarScanParam_t* scanparam, long nbins, long nrays, void* data, RaveDataType type);

/**
 * Creates a data field with the specified dimensions and type. The data till be initialized to 0.
 * @param[in] scanparam - self
 * @param[in] nbins - number of bins
 * @param[in] nrays - number of rays
 * @param[in] type - the type of the data
 * @returns 1 on success otherwise 0
 */
int PolarScanParam_createData(PolarScanParam_t* scanparam, long nbins, long nrays, RaveDataType type);

/**
 * Returns a pointer to the internal data storage.
 * @param[in] scanparam - self
 * @return the internal data pointer (NOTE! Do not release this pointer)
 */
void* PolarScanParam_getData(PolarScanParam_t* scanparam);

/**
 * Returns the number of bins
 * @param[in] scanparam - self
 * @return the number of bins
 */
long PolarScanParam_getNbins(PolarScanParam_t* scanparam);

/**
 * Returns the number of rays/scan
 * @param[in] scanparam - self
 * @return the number of rays
 */
long PolarScanParam_getNrays(PolarScanParam_t* scanparam);

/**
 * Returns the data type
 * @param[in] scanparam - self
 * @return the data type
 */
RaveDataType PolarScanParam_getDataType(PolarScanParam_t* scan);

/**
 * Returns the value at the specified index.
 * @param[in] scanparam - self
 * @param[in] bin - the bin index
 * @param[in] ray - the ray index
 * @param[out] v - the data at the specified index
 * @return the type of data
 */
RaveValueType PolarScanParam_getValue(PolarScanParam_t* scanparam, int bin, int ray, double* v);

/**
 * Returns the linear converted value at the specified index. That is,
 * offset + gain * value;
 * @param[in] scanparam - self
 * @param[in] bin - the bin index
 * @param[in] ray - the ray index
 * @param[out] v - the data at the specified index
 * @return the type of data
 */
RaveValueType PolarScanParam_getConvertedValue(PolarScanParam_t* scanparam, int bin, int ray, double* v);

/**
 * Adds a rave attribute to the parameter.
 * @param[in] scanparam - self
 * @param[in] attribute - the attribute
 * @return 1 on success otherwise 0
 */
int PolarScanParam_addAttribute(PolarScanParam_t* scanparam,
  RaveAttribute_t* attribute);

/**
 * Returns the rave attribute that is named accordingly.
 * @param[in] scanparam - self
 * @param[in] name - the name of the attribute
 * @returns the attribute if found otherwise NULL
 */
RaveAttribute_t* PolarScanParam_getAttribute(PolarScanParam_t* scanparam,
  const char* name);

/**
 * Returns a list of attribute names. Release with \@ref #RaveList_freeAndDestroy.
 * @param[in] scanparam - self
 * @returns a list of attribute names
 */
RaveList_t* PolarScanParam_getAttributeNames(PolarScanParam_t* scanparam);

/**
 * Returns a list of attribute values that should be stored for this parameter. Corresponding
 * members will also be added as attribute values. E.g. gain will be stored
 * as a double with name what/gain.
 * @param[in] scanparam - self
 * @returns a list of RaveAttributes.
 */
RaveObjectList_t* PolarScanParam_getAttributeValues(PolarScanParam_t* scanparam);

#endif
