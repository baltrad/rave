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
 * Defines the functions available when working with cartesian products.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 *
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-16
 */
#ifndef CARTESIAN_H
#define CARTESIAN_H
#include "rave_transform.h"
#include "projection.h"
#include "area.h"
#include "rave_object.h"
#include "rave_types.h"
#include "rave_list.h"
#include "raveobject_list.h"
#include "rave_attribute.h"

/**
 * Defines a Cartesian product
 */
typedef struct _Cartesian_t Cartesian_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType Cartesian_TYPE;

/**
 * Sets the nominal time.
 * @param[in] cartesian - self
 * @param[in] value - the time in the format HHmmss
 * @returns 1 on success, otherwise 0
 */
int Cartesian_setTime(Cartesian_t* cartesian, const char* value);

/**
 * Returns the nominal time.
 * @param[in] cartesian - self
 * @returns the nominal time (or NULL if there is none)
 */
const char* Cartesian_getTime(Cartesian_t* cartesian);

/**
 * Sets the nominal date.
 * @param[in] cartesian - self
 * @param[in] value - the date in the format YYYYMMDD
 * @returns 1 on success, otherwise 0
 */
int Cartesian_setDate(Cartesian_t* cartesian, const char* value);

/**
 * Returns the nominal date.
 * @param[in] cartesian - self
 * @returns the nominal time (or NULL if there is none)
 */
const char* Cartesian_getDate(Cartesian_t* cartesian);

/**
 * Sets the source.
 * @param[in] cartesian - self
 * @param[in] value - the source
 * @returns 1 on success, otherwise 0
 */
int Cartesian_setSource(Cartesian_t* cartesian, const char* value);

/**
 * Returns the source.
 * @param[in] cartesian - self
 * @returns the source or NULL if there is none
 */
const char* Cartesian_getSource(Cartesian_t* cartesian);

/**
 * Sets the object type this cartesian product should represent.
 * @param[in] cartesian - self
 * @param[in] type - the object type
 * @returns 1 if the specified object type is supported, otherwise 0
 */
int Cartesian_setObjectType(Cartesian_t* cartesian, Rave_ObjectType type);

/**
 * Returns the object type this cartesian product represents.
 * @param[in] cartesian - self
 * @returns the object type
 */
Rave_ObjectType Cartesian_getObjectType(Cartesian_t* cartesian);

/**
 * Returns the xsize
 * @param[in] cartesian - the cartesian product
 * @return the xsize
 */
long Cartesian_getXSize(Cartesian_t* cartesian);

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
 * Sets the product this cartesian represents.
 * @param[in] cartesian  self
 * @param[in] type - the product type
 * @returns 1 if the operation was successful, otherwise 0
 */
int Cartesian_setProduct(Cartesian_t* cartesian, Rave_ProductType type);

/**
 * Returns the product this cartesian represents.
 * @param[in] cartesian - self
 * @returns the product type
 */
Rave_ProductType Cartesian_getProduct(Cartesian_t* cartesian);

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
 * Returns the x index
 * Evaluated as: (x - lowerLeft.x)/xscale
 * @param[in] cartesian - the cartesian product
 * @param[in] x - the x position in the area definition
 * @returns the x index
 */
long Cartesian_getIndexX(Cartesian_t* cartesian, double x);

/**
 * Returns the y index
 * Evaluated as: (upperRight.y - y)/yscale
 * @param[in] cartesian - the cartesian product
 * @param[in] y - the y position in the area definition
 * @returns the y index
 */
long Cartesian_getIndexY(Cartesian_t* cartesian, double y);

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
 * @returns 1 on success, otherwise 0
 */
int Cartesian_setQuantity(Cartesian_t* cartesian, const char* quantity);

/**
 * Returns the quantity
 * @param[in] cartesian - the cartesian product
 * @return the quantity
 */
const char* Cartesian_getQuantity(Cartesian_t* cartesian);

/**
 * Sets the gain.
 * @param[in] cartesian - the cartesian product
 * @param[in] gain - the gain (MAY NOT BE 0.0)
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
 * Scales the value v according to gain and offset before setting it.
 * I.e. same as Cartesian_setValue(cartesian, x, y, (v - offset)/gain)
 */
int Cartesian_setConvertedValue(Cartesian_t* cartesian, long x, long y, double v);

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
 * Returns the converted value at the specified x and y position.
 * @param[in] cartesian - the cartesian product
 * @param[in] x - the x index
 * @param[in] y - the y index
 * @param[out] v - the data at the specified index
 * @return the type of data
 */
RaveValueType Cartesian_getConvertedValue(Cartesian_t* cartesian, long x, long y, double* v);

/**
 * Initializes a cartesian product with values from the area.
 * @param[in] cartesian - self
 * @param[in] area - the area
 * @param[in] datatype - the type of data
 * @returns 1 on success, otherwise 0
 */
int Cartesian_init(Cartesian_t* cartesian, Area_t* area, RaveDataType datatype);

/**
 * Returns the mean value over a NxN square around the specified x and y position.
 * @param[in] cartesian - the cartesian product
 * @param[in] x - the x index
 * @param[in] y - the y index
 * @param[in] N - the N size
 * @param[out] v - the data at the specified index
 * @return the type of data
 */
RaveValueType Cartesian_getMean(Cartesian_t* cartesian, long x, long y, int N, double* v);

/**
 * Verifies that all preconditions are met in order to perform
 * a transformation.
 * @param[in] cartesian - the cartesian product
 * @returns 1 if the cartesian product is ready, otherwise 0.
 */
int Cartesian_isTransformable(Cartesian_t* cartesian);

/**
 * Adds a rave attribute to the cartesian product. If attribute maps to the
 * member attributes it will be used to set the specific member
 * instead.
 * @param[in] cartesian - self
 * @param[in] attribute - the attribute
 * @return 1 on success otherwise 0
 */
int Cartesian_addAttribute(Cartesian_t* cartesian, RaveAttribute_t* attribute);

/**
 * Returns the rave attribute that is named accordingly.
 * @param[in] cartesian - self
 * @param[in] name - the name of the attribute
 * @returns the attribute if found otherwise NULL
 */
RaveAttribute_t* Cartesian_getAttribute(Cartesian_t* cartesian, const char* name);

/**
 * Returns a list of attribute names. Release with \@ref #RaveList_freeAndDestroy.
 * @param[in] cartesian - self
 * @returns a list of attribute names
 */
RaveList_t* Cartesian_getAttributeNames(Cartesian_t* cartesian);

/**
 * Returns a list of attribute values that should be stored for this cartesian product.
 * Corresponding members will also be added as attribute values.
 * @param[in] cartesian - self
 * @param[in] otype - what type of attributes that should be returned, if it is for a cartesian image or a image
 * belonging to a cartesian volume
 * @returns a list of RaveAttributes.
 */
RaveObjectList_t* Cartesian_getAttributeValues(Cartesian_t* cartesian);

/**
 * Returns if the cartesian product has got the specified attribute.
 * @param[in] cartesian - self
 * @param[in] name - what to look for
 * @returns 1 if the attribute exists, otherwise 0
 */
int Cartesian_hasAttribute(Cartesian_t* cartesian, const char* name);

/**
 * Validates if the cartesian is a valid cartesian product in the means
 * of storing it.
 * @param[in] cartesian - self
 * @param[in] otype - what this cartesian belongs to, e.g IMAGE for self contained
 * and CVOL for a member of a volume.
 * @returns 1 if cartesian is valid, otherwise 0
 */
int Cartesian_isValid(Cartesian_t* cartesian, Rave_ObjectType otype);

/**
 * Adds the lon lat corner extent to the attribute list. If llX, llY, urX and urY are all 0.0, then
 * nothing will be added to the attribute list.
 * @param[in] list - the list to add the attributes to
 * @param[in] projection - the projection to use for converting to corner coordinates
 * @param[in] llX - the lower left X coordinate
 * @param[in] llY - the lower left Y coordinate
 * @param[in] urX - the upper right X coordinate
 * @param[in] urY - the upper right Y coordinate
 * @returns 1 on success otherwise 0
 */
int CartesianHelper_addLonLatExtentToAttributeList(RaveObjectList_t* list, Projection_t* projection, double llX, double llY, double urX, double urY);

#endif
