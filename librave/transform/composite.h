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
 * Provides functionality for creating composites.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-01-19
 */
#ifndef COMPOSITE_H
#define COMPOSITE_H

#include "rave_object.h"
#include "rave_types.h"
#include "cartesian.h"
#include "area.h"

/**
 * What type of selection variant to use
 */
typedef enum CompositeSelectionMethod_t {
  CompositeSelectionMethod_NEAREST = 0, /**< Nearest radar defines pixel to use (default) */
  CompositeSelectionMethod_HEIGHT,       /**< Pixel closest to ground defines pixel to use */
  CompositeSelectionMethod_POO           /**< Create composite by checking poo-fields */
} CompositeSelectionMethod_t;

/**
 * Defines a Composite generator
 */
typedef struct _Composite_t Composite_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType Composite_TYPE;

/**
 * Adds one RaveCoreObject, currently, the only supported type is volumes but
 * this might be enhanced in the future to also allow for cartesian products
 * to be added.
 * @param[in] composite - self
 * @param[in] object - the item to be added to the composite
 * @returns 1 on success, otherwise 0
 */
int Composite_add(Composite_t* composite, RaveCoreObject* object);

/**
 * Sets the product type that should be generated when generating the
 * composite.
 * @param[in] composite - self
 * @param[in] type - the product type, currently only PCAPPI supported.
 */
void Composite_setProduct(Composite_t* composite, Rave_ProductType type);

/**
 * Returns the product type
 * @returns the product type
 */
Rave_ProductType Composite_getProduct(Composite_t* composite);

/**
 * Sets the selection method to use. @see \ref #CompositeSelectionMethod_t.
 * @param[in] self - self
 * @param[in] method - the method to use
 * @return 1 on success otherwise 0
 */
int Composite_setSelectionMethod(Composite_t* self, CompositeSelectionMethod_t method);

/**
 * Returns the selection method. @see \ref #CompositeSelectionMethod_t
 * @param[in] self - self
 * @return the selection method
 */
CompositeSelectionMethod_t Composite_getSelectionMethod(Composite_t* self);

/**
 * Sets the height that should be used when generating a
 * composite as CAPPI or PCAPPI.
 * @param[in] composite - self
 * @param[in] height - the height
 */
void Composite_setHeight(Composite_t* composite, double height);

/**
 * Returns the height that is used for composite generation.
 * @param[in] composite - self
 * @returns the height
 */
double Composite_getHeight(Composite_t* composite);

/**
 * Sets the elevation angle that should be used when generating a
 * composite as PPI.
 * @param[in] composite - self
 * @param[in] angle - the angle in radians
 */
void Composite_setElevationAngle(Composite_t* composite, double angle);

/**
 * Returns the elevation angle that is used for composite generation.
 * @param[in] composite - self
 * @returns the height
 */
double Composite_getElevationAngle(Composite_t* composite);

/**
 * The quantity to use for this composite
 * @param[in] composite - self
 * @param[in] quantity - the quantity, defaults to DBZH
 * @returns 1 on success otherwise 0
 */
int Composite_setQuantity(Composite_t* composite, const char* quantity);

/**
 * Returns the quantity that is of interest when generating the composite
 * @param[in] composite - self
 * @returns the quantity
 */
const char* Composite_getQuantity(Composite_t* composite);

/**
 * Sets the offset to be used in the composite
 * @param[in] composite - self
 * @param[in] offset - the offset
 */
void Composite_setOffset(Composite_t* composite, double offset);

/**
 * Returns the offset that should be used in the composite
 * @param[in] composite - self
 * @returns the offset
 */
double Composite_getOffset(Composite_t* composite);

/**
 * Sets the gain to be used in the composite
 * @param[in] composite - self
 * @param[in] gain - the gain
 */
void Composite_setGain(Composite_t* composite, double gain);

/**
 * Returns the gain that should be used in the composite
 * @param[in] composite - self
 * @returns the gain
 */
double Composite_getGain(Composite_t* composite);

/**
 * Sets the nominal time.
 * @param[in] composite - self
 * @param[in] value - the time in the format HHmmss
 * @returns 1 on success, otherwise 0
 */
int Composite_setTime(Composite_t* composite, const char* value);

/**
 * Returns the nominal time.
 * @param[in] composite - self
 * @returns the nominal time (or NULL if there is none)
 */
const char* Composite_getTime(Composite_t* composite);

/**
 * Sets the nominal date.
 * @param[in] composite - self
 * @param[in] value - the date in the format YYYYMMDD
 * @returns 1 on success, otherwise 0
 */
int Composite_setDate(Composite_t* composite, const char* value);

/**
 * Returns the nominal date.
 * @param[in] composite - self
 * @returns the nominal time (or NULL if there is none)
 */
const char* Composite_getDate(Composite_t* composite);

/**
 * Generates a composite according to the nearest radar principle.
 * @param[in] composite - self
 * @param[in] area - the area that should be used for defining the composite.
 * @param[in] qualityflags - A list of char pointers identifying how/task values in the quality fields of the polar data.
 *            Each entry in this list will result in the atempt to generate a corresponding quality field
 *            in the resulting cartesian product. (MAY BE NULL)
 * @returns the generated composite.
 */
Cartesian_t* Composite_nearest(Composite_t* composite, Area_t* area, RaveList_t* qualityflags);

#endif /* COMPOSITE_H */
