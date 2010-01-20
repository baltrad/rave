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
 * Generates a composite according to the nearest radar principle.
 * @param[in] composite - self
 * @param[in] area - the area that should be used for defining the composite.
 * @param[in] height - the height where the product should be generated at
 * @returns the generated composite.
 */
Cartesian_t* Composite_nearest(Composite_t* composite, Area_t* area, double height);

#endif /* COMPOSITE_H */
