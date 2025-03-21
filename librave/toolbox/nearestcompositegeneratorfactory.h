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
 * The NEAREST composite factory
 *
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-10
 */
#ifndef NEAREST_COMPOSITE_GENERATOR_FACTORY_H
#define NEAREST_COMPOSITE_GENERATOR_FACTORY_H

#include "compositearguments.h"
#include "compositegeneratorfactory.h"
#include "compositegenerator.h"
#include "cartesian.h"
#include "rave_object.h"
#include "rave_types.h"
#include "raveobject_list.h"
#include "rave_field.h"

/**
 * Defines a Composite generator plugin
 */
typedef struct _NearestCompositeGeneratorFactory_t NearestCompositeGeneratorFactory_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType NearestCompositeGeneratorFactory_TYPE;

/**
 * @returns the name of this factory
 */
const char* NearestCompositeGeneratorFactory_getName(CompositeGeneratorFactory_t* self);

/**
 * @returns the default id of this factory
 */
const char* NearestCompositeGeneratorFactory_getDefaultId(CompositeGeneratorFactory_t* self);

/**
 * @returns if this factory can handle the generator request or not
 */
int NearestCompositeGeneratorFactory_canHandle(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments);

/**
 * Sets the factory with properties
 * @param[in] self - self
 * @param[in] properties - the properties
 */
 int NearestCompositeGeneratorFactory_setProperties(CompositeGeneratorFactory_t* self, RaveProperties_t* properties);

 /**
 * Returns properties from the factory
 * @param[in] self - self
 * @return properties - the properties
 */
 RaveProperties_t* NearestCompositeGeneratorFactory_getProperties(CompositeGeneratorFactory_t* self);

/**
 * @returns the result from the generation
 */
Cartesian_t* NearestCompositeGeneratorFactory_generate(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments);

/**
 * The factory creation method.
 * @return a new instance of the factory on success
 */
CompositeGeneratorFactory_t* NearestCompositeGeneratorFactory_create(CompositeGeneratorFactory_t* self);

#endif /* NEAREST_COMPOSITE_GENERATOR_FACTORY_H */
