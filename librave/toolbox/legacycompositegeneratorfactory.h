/* --------------------------------------------------------------------
Copyright (C) 2024 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Interface for defining your own compositing factory method. If CompositeGeneratorFactory_canHandle(...) returns true,
 * then CompositeGeneratorFactory_generate(....) will be called. 
 *
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-10-10
 */
#ifndef LEGACY_COMPOSITE_GENERATOR_FACTORY_H
#define LEGACY_COMPOSITE_GENERATOR_FACTORY_H

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
typedef struct _LegacyCompositeGeneratorFactory_t LegacyCompositeGeneratorFactory_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType LegacyCompositeGeneratorFactory_TYPE;

/**
 * @returns the name of this factory
 */
const char* LegacyCompositeGeneratorFactory_getName(CompositeGeneratorFactory_t* self);

/**
 * @returns the default id of this factory
 */
const char* LegacyCompositeGeneratorFactory_getDefaultId(CompositeGeneratorFactory_t* self);

/**
 * @returns if this factory can handle the generator request or not
 */
int LegacyCompositeGeneratorFactory_canHandle(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments);

/**
 * Sets the factory with properties
 * @param[in] self - self
 * @param[in] properties - the properties
 */
 int LegacyCompositeGeneratorFactory_setProperties(CompositeGeneratorFactory_t* self, RaveProperties_t* properties);

 /**
 * Returns properties from the factory
 * @param[in] self - self
 * @return properties - the properties
 */
 RaveProperties_t* LegacyCompositeGeneratorFactory_getProperties(CompositeGeneratorFactory_t* self);

/**
 * @returns the result from the generation
 */
Cartesian_t* LegacyCompositeGeneratorFactory_generate(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments);

/**
 * The factory creation method.
 * @return a new instance of the factory on success
 */
CompositeGeneratorFactory_t* LegacyCompositeGeneratorFactory_create(CompositeGeneratorFactory_t* self);

#endif /* LEGACY_COMPOSITE_GENERATOR_FACTORY_H */
