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
 * Factory method for ACQVA composite generation 
 *
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-14
 */
#ifndef ACQVA_COMPOSITE_GENERATOR_FACTORY_H
#define ACQVA_COMPOSITE_GENERATOR_FACTORY_H

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
typedef struct _AcqvaCompositeGeneratorFactory_t AcqvaCompositeGeneratorFactory_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType AcqvaCompositeGeneratorFactory_TYPE;

/**
 * @returns the name of this plugin
 */
const char* AcqvaCompositeGeneratorFactory_getName(CompositeGeneratorFactory_t* self);

/**
 * @returns the default id of this factory
 */
const char* AcqvaCompositeGeneratorFactory_getDefaultId(CompositeGeneratorFactory_t* self);

/**
 * @returns if this factory can handle the generator request or not
 */
int AcqvaCompositeGeneratorFactory_canHandle(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments);

/**
 * Sets the factory with properties
 * @param[in] self - self
 * @param[in] properties - the properties
 */
 int AcqvaCompositeGeneratorFactory_setProperties(CompositeGeneratorFactory_t* self, RaveProperties_t* properties);

 /**
 * Returns properties from the factory
 * @param[in] self - self
 * @return properties - the properties
 */
 RaveProperties_t* AcqvaCompositeGeneratorFactory_getProperties(CompositeGeneratorFactory_t* self);

/**
 * @returns the result from the generation
 */
Cartesian_t* AcqvaCompositeGeneratorFactory_generate(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments);

/**
 * Return a new instance of the factory
 * @return 1 on success otherwise 0
 */
CompositeGeneratorFactory_t* AcqvaCompositeGeneratorFactory_create(CompositeGeneratorFactory_t* self);

#endif /* ACQVA_COMPOSITE_GENERATOR_FACTORY_H */
