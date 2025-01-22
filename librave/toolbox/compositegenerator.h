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
 * Generator for generating composites using various approaches. You are able to register
 * your own composite methods so that you get a standard methodology when creating them.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-10-09
 */
#ifndef COMPOSITEGENERATOR_H
#define COMPOSITEGENERATOR_H

#include "cartesian.h"
#include "compositegeneratorfactory.h"
#include "compositearguments.h"
#include "rave_object.h"
#include "rave_types.h"
#include "raveobject_hashtable.h"
#include "limits.h"


/**
 * Defines a Composite generator
 */
typedef struct _CompositeGenerator_t CompositeGenerator_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType CompositeGenerator_TYPE;

/**
 * Creates a default composite generator with all generators provided by rave-toolbox.
 * This should probably be refactored into a compositemanager but for now this is good enough.
 * @return an initialized composite generator
 */
CompositeGenerator_t* CompositeGenerator_create(void);

/**
 * Registers a generator factory in the the generator.
 * @param[in] generator - self
 * @param[in] id - the id of the factory
 * @param[in] facgtory - the actual factory
 * @return 1 if factory added successfully, otherwise 0
 */
int CompositeGenerator_register(CompositeGenerator_t* generator, const char* id, CompositeGeneratorFactory_t* factory);

/**
 * Returns a list of registered factory ids.
 * NOTE: Remember to use RaveList_freeAndDestroy to release all memory in the returned RaveList_t.
 * @param[in] generator - self
 * @return a list of registered factory ids
 */
RaveList_t* CompositeGenerator_getFactoryIDs(CompositeGenerator_t* generator);

/**
 * Removes the factory with specified id. If id doesn't exist nothing will be done,
 * @param[in] generator - self
 * @param[in] id - the id of the factory to remove
 */
void CompositeGenerator_unregister(CompositeGenerator_t* generator, const char* id);

/**
 * Generates a composite according to the configured parameters in the composite structure.
 * @param[in] generator - self
 * @param[in] arguments - the arguments
 * @returns the generated composite.
 */
Cartesian_t* CompositeGenerator_generate(CompositeGenerator_t* generator, CompositeArguments_t* arguments);

#endif /* COMPOSITE_H */
