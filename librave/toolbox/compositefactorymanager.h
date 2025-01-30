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
 * Registry for keeping track of available composite generator factories.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-29
 */
#ifndef COMPOSITEFACTORYMANAGER_H
#define COMPOSITEFACTORYMANAGER_H

#include "compositegeneratorfactory.h"
#include "rave_object.h"
#include "rave_types.h"
#include "raveobject_hashtable.h"


/**
 * Defines a Composite generator
 */
typedef struct _CompositeFactoryManager_t CompositeFactoryManager_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType CompositeFactoryManager_TYPE;

/**
 * Adds generator factory in the manager.
 * @param[in] manager - self
 * @param[in] id - the id of the factory
 * @param[in] factory - the actual factory
 * @return 1 if factory added successfully, otherwise 0
 */
int CompositeFactoryManager_add(CompositeFactoryManager_t* manager, CompositeGeneratorFactory_t* factory);

/**
 * Returns a list of registered factory names.
 * NOTE: Remember to use RaveList_freeAndDestroy to release all memory in the returned RaveList_t.
 * @param[in] manager - self
 * @return a list of registered factory names
 */
RaveList_t* CompositeFactoryManager_getRegisteredFactoryNames(CompositeFactoryManager_t* manager);

/**
 * Returns the registered factories.
 * @param[in] manager - self
 * @return the registered factories
 */
RaveObjectList_t* CompositeFactoryManager_getRegisteredFactories(CompositeFactoryManager_t* manager);

/**
 * Removes the factory with specified id. If id doesn't exist nothing will be done,
 * @param[in] manager - self
 * @param[in] id - the id of the factory to remove
 */
void CompositeFactoryManager_remove(CompositeFactoryManager_t* manager, const char* name);

/**
 * Returns the factory with specified name.
 * @param[in] manager - self
 * @param[in] name - name of the factory
 * @return the found factory or NULL if not
 */
CompositeGeneratorFactory_t* CompositeFactoryManager_get(CompositeFactoryManager_t* manager, const char* name);

/**
 * Returns the number of registered factories.
 * @param[in] manager - self
 * @return number of registered factories
 */
int CompositeFactoryManager_size(CompositeFactoryManager_t* manager);

/**
 * Returns if the specified factory class is registered or not.
 * @param[in] manager - self
 * @param[in] name - queried name
 * @return 1 if registered otherwise 0
 */
int CompositeFactoryManager_isRegistered(CompositeFactoryManager_t* manager, const char* name);

#endif /* COMPOSITE_H */
