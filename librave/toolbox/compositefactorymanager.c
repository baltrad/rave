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
 * Registry for keeping track of available composite generator factories.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-29
 */
#include "compositefactorymanager.h"
#include "rave_object.h"
#include "raveobject_hashtable.h"
#include "raveobject_list.h"
#include "rave_types.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <stdio.h>
#include <string.h>
#include <strings.h>

#include "nearestcompositegeneratorfactory.h"
#include "legacycompositegeneratorfactory.h"
#include "acqvacompositegeneratorfactory.h"

/**
 * Represents the composite manager product.
 */
struct _CompositeFactoryManager_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveObjectHashTable_t* factories; /**< the factories */
};

/*@{ Private functions */

/**
 * Creates the default factories provided by the rave toolbox.
 * @param[in] manager - self
 * @return 1 on success otherwise 0
 */
static int CompositeFactoryManagerInternal_createDefault(CompositeFactoryManager_t* manager)
{
  CompositeGeneratorFactory_t* factory = NULL;

  /** Legacy composite handling */
  factory = RAVE_OBJECT_NEW(&LegacyCompositeGeneratorFactory_TYPE);
  if (factory == NULL || !CompositeFactoryManager_add(manager, factory)) {
    goto fail;
  }
  RAVE_OBJECT_RELEASE(factory);

  /** Nearest composite handling */
  factory = RAVE_OBJECT_NEW(&NearestCompositeGeneratorFactory_TYPE);
  if (factory == NULL || !CompositeFactoryManager_add(manager, factory)) {
    goto fail;
  }
  RAVE_OBJECT_RELEASE(factory);

  /** Acqva composite handling */
  factory = RAVE_OBJECT_NEW(&AcqvaCompositeGeneratorFactory_TYPE);
  if (factory == NULL || !CompositeFactoryManager_add(manager, factory)) {
    goto fail;
  }
  RAVE_OBJECT_RELEASE(factory);
  return 1;
fail:
  return 0;
}

/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int CompositeFactoryManager_constructor(RaveCoreObject* obj)
{
  CompositeFactoryManager_t* this = (CompositeFactoryManager_t*)obj;
  this->factories = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  if (this->factories == NULL) {
    goto fail;
  }
  return CompositeFactoryManagerInternal_createDefault(this);
fail:
  RAVE_OBJECT_RELEASE(this->factories);
  return 0;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int CompositeFactoryManager_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  CompositeFactoryManager_t* this = (CompositeFactoryManager_t*)obj;
  CompositeFactoryManager_t* src = (CompositeFactoryManager_t*)srcobj;
  this->factories = RAVE_OBJECT_CLONE(src->factories);
  if (this->factories == NULL) {
    goto fail;
  }
  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->factories);
  return 0;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void CompositeFactoryManager_destructor(RaveCoreObject* obj)
{
  CompositeFactoryManager_t* this = (CompositeFactoryManager_t*)obj;
  RAVE_OBJECT_RELEASE(this->factories);
}

/*@} End of Private functions */

/*@{ Interface functions */
int CompositeFactoryManager_add(CompositeFactoryManager_t* manager, CompositeGeneratorFactory_t* factory)
{
  RAVE_ASSERT((manager != NULL), "manager == NULL");
  if (factory == NULL) {
    return 0;
  }
  if (RaveObjectHashTable_exists(manager->factories, CompositeGeneratorFactory_getName(factory))) {
    RAVE_INFO1("Replacing generator factory: %s", CompositeGeneratorFactory_getName(factory));
  }
  if (!RaveObjectHashTable_put(manager->factories, CompositeGeneratorFactory_getName(factory), (RaveCoreObject*)factory)) {
    RAVE_ERROR1("Failed to add/replace factory: %s", CompositeGeneratorFactory_getName(factory));
    return 0;
  }
  return 1;
}


RaveList_t* CompositeFactoryManager_getRegisteredFactoryNames(CompositeFactoryManager_t* manager)
{
  RAVE_ASSERT((manager != NULL), "manager == NULL");
  return RaveObjectHashTable_keys(manager->factories);
}

RaveObjectList_t* CompositeFactoryManager_getRegisteredFactories(CompositeFactoryManager_t* manager)
{
  RAVE_ASSERT((manager != NULL), "manager == NULL");
  return RaveObjectHashTable_values(manager->factories);
}

void CompositeFactoryManager_remove(CompositeFactoryManager_t* manager, const char* name)
{
  CompositeGeneratorFactory_t* obj = NULL;
  RAVE_ASSERT((manager != NULL), "manager == NULL");
  if (name == NULL) {
    return;
  }
  obj = (CompositeGeneratorFactory_t*)RaveObjectHashTable_remove(manager->factories, name);
  RAVE_OBJECT_RELEASE(obj);
}

CompositeGeneratorFactory_t* CompositeFactoryManager_get(CompositeFactoryManager_t* manager, const char* name)
{
  CompositeGeneratorFactory_t* result = NULL;
  RAVE_ASSERT((manager != NULL), "manager == NULL");
  result = (CompositeGeneratorFactory_t*)RaveObjectHashTable_get(manager->factories, name);
  return result;
}

int CompositeFactoryManager_size(CompositeFactoryManager_t* manager)
{
  RAVE_ASSERT((manager != NULL), "manager == NULL");
  return RaveObjectHashTable_size(manager->factories);
}

int CompositeFactoryManager_isRegistered(CompositeFactoryManager_t* manager, const char* name)
{
  RAVE_ASSERT((manager != NULL), "manager == NULL");
  return RaveObjectHashTable_exists(manager->factories, name);
}

/*@{ Interface functions */

/*@} End of Interface functions */

RaveCoreObjectType CompositeFactoryManager_TYPE = {
    "CompositeFactoryManager",
    sizeof(CompositeFactoryManager_t),
    CompositeFactoryManager_constructor,
    CompositeFactoryManager_destructor,
    CompositeFactoryManager_copyconstructor
};

