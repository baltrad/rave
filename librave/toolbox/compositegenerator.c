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
 * Provides functionality for creating composites.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-10-10
 */
#include "compositegenerator.h"
#include "compositearguments.h"
#include "compositegeneratorfactory.h"
#include "polarvolume.h"
#include "rave_object.h"
#include "raveobject_hashtable.h"
#include "raveobject_list.h"
#include "rave_types.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <stdio.h>

#include "legacycompositegeneratorfactory.h"
#include "acqvacompositegeneratorfactory.h"

/**
 * Represents the cartesian product.
 */
struct _CompositeGenerator_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveObjectHashTable_t* factories; /**< the factories */
};

/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int CompositeGenerator_constructor(RaveCoreObject* obj)
{
  CompositeGenerator_t* this = (CompositeGenerator_t*)obj;
  this->factories = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  if (this->factories == NULL) {
    return 0;
  }
  return 1;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int CompositeGenerator_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  CompositeGenerator_t* this = (CompositeGenerator_t*)obj;
  CompositeGenerator_t* src = (CompositeGenerator_t*)srcobj;
  this->factories = RAVE_OBJECT_CLONE(src->factories);
  if (this->factories == NULL) {
    return 0;
  }
  return 1;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void CompositeGenerator_destructor(RaveCoreObject* obj)
{
  CompositeGenerator_t* this = (CompositeGenerator_t*)obj;
  RAVE_OBJECT_RELEASE(this->factories);
}

/*@} End of Private functions */

/*@{ Interface functions */
CompositeGenerator_t* CompositeGenerator_create(void)
{
  CompositeGenerator_t* result = NULL;
  CompositeGeneratorFactory_t* factory = NULL;

  result = RAVE_OBJECT_NEW(&CompositeGenerator_TYPE);
  if (result == NULL) {
    goto fail;
  }

  /** Legacy composite handling */
  factory = RAVE_OBJECT_NEW(&LegacyCompositeGeneratorFactory_TYPE);
  if (factory == NULL || !RaveObjectHashTable_put(result->factories, "legacy", (RaveCoreObject*)factory)) {
    goto fail;
  }
  RAVE_OBJECT_RELEASE(factory);

  factory = RAVE_OBJECT_NEW(&AcqvaCompositeGeneratorFactory_TYPE);
  if (factory == NULL || !RaveObjectHashTable_put(result->factories, "acqva", (RaveCoreObject*)factory)) {
    goto fail;
  }
  RAVE_OBJECT_RELEASE(factory);

  return result;
fail:
  RAVE_OBJECT_RELEASE(result);
  return NULL;
}

int CompositeGenerator_register(CompositeGenerator_t* generator, const char* id, CompositeGeneratorFactory_t* factory)
{
  RAVE_ASSERT((generator != NULL), "generator == NULL");
  if (id == NULL || factory == NULL) {
    RAVE_ERROR0("Must provide both id and factory");
    return 0;
  }
  return RaveObjectHashTable_put(generator->factories, id, (RaveCoreObject*)factory);
}

RaveList_t* CompositeGenerator_getFactoryIDs(CompositeGenerator_t* generator)
{
  RAVE_ASSERT((generator != NULL), "generator == NULL");
  return RaveObjectHashTable_keys(generator->factories);
}

void CompositeGenerator_unregister(CompositeGenerator_t* generator, const char* id)
{
  CompositeGeneratorFactory_t* obj = NULL;
  RAVE_ASSERT((generator != NULL), "generator == NULL");
  obj = (CompositeGeneratorFactory_t*)RaveObjectHashTable_remove(generator->factories, id);
  if (obj != NULL) {
    RAVE_OBJECT_RELEASE(obj);
  }
}

Cartesian_t* CompositeGenerator_generate(CompositeGenerator_t* generator, CompositeArguments_t* arguments)
{
  Cartesian_t* result = NULL;
  CompositeGeneratorFactory_t* factory = NULL;

  RAVE_ASSERT((generator != NULL), "generator == NULL");
  if (arguments == NULL) {
    RAVE_ERROR0("Must provide arguments when generating a product");
    return NULL;
  }

  if (CompositeArguments_getStrategy(arguments) != NULL) {
    factory = (CompositeGeneratorFactory_t*)RaveObjectHashTable_get(generator->factories, CompositeArguments_getStrategy(arguments));
    // If there is no factory with registered name, then we will check for any factory that can support provided arguments
  }

  if (factory == NULL) {
    RaveObjectList_t* values = RaveObjectHashTable_values(generator->factories);

    if (values != NULL) {
      int i = 0, nlen = 0;
      nlen = RaveObjectList_size(values);
      for (i = 0; factory == NULL && i < nlen; i++) {
        CompositeGeneratorFactory_t* xf = (CompositeGeneratorFactory_t*)RaveObjectList_get(values, i);

        if (xf != NULL && CompositeGeneratorFactory_canHandle(xf, arguments)) {
          factory = RAVE_OBJECT_COPY(xf);
        }
        RAVE_OBJECT_RELEASE(xf);
      }
    }
    RAVE_OBJECT_RELEASE(values);
  }

  if (factory != NULL) {
    fprintf(stderr, "Using worker: %s\n", CompositeGeneratorFactory_getName(factory));
    CompositeGeneratorFactory_t* worker = CompositeGeneratorFactory_create(factory);
    if (worker != NULL) {
      result = CompositeGeneratorFactory_generate(worker, arguments);
    }
    RAVE_OBJECT_RELEASE(worker);
  }
  RAVE_OBJECT_RELEASE(factory);
  return result;
}

/*@{ Interface functions */

/*@} End of Interface functions */

RaveCoreObjectType CompositeGenerator_TYPE = {
    "CompositeGenerator",
    sizeof(CompositeGenerator_t),
    CompositeGenerator_constructor,
    CompositeGenerator_destructor,
    CompositeGenerator_copyconstructor
};

