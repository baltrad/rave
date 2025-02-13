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
 * The RATE composite factory
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-10-10
 */
#include "ratecompositegeneratorfactory.h"
#include "compositearguments.h"
#include "compositeengine.h"
#include "polarvolume.h"
#include "rave_attribute.h"
#include "rave_list.h"
#include "rave_object.h"
#include "raveobject_list.h"
#include "rave_types.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_utilities.h"
#include "composite.h"
#include <strings.h>
#include <string.h>
#include <stdio.h>

typedef struct _RateCompositeGeneratorFactory_t {
  RAVE_OBJECT_HEAD /**< Always on top */
  COMPOSITE_GENERATOR_FACTORY_HEAD /**< composite generator plugin specifics */
  CompositeEngine_t* engine; /**< the compositing engine */
} RateCompositeGeneratorFactory_t;

int RateCompositeGeneratorFactory_getPolarValueAtPosition(void* extradata, CompositeArguments_t* arguments, RaveCoreObject* object, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue);

/*@{ Private functions */
/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int RateCompositeGeneratorFactory_constructor(RaveCoreObject* obj)
{
  RateCompositeGeneratorFactory_t* this = (RateCompositeGeneratorFactory_t*)obj;
  this->getName = RateCompositeGeneratorFactory_getName;
  this->getDefaultId = RateCompositeGeneratorFactory_getDefaultId;
  this->canHandle = RateCompositeGeneratorFactory_canHandle;
  this->generate = RateCompositeGeneratorFactory_generate;
  this->create = RateCompositeGeneratorFactory_create;
  this->engine = RAVE_OBJECT_NEW(&CompositeEngine_TYPE);
  if (this->engine == NULL) {
    RAVE_ERROR0("Failed to create compositing engine");
    goto fail;
  }

  if (!CompositeEngine_registerPolarValueAtPositionFunction(this->engine, "RATE", RateCompositeGeneratorFactory_getPolarValueAtPosition)) {
    RAVE_ERROR0("Failed to set getPolarValueAtPosition function for RATE");
    goto fail;
  }

  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->engine);
  return 0;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int RateCompositeGeneratorFactory_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RateCompositeGeneratorFactory_t* this = (RateCompositeGeneratorFactory_t*)obj;
  RateCompositeGeneratorFactory_t* src = (RateCompositeGeneratorFactory_t*)srcobj;
  this->getName = RateCompositeGeneratorFactory_getName;
  this->getDefaultId = RateCompositeGeneratorFactory_getDefaultId;
  this->canHandle = RateCompositeGeneratorFactory_canHandle;
  this->generate = RateCompositeGeneratorFactory_generate;
  this->create = RateCompositeGeneratorFactory_create;
  this->engine = RAVE_OBJECT_CLONE(src->engine);
  if (this->engine == NULL) {
    RAVE_ERROR0("Failed to clone compositing engine");
    goto fail;
  }

  if (!CompositeEngine_registerPolarValueAtPositionFunction(this->engine, "RATE", RateCompositeGeneratorFactory_getPolarValueAtPosition)) {
    RAVE_ERROR0("Failed to set getPolarValueAtPosition function for RATE");
    goto fail;
  }

  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->engine);
  return 0;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void RateCompositeGeneratorFactory_destructor(RaveCoreObject* obj)
{
  RateCompositeGeneratorFactory_t* this = (RateCompositeGeneratorFactory_t*)obj;
  RAVE_OBJECT_RELEASE(this->engine);
}

int RateCompositeGeneratorFactory_getPolarValueAtPosition(void* extradata, CompositeArguments_t* arguments, RaveCoreObject* object, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue)
{
  RateCompositeGeneratorFactory_t* self = (RateCompositeGeneratorFactory_t*)extradata;
  int result = 0;
  if (quantity == NULL) {
    return 0;
  }
  if (strcasecmp("RATE", quantity) == 0) {
    result = CompositeEngineUtility_getPolarValueAtPosition(extradata, arguments, object, "DBZH", navinfo, qiFieldName, otype, ovalue, qivalue);
  } else {
    result = CompositeEngineUtility_getPolarValueAtPosition(extradata, arguments, object, quantity, navinfo, qiFieldName, otype, ovalue, qivalue);
  }

  return result;
}


/*@} End of Private functions */

/*@{ Interface functions */

/**
 * @returns the unique name for this composite generator plugin
 */
const char* RateCompositeGeneratorFactory_getName(CompositeGeneratorFactory_t* self)
{
  return "RateCompositeGenerator";
}

/**
 * @returns the default id of this factory
 */
const char* RateCompositeGeneratorFactory_getDefaultId(CompositeGeneratorFactory_t* self)
{
  return "rate";
}

/**
 * @returns if this plugin can handle the generator request
 */
int RateCompositeGeneratorFactory_canHandle(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments)
{
  const char* productid;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (arguments == NULL) {
    return 0;
  }
  return 1;
}

/**
 * @returns the result from the generation
 */
Cartesian_t* RateCompositeGeneratorFactory_generate(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments)
{
  return CompositeEngine_generate(((RateCompositeGeneratorFactory_t*)self)->engine, arguments, (void*)self);
}

/**
 * The initializing function so that we know what composite generator we are working with.
 * Note, this function will be called by the composite module and will override any previous calls.
 * @return 1 on success otherwise 0
 */
CompositeGeneratorFactory_t* RateCompositeGeneratorFactory_create(CompositeGeneratorFactory_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (RAVE_OBJECT_CHECK_TYPE(self, &RateCompositeGeneratorFactory_TYPE)) {
    return RAVE_OBJECT_CLONE(self);
  }
  RAVE_ERROR0("Something is wrong in implementation. Should not arrive here unless type is a RateCompositeGeneratorFactory_TYPE");
  return NULL;
}

/*@} End of Interface functions */

RaveCoreObjectType RateCompositeGeneratorFactory_TYPE = {
    "RateCompositeGeneratorFactory",
    sizeof(RateCompositeGeneratorFactory_t),
    RateCompositeGeneratorFactory_constructor,
    RateCompositeGeneratorFactory_destructor,
    RateCompositeGeneratorFactory_copyconstructor
};
