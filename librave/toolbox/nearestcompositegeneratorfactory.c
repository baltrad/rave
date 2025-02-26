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
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-10-10
 */
#include "nearestcompositegeneratorfactory.h"
#include "compositeenginefunctions.h"
#include "cartesian.h"
#include "composite_utils.h"
#include "compositearguments.h"
#include "compositeengine.h"
#include "compositeengineqc.h"
#include "polarvolume.h"
#include "rave_attribute.h"
#include "rave_list.h"
#include "rave_object.h"
#include "rave_properties.h"
#include "rave_value.h"
#include "raveobject_hashtable.h"
#include "raveobject_list.h"
#include "rave_types.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_utilities.h"
#include "raveutil.h"
#include "composite.h"
#include <strings.h>
#include <string.h>
#include <stdio.h>

typedef struct _NearestCompositeGeneratorFactory_t {
  RAVE_OBJECT_HEAD /**< Always on top */
  COMPOSITE_GENERATOR_FACTORY_HEAD /**< composite generator plugin specifics */
  CompositeEngine_t* engine; /**< the compositing engine */
  CompositeEngineOvershootingQcHandler_t* overshooting;
} NearestCompositeGeneratorFactory_t;

static const char* SUPPORTED_PRODUCTS[]={
  "PPI",
  "CAPPI",
  "PCAPPI",
  "MAX",
  "PMAX",
  NULL
};

static int NearestCompositeGeneratorFactory_onStarting(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings);

static int NearestCompositeGeneratorFactory_getPolarValueAtPosition(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue);

static int NearestCompositeGeneratorFactory_getQualityValue(CompositeEngine_t* self, void* extradata, CompositeArguments_t* args, RaveCoreObject* obj, const char* quantity, const char* qfieldname, PolarNavigationInfo* navinfo, double* v);

/*@{ Private functions */
/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int NearestCompositeGeneratorFactory_constructor(RaveCoreObject* obj)
{
  NearestCompositeGeneratorFactory_t* this = (NearestCompositeGeneratorFactory_t*)obj;
  CompositeQualityFlagDefinition_t* definition = NULL;

  this->getName = NearestCompositeGeneratorFactory_getName;
  this->getDefaultId = NearestCompositeGeneratorFactory_getDefaultId;
  this->canHandle = NearestCompositeGeneratorFactory_canHandle;
  this->setProperties = NearestCompositeGeneratorFactory_setProperties;
  this->getProperties = NearestCompositeGeneratorFactory_getProperties;
  this->generate = NearestCompositeGeneratorFactory_generate;
  this->create = NearestCompositeGeneratorFactory_create;
  this->engine = NULL;
  this->overshooting = NULL;

  this->engine = RAVE_OBJECT_NEW(&CompositeEngine_TYPE);
  if (this->engine == NULL) {
    RAVE_ERROR0("Failed to create compositing engine");
    goto fail;
  }

  this->overshooting = RAVE_OBJECT_NEW(&CompositeEngineOvershootingQcHandler_TYPE);
  if (this->overshooting == NULL) {
    RAVE_ERROR0("Failed to create overshooting handler");
    goto fail;
  }

  if (!CompositeEngine_setOnStartingFunction(this->engine, NearestCompositeGeneratorFactory_onStarting)) {
    RAVE_ERROR0("Failed to set the onStarting function");
    goto fail;
  }

  if (!CompositeEngine_setDefaultPolarValueAtPositionFunction(this->engine, NearestCompositeGeneratorFactory_getPolarValueAtPosition)) {
    RAVE_ERROR0("Failed to set getPolarValueAtPosition function for ANY");
    goto fail;
  }

  if (!CompositeEngine_registerPolarValueAtPositionFunction(this->engine, "RATE", NearestCompositeGeneratorFactory_getPolarValueAtPosition)) {
    RAVE_ERROR0("Failed to set getPolarValueAtPosition function for RATE");
    goto fail;
  }

  if (!CompositeEngine_setGetQualityValueFunction(this->engine, NearestCompositeGeneratorFactory_getQualityValue)) {
    RAVE_ERROR0("Failed to set getQualityValue function for NEAREST");
    goto fail;
  }

  definition = CompositeEngineQcHandler_getFlagDefinition(this->overshooting);
  if (definition == NULL || ! CompositeEngine_registerQualityFlagDefinition(this->engine, definition)) {
    RAVE_ERROR0("Failed to register overshooting definition");
    goto fail;
  }
  RAVE_OBJECT_RELEASE(definition);

  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->engine);
  RAVE_OBJECT_RELEASE(this->overshooting);
  RAVE_OBJECT_RELEASE(definition);

  return 0;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int NearestCompositeGeneratorFactory_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  NearestCompositeGeneratorFactory_t* this = (NearestCompositeGeneratorFactory_t*)obj;
  NearestCompositeGeneratorFactory_t* src = (NearestCompositeGeneratorFactory_t*)srcobj;
  this->getName = NearestCompositeGeneratorFactory_getName;
  this->getDefaultId = NearestCompositeGeneratorFactory_getDefaultId;
  this->canHandle = NearestCompositeGeneratorFactory_canHandle;
  this->setProperties = NearestCompositeGeneratorFactory_setProperties;
  this->getProperties = NearestCompositeGeneratorFactory_getProperties;
  this->generate = NearestCompositeGeneratorFactory_generate;
  this->create = NearestCompositeGeneratorFactory_create;
  this->engine = NULL;
  this->overshooting = NULL;

  this->engine = RAVE_OBJECT_CLONE(src->engine);
  if (this->engine == NULL) {
    RAVE_ERROR0("Failed to clone compositing engine");
    goto fail;
  }

  this->overshooting = RAVE_OBJECT_CLONE(src->overshooting);
  if (this->overshooting == NULL) {
    RAVE_ERROR0("Failed to clone overshooting handler");
    goto fail;
  }

  if (!CompositeEngine_setOnStartingFunction(this->engine, NearestCompositeGeneratorFactory_onStarting)) {
    RAVE_ERROR0("Failed to set the onStarting function");
    goto fail;
  }

  if (!CompositeEngine_setDefaultPolarValueAtPositionFunction(this->engine, NearestCompositeGeneratorFactory_getPolarValueAtPosition)) {
    RAVE_ERROR0("Failed to set getPolarValueAtPosition function for ANY");
    goto fail;
  }

  if (!CompositeEngine_registerPolarValueAtPositionFunction(this->engine, "RATE", NearestCompositeGeneratorFactory_getPolarValueAtPosition)) {
    RAVE_ERROR0("Failed to set getPolarValueAtPosition function for RATE");
    goto fail;
  }

  if (!CompositeEngine_setGetQualityValueFunction(this->engine, NearestCompositeGeneratorFactory_getQualityValue)) {
    RAVE_ERROR0("Failed to set getQualityValue function for NEAREST");
    goto fail;
  }

  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->engine);
  RAVE_OBJECT_RELEASE(this->overshooting);
  return 0;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void NearestCompositeGeneratorFactory_destructor(RaveCoreObject* obj)
{
  NearestCompositeGeneratorFactory_t* this = (NearestCompositeGeneratorFactory_t*)obj;
  RAVE_OBJECT_RELEASE(this->engine);
  RAVE_OBJECT_RELEASE(this->overshooting);
}

static int NearestCompositeGeneratorFactory_onStarting(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings)
{
  NearestCompositeGeneratorFactory_t* self = (NearestCompositeGeneratorFactory_t*)extradata;
  int result = 0;
  RaveProperties_t* properties = CompositeEngine_getProperties(engine);
  if (!CompositeEngineFunctions_prepareRATE(engine, arguments, bindings, nbindings)) {
    RAVE_ERROR0("Failed to prepare RATE coefficients");
    goto fail;
  }
  CompositeEngineQcHandler_initialize(self->overshooting, extradata, properties, arguments, bindings, nbindings);

  result = 1;
fail:
  RAVE_OBJECT_RELEASE(properties);
  return result;
}

static int NearestCompositeGeneratorFactory_getPolarValueAtPosition(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue)
{
  int result = 0;
  
  if (quantity == NULL) {
    return 0;
  }
  if (strcasecmp("RATE", quantity) == 0) {
    result = CompositeEngineFunctions_getRATEValueAtPosition(engine, extradata, arguments, binding, quantity, navinfo, qiFieldName, otype, ovalue, qivalue);
  } else {
    result = CompositeEngineUtility_getPolarValueAtPosition(engine, extradata, arguments, binding, quantity, navinfo, qiFieldName, otype, ovalue, qivalue);
  }

  return result;
}

static int NearestCompositeGeneratorFactory_getQualityValue(CompositeEngine_t* self, void* extradata, CompositeArguments_t* args, RaveCoreObject* obj, const char* quantity, const char* qfieldname, PolarNavigationInfo* navinfo, double* v)
{
  int result = 0;
  NearestCompositeGeneratorFactory_t* this = (NearestCompositeGeneratorFactory_t*)extradata;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((obj != NULL), "obj == NULL");
  RAVE_ASSERT((navinfo != NULL), "navinfo == NULL");

  if (strcmp("se.smhi.detector.poo", qfieldname) == 0) {
    result = CompositeEngineQcHandler_getQualityValue(this->overshooting, extradata, args, obj, quantity, qfieldname, navinfo,  v);
  }

  return result;
}

/*@} End of Private functions */

/*@{ Interface functions */

/**
 * @returns the unique name for this composite generator plugin
 */
const char* NearestCompositeGeneratorFactory_getName(CompositeGeneratorFactory_t* self)
{
  return "NearestCompositeGenerator";
}

/**
 * @returns the default id of this factory
 */
const char* NearestCompositeGeneratorFactory_getDefaultId(CompositeGeneratorFactory_t* self)
{
  return "nearest";
}

/**
 * @returns if this plugin can handle the generator request
 */
int NearestCompositeGeneratorFactory_canHandle(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments)
{
  const char* productid;
  int result = 0;
  RaveAttribute_t* attr =  NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (arguments == NULL) {
    return 0;
  }
  productid = CompositeArguments_getProduct(arguments);
  if (productid != NULL) {
    if (!RaveUtilities_arrayContains(SUPPORTED_PRODUCTS, productid, 0)) {
      return 0;
    }
  }

  attr = CompositeArguments_getArgument(arguments, "interpolation_method");
  if (attr != NULL) {
    char* value = NULL;
    RaveAttribute_getString(attr, &value);
    if (value == NULL || strcasecmp("NEAREST", value) != 0) {
      goto fail;
    }
  }
  result = 1;
fail:
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

int NearestCompositeGeneratorFactory_setProperties(CompositeGeneratorFactory_t* self, RaveProperties_t* properties)
{
  NearestCompositeGeneratorFactory_t* factory = (NearestCompositeGeneratorFactory_t*)self;
  RAVE_ASSERT((factory != NULL), "self == NULL");
  CompositeEngine_setProperties(factory->engine, properties);
  return 1;
}

RaveProperties_t* NearestCompositeGeneratorFactory_getProperties(CompositeGeneratorFactory_t* self)
{
  NearestCompositeGeneratorFactory_t* factory = (NearestCompositeGeneratorFactory_t*)self;
  RAVE_ASSERT((factory != NULL), "factory == NULL");
  return CompositeEngine_getProperties(factory->engine);
}

/**
 * @returns the result from the generation
 */
Cartesian_t* NearestCompositeGeneratorFactory_generate(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments)
{
  return CompositeEngine_generate(((NearestCompositeGeneratorFactory_t*)self)->engine, arguments, (void*)self);
}

/**
 * The initializing function so that we know what composite generator we are working with.
 * Note, this function will be called by the composite module and will override any previous calls.
 * @return 1 on success otherwise 0
 */
CompositeGeneratorFactory_t* NearestCompositeGeneratorFactory_create(CompositeGeneratorFactory_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (RAVE_OBJECT_CHECK_TYPE(self, &NearestCompositeGeneratorFactory_TYPE)) {
    return RAVE_OBJECT_CLONE(self);
  }
  RAVE_ERROR0("Something is wrong in implementation. Should not arrive here unless type is a NearestCompositeGeneratorFactory_TYPE");
  return NULL;
}

/*@} End of Interface functions */

RaveCoreObjectType NearestCompositeGeneratorFactory_TYPE = {
    "NearestCompositeGeneratorFactory",
    sizeof(NearestCompositeGeneratorFactory_t),
    NearestCompositeGeneratorFactory_constructor,
    NearestCompositeGeneratorFactory_destructor,
    NearestCompositeGeneratorFactory_copyconstructor
};
