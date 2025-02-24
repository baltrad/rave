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
#include "compositearguments.h"
#include "compositeengine.h"
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
} NearestCompositeGeneratorFactory_t;

static const char* SUPPORTED_PRODUCTS[]={
  "PPI",
  "CAPPI",
  "PCAPPI",
  "MAX",
  "PMAX",
  NULL
};

int NearestCompositeGeneratorFactory_getPolarValueAtPosition(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue);

/*@{ Private functions */
/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int NearestCompositeGeneratorFactory_constructor(RaveCoreObject* obj)
{
  NearestCompositeGeneratorFactory_t* this = (NearestCompositeGeneratorFactory_t*)obj;
  this->getName = NearestCompositeGeneratorFactory_getName;
  this->getDefaultId = NearestCompositeGeneratorFactory_getDefaultId;
  this->canHandle = NearestCompositeGeneratorFactory_canHandle;
  this->setProperties = NearestCompositeGeneratorFactory_setProperties;
  this->getProperties = NearestCompositeGeneratorFactory_getProperties;
  this->generate = NearestCompositeGeneratorFactory_generate;
  this->create = NearestCompositeGeneratorFactory_create;

  this->engine = RAVE_OBJECT_NEW(&CompositeEngine_TYPE);
  if (this->engine == NULL) {
    RAVE_ERROR0("Failed to create compositing engine");
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

  this->engine = RAVE_OBJECT_CLONE(src->engine);
  if (this->engine == NULL) {
    RAVE_ERROR0("Failed to clone compositing engine");
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

  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->engine);
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
}

int NearestCompositeGeneratorFactory_getPolarValueAtPosition(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue)
{
  NearestCompositeGeneratorFactory_t* self = (NearestCompositeGeneratorFactory_t*)extradata;
  RaveProperties_t* properties = NULL;
  int result = 0;
  
  if (quantity == NULL) {
    return 0;
  }
  properties = CompositeEngine_getProperties(engine);
  if (strcasecmp("RATE", quantity) == 0) {
    double zr_a = 200.0, zr_b = 1.6;
    if (binding->source != NULL && binding->value == NULL) {
      /* We create a cache by looking up the zr-coefficients for the object and store it in the binding value so that we only
       * have to do it once.
       */
      if (properties != NULL) {
        RaveValue_t* value = RaveProperties_get(properties, "rave.rate.zr.coefficients");
        if (value != NULL && RaveValue_type(value) == RaveValue_Type_Hashtable) {
          RaveObjectHashTable_t* rates = RaveValue_toHashTable(value);
          if (RaveObjectHashTable_exists(rates, OdimSource_getNod(binding->source))) {
            RaveValue_t* zr = (RaveValue_t*)RaveObjectHashTable_get(rates, OdimSource_getNod(binding->source));
            if (zr != NULL && RaveValue_type(zr) == RaveValue_Type_DoubleArray) {
              double* v = NULL;
              int len = 0;
              RaveValue_getDoubleArray(zr, &v, &len);
              if (len == 2) {
                binding->value = RAVE_OBJECT_COPY(zr);
              } else {
                RAVE_ERROR1("rave.rate.zr.coefficients coefficient for %s could not be read, should be (zr_a, zr_b)", OdimSource_getNod(binding->source));
              }
            }
            RAVE_OBJECT_RELEASE(zr);
          }
          RAVE_OBJECT_RELEASE(rates);
        }
        RAVE_OBJECT_RELEASE(value);
      }
    } else if (binding->value != NULL) {
      double* v = NULL;
      int len = 0;
      RaveValue_getDoubleArray(binding->value, &v, &len);
      zr_a = v[0];
      zr_b = v[1];
    }
    result = CompositeEngineUtility_getPolarValueAtPosition(engine, extradata, arguments, binding, "DBZH", navinfo, qiFieldName, otype, ovalue, qivalue);
    if (*otype == RaveValueType_DATA) {
      double rr = dBZ2R(*ovalue, zr_a, zr_b);
      *ovalue = rr;
    }
  } else {
    result = CompositeEngineUtility_getPolarValueAtPosition(engine, extradata, arguments, binding, "DBZH", navinfo, qiFieldName, otype, ovalue, qivalue);
  }

  RAVE_OBJECT_RELEASE(properties);
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
