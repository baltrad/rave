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
#include "cartesian.h"
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
  RaveObjectHashTable_t* poofields; /**< if poo quality field should be created, cash it here */
} NearestCompositeGeneratorFactory_t;

static const char* SUPPORTED_PRODUCTS[]={
  "PPI",
  "CAPPI",
  "PCAPPI",
  "MAX",
  "PMAX",
  NULL
};

#define HOWTASK_POO_GAIN 1.0/UCHAR_MAX
#define HOWTASK_POO_OFFSET 0.0
#define HOWTASK_POO_DATATYPE RaveDataType_UCHAR

static int NearestCompositeGeneratorFactory_onStarting(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings);

static int NearestCompositeGeneratorFactory_getPolarValueAtPosition(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue);

static int NearestCompositeGeneratorFactory_getQualityValue(CompositeEngine_t* self, void* extradata, CompositeArguments_t* args, RaveCoreObject* obj, const char* qfieldname, PolarNavigationInfo* navinfo, double* v);

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
  this->poofields = NULL;

  this->engine = RAVE_OBJECT_NEW(&CompositeEngine_TYPE);
  if (this->engine == NULL) {
    RAVE_ERROR0("Failed to create compositing engine");
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

  if (!CompositeEngine_registerQualityFlagDefinition(this->engine, "se.smhi.detector.poo", HOWTASK_POO_DATATYPE, HOWTASK_POO_OFFSET, HOWTASK_POO_GAIN)) {
    RAVE_ERROR0("Failed to register quality flag definition for: se.smhi.detector.poo");
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
  this->poofields = NULL;

  this->engine = RAVE_OBJECT_CLONE(src->engine);
  if (this->engine == NULL) {
    RAVE_ERROR0("Failed to clone compositing engine");
    goto fail;
  }

  if (src->poofields != NULL) {
    this->poofields = RAVE_OBJECT_CLONE(src->poofields);
    if (this->poofields == NULL) {
      RAVE_ERROR0("Failed to clone poo fields");
      goto fail;
    }
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
  RAVE_OBJECT_RELEASE(this->poofields);
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
  RAVE_OBJECT_RELEASE(this->poofields);
}

/**
 * Will traverse all objects in the list and atempt to find a scan that contains a
 * quality field that has got a how/task value == se.smhi.detector.poo.
 * All scans that contains such a field will get a scan set in the resulting
 * hash table with the quality data set as the default (and only) parameter.
 * @param[in] composite - the composite
 * @return a hash table
 */
 static RaveObjectHashTable_t* CompositeEngineUtility_getQualityScanFields(CompositeEngine_t* engine, CompositeEngineObjectBinding_t* bindings, int nbindings, const char* qualityFieldName)
 {
   RaveObjectHashTable_t* result = NULL;
   RaveObjectHashTable_t* scans = NULL;
   int i = 0;
   int status = 1;
 
   scans = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
   if (scans == NULL) {
     RAVE_ERROR0("Failed to allocate memory for object hash table");
     goto done;
   }
 
   for (i = 0; status == 1 && i < nbindings; i++) {
     if (RAVE_OBJECT_CHECK_TYPE(bindings[i].object, &PolarScan_TYPE)) {
       RaveField_t* field = PolarScan_findAnyQualityFieldByHowTask((PolarScan_t*)bindings[i].object, qualityFieldName);
       if (field != NULL) {
         PolarScan_t* scan = PolarScan_createFromScanAndField((PolarScan_t*)bindings[i].object, field);
         if (scan == NULL || !RaveObjectHashTable_put(scans, PolarScan_getSource(scan), (RaveCoreObject*)scan)) {
           RAVE_ERROR0("Failed to add poo scan to hash table");
           status = 0;
         }
         RAVE_OBJECT_RELEASE(scan);
       }
       RAVE_OBJECT_RELEASE(field);
     } else if (RAVE_OBJECT_CHECK_TYPE(bindings[i].object, &PolarVolume_TYPE)) {
       PolarScan_t* pooscan = PolarVolume_findAnyScanWithQualityFieldByHowTask((PolarVolume_t*)bindings[i].object, qualityFieldName);
       if (pooscan != NULL) {
         RaveField_t* field = PolarScan_findAnyQualityFieldByHowTask(pooscan, qualityFieldName);
         if (field != NULL) {
           PolarScan_t* scan = PolarScan_createFromScanAndField(pooscan, field);
           if (scan == NULL || !RaveObjectHashTable_put(scans, PolarScan_getSource(scan), (RaveCoreObject*)scan)) {
             RAVE_ERROR0("Failed to scan to hash table");
             status = 0;
           }
           RAVE_OBJECT_RELEASE(scan);
         }
         RAVE_OBJECT_RELEASE(field);
       }
       RAVE_OBJECT_RELEASE(pooscan);
     }
   }
   result = RAVE_OBJECT_COPY(scans);
done:
   RAVE_OBJECT_RELEASE(scans);
   return result;
}

static int NearestCompositeGeneratorFactory_onStarting(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings)
{
  NearestCompositeGeneratorFactory_t* self = (NearestCompositeGeneratorFactory_t*)extradata;

  RAVE_OBJECT_RELEASE(self->poofields);
  if (CompositeArguments_hasQualityFlag(arguments, "se.smhi.detector.poo")) {
    self->poofields = CompositeEngineUtility_getQualityScanFields(engine, bindings, nbindings, "se.smhi.detector.poo");
  }

  return 1;
}

static int NearestCompositeGeneratorFactory_getPolarValueAtPosition(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue)
{
  /*NearestCompositeGeneratorFactory_t* self = (NearestCompositeGeneratorFactory_t*)extradata;*/
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

static int NearestCompositeGeneratorFactory_getQualityValue(CompositeEngine_t* self, void* extradata, CompositeArguments_t* args, RaveCoreObject* obj, const char* qfieldname, PolarNavigationInfo* navinfo, double* v)
{
  int result = 0;
  PolarScan_t* pooscan = NULL;
  NearestCompositeGeneratorFactory_t* this = (NearestCompositeGeneratorFactory_t*)extradata;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((obj != NULL), "obj == NULL");
  RAVE_ASSERT((navinfo != NULL), "navinfo == NULL");

  if (strcmp("se.smhi.detector.poo", qfieldname) == 0) {
    /* We know how to handle poo */
    if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE) && navinfo->ei >= 0 && navinfo->ri >= 0 && navinfo->ai >= 0) {
      const char* source = PolarVolume_getSource((PolarVolume_t*)obj);
      if (source != NULL) {
        pooscan = (PolarScan_t*)RaveObjectHashTable_get(this->poofields, source);
      }
    } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE) && navinfo->ri >= 0 && navinfo->ai >= 0) {
      const char* source = PolarScan_getSource((PolarScan_t*)obj);
      if (source != NULL ) {
        pooscan = (PolarScan_t*)RaveObjectHashTable_get(this->poofields, source);
      }
    }
    if (pooscan != NULL) {
      double value = 0.0;
      RaveValueType t = PolarScan_getNearest(pooscan, navinfo->lon, navinfo->lat, 1, &value);

      if (t != RaveValueType_DATA) {
        value = 0.0;
      }

      *v = (value - HOWTASK_POO_OFFSET) / HOWTASK_POO_GAIN;

      result = 1;
    }
  }

  RAVE_OBJECT_RELEASE(pooscan);
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
