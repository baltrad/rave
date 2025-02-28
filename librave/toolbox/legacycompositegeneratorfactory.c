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
#include "legacycompositegeneratorfactory.h"
#include "compositearguments.h"
#include "polarvolume.h"
#include "rave_attribute.h"
#include "rave_list.h"
#include "rave_object.h"
#include "rave_properties.h"
#include "raveobject_list.h"
#include "rave_types.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_utilities.h"
#include "composite.h"
#include <strings.h>
#include <string.h>
#include <stdio.h>
#include "poo_composite_algorithm.h"

static const char* SUPPORTED_PRODUCTS[]={
  "PPI",
  "CAPPI",
  "PCAPPI",
  "MAX",
  "PMAX",
  NULL
};

typedef struct _LegacyCompositeGeneratorFactory_t {
  RAVE_OBJECT_HEAD /**< Always on top */
  COMPOSITE_GENERATOR_FACTORY_HEAD /**< composite generator plugin specifics */
  RaveProperties_t* properties; /**< the properties */
} LegacyCompositeGeneratorFactory_t;


typedef struct InterpolationMethodMapping_t {
  const char* methodstr;
  CompositeInterpolationMethod_t method;
} InterpolationMethodMapping_t;

static InterpolationMethodMapping_t INTERPOLATION_MAPPING[] = {
  {"NEAREST", CompositeInterpolationMethod_NEAREST},
  {"LINEAR_HEIGHT", CompositeInterpolationMethod_LINEAR_HEIGHT},
  {"LINEAR_RANGE", CompositeInterpolationMethod_LINEAR_RANGE},
  {"LINEAR_AZIMUTH", CompositeInterpolationMethod_LINEAR_AZIMUTH},
  {"LINEAR_RANGE_AND_AZIMUTH", CompositeInterpolationMethod_LINEAR_RANGE_AND_AZIMUTH},
  {"LINEAR_3D", CompositeInterpolationMethod_LINEAR_3D},
  {"QUADRATIC_HEIGHT", CompositeInterpolationMethod_QUADRATIC_HEIGHT},
  {"QUADRATIC_3D", CompositeInterpolationMethod_QUADRATIC_3D},
  {NULL, CompositeInterpolationMethod_NEAREST}
};


/*@{ Private functions */
/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int LegacyCompositeGeneratorFactory_constructor(RaveCoreObject* obj)
{
  LegacyCompositeGeneratorFactory_t* this = (LegacyCompositeGeneratorFactory_t*)obj;
  this->getName = LegacyCompositeGeneratorFactory_getName;
  this->getDefaultId = LegacyCompositeGeneratorFactory_getDefaultId;
  this->canHandle = LegacyCompositeGeneratorFactory_canHandle;
  this->setProperties = LegacyCompositeGeneratorFactory_setProperties;
  this->getProperties = LegacyCompositeGeneratorFactory_getProperties;
  this->generate = LegacyCompositeGeneratorFactory_generate;
  this->create = LegacyCompositeGeneratorFactory_create;
  this->properties = NULL;
  return 1;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int LegacyCompositeGeneratorFactory_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  LegacyCompositeGeneratorFactory_t* this = (LegacyCompositeGeneratorFactory_t*)obj;
  LegacyCompositeGeneratorFactory_t* src = (LegacyCompositeGeneratorFactory_t*)srcobj;
  this->getName = LegacyCompositeGeneratorFactory_getName;
  this->getDefaultId = LegacyCompositeGeneratorFactory_getDefaultId;
  this->canHandle = LegacyCompositeGeneratorFactory_canHandle;
  this->setProperties = LegacyCompositeGeneratorFactory_setProperties;
  this->getProperties = LegacyCompositeGeneratorFactory_getProperties;
  this->generate = LegacyCompositeGeneratorFactory_generate;
  this->create = LegacyCompositeGeneratorFactory_create;
  this->properties = NULL;
  if (src->properties != NULL) {
    this->properties = RAVE_OBJECT_CLONE(src->properties);
    if (this->properties == NULL) {
      RAVE_ERROR0("Failed to clone properties");
      goto fail;
    }
  }
  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->properties);
  return 0;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void LegacyCompositeGeneratorFactory_destructor(RaveCoreObject* obj)
{
  LegacyCompositeGeneratorFactory_t* this = (LegacyCompositeGeneratorFactory_t*)obj;
  RAVE_OBJECT_RELEASE(this->properties);
}

static CompositeInterpolationMethod_t LegacyCompositeGeneratorFactoryInternal_getInterpolationMethod(const char* method)
{
  int ctr = 0;
  while (INTERPOLATION_MAPPING[ctr].methodstr != NULL) {
    if (strcasecmp(INTERPOLATION_MAPPING[ctr].methodstr, method) == 0) {
      return INTERPOLATION_MAPPING[ctr].method;
    }
    ctr++;
  }
  return CompositeInterpolationMethod_NEAREST;
}

/*@} End of Private functions */

/*@{ Interface functions */

/**
 * @returns the unique name for this composite generator plugin
 */
const char* LegacyCompositeGeneratorFactory_getName(CompositeGeneratorFactory_t* self)
{
  return "LegacyCompositeGenerator";
}


/**
 * @returns the default id of this factory
 */
const char* LegacyCompositeGeneratorFactory_getDefaultId(CompositeGeneratorFactory_t* self)
{
  return "legacy";
}

/**
 * @returns if this plugin can handle the generator request
 */
int LegacyCompositeGeneratorFactory_canHandle(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments)
{
  const char* productid;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (arguments == NULL) {
    return 0;
  }
  productid = CompositeArguments_getProduct(arguments);
  if (productid != NULL) {
    if (!RaveUtilities_arrayContains(SUPPORTED_PRODUCTS, productid, 0)) {
      return 0;
    }
    if (strcasecmp("PMAX", productid) == 0 || strcasecmp("MAX", productid) == 0) {
      RaveAttribute_t* attr = CompositeArguments_getArgument(arguments, "interpolation_method");
      if (attr != NULL) {
        char* value = NULL;
        RaveAttribute_getString(attr, &value);
        if (value == NULL || strcasecmp("NEAREST", value) != 0) {
          RAVE_OBJECT_RELEASE(attr);
          return 0;
        }
      }
      RAVE_OBJECT_RELEASE(attr);
    }
  }

  return 1;
}

int LegacyCompositeGeneratorFactory_setProperties(CompositeGeneratorFactory_t* self, RaveProperties_t* properties)
{
  LegacyCompositeGeneratorFactory_t* factory = (LegacyCompositeGeneratorFactory_t*)self;
  RAVE_ASSERT((factory != NULL), "self == NULL");
  RAVE_OBJECT_RELEASE(factory->properties);
  if (properties != NULL) {
    factory->properties = RAVE_OBJECT_COPY(properties);
  }
  return 1;
}

RaveProperties_t* LegacyCompositeGeneratorFactory_getProperties(CompositeGeneratorFactory_t* self)
{
  LegacyCompositeGeneratorFactory_t* factory = (LegacyCompositeGeneratorFactory_t*)self;
  RAVE_ASSERT((factory != NULL), "factory == NULL");
  return RAVE_OBJECT_COPY(factory->properties);
}

/**
 * @returns the result from the generation
 */
Cartesian_t* LegacyCompositeGeneratorFactory_generate(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments)
{
  Composite_t* composite = NULL;
  Cartesian_t* result = NULL;
  RaveAttribute_t* attr = NULL;
  Area_t* area = NULL;
  RaveList_t* qualityflags = NULL;
  CompositeAlgorithm_t* algorithm = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (arguments == NULL) {
    RAVE_ERROR0("Must provide arguments when generating the composite");
    return NULL;
  }
  area = CompositeArguments_getArea(arguments);
  if (area == NULL) {
    RAVE_ERROR0("Missing area in arguments");
    return NULL;
  }

  composite = RAVE_OBJECT_NEW(&Composite_TYPE);
  if (composite != NULL) {
    int i = 0;
    int nobjects = CompositeArguments_getNumberOfObjects(arguments);
    int nrparams = CompositeArguments_getParameterCount(arguments);
    Rave_ProductType prodtype = CompositeArguments_getProductType(arguments);

    if (prodtype == Rave_ProductType_UNDEFINED) {
      if (CompositeArguments_getProduct(arguments) != NULL) {
        RAVE_ERROR1("Can't support product: %s\n", CompositeArguments_getProduct(arguments));
        goto done;
      } else {
        RAVE_ERROR0("No product has been set\n");
        goto done;
      }
    }
    Composite_setProduct(composite, prodtype);

    Composite_setSelectionMethod(composite, CompositeSelectionMethod_NEAREST);
    attr = CompositeArguments_getArgument(arguments, "selection_method");
    if (attr != NULL) {
      char* v = NULL;
      if (RaveAttribute_getString(attr, &v)) {
        if (strcasecmp("HEIGHT_ABOVE_SEALEVEL", v) == 0) {
          Composite_setSelectionMethod(composite, CompositeSelectionMethod_HEIGHT);
        }
      }
      RAVE_OBJECT_RELEASE(attr);
    }

    Composite_setInterpolationMethod(composite, CompositeInterpolationMethod_NEAREST);
    attr = CompositeArguments_getArgument(arguments, "interpolation_method");
    if (attr != NULL) {
      char* v = NULL;
      if (RaveAttribute_getString(attr, &v)) {
        Composite_setInterpolationMethod(composite, LegacyCompositeGeneratorFactoryInternal_getInterpolationMethod(v));
      }
      RAVE_OBJECT_RELEASE(attr);
    }

    attr = CompositeArguments_getArgument(arguments, "quality_indicator_field");
    if (attr != NULL) {
      char* v = NULL;
      if (RaveAttribute_getString(attr, &v)) {
        Composite_setQualityIndicatorFieldName(composite, v);
      }
      RAVE_OBJECT_RELEASE(attr);
    }

    Composite_setHeight(composite, CompositeArguments_getHeight(arguments));
    Composite_setElevationAngle(composite, CompositeArguments_getElevationAngle(arguments));
    Composite_setRange(composite, CompositeArguments_getRange(arguments));

    attr = CompositeArguments_getArgument(arguments, "quality_indicator_field");
    if (attr != NULL) {
      char* v = NULL;
      if (RaveAttribute_getString(attr, &v)) {
        Composite_setQualityIndicatorFieldName(composite, v);
      }
      RAVE_OBJECT_RELEASE(attr);
    }

    for (i = 0; i < nrparams; i++) {
      double gain = 0.0, offset = 0.0, nodata = 0.0, undetect = 0.0;
      RaveDataType datatype = RaveDataType_UCHAR;
      const char* quantity = CompositeArguments_getParameterAtIndex(arguments, i, &gain, &offset, &datatype, &nodata, &undetect);
      if (!Composite_addParameter(composite, quantity, gain, offset, -30.0)) {
        RAVE_ERROR0("Could not add parameter to composite generator");
        goto done;
      }
    }

    for (i = 0; i < nobjects; i++) {
      RaveCoreObject* object = CompositeArguments_getObject(arguments, i);
      if (object == NULL || !Composite_add(composite, object)) {
        RAVE_ERROR0("Failed to add object to composite");
        RAVE_OBJECT_RELEASE(object);
        goto done;
      }
      RAVE_OBJECT_RELEASE(object);
    }
    
    Composite_setTime(composite, CompositeArguments_getTime(arguments));
    Composite_setDate(composite, CompositeArguments_getDate(arguments));

    algorithm = RAVE_OBJECT_NEW(&PooCompositeAlgorithm_TYPE);
    if (algorithm == NULL) {
      RAVE_ERROR0("Failed to add poo algorithm to composite generator");
      goto done;
    }
    Composite_setAlgorithm(composite, algorithm);

    qualityflags = CompositeArguments_getQualityFlags(arguments);

    result = Composite_generate(composite, area, qualityflags);
  }
done:
  RAVE_OBJECT_RELEASE(composite);
  RAVE_OBJECT_RELEASE(algorithm);
  RAVE_OBJECT_RELEASE(attr);
  RAVE_OBJECT_RELEASE(area);
  if (qualityflags != NULL) {
    RaveList_freeAndDestroy(&qualityflags);
  }

  return result;
}

/**
 * The initializing function so that we know what composite generator we are working with.
 * Note, this function will be called by the composite module and will override any previous calls.
 * @return 1 on success otherwise 0
 */
CompositeGeneratorFactory_t* LegacyCompositeGeneratorFactory_create(CompositeGeneratorFactory_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (RAVE_OBJECT_CHECK_TYPE(self, &LegacyCompositeGeneratorFactory_TYPE)) {
    return RAVE_OBJECT_CLONE(self);
  }
  RAVE_ERROR0("Something is wrong in implementation. Should not arrive here unless type is a LegacyCompositeGeneratorFactory_TYPE");
  return NULL;
}

/*@} End of Interface functions */

RaveCoreObjectType LegacyCompositeGeneratorFactory_TYPE = {
    "LegacyCompositeGeneratorFactory",
    sizeof(LegacyCompositeGeneratorFactory_t),
    LegacyCompositeGeneratorFactory_constructor,
    LegacyCompositeGeneratorFactory_destructor,
    LegacyCompositeGeneratorFactory_copyconstructor
};
