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

int RateCompositeGeneratorFactory_getLonLat(void* extradata, RaveCoreObject* object, ProjectionPipeline_t* pipeline, double herex, double herey, double* olon, double* olat);

int RateCompositeGeneratorFactory_selectRadarData(void* extradata, RaveCoreObject* object, double olon, double olat, CompositeUtilValue_t* cvalues, int ncvalues);

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

  if (!CompositeEngine_setLonLatFunction(this->engine, &RateCompositeGeneratorFactory_getLonLat)) {
    RAVE_ERROR0("Failed to set lon lat function!!!!");
    goto fail;
  }

  if (!CompositeEngine_setSelectRadarDataFunction(this->engine, &RateCompositeGeneratorFactory_selectRadarData)) {
    RAVE_ERROR0("Failed to set select radar data function!!!!");
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

  if (!CompositeEngine_setLonLatFunction(this->engine, &RateCompositeGeneratorFactory_getLonLat)) {
    RAVE_ERROR0("Failed to set lon lat function!!!!");
    goto fail;
  }

  if (!CompositeEngine_setSelectRadarDataFunction(this->engine, &RateCompositeGeneratorFactory_selectRadarData)) {
    RAVE_ERROR0("Failed to set select radar data function!!!!");
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

static CompositeInterpolationMethod_t RateCompositeGeneratorFactoryInternal_getInterpolationMethod(const char* method)
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

int RateCompositeGeneratorFactory_getLonLat(void* extradata, RaveCoreObject* object, ProjectionPipeline_t* pipeline, double herex, double herey, double* olon, double* olat)
{
  fprintf(stderr, "Getting lon/lat data\n");
  return 0;
}

int RateCompositeGeneratorFactory_selectRadarData(void* extradata, RaveCoreObject* object, double olon, double olat, CompositeUtilValue_t* cvalues, int ncvalues)
{
  fprintf(stderr, "Getting select radar data\n");
  return 0;
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
//   Composite_t* composite = NULL;
//   Cartesian_t* result = NULL;
//   RaveAttribute_t* attr = NULL;
//   Area_t* area = NULL;
//   RaveList_t* qualityflags = NULL;

//   RAVE_ASSERT((self != NULL), "self == NULL");
//   if (arguments == NULL) {
//     RAVE_ERROR0("Must provide arguments when generating the composite");
//     return NULL;
//   }
//   area = CompositeArguments_getArea(arguments);
//   if (area == NULL) {
//     RAVE_ERROR0("Missing area in arguments");
//     return NULL;
//   }

//   composite = RAVE_OBJECT_NEW(&Composite_TYPE);
//   if (composite != NULL) {
//     int i = 0;
//     int nobjects = CompositeArguments_getNumberOfObjects(arguments);
//     int nrparams = CompositeArguments_getParameterCount(arguments);
//     Rave_ProductType prodtype = CompositeArguments_getProductType(arguments);

//     if (prodtype == Rave_ProductType_UNDEFINED) {
//       if (CompositeArguments_getProduct(arguments) != NULL) {
//         RAVE_ERROR1("Can't support product: %s\n", CompositeArguments_getProduct(arguments));
//         goto done;
//       } else {
//         RAVE_ERROR0("No product has been set\n");
//         goto done;
//       }
//     }
//     Composite_setProduct(composite, prodtype);

//     Composite_setSelectionMethod(composite, CompositeSelectionMethod_NEAREST);
//     attr = CompositeArguments_getArgument(arguments, "selection_method");
//     if (attr != NULL) {
//       char* v = NULL;
//       if (RaveAttribute_getString(attr, &v)) {
//         if (strcasecmp("HEIGHT_ABOVE_SEALEVEL", v) == 0) {
//           Composite_setSelectionMethod(composite, CompositeSelectionMethod_HEIGHT);
//         }
//       }
//       RAVE_OBJECT_RELEASE(attr);
//     }

//     Composite_setInterpolationMethod(composite, CompositeInterpolationMethod_NEAREST);
//     attr = CompositeArguments_getArgument(arguments, "interpolation_method");
//     if (attr != NULL) {
//       char* v = NULL;
//       if (RaveAttribute_getString(attr, &v)) {
//         Composite_setInterpolationMethod(composite, LegacyCompositeGeneratorFactoryInternal_getInterpolationMethod(v));
//       }
//       RAVE_OBJECT_RELEASE(attr);
//     }

//     attr = CompositeArguments_getArgument(arguments, "quality_indicator_field");
//     if (attr != NULL) {
//       char* v = NULL;
//       if (RaveAttribute_getString(attr, &v)) {
//         Composite_setQualityIndicatorFieldName(composite, v);
//       }
//       RAVE_OBJECT_RELEASE(attr);
//     }

//     Composite_setHeight(composite, CompositeArguments_getHeight(arguments));
//     Composite_setElevationAngle(composite, CompositeArguments_getElevationAngle(arguments));
//     Composite_setRange(composite, CompositeArguments_getRange(arguments));

//     attr = CompositeArguments_getArgument(arguments, "quality_indicator_field");
//     if (attr != NULL) {
//       char* v = NULL;
//       if (RaveAttribute_getString(attr, &v)) {
//         Composite_setQualityIndicatorFieldName(composite, v);
//       }
//       RAVE_OBJECT_RELEASE(attr);
//     }

//     for (i = 0; i < nrparams; i++) {
//       double gain = 0.0, offset = 0.0, nodata = 0.0, undetect = 0.0;
//       RaveDataType datatype = RaveDataType_UCHAR;
//       const char* quantity = CompositeArguments_getParameterAtIndex(arguments, i, &gain, &offset, &datatype, &nodata, &undetect);
//       if (!Composite_addParameter(composite, quantity, gain, offset, -30.0)) {
//         RAVE_ERROR0("Could not add parameter to composite generator");
//         goto done;
//       }
//     }

//     for (i = 0; i < nobjects; i++) {
//       RaveCoreObject* object = CompositeArguments_getObject(arguments, i);
//       if (object == NULL || !Composite_add(composite, object)) {
//         RAVE_ERROR0("Failed to add object to composite");
//         RAVE_OBJECT_RELEASE(object);
//         goto done;
//       }
//       RAVE_OBJECT_RELEASE(object);
//     }
    
//     Composite_setTime(composite, CompositeArguments_getTime(arguments));
//     Composite_setDate(composite, CompositeArguments_getDate(arguments));

//     qualityflags = CompositeArguments_getQualityFlags(arguments);

//     result = Composite_generate(composite, area, qualityflags);
//   }
// done:
//   RAVE_OBJECT_RELEASE(composite);
//   RAVE_OBJECT_RELEASE(attr);
//   RAVE_OBJECT_RELEASE(area);
//   if (qualityflags != NULL) {
//     RaveList_freeAndDestroy(&qualityflags);
//   }

//   return result;
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
