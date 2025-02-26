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
 * Provides functionality for creating acqva composites.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-14
 */
#include "acqvacompositegeneratorfactory.h"
#include "composite_utils.h"
#include "compositearguments.h"
#include "compositeengine.h"
#include "compositeengineqc.h"
#include "polarvolume.h"
#include "raveobject_list.h"
#include "rave_types.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_utilities.h"
#include <stdio.h>
#include <string.h>



typedef struct _AcqvaCompositeGeneratorFactory_t {
  RAVE_OBJECT_HEAD /**< Always on top */
  COMPOSITE_GENERATOR_FACTORY_HEAD /**< composite generator plugin specifics */
  CompositeEngine_t* engine; /**<the engine */
  CompositeEngineOvershootingQcHandler_t* overshooting;
} AcqvaCompositeGeneratorFactory_t;
/*@{ Private functions */

static int AcqvaCompositeGeneratorFactory_onStarting(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings);

/**
 * The function locating the lowest usable data value
 */
static int AcqvaCompositeGeneratorFactoryInternal_selectRadarData(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, int index, double olon, double olat, struct CompositeEngineRadarData_t* cvalues, int ncvalues);

static int AcqvaCompositeGeneratorFactory_getQualityValue(CompositeEngine_t* self, void* extradata, CompositeArguments_t* args, RaveCoreObject* obj, const char* quantity, const char* qfieldname, PolarNavigationInfo* navinfo, double* v);

/**
 * The quality field that contains the acqva information.
 */
#define ACQVA_QUALITY_FIELD_NAME "se.smhi.acqva"

/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int AcqvaCompositeGeneratorFactory_constructor(RaveCoreObject* obj)
{
  AcqvaCompositeGeneratorFactory_t* this = (AcqvaCompositeGeneratorFactory_t*)obj;
  this->getName = AcqvaCompositeGeneratorFactory_getName;
  this->getDefaultId = AcqvaCompositeGeneratorFactory_getDefaultId;
  this->canHandle = AcqvaCompositeGeneratorFactory_canHandle;
  this->setProperties = AcqvaCompositeGeneratorFactory_setProperties;
  this->getProperties = AcqvaCompositeGeneratorFactory_getProperties;
  this->generate = AcqvaCompositeGeneratorFactory_generate;
  this->create = AcqvaCompositeGeneratorFactory_create;
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

  if (!CompositeEngine_setOnStartingFunction(this->engine, AcqvaCompositeGeneratorFactory_onStarting)) {
    RAVE_ERROR0("Failed to set the onStarting function");
    goto fail;
  }

  if (!CompositeEngine_setSelectRadarDataFunction(this->engine, AcqvaCompositeGeneratorFactoryInternal_selectRadarData)) {
    RAVE_ERROR0("Failed to set selectRadarData function");
    goto fail;
  }

  if (!CompositeEngine_setGetQualityValueFunction(this->engine, AcqvaCompositeGeneratorFactory_getQualityValue)) {
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
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int AcqvaCompositeGeneratorFactory_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  AcqvaCompositeGeneratorFactory_t* this = (AcqvaCompositeGeneratorFactory_t*)obj;
  AcqvaCompositeGeneratorFactory_t* src = (AcqvaCompositeGeneratorFactory_t*)srcobj;
  this->getName = AcqvaCompositeGeneratorFactory_getName;
  this->getDefaultId = AcqvaCompositeGeneratorFactory_getDefaultId;
  this->canHandle = AcqvaCompositeGeneratorFactory_canHandle;
  this->setProperties = AcqvaCompositeGeneratorFactory_setProperties;
  this->getProperties = AcqvaCompositeGeneratorFactory_getProperties;
  this->generate = AcqvaCompositeGeneratorFactory_generate;
  this->create = AcqvaCompositeGeneratorFactory_create;
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

  if (!CompositeEngine_setOnStartingFunction(this->engine, AcqvaCompositeGeneratorFactory_onStarting)) {
    RAVE_ERROR0("Failed to set the onStarting function");
    goto fail;
  }

  if (!CompositeEngine_setSelectRadarDataFunction(this->engine, AcqvaCompositeGeneratorFactoryInternal_selectRadarData)) {
    RAVE_ERROR0("Failed to set selectRadarData function");
    goto fail;
  }

  if (!CompositeEngine_setGetQualityValueFunction(this->engine, AcqvaCompositeGeneratorFactory_getQualityValue)) {
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
static void AcqvaCompositeGeneratorFactory_destructor(RaveCoreObject* obj)
{
  AcqvaCompositeGeneratorFactory_t* this = (AcqvaCompositeGeneratorFactory_t*)obj;
  RAVE_OBJECT_RELEASE(this->engine);
  RAVE_OBJECT_RELEASE(this->overshooting);
}

static int AcqvaCompositeGeneratorFactory_onStarting(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings)
{
  AcqvaCompositeGeneratorFactory_t* self = (AcqvaCompositeGeneratorFactory_t*)extradata;
  RaveProperties_t* properties = CompositeEngine_getProperties(engine);

  CompositeEngineQcHandler_initialize(self->overshooting, extradata, properties, arguments, bindings, nbindings);

  RAVE_OBJECT_RELEASE(properties);

  return 1;
}

/**
 * Identifies the lowest possible value to use in this volume. It will use the quality field named
 * ACQVA_QUALITY_FIELD_NAME. If found quality value at specified lon/lat is != 0, it will assume
 * that the value is usable. Note, this requires that the volume is sorted in asacending order.
 * 
 * @param[in] self - self
 * @param[in] pvol - the polar volume
 * @param[in] lon - the longitude
 * @param[in] lat - the latitude
 * @param[out] height - the height for the found position (only set if value found)
 * @param[out] elangle - the elevation angle for the found position (only set if value found)
 * @param[out] ray - the azimuth index for the found position (only set if value found)
 * @param[out] bin - the bin index for the found position (only set if value found)
 * @param[out] eindex - the elevation index in the vollume for the found position (only set if value found)
 * @param[in,out] navinfo  - all the relevant navinfo for the found value
 * @return 1 if value is found, otherwise 0
 */
static int AcqvaCompositeGeneratorFactoryInternal_findLowestUsableValue(AcqvaCompositeGeneratorFactory_t* self, PolarVolume_t* pvol, 
  double lon, double lat, double* height, 
  double* elangle, int* ray, int* bin, int* eindex, PolarNavigationInfo* outnavinfo)
{
  int nrelevs = 0, i = 0, found = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (pvol == NULL) {
    RAVE_ERROR0("Providing pvol == NULL");
    return 0;
  }
  nrelevs = PolarVolume_getNumberOfScans(pvol);
  for (i = 0; !found && i < nrelevs; i++) {
    PolarNavigationInfo navinfo;
    PolarScan_t* scan = PolarVolume_getScan(pvol, i);
    if (PolarScan_getNearestNavigationInfo(scan, lon, lat, &navinfo)) {
      double v = 0.0;
      if (PolarScan_getQualityValueAt(scan, NULL, navinfo.ri, navinfo.ai, ACQVA_QUALITY_FIELD_NAME, 1, &v)) {
        if (v != 0.0) {
          *height = navinfo.actual_height;
          *elangle = navinfo.elevation;
          *ray = navinfo.ai;
          *bin = navinfo.ri;
          *eindex = i;
          *outnavinfo = navinfo;
          found = 1;
        }
      }
    }
    RAVE_OBJECT_RELEASE(scan);
  }
  return found;
}

/**
 * Assumes that the binding contains a polar volume. If the polar volume contains a usable value (located from bottom and up) it will
 * compare the values height with the previously stored minimum distance. If the new height is closer to the ground, then this value
 * and navigation information will be saved. 
 * @param[in] engine - self
 * @param[in] arguments - the arguments
 * @param[in] binding - the binding between volume, pipeline and other miscellaneous values
 * @param[in] index - the index of the object. The binding is basically &bindings[index].
 * @param[in] olon - the longitude for data
 * @param[in] olat - the latitude for data
 * @param[in] cvalues - the array of cartesian parameter values
 * @param[in] ncvalues - number of items in the array cvalues
 * @return always 1
 */
static int AcqvaCompositeGeneratorFactoryInternal_selectRadarData(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, int index, double olon, double olat, struct CompositeEngineRadarData_t* cvalues, int ncvalues)
{
  double dist = 0.0, maxdist = 0.0;
  AcqvaCompositeGeneratorFactory_t* self = (AcqvaCompositeGeneratorFactory_t*)extradata;

  dist = PolarVolume_getDistance((PolarVolume_t*)binding->object, olon, olat);
  maxdist = PolarVolume_getMaxDistance((PolarVolume_t*)binding->object);

  if (dist <= maxdist) {
    double height=0.0, elangle=0.0;
    int ray=0, bin=0, eindex=0, cindex = 0;
    PolarNavigationInfo navinfo;
    if (AcqvaCompositeGeneratorFactoryInternal_findLowestUsableValue(self, (PolarVolume_t*)binding->object, olon, olat, &height, &elangle, &ray, &bin, &eindex, &navinfo)) {
      for (cindex = 0; cindex < ncvalues; cindex++) {
        RaveValueType otype = RaveValueType_NODATA;
        double v = 0.0;
        otype = PolarVolume_getConvertedParameterValueAt((PolarVolume_t*)binding->object, cvalues[cindex].name, eindex, bin, ray, &v);
        if (otype != RaveValueType_NODATA) {
          if (cvalues[cindex].mindist > height) {
            cvalues[cindex].mindist = height;
            cvalues[cindex].value = v;
            cvalues[cindex].vtype = otype;
            cvalues[cindex].navinfo = navinfo;
            cvalues[cindex].radarindex = index;
            cvalues[cindex].radardist = cvalues[cindex].navinfo.actual_range;
          }
        }
      }
    }
  }
  return 1;
}

static int AcqvaCompositeGeneratorFactory_getQualityValue(CompositeEngine_t* self, void* extradata, CompositeArguments_t* args, RaveCoreObject* obj, const char* quantity, const char* qfieldname, PolarNavigationInfo* navinfo, double* v)
{
  int result = 0;
  AcqvaCompositeGeneratorFactory_t* this = (AcqvaCompositeGeneratorFactory_t*)extradata;

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

const char* AcqvaCompositeGeneratorFactory_getName(CompositeGeneratorFactory_t* self)
{
  return "AcqvaCompositeGenerator";
}

/**
 * @returns the default id of this factory
 */
const char* AcqvaCompositeGeneratorFactory_getDefaultId(CompositeGeneratorFactory_t* self)
{
  return "acqva";
}

int AcqvaCompositeGeneratorFactory_canHandle(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments)
{
  const char* productid;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (arguments == NULL) {
    return 0;
  }
  productid = CompositeArguments_getProduct(arguments);
  if (productid == NULL || strcasecmp("ACQVA", productid) != 0) {
    return 0;
  }
  return 1;
}

int AcqvaCompositeGeneratorFactory_setProperties(CompositeGeneratorFactory_t* self, RaveProperties_t* properties)
{
  AcqvaCompositeGeneratorFactory_t* factory = (AcqvaCompositeGeneratorFactory_t*)self;
  RAVE_ASSERT((factory != NULL), "self == NULL");
  CompositeEngine_setProperties(factory->engine, properties);
  return 1;
}

RaveProperties_t* AcqvaCompositeGeneratorFactory_getProperties(CompositeGeneratorFactory_t* self)
{
  AcqvaCompositeGeneratorFactory_t* factory = (AcqvaCompositeGeneratorFactory_t*)self;
  RAVE_ASSERT((factory != NULL), "factory == NULL");
  return CompositeEngine_getProperties(factory->engine);
}

Cartesian_t* AcqvaCompositeGeneratorFactory_generate(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments)
{
  int i = 0, nobjects = 0;

  nobjects = CompositeArguments_getNumberOfObjects(arguments);
  for (i = 0; i < nobjects; i++) {
    RaveCoreObject* obj = CompositeArguments_getObject(arguments, i);
    if (!RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
      RAVE_ERROR0("Acqva can only process volumes");
      RAVE_OBJECT_RELEASE(obj);
      return NULL;
    }
    RAVE_OBJECT_RELEASE(obj);
  }
  return CompositeEngine_generate(((AcqvaCompositeGeneratorFactory_t*)self)->engine, arguments, (void*)self);
}

CompositeGeneratorFactory_t* AcqvaCompositeGeneratorFactory_create(CompositeGeneratorFactory_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (RAVE_OBJECT_CHECK_TYPE(self, &AcqvaCompositeGeneratorFactory_TYPE)) {
    return RAVE_OBJECT_CLONE(self);
  }
  RAVE_ERROR0("Something is wrong in implementation. Should not arrive here unless type is a AcqvaCompositeGeneratorFactory_TYPE");
  return NULL;
}

/*@} End of Interface functions */

RaveCoreObjectType AcqvaCompositeGeneratorFactory_TYPE = {
    "AcqvaCompositeGeneratorFactory",
    sizeof(AcqvaCompositeGeneratorFactory_t),
    AcqvaCompositeGeneratorFactory_constructor,
    AcqvaCompositeGeneratorFactory_destructor,
    AcqvaCompositeGeneratorFactory_copyconstructor
};
