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
 * Provides a base engine for creating composites.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-31
 */
#include "compositeengine.h"
#include "composite_utils.h"
#include "compositearguments.h"
#include "polarvolume.h"
#include "rave_properties.h"
#include "raveobject_hashtable.h"
#include "raveobject_list.h"
#include "rave_types.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_datetime.h"
#include "projection_pipeline.h"
#include <string.h>
#include "rave_field.h"
#include <float.h>
#include <stdio.h>
#include <math.h>


/**
 * Represents the cartesian product.
 */
struct _CompositeEngine_t {
  RAVE_OBJECT_HEAD /** Always on top */
  composite_engine_onStarting_fun onStarting; /**< Called just before the compositing loop is started. I.e. after basic setup like creating quality flags, object bindings ... */
  composite_engine_onFinished_fun onFinished; /**< Called just after the compositing loop has finished. */
  composite_engine_getLonLat_fun getLonLat;   /**< Calculates the lon/lat from the x/y surface coordinate */
  composite_engine_selectRadarData_fun selectRadarData; /**< From the lon/lat the wanted radar data is returned */
  composite_engine_getPolarValueAtPosition_fun getPolarValueAtPosition; /**< gets a polar value at specified position */
  composite_engine_setRadarData_fun setRadarData; /**< sets the radar data in the composite */
  composite_engine_addQualityFlagsToCartesian_fun addQualityFlagsToCartesian; /**< adds quality flags to the cartesian product */
  composite_engine_getQualityValue_fun getQualityValue; /**< returns the quality value at specified location */
  composite_engine_fillQualityInformation_fun fillQualityInformation; /**< fills the quality fields */
  RaveObjectHashTable_t* polarValueAtPositionMapping; /**< a mapping between requested quantity and a call for getPolarValueAtPosition */
  RaveProperties_t* properties; /**< properties that is required to setup the factory and engine */
  RaveObjectHashTable_t* qualityFlagDefinitions; /**< the quality flag definitions */
};

static CompositeQualityFlagSettings_t COMPOSITE_ENGINE_QUALITY_FLAG_DEFINITIONS[] = {
  {COMPOSITE_ENGINE_DISTANCE_TO_RADAR_HOW_TASK, RaveDataType_UCHAR, 0.0, COMPOSITE_ENGINE_DISTANCE_TO_RADAR_RESOLUTION},
  {COMPOSITE_ENGINE_HEIGHT_ABOVE_SEA_HOW_TASK, RaveDataType_UCHAR, 0.0, COMPOSITE_ENGINE_HEIGHT_RESOLUTION},
  {COMPOSITE_ENGINE_RADAR_INDEX_HOW_TASK, RaveDataType_UCHAR, 0.0, 1.0},
  {NULL, RaveDataType_UNDEFINED, 0.0, 0.0}
};

/*@{ Private CompositeEnginePolarValueFunction class */
/**
 * The class definition
 */
typedef struct CompositeEnginePolarValueFunction_t {
  RAVE_OBJECT_HEAD /** Always on top */
  composite_engine_getPolarValueAtPosition_fun getPolarValueAtPosition;  
} CompositeEnginePolarValueFunction_t;

static int CompositeEnginePolarValueFunction_constructor(RaveCoreObject* obj)
{
  CompositeEnginePolarValueFunction_t* this = (CompositeEnginePolarValueFunction_t*)obj;
  this->getPolarValueAtPosition = NULL;
  return 1;
}
static int CompositeEnginePolarValueFunction_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  CompositeEnginePolarValueFunction_t* this = (CompositeEnginePolarValueFunction_t*)obj;
  CompositeEnginePolarValueFunction_t* src = (CompositeEnginePolarValueFunction_t*)srcobj;
  this->getPolarValueAtPosition = src->getPolarValueAtPosition;
  return 1;
}
static void CompositeEnginePolarValueFunction_destructor(RaveCoreObject* obj)
{
  // CompositeEnginePolarValueFunction_t* this = (CompositeEnginePolarValueFunction_t*)obj;
}

RaveCoreObjectType CompositeEnginePolarValueFunction_TYPE = {
    "CompositeEnginePolarValueFunction",
    sizeof(CompositeEnginePolarValueFunction_t),
    CompositeEnginePolarValueFunction_constructor,
    CompositeEnginePolarValueFunction_destructor,
    CompositeEnginePolarValueFunction_copyconstructor
};
/*@} End of Private CompositeEnginePolarValueFunction class */

/*@{ Private functions */

static int CompositeEngineInternal_onStarting(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings);

static int CompositeEngineInternal_onFinished(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings);

static int CompositeEngineInternal_getLonLat(CompositeEngine_t* engine, void* extradata, CompositeEngineObjectBinding_t* binding, double herex, double herey, double* olon, double* olat);

static int CompositeEngineInternal_selectRadarData(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, int index, double olon, double olat, CompositeEngineRadarData_t* cvalues, int ncvalues);

static int CompositeEngineInternal_getPolarValueAtPosition(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue);

static int CompositeEngineInternal_setRadarData(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, Cartesian_t* cartesian, double olon, double olat, long x, long y, CompositeEngineRadarData_t* cvalues, int ncvalues);

static int CompositeEngineInternal_getQualityValue(CompositeEngine_t* self, void* extradata, CompositeArguments_t* args, RaveCoreObject* obj, const char* quantity, const char* qfieldname, PolarNavigationInfo* navinfo, double* v);

static int CompositeEngineInternal_addQualityFlagsToCartesian(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, Cartesian_t* cartesian);

static int CompositeEngineInternal_fillQualityInformation(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, long x, long y, CartesianParam_t* param, double radardist, int radarindex, PolarNavigationInfo* info);



/**
 * Registers the default quality flag definitions.
 */
static int CompositeEngineInternal_registerDefaultQualityFlagDefinitions(CompositeEngine_t* engine)
{
  return CompositeUtils_registerQualityFlagDefinitionFromSettings(engine->qualityFlagDefinitions, COMPOSITE_ENGINE_QUALITY_FLAG_DEFINITIONS);
}

/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int CompositeEngine_constructor(RaveCoreObject* obj)
{
  CompositeEngine_t* this = (CompositeEngine_t*)obj;
  this->onStarting = CompositeEngineInternal_onStarting;
  this->onFinished = CompositeEngineInternal_onFinished;
  this->getLonLat = CompositeEngineInternal_getLonLat;
  this->selectRadarData = CompositeEngineInternal_selectRadarData;
  this->getPolarValueAtPosition = CompositeEngineInternal_getPolarValueAtPosition;
  this->setRadarData = CompositeEngineInternal_setRadarData;
  this->addQualityFlagsToCartesian = CompositeEngineInternal_addQualityFlagsToCartesian;
  this->getQualityValue = CompositeEngineInternal_getQualityValue;
  this->fillQualityInformation = CompositeEngineInternal_fillQualityInformation;
  this->properties = NULL;
  this->qualityFlagDefinitions = NULL;

  this->polarValueAtPositionMapping = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  this->qualityFlagDefinitions = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  if (this->polarValueAtPositionMapping == NULL || this->qualityFlagDefinitions == NULL) {
    RAVE_ERROR0("Failed to allocate memory for mapping");
    goto fail;
  }

  if (!CompositeEngineInternal_registerDefaultQualityFlagDefinitions(this)) {
    goto fail;
  }

  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->polarValueAtPositionMapping);
  RAVE_OBJECT_RELEASE(this->qualityFlagDefinitions);
  return 0;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int CompositeEngine_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  CompositeEngine_t* this = (CompositeEngine_t*)obj;
  CompositeEngine_t* src = (CompositeEngine_t*)srcobj;
  this->onStarting = src->onStarting;
  this->onFinished = src->onFinished;
  this->getLonLat = src->getLonLat;
  this->selectRadarData = src->selectRadarData;
  this->getPolarValueAtPosition = src->getPolarValueAtPosition; 
  this->setRadarData = src->setRadarData;
  this->addQualityFlagsToCartesian = src->addQualityFlagsToCartesian;
  this->getQualityValue = src->getQualityValue;
  this->fillQualityInformation = src->fillQualityInformation;
  this->properties = NULL;

  this->polarValueAtPositionMapping = RAVE_OBJECT_CLONE(src->polarValueAtPositionMapping);
  this->qualityFlagDefinitions = RAVE_OBJECT_CLONE(src->qualityFlagDefinitions);
  if (this->polarValueAtPositionMapping == NULL  || this->qualityFlagDefinitions == NULL) {
    RAVE_ERROR0("Failed to allocate memory for mapping");
    goto fail;
  }

  if (src->properties != NULL) {
    this->properties = RAVE_OBJECT_CLONE(src->properties);
    if (this->properties == NULL) {
      RAVE_ERROR0("Failed to clone properties");
      goto fail;
    }
  }
  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->polarValueAtPositionMapping);
  RAVE_OBJECT_RELEASE(this->qualityFlagDefinitions);
  RAVE_OBJECT_RELEASE(this->properties);
  return 0;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void CompositeEngine_destructor(RaveCoreObject* obj)
{
  CompositeEngine_t* this = (CompositeEngine_t*)obj;
  RAVE_OBJECT_RELEASE(this->polarValueAtPositionMapping);
  RAVE_OBJECT_RELEASE(this->properties);
  RAVE_OBJECT_RELEASE(this->qualityFlagDefinitions);
}

/**
 * Returns the position that is closest to the specified lon/lat according to
 * the composites attributes like type/elevation/height/etc.
 * @param[in] composite - self
 * @param[in] object - the data object
 * @param[in] plon - the longitude
 * @param[in] plat - the latitude
 * @param[out] nav - the navigation information
 * @return 1 if hit or 0 if outside
 */
static int CompositeEngineInternal_nearestPosition(
  CompositeArguments_t* arguments,
  RaveCoreObject* object,
  double plon,
  double plat,
  PolarNavigationInfo* nav)
{
  int result = 0;
  Rave_CompositingProduct productType = Rave_CompositingProduct_UNDEFINED;

  RAVE_ASSERT((arguments != NULL), "composite == NULL");
  RAVE_ASSERT((nav != NULL), "nav == NULL");
  
  productType = CompositeArguments_getCompositingProduct(arguments);

  if (object != NULL) {
    if (RAVE_OBJECT_CHECK_TYPE(object, &PolarScan_TYPE)) {
      if (productType == Rave_CompositingProduct_PPI ||
          productType == Rave_CompositingProduct_PCAPPI ||
          productType == Rave_CompositingProduct_PMAX) {
        result = PolarScan_getNearestNavigationInfo((PolarScan_t*)object, plon, plat, nav);
      }
    } else if (RAVE_OBJECT_CHECK_TYPE(object, &PolarVolume_TYPE)) {
      if (productType == Rave_CompositingProduct_PCAPPI ||
          productType == Rave_CompositingProduct_CAPPI ||
          productType == Rave_CompositingProduct_PMAX) {
        int insidee = (productType == Rave_CompositingProduct_PCAPPI || productType == Rave_CompositingProduct_PMAX)?0:1;
        result = PolarVolume_getNearestNavigationInfo((PolarVolume_t*)object,
                                                      plon,
                                                      plat,
                                                      CompositeArguments_getHeight(arguments),
                                                      insidee,
                                                      nav);
      } else if (productType == Rave_CompositingProduct_PPI) {
        PolarScan_t* scan = PolarVolume_getScanClosestToElevation((PolarVolume_t*)object,
                                                                  CompositeArguments_getElevationAngle(arguments),
                                                                  0);
        if (scan == NULL) {
          RAVE_ERROR1("Failed to fetch scan nearest to elevation %g",
                      CompositeArguments_getElevationAngle(arguments));
          goto done;
        }
        result = PolarScan_getNearestNavigationInfo((PolarScan_t*)scan, plon, plat, nav);
        nav->ei = PolarVolume_indexOf((PolarVolume_t*)object, scan);
        RAVE_OBJECT_RELEASE(scan);
      }
    }
  }

done:
  return result;
}

/*@{ Composite Engine internal function pointers */
static int CompositeEngineInternal_onStarting(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings)
{
  return 1;
}

static int CompositeEngineInternal_onFinished(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings)
{
  return 1;
}

static int CompositeEngineInternal_getLonLat(CompositeEngine_t* engine, void* extradata, CompositeEngineObjectBinding_t* binding, double herex, double herey, double* olon, double* olat)
{
  return CompositeEngineUtility_getLonLat(engine, binding, herex, herey, olon, olat);
}

static int CompositeEngineInternal_selectRadarData(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, int index, double olon, double olat, CompositeEngineRadarData_t* cvalues, int ncvalues)
{
  return CompositeEngineUtility_selectRadarData(engine, extradata, arguments, binding, index, olon, olat, cvalues, ncvalues);
}

static int CompositeEngineInternal_getPolarValueAtPosition(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue)
{
  return CompositeEngineUtility_getPolarValueAtPosition(engine, extradata, arguments, binding, quantity, navinfo, qiFieldName, otype, ovalue, qivalue);
}

static int CompositeEngineInternal_setRadarData(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, Cartesian_t* cartesian, double olon, double olat, long x, long y, CompositeEngineRadarData_t* cvalues, int ncvalues)
{
  return CompositeEngineUtility_setRadarData(engine, extradata, arguments, cartesian, olon, olat, x, y, cvalues, ncvalues);
}

static int CompositeEngineInternal_addQualityFlagsToCartesian(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, Cartesian_t* cartesian)
{
  return CompositeUtils_addQualityFlagsToCartesian(arguments, cartesian, engine->qualityFlagDefinitions);
}

static int CompositeEngineInternal_getQualityValue(CompositeEngine_t* self, void* extradata, CompositeArguments_t* args, RaveCoreObject* obj, const char* quantity, const char* qfieldname, PolarNavigationInfo* navinfo, double* v)
{
  /* Return 0 since we want the default behavior to handle the quality flag, see \ref CompositeEngineUtility_fillQualityInformation */
  return 0;
}

static int CompositeEngineInternal_fillQualityInformation(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, long x, long y, CartesianParam_t* param, double radardist, int radarindex, PolarNavigationInfo* info)
{
  return CompositeEngineUtility_fillQualityInformation(engine, extradata, arguments, x, y, param, radardist, radarindex, info);
}
/*@} End of Composite Engine internal function pointers */

/*@{ Composite Engine function pointer setters */

int CompositeEngine_setOnStartingFunction(CompositeEngine_t* self, composite_engine_onStarting_fun onStarting)
{
  if (onStarting != NULL) {
    self->onStarting = onStarting;
  } else {
    self->onStarting = CompositeEngineInternal_onStarting;
  }
  return 1;
}

int CompositeEngine_setOnFinishedFunction(CompositeEngine_t* self, composite_engine_onFinished_fun onFinished)
{
  if (onFinished != NULL) {
    self->onFinished = onFinished;
  } else {
    self->onFinished = CompositeEngineInternal_onFinished;
  }
  return 1;
}


int CompositeEngine_setLonLatFunction(CompositeEngine_t* self, composite_engine_getLonLat_fun getLonLat)
{
  if (getLonLat != NULL) {
    self->getLonLat = getLonLat;
  } else {
    self->getLonLat = CompositeEngineInternal_getLonLat;
  }
  return 1;
}

int CompositeEngine_setSelectRadarDataFunction(CompositeEngine_t* self, composite_engine_selectRadarData_fun selectRadarData)
{
  if (selectRadarData != NULL) {
    self->selectRadarData = selectRadarData;
  } else {
    self->selectRadarData = CompositeEngineInternal_selectRadarData;
  }
  return 1;
}

int CompositeEngine_setDefaultPolarValueAtPositionFunction(CompositeEngine_t* self, composite_engine_getPolarValueAtPosition_fun getPolarValueAtPosition)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (getPolarValueAtPosition != NULL) {
    self->getPolarValueAtPosition = getPolarValueAtPosition;
  } else {
    self->getPolarValueAtPosition = CompositeEngineInternal_getPolarValueAtPosition;
  }
  return 1;
}

int CompositeEngine_registerPolarValueAtPositionFunction(CompositeEngine_t* self, const char* quantity, composite_engine_getPolarValueAtPosition_fun getPolarValueAtPosition)
{
  CompositeEnginePolarValueFunction_t* function = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (quantity == NULL) {
    RAVE_ERROR0("Must provide a quantity");
    goto fail;
  }

  if (getPolarValueAtPosition == NULL) {
    RaveCoreObject* obj = RaveObjectHashTable_get(self->polarValueAtPositionMapping, quantity);
    RAVE_OBJECT_RELEASE(obj);
  } else {
    function = RAVE_OBJECT_NEW(&CompositeEnginePolarValueFunction_TYPE);
    if (function == NULL) {
      RAVE_ERROR0("Could not create function object");
      goto fail;
    }
    function->getPolarValueAtPosition = getPolarValueAtPosition;
    if (!RaveObjectHashTable_put(self->polarValueAtPositionMapping, quantity, (RaveCoreObject*)function)) {
      RAVE_ERROR0("Failed to add function to mapping");
      goto fail;
    }
    RAVE_OBJECT_RELEASE(function);
  }
  return 1;
fail:
  RAVE_OBJECT_RELEASE(function);
  return 0;
}

int CompositeEngine_setSetRadarDataFunction(CompositeEngine_t* self, composite_engine_setRadarData_fun setRadarData)
{
  if (setRadarData != NULL) {
    self->setRadarData = setRadarData;
  } else {
    self->setRadarData = CompositeEngineInternal_setRadarData;
  }
  return 1;
}

int CompositeEngine_setAddQualityFlagsToCartesianFunction(CompositeEngine_t* self, composite_engine_addQualityFlagsToCartesian_fun addQualityFlagsToCartesian)
{
  if (addQualityFlagsToCartesian != NULL) {
    self->addQualityFlagsToCartesian = addQualityFlagsToCartesian;
  } else {
    self->addQualityFlagsToCartesian = CompositeEngineInternal_addQualityFlagsToCartesian;
  }
  return 1;
}


int CompositeEngine_setGetQualityValueFunction(CompositeEngine_t* self, composite_engine_getQualityValue_fun getQualityValue)
{
  if (getQualityValue != NULL) {
    self->getQualityValue = getQualityValue;
  } else {
    self->getQualityValue = CompositeEngineInternal_getQualityValue;
  }
  return 1;  
}

int CompositeEngine_setFillQualityInformationFunction(CompositeEngine_t* self, composite_engine_fillQualityInformation_fun fillQualityInformation)
{
  if (fillQualityInformation != NULL) {
    self->fillQualityInformation = fillQualityInformation;
  } else {
    self->fillQualityInformation = CompositeEngineInternal_fillQualityInformation;
  }
  return 1;  
}
/*@} End of Composite Engine function pointer setters */

/*@{ Composite Engine functions called from generate */

int CompositeEngineFunction_onStarting(CompositeEngine_t* self, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings)
{
  return self->onStarting(self, extradata, arguments, bindings, nbindings);
}

int CompositeEngineFunction_onFinished(CompositeEngine_t* self, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings)
{
  return self->onFinished(self, extradata, arguments, bindings, nbindings);
}

int CompositeEngineFunction_getLonLat(CompositeEngine_t* self, void* extradata, CompositeEngineObjectBinding_t* binding, double herex, double herey, double* olon, double* olat)
{
  return self->getLonLat(self, extradata, binding, herex, herey, olon, olat);
}

int CompositeEngineFunction_selectRadarData(CompositeEngine_t* self, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, int index, double olon, double olat, CompositeEngineRadarData_t* cvalues, int ncvalues)
{
  return self->selectRadarData(self, extradata, arguments, binding, index, olon, olat, cvalues, ncvalues);
}

int CompositeEngineFunction_getPolarValueAtPosition(CompositeEngine_t* self, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue)
{
  return self->getPolarValueAtPosition(self, extradata, arguments, binding, quantity, navinfo, qiFieldName, otype, ovalue, qivalue);
}

int CompositeEngineFunction_setRadarData(CompositeEngine_t* self, void* extradata, CompositeArguments_t* arguments, Cartesian_t* cartesian, double olon, double olat, long x, long y, CompositeEngineRadarData_t* cvalues, int ncvalues)
{
  return self->setRadarData(self, extradata, arguments, cartesian, olon, olat, x, y, cvalues, ncvalues);
}

int CompositeEngineFunction_addQualityFlagsToCartesian(CompositeEngine_t* self, void* extradata, CompositeArguments_t* arguments, Cartesian_t* cartesian)
{
  return self->addQualityFlagsToCartesian(self, extradata, arguments, cartesian);
}

int CompositeEngineFunction_getQualityValue(CompositeEngine_t* self, void* extradata, CompositeArguments_t* args, RaveCoreObject* obj, const char* quantity, const char* qfieldname, PolarNavigationInfo* navinfo, double* v)
{
  return self->getQualityValue(self, extradata, args, obj, quantity, qfieldname, navinfo, v);
}

int CompositeEngineFunction_fillQualityInformation(CompositeEngine_t* self, void* extradata, CompositeArguments_t* arguments, long x, long y, CartesianParam_t* param, double radardist, int radarindex, PolarNavigationInfo* info)
{
  return self->fillQualityInformation(self, extradata, arguments, x, y, param, radardist, radarindex, info);
}
/*@} End of Composite Engine functions called from generate */

/*@{ Composite Engine utility functions */

int CompositeEngineUtility_getLonLat(CompositeEngine_t* engine, CompositeEngineObjectBinding_t* binding, double herex, double herey, double* olon, double* olat)
{
  return ProjectionPipeline_fwd(binding->pipeline, herex, herey, olon, olat);
}

int CompositeEngineUtility_selectRadarData(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, int index, double olon, double olat, CompositeEngineRadarData_t* cvalues, int ncvalues)
{
  double dist = 0.0, maxdist = 0.0, rdist = 0.0;
  int cindex = 0;
  RaveAttribute_t* selectionMethodAttr = NULL;
  PolarNavigationInfo navinfo = {0};
  int useHeightSelection = 0;
  const char* qiFieldName;

  if (arguments == NULL) {
    RAVE_ERROR0("Must provide arguments");
    return 0;
  }

  if (RAVE_OBJECT_CHECK_TYPE(binding->object, &PolarVolume_TYPE)) {
    dist = PolarVolume_getDistance((PolarVolume_t*)binding->object, olon, olat);
    maxdist = PolarVolume_getMaxDistance((PolarVolume_t*)binding->object);
  } else if (RAVE_OBJECT_CHECK_TYPE(binding->object, &PolarScan_TYPE)) {
    dist = PolarScan_getDistance((PolarScan_t*)binding->object, olon, olat);
    maxdist = PolarScan_getMaxDistance((PolarScan_t*)binding->object);
  }

  qiFieldName = CompositeArguments_getQIFieldName(arguments);

  // If not HEIGHT, we will assume nearest to radar
  selectionMethodAttr = CompositeArguments_getArgument(arguments, "selection_method");
  if (selectionMethodAttr != NULL) {
    char* selectionMethodStr = NULL;
    RaveAttribute_getString(selectionMethodAttr, &selectionMethodStr);
    if (selectionMethodStr != NULL && strcasecmp("HEIGHT_ABOVE_SEALEVEL", selectionMethodStr) == 0) {
      useHeightSelection = 1;
    }
  }
  RAVE_OBJECT_RELEASE(selectionMethodAttr);

  if (dist <= maxdist) {
    if (CompositeEngineInternal_nearestPosition(arguments, binding->object, olon, olat, &navinfo)) {
      rdist = dist; /* Remember distance to radar */

      if (useHeightSelection) {
        dist = navinfo.actual_height;
      }

      for (cindex = 0; cindex < ncvalues; cindex++) {
        RaveValueType otype = RaveValueType_NODATA;
        double ovalue = 0.0, qivalue = 0.0;
        if (cvalues[cindex].getPolarValueAtPosition != NULL) {
          cvalues[cindex].getPolarValueAtPosition(engine, extradata, arguments, binding, cvalues[cindex].name, &navinfo, qiFieldName, &otype, &ovalue, &qivalue);
        } else {
          CompositeEngineUtility_getPolarValueAtPosition(engine, extradata, arguments, binding, cvalues[cindex].name, &navinfo, qiFieldName, &otype, &ovalue, &qivalue);
        }

        if (otype == RaveValueType_DATA || otype == RaveValueType_UNDETECT) {
          if (cvalues[cindex].vtype != RaveValueType_DATA && cvalues[cindex].vtype != RaveValueType_UNDETECT) {
            /* First time */
            cvalues[cindex].vtype = otype;
            cvalues[cindex].value = ovalue;
            cvalues[cindex].mindist = dist;
            cvalues[cindex].radardist = rdist;
            cvalues[cindex].radarindex = index;
            cvalues[cindex].navinfo = navinfo;
            cvalues[cindex].qivalue = qivalue;
          } else if (
            qiFieldName != NULL &&
              ((qivalue > cvalues[cindex].qivalue) ||
              (qivalue == cvalues[cindex].qivalue && dist < cvalues[cindex].mindist))) {
            cvalues[cindex].vtype = otype;
            cvalues[cindex].value = ovalue;
            cvalues[cindex].mindist = dist;
            cvalues[cindex].radardist = rdist;
            cvalues[cindex].radarindex = index;
            cvalues[cindex].navinfo = navinfo;
            cvalues[cindex].qivalue = qivalue;
          } else if (qiFieldName == NULL && dist < cvalues[cindex].mindist) {
            cvalues[cindex].vtype = otype;
            cvalues[cindex].value = ovalue;
            cvalues[cindex].mindist = dist;
            cvalues[cindex].radardist = rdist;
            cvalues[cindex].radarindex = index;
            cvalues[cindex].navinfo = navinfo;
            cvalues[cindex].qivalue = qivalue;
          }
        }
      }
    }
  }
  return 1;
}

int CompositeEngineUtility_getPolarValueAtPosition(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue)
{
  return CompositeUtils_getPolarValueAtPosition(binding->object, quantity, navinfo, qiFieldName, otype, ovalue, qivalue);  
}

int CompositeEngineUtility_setRadarData(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, Cartesian_t* cartesian, double olon, double olat, long x, long y, CompositeEngineRadarData_t* cvalues, int ncvalues)
{
  int cindex = 0;
  Rave_CompositingProduct prodtype = Rave_CompositingProduct_UNDEFINED;
  double range = 0.0;
  const char* qiFieldName;

  if (arguments == NULL) {
    RAVE_ERROR0("Must provide arguments");
    return 0;
  }
  if (cartesian == NULL) {
    RAVE_ERROR0("Must provide cartesian");
    return 0;
  }
  if (cvalues == NULL || ncvalues == 0) {
    RAVE_ERROR0("No values to set");
    return 0;
  }

  prodtype = CompositeArguments_getCompositingProduct(arguments);
  range = CompositeArguments_getRange(arguments);
  qiFieldName = CompositeArguments_getQIFieldName(arguments);

  /** We don't want to calculate vertical max for each overlapping object so we locate vertical max in setter instead */
  for (cindex = 0; cindex < ncvalues; cindex++) {
    double vvalue = cvalues[cindex].value;
    double vtype = cvalues[cindex].vtype;
    PolarNavigationInfo info = cvalues[cindex].navinfo;
    if (vtype != RaveValueType_NODATA && prodtype == Rave_CompositingProduct_PMAX && cvalues[cindex].radardist < range) {
      RaveValueType ntype = RaveValueType_NODATA;
      double nvalue = 0.0;
      RaveCoreObject* object = CompositeArguments_getObject(arguments, cvalues[cindex].radarindex);
      if (object != NULL) {
        if (vtype == RaveValueType_UNDETECT) {
          /* Undetect should not affect navigation information */
          CompositeUtils_getVerticalMaxValue(object, cvalues[cindex].name, qiFieldName, olon, olat, &ntype, &nvalue, NULL, NULL);
        } else {
          CompositeUtils_getVerticalMaxValue(object, cvalues[cindex].name, qiFieldName, olon, olat, &ntype, &nvalue, &info, NULL);
        }
        if (ntype != RaveValueType_NODATA) {
          vtype = ntype;
          vvalue = nvalue;
        } else {
          /* If we find nodata then we really should use the original navi, void* extradatagation information since there must be something wrong */
          info = cvalues[cindex].navinfo;
        }
      }
      RAVE_OBJECT_RELEASE(object);
    }

    CartesianParam_setConvertedValue(cvalues[cindex].parameter, x, y, vvalue, vtype);

    if ((vtype == RaveValueType_DATA || vtype == RaveValueType_UNDETECT) &&
      cvalues[cindex].radarindex >= 0 && CompositeArguments_getNumberOfQualityFlags(arguments) > 0) {
      CompositeEngineFunction_fillQualityInformation(engine, extradata, arguments, x, y, cvalues[cindex].parameter, cvalues[cindex].radardist, cvalues[cindex].radarindex, &info);
    }
  }
  return 1;  
}

int CompositeEngineUtility_addQualityFlagsToCartesian(CompositeEngine_t* engine, CompositeArguments_t* arguments, Cartesian_t* cartesian)
{
  return CompositeUtils_addQualityFlagsToCartesian(arguments, cartesian, engine->qualityFlagDefinitions);
}

int CompositeEngineUtility_fillQualityInformation(CompositeEngine_t* self, void* extradata, CompositeArguments_t* arguments, long x, long y, CartesianParam_t* param, double radardist, int radarindex, PolarNavigationInfo* navinfo)
{
  int nfields = 0, i = 0;
  const char* quantity;
  RAVE_ASSERT((arguments != NULL), "arguments == NULL");
  RAVE_ASSERT((param != NULL), "param == NULL");
  RAVE_ASSERT((navinfo != NULL), "navinfo == NULL");

  nfields = CartesianParam_getNumberOfQualityFields(param);
  quantity = CartesianParam_getQuantity(param);

  for (i = 0; i < nfields; i++) {
    RaveField_t* field = NULL;
    RaveAttribute_t* attribute = NULL;
    char* name = NULL;
    double v = 0.0;
    int valuefetched = 0;

    field = CartesianParam_getQualityField(param, i);
    if (field != NULL) {
      attribute = RaveField_getAttribute(field, "how/task");
    }
    if (attribute != NULL) {
      RaveAttribute_getString(attribute, &name);
    }

    if (name != NULL) {
      RaveCoreObject* obj = CompositeArguments_getObject(arguments, radarindex);
      if (obj != NULL) {
        if (strcmp(COMPOSITE_ENGINE_DISTANCE_TO_RADAR_HOW_TASK, name) == 0) {
          RaveField_setValue(field, x, y, radardist/COMPOSITE_ENGINE_DISTANCE_TO_RADAR_RESOLUTION);
        } else if (strcmp(COMPOSITE_ENGINE_RADAR_INDEX_HOW_TASK, name) == 0) {
          RaveField_setValue(field, x, y, (double)CompositeArguments_getObjectRadarIndexValue(arguments, radarindex));
        } else if (CompositeEngineFunction_getQualityValue(self, extradata, arguments, obj, quantity, name, navinfo, &v)) {
          RaveField_setValue(field, x, y, (double)v);
        } else {
          if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
            if (navinfo->ei >= 0 && navinfo->ri >= 0 && navinfo->ai >= 0 ) {
              valuefetched = PolarVolume_getQualityValueAt((PolarVolume_t*)obj, quantity, navinfo->ei, navinfo->ri, navinfo->ai, name, 1, &v);
            }
          } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
            if (navinfo->ri >= 0 && navinfo->ai >= 0) {
              valuefetched = PolarScan_getQualityValueAt((PolarScan_t*)obj, quantity, navinfo->ri, navinfo->ai, name, 1, &v);
            }
          }

          if (valuefetched) {
            v = (v - COMPOSITE_ENGINE_DEFAULT_QUALITY_FIELDS_OFFSET) / COMPOSITE_ENGINE_DEFAULT_QUALITY_FIELDS_GAIN;
          }

          RaveField_setValue(field, x, y, v);
        }
      }
      RAVE_OBJECT_RELEASE(obj);
    }

    RAVE_OBJECT_RELEASE(field);
    RAVE_OBJECT_RELEASE(attribute);
  }
  return 1;
}
/*@} End of Composite Engine utility functions */

void CompositeEngine_setProperties(CompositeEngine_t* self, RaveProperties_t* properties)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_OBJECT_RELEASE(self->properties);
  if (properties != NULL) {
    self->properties = RAVE_OBJECT_COPY(properties);
  }
}

RaveProperties_t* CompositeEngine_getProperties(CompositeEngine_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RAVE_OBJECT_COPY(self->properties);
}

int CompositeEngine_registerQualityFlagDefinition(CompositeEngine_t* self, CompositeQualityFlagDefinition_t* definition)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (definition != NULL) {
    result = CompositeUtils_registerQualityFlagDefinition(self->qualityFlagDefinitions, definition);
  }
  return result;
}

Cartesian_t* CompositeEngine_generate(CompositeEngine_t* self, CompositeArguments_t* arguments, void* extradata)
{
  Cartesian_t *cartesian = NULL, *result = NULL;
  CompositeEngineRadarData_t* cvalues = NULL;
  CompositeEngineObjectBinding_t* bindings = NULL;
  OdimSources_t* sources = NULL;

  int x = 0, y = 0, i = 0, xsize = 0, ysize = 0, nradars = 0, nbindings = 0;
  int nentries = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (!CompositeUtils_isValidCartesianArguments(arguments)) {
    RAVE_ERROR0("Ensure that cartesian arguments are valid");
    return NULL;
  }

  cartesian = CompositeUtils_createCartesianFromArguments(arguments);
  if (cartesian == NULL) {
    goto fail;
  }

  cvalues = CompositeEngineUtility_createRadarData(arguments, cartesian, &nentries);
  if (cvalues == NULL) {
    goto fail;
  }

  /* Register the function pointers if necessary */
  for (i = 0; i < nentries; i++) {
    CompositeEnginePolarValueFunction_t* f = (CompositeEnginePolarValueFunction_t*)RaveObjectHashTable_get(self->polarValueAtPositionMapping, cvalues[i].name);
    if (f != NULL) {
      cvalues[i].getPolarValueAtPosition = f->getPolarValueAtPosition;
    }
    RAVE_OBJECT_RELEASE(f);
  }

  xsize = Cartesian_getXSize(cartesian);
  ysize = Cartesian_getYSize(cartesian);
  nradars = CompositeArguments_getNumberOfObjects(arguments);

  if (!CompositeEngineFunction_addQualityFlagsToCartesian(self, extradata, arguments, cartesian)) {
    RAVE_ERROR0("Failed to add quality flags to cartesian product");
    goto fail;
  }

  if (self->properties != NULL) {
    sources = RaveProperties_getOdimSources(self->properties);
  }

  bindings = CompositeEngineObjectBinding_createObjectBinding(arguments, cartesian, &nbindings, sources);
  if (bindings == NULL || nbindings != nradars) {
    RAVE_ERROR0("Could not create a proper pipeline binding");
    goto fail;
  }

  for (int i = 0; i < nbindings; i++) {
    RaveCoreObject* obj = CompositeArguments_getObject(arguments, i);
    if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
      PolarVolume_sortByElevations((PolarVolume_t*)obj, 1);
    }
    RAVE_OBJECT_RELEASE(obj);
  }

  if (!CompositeEngineFunction_onStarting(self, extradata, arguments, bindings, nbindings)) {
    goto fail;
  }

  for (y = 0; y < ysize; y++) {
    double herey = Cartesian_getLocationY(cartesian, y);
    for (x = 0; x < xsize; x++) {
      double herex = Cartesian_getLocationX(cartesian, x);
      double olon = 0.0, olat = 0.0;
      CompositeEngineUtility_resetRadarData(arguments, cvalues, nentries);

      for (i = 0; i < nbindings; i++) {
        /* We will go from surface coords into the lonlat projection assuming that a polar volume uses a lonlat projection*/
        if (!CompositeEngineFunction_getLonLat(self, extradata, &bindings[i], herex, herey, &olon, &olat)) {
          RAVE_WARNING0("Failed to get radar data for wanted coordinate");
        } else {
          if (!CompositeEngineFunction_selectRadarData(self, extradata, arguments, &bindings[i], i, olon, olat, cvalues, nentries)) {
            RAVE_ERROR0("Failed to get radar data");
          }
        }
      }
      CompositeEngineFunction_setRadarData(self, extradata, arguments, cartesian, olon, olat, x, y, cvalues, nentries);
    }
  }

  if (!CompositeEngineFunction_onFinished(self, extradata, arguments, bindings, nbindings)) {
    goto fail;
  }

  result = RAVE_OBJECT_COPY(cartesian);

fail:
  RAVE_OBJECT_RELEASE(cartesian);
  CompositeEngineUtility_freeRadarData(&cvalues, nentries);
  CompositeEngineObjectBinding_releaseObjectBinding(&bindings, nbindings);
  RAVE_OBJECT_RELEASE(sources);
  return result;
}

CompositeEngineRadarData_t* CompositeEngineUtility_createRadarData(CompositeArguments_t* arguments, Cartesian_t* cartesian, int* nentries)
{
  CompositeEngineRadarData_t* result = NULL;
  int nparam = 0;
  if (arguments == NULL) {
    RAVE_ERROR0("Must provide arguments");
    return NULL;
  }
  nparam = CompositeArguments_getParameterCount(arguments);
  if (nparam > 0) {
    int i = 0;

    result = RAVE_MALLOC(sizeof(CompositeEngineRadarData_t) * nparam);
    if (result == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for composite values");
    } else {
      memset(result, 0, sizeof(CompositeEngineRadarData_t) * nparam);
    }

    for (i = 0; i < nparam; i++) {
      const char* quantity = CompositeArguments_getParameterName(arguments, i);
      CartesianParam_t* parameter = Cartesian_getParameter(cartesian, quantity);
      if (parameter == NULL) {
        RAVE_ERROR0("Failed to get parameter from cartesian.");
        CompositeEngineUtility_freeRadarData(&result, nparam);
        return NULL;
      }
      result[i].parameter = parameter;
      result[i].name = (const char*)CompositeArguments_getParameterName(arguments, i);
    }
    *nentries = nparam;
  }
  return result;  
}

void CompositeEngineUtility_resetRadarData(CompositeArguments_t* arguments, CompositeEngineRadarData_t* cvalues, int nentries)
{
  if (cvalues != NULL) {
    int i = 0;
    for (i = 0; i < nentries; i++) {
      cvalues[i].mindist = 1e10;
      cvalues[i].radarindex = -1;
      cvalues[i].vtype = RaveValueType_NODATA;
    }
  }
}

void CompositeEngineUtility_freeRadarData(CompositeEngineRadarData_t** cvalues, int nparam)
{
  if (cvalues != NULL && *cvalues != NULL) {
    int i = 0;
    for (i = 0; i < nparam; i++) {
      RAVE_OBJECT_RELEASE((*cvalues)[i].parameter);
    }
    RAVE_FREE(*cvalues);
    *cvalues = NULL;
  }
}

/*@} End of Interface functions */

RaveCoreObjectType CompositeEngine_TYPE = {
    "CompositeEngine",
    sizeof(CompositeEngine_t),
    CompositeEngine_constructor,
    CompositeEngine_destructor,
    CompositeEngine_copyconstructor
};

