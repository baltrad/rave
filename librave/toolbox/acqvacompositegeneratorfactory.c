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
#include "cartesian.h"
#include "compositeenginebase.h"
#include "compositeenginefunctions.h"
#include "composite_utils.h"
#include "compositearguments.h"
#include "compositeengine.h"
#include "compositeengineqc.h"
#include "polarscan.h"
#include "polarscanparam.h"
#include "polarvolume.h"
#include "rave_attribute.h"
#include "rave_field.h"
#include "rave_properties.h"
#include "rave_value.h"
#include "raveobject_list.h"
#include "rave_types.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_utilities.h"
#include <stdio.h>
#include <string.h>
#include "rave_io.h"


typedef struct _AcqvaCompositeGeneratorFactory_t {
  RAVE_OBJECT_HEAD /**< Always on top */
  COMPOSITE_GENERATOR_FACTORY_HEAD /**< composite generator plugin specifics */
  CompositeEngine_t* engine; /**<the engine */
  CompositeEngineOvershootingQcHandler_t* overshooting;
} AcqvaCompositeGeneratorFactory_t;
/*@{ Private functions */

static int AcqvaCompositeGeneratorFactory_onStarting(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings);

static int AcqvaCompositeGeneratorFactory_onFinished(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings);

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
  CompositeQualityFlagDefinition_t* definition = NULL;

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

  if (!CompositeEngine_setOnFinishedFunction(this->engine, AcqvaCompositeGeneratorFactory_onFinished)) {
    RAVE_ERROR0("Failed to set the onFinished function");
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

  if (!CompositeEngine_setOnFinishedFunction(this->engine, AcqvaCompositeGeneratorFactory_onFinished)) {
    RAVE_ERROR0("Failed to set the onFinished function");
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

PolarVolume_t* AcqvaCompositeGeneratorFactoryInternal_loadCluttermap(const char* cluttermap_dir, OdimSource_t* source)
{
  RaveIO_t* rio = NULL;
  PolarVolume_t* result = NULL;
  if (OdimSource_getNod(source) != NULL) {
    char buff[512];
    snprintf(buff, 512, "%s/%s.h5", cluttermap_dir, OdimSource_getNod(source));
    rio = RaveIO_open(buff, 0, NULL);
    if (rio == NULL) {
      RAVE_ERROR1("Could not identify cluttermap '%s'", buff);
      goto fail;
    }
    if (RaveIO_getObjectType(rio) == Rave_ObjectType_PVOL) {
      result = (PolarVolume_t*)RaveIO_getObject(rio);
    }
  } else {
    RAVE_ERROR0("OdimSource does not contain NOD");
  }

fail:
  RAVE_OBJECT_RELEASE(rio);
  return result;
}

static int AcqvaCompositeGeneratorFactoryInternal_updateWithCluttermaps(AcqvaCompositeGeneratorFactory_t* self, RaveProperties_t* properties, CompositeEngineObjectBinding_t* bindings, int nbindings)
{
  int i = 0;
  int failed = 0;
  const char* cluttermap_dir = NULL;

  if (RaveProperties_hasProperty(properties, "rave.acqva.cluttermap.dir")) {
    RaveValue_t* property = RaveProperties_get(properties, "rave.acqva.cluttermap.dir");
    if (RaveValue_type(property) == RaveValue_Type_String) {
      cluttermap_dir = RaveValue_toString(property);
    }
    RAVE_OBJECT_RELEASE(property);
  }

  for (i = 0; !failed && i < nbindings; i++) {
    int nscans = 0, j = 0;
    PolarVolume_t* cluttermap = NULL;
    /* We can only update volume if we have sources, a cluttermap dir and a mapping object */
    if (bindings[i].source != NULL && cluttermap_dir != NULL) {
      cluttermap = AcqvaCompositeGeneratorFactoryInternal_loadCluttermap(cluttermap_dir, bindings[i].source);
    }
    nscans = PolarVolume_getNumberOfScans((PolarVolume_t*)bindings[i].object);
    for (j = 0; !failed && j < nscans; j++) {
      PolarScan_t* scan = PolarVolume_getScan((PolarVolume_t*)bindings[i].object, j);
      RaveField_t* qfield = PolarScan_getQualityFieldByHowTask(scan, ACQVA_QUALITY_FIELD_NAME);
      if (qfield == NULL && cluttermap != NULL) {
        PolarScan_t* cmapscan = PolarVolume_getScanClosestToElevation(cluttermap, PolarScan_getElangle(scan), 0);
        if (cmapscan != NULL && fabs(PolarScan_getElangle(scan) - PolarScan_getElangle(cmapscan)) < 0.0001) {
          PolarScanParam_t* param = PolarScan_getParameter(cmapscan, "ACQVA");
          if (param != NULL) {
            qfield = PolarScanParam_toField(param);
            if (qfield != NULL) {
              RaveAttribute_t* attr = RaveAttributeHelp_createString("how/task", ACQVA_QUALITY_FIELD_NAME);
              if (attr == NULL || !RaveField_addAttribute(qfield, attr)) {
                RAVE_ERROR0("Could not create rave attribute");
                failed = 1;
              }
              RAVE_OBJECT_RELEASE(attr);

              attr = RaveAttributeHelp_createString("how/acqva_remove_me", "YES");
              if (attr == NULL || !RaveField_addAttribute(qfield, attr)) {
                RAVE_ERROR0("Could not create rave attribute");
                failed = 1;
              }
              RAVE_OBJECT_RELEASE(attr);

              if (!PolarScan_addQualityField(scan, qfield)) {
                RAVE_ERROR0("Failed to add ACQVA quality field to scan");
                failed = 1;
              }
            }
          } else {
            RAVE_ERROR0("No ACQVA parameter in cluttermap");
            failed = 1;
          }
          RAVE_OBJECT_RELEASE(param);
        } else {
          RAVE_ERROR1("Could not find a matching scan for %s", OdimSource_getNod(bindings[i].source));
          failed = 1;
        }
        RAVE_OBJECT_RELEASE(cmapscan);
      } else if (qfield == NULL) {
        RAVE_ERROR1("Can not create ACQVA product since %s does not have any cluttermap associated", OdimSource_getNod(bindings[i].source));
        failed = 1;
      }
      RAVE_OBJECT_RELEASE(qfield);
      RAVE_OBJECT_RELEASE(scan);
    }
    RAVE_OBJECT_RELEASE(cluttermap);
  }

  if (failed) {
    return 0;
  }
  return 1;
}

static int AcqvaCompositeGeneratorFactory_onStarting(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings)
{
  AcqvaCompositeGeneratorFactory_t* self = (AcqvaCompositeGeneratorFactory_t*)extradata;
  int result = 0;

  RaveProperties_t* properties = CompositeEngine_getProperties(engine);
  if (!CompositeEngineFunctions_prepareRATE(engine, arguments, bindings, nbindings)) {
    RAVE_ERROR0("Failed to prepare RATE coefficients");
    goto fail;
  }

  if (!AcqvaCompositeGeneratorFactoryInternal_updateWithCluttermaps(self, properties, bindings, nbindings)) {
    RAVE_ERROR0("Failed to update volumes with cluttermaps");
    goto fail;;
  }

  CompositeEngineQcHandler_initialize(self->overshooting, extradata, properties, arguments, bindings, nbindings);

  result = 1;
fail:
  RAVE_OBJECT_RELEASE(properties);
  return result;
}

static int AcqvaCompositeGeneratorFactory_onFinished(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings)
{
  //AcqvaCompositeGeneratorFactory_t* self = (AcqvaCompositeGeneratorFactory_t*)extradata;
  int i = 0;
  for (i = 0; i < nbindings; i++) {
    int j = 0, nscans = PolarVolume_getNumberOfScans((PolarVolume_t*)bindings[i].object);
    for (j = 0; j < nscans; j++) {
      PolarScan_t* scan = PolarVolume_getScan((PolarVolume_t*)bindings[i].object, j);
      if (scan != NULL) {
        RaveField_t* qfield = PolarScan_getQualityFieldByHowTask(scan, ACQVA_QUALITY_FIELD_NAME);
        if (qfield != NULL && RaveField_hasAttribute(qfield, "how/acqva_remove_me")) {
          PolarScan_removeQualityFieldByHowTask(scan, ACQVA_QUALITY_FIELD_NAME);
        }
        RAVE_OBJECT_RELEASE(qfield);
      }
      RAVE_OBJECT_RELEASE(scan);
    }
  }
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
        if (strcasecmp("RATE", cvalues[cindex].name) == 0) {
          otype = PolarVolume_getConvertedParameterValueAt((PolarVolume_t*)binding->object, "DBZH", eindex, bin, ray, &v);
          if (otype == RaveValueType_DATA) {
            v = CompositeEngineFunction_convertDbzToRate(binding, otype, v, DEFAULT_ZR_A, DEFAULT_ZR_B);
          }
        } else {
          otype = PolarVolume_getConvertedParameterValueAt((PolarVolume_t*)binding->object, cvalues[cindex].name, eindex, bin, ray, &v);
        }
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

  if (strcasecmp("se.smhi.detector.poo", qfieldname) == 0) {
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
  Cartesian_t* result = NULL;

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
  result = CompositeEngine_generate(((AcqvaCompositeGeneratorFactory_t*)self)->engine, arguments, (void*)self);
  if (result != NULL) {
    Cartesian_setProduct(result, Rave_ProductType_COMP);
  }
  return result;
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
