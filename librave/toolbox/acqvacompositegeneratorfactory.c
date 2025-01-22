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
} AcqvaCompositeGeneratorFactory_t;
/*@{ Private functions */


/** The resolution to use for scaling the distance from pixel to used radar. */
/** By multiplying the values in the distance field by 2000, we get the value in unit meters. */
#define DISTANCE_TO_RADAR_RESOLUTION 2000.0

/** Same for height, scaled to 100 m resolution up to 25.5 km */
#define HEIGHT_RESOLUTION 100.0

/** The name of the task for specifying distance to radar */
#define DISTANCE_TO_RADAR_HOW_TASK "se.smhi.composite.distance.radar"

/** The name of the task for specifying height above sea level */
#define HEIGHT_ABOVE_SEA_HOW_TASK "se.smhi.composite.height.radar"

/** The name of the task for indexing the radars used */
#define RADAR_INDEX_HOW_TASK "se.smhi.composite.index.radar"

static CompositeQualityFlagSettings_t ACQVA_QUALITY_FLAG_DEFINITIONS[] = {
  {DISTANCE_TO_RADAR_HOW_TASK, RaveDataType_UCHAR, 0.0, DISTANCE_TO_RADAR_RESOLUTION},
  {HEIGHT_ABOVE_SEA_HOW_TASK, RaveDataType_UCHAR, 0.0, HEIGHT_RESOLUTION},
  {RADAR_INDEX_HOW_TASK, RaveDataType_UCHAR, 0.0, 1.0},
  {NULL, RaveDataType_UNDEFINED, 0.0, 0.0}
};

/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int AcqvaCompositeGeneratorFactory_constructor(RaveCoreObject* obj)
{
  AcqvaCompositeGeneratorFactory_t* this = (AcqvaCompositeGeneratorFactory_t*)obj;
  this->getName = AcqvaCompositeGeneratorFactory_getName;
  this->canHandle = AcqvaCompositeGeneratorFactory_canHandle;
  this->generate = AcqvaCompositeGeneratorFactory_generate;
  this->create = AcqvaCompositeGeneratorFactory_create;
  return 1;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int AcqvaCompositeGeneratorFactory_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  AcqvaCompositeGeneratorFactory_t* this = (AcqvaCompositeGeneratorFactory_t*)obj;
  //AcqvaCompositeGeneratorFactory_t* src = (AcqvaCompositeGeneratorFactory_t*)srcobj;
  this->getName = AcqvaCompositeGeneratorFactory_getName;
  this->canHandle = AcqvaCompositeGeneratorFactory_canHandle;
  this->generate = AcqvaCompositeGeneratorFactory_generate;
  this->create = AcqvaCompositeGeneratorFactory_create;

  return 1;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void AcqvaCompositeGeneratorFactory_destructor(RaveCoreObject* obj)
{
  //AcqvaCompositeGeneratorFactory_t* this = (AcqvaCompositeGeneratorFactory_t*)obj;
}

int AcqvaCompositeGeneratorFactoryInternal_findLowestUsableValue(CompositeGeneratorFactory_t* self, PolarVolume_t* pvol, 
  double lon, double lat, const char* qfieldname, double* height, 
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
      if (PolarScan_getQualityValueAt(scan, NULL, navinfo.ri, navinfo.ai, qfieldname, 1, &v)) {
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
 * Uses the navigation information of the value positions and fills all 
 * associated cartesian quality with the composite objects quality fields. If
 * there is more than one value position in the valuePositions-array, an 
 * an interpolation of the quality value will be performed, along the dimensions
 * defined in the interpolationDimensions-array.
 * 
 * @param[in] composite - self
 * @param[in] x - x coordinate
 * @param[in] y - y coordinate
 * @param[in] cvalues - the composite values
 * @param[in] interpolationDimensions - dimensions to perform interpolation in                      
 */
static void AcqvaCompositeGeneratorFactoryInternal_fillQualityInformation(
  CompositeArguments_t* arguments,
  int x,
  int y,
  CompositeUtilValue_t* cvalues)
{
  int nfields = 0, i = 0;
  const char* quantity;
  CartesianParam_t* param = NULL;
  double radardist = 0;
  int radarindex = 0;

  RAVE_ASSERT((arguments != NULL), "self == NULL");
  RAVE_ASSERT((cvalues != NULL), "cvalues == NULL");

  param = cvalues->parameter;
  radardist = cvalues->radardist;
  radarindex = cvalues->radarindex;

  nfields = CartesianParam_getNumberOfQualityFields(param);
  quantity = CartesianParam_getQuantity(param);

  for (i = 0; i < nfields; i++) {
    RaveField_t* field = NULL;
    RaveAttribute_t* attribute = NULL;
    char* name = NULL;
    double value = 0.0;

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
        if (strcmp(DISTANCE_TO_RADAR_HOW_TASK, name) == 0) {
          value = radardist/DISTANCE_TO_RADAR_RESOLUTION;
        } else if (strcmp(HEIGHT_ABOVE_SEA_HOW_TASK, name) == 0) {
          value = cvalues->navinfo.actual_height / HEIGHT_RESOLUTION;
        } else if (strcmp(RADAR_INDEX_HOW_TASK, name) == 0) {
          //value = (double)Acqva_getRadarIndexValue(self, radarindex);
        } else {
          // if (AcqvaInternal_getQualityValueAtPosition(self, obj, quantity, name, &cvalues->navinfo, &value)) {
          //   value = (value - ACQVA_QUALITY_FIELDS_OFFSET) / ACQVA_QUALITY_FIELDS_GAIN;
          // }
        }
        RaveField_setValue(field, x, y, value);
      }
      RAVE_OBJECT_RELEASE(obj);
    }

    RAVE_OBJECT_RELEASE(field);
    RAVE_OBJECT_RELEASE(attribute);
  }
}

/*@} End of Private functions */

/*@{ Interface functions */

const char* AcqvaCompositeGeneratorFactory_getName(CompositeGeneratorFactory_t* self)
{
  return "AcqvaCompositeGenerator";
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

Cartesian_t* AcqvaCompositeGeneratorFactory_generate(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments)
{
  Cartesian_t *cartesian = NULL, *result = NULL;
  CompositeUtilValue_t* cvalues = NULL;
  CompositeRaveObjectBinding_t* pipelineBinding = NULL;
  int x = 0, y = 0, i = 0, xsize = 0, ysize = 0, nradars = 0, nbindings = 0, nqualityflags = 0;
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

  cvalues = CompositeUtils_createCompositeValues(arguments, cartesian, &nentries);
  if (cvalues == NULL) {
    goto fail;
  }

  xsize = Cartesian_getXSize(cartesian);
  ysize = Cartesian_getYSize(cartesian);
  nradars = CompositeArguments_getNumberOfObjects(arguments);
  nqualityflags = CompositeArguments_getNumberOfQualityFlags(arguments);

  if (!CompositeUtils_addQualityFlagsToCartesian(arguments, cartesian, ACQVA_QUALITY_FLAG_DEFINITIONS, &nqualityflags)) {
    RAVE_ERROR0("Failed to add quality flags to product");
    goto fail;
  }

  pipelineBinding = CompositeUtils_createRaveObjectBinding(arguments, cartesian, &nbindings);
  if (pipelineBinding == NULL || nbindings != nradars) {
    RAVE_ERROR0("Could not create a proper pipeline binding");
    goto fail;
  }

  for (y = 0; y < ysize; y++) {
    double herey = Cartesian_getLocationY(cartesian, y);
    for (x = 0; x < xsize; x++) {
      int cindex = 0;
      double herex = Cartesian_getLocationX(cartesian, x);
      double olon = 0.0, olat = 0.0;
      CompositeUtils_resetCompositeValues(arguments, cvalues, nentries);

      for (i = 0; i < nbindings; i++) {
        RaveCoreObject* obj = NULL;
        ProjectionPipeline_t* pipeline = NULL;
        obj = RAVE_OBJECT_COPY(pipelineBinding[i].object);
        pipeline = RAVE_OBJECT_COPY(pipelineBinding[i].pipeline);

        /* We will go from surface coords into the lonlat projection assuming that a polar volume uses a lonlat projection*/
        if (!ProjectionPipeline_fwd(pipeline, herex, herey, &olon, &olat)) {
          RAVE_WARNING0("Failed to transform from composite into polar coordinates");
        } else {
          double dist = 0.0;
          double maxdist = 0.0;
          if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
            dist = PolarVolume_getDistance((PolarVolume_t*)obj, olon, olat);
            maxdist = PolarVolume_getMaxDistance((PolarVolume_t*)obj);
          } else {
            RAVE_ERROR0("ACQVA currently only handles polar volumes");
            goto fail;
          }
          if (dist <= maxdist) {
            double height=0.0, elangle=0.0;
            int ray=0, bin=0, eindex=0;
            PolarNavigationInfo navinfo;
            if (AcqvaCompositeGeneratorFactoryInternal_findLowestUsableValue(self, (PolarVolume_t*)obj, 
              olon, olat, "se.smhi.acqva", &height, &elangle, &ray, &bin, &eindex, &navinfo)) {
              for (cindex = 0; cindex < nentries; cindex++) {
                RaveValueType otype = RaveValueType_NODATA;
                double v = 0.0;
                otype = PolarVolume_getConvertedParameterValueAt((PolarVolume_t*)obj, cvalues[cindex].name, eindex, bin, ray, &v);
                if (otype != RaveValueType_NODATA) {
                  if (cvalues[cindex].mindist > height) {
                    cvalues[cindex].mindist = height;
                    cvalues[cindex].value = v;
                    cvalues[cindex].vtype = otype;
                    cvalues[cindex].navinfo = navinfo;
                    cvalues[cindex].radarindex = i;
                    cvalues[cindex].radardist = cvalues[cindex].navinfo.actual_range;
                  }
                }
              }
            }
          }
        }
        RAVE_OBJECT_RELEASE(pipeline);
        RAVE_OBJECT_RELEASE(obj);
      }

      for (cindex = 0; cindex < nentries; cindex++) {
        double vvalue = cvalues[cindex].value;
        int vtype = cvalues[cindex].vtype;
        CartesianParam_setConvertedValue(cvalues[cindex].parameter, x, y, vvalue, vtype);
        if ((vtype == RaveValueType_DATA || vtype == RaveValueType_UNDETECT) &&
            cvalues[cindex].radarindex >= 0 && nqualityflags > 0) {
//          AcqvaInternal_fillQualityInformation(self, x, y, &cvalues[cindex]);
        }        
      }
    }
  }

  result = RAVE_OBJECT_COPY(cartesian);
fail:
  RAVE_OBJECT_RELEASE(cartesian);
  CompositeUtils_freeCompositeValueParameters(&cvalues, nentries);
  CompositeUtils_releaseRaveObjectBinding(&pipelineBinding, nbindings);
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
