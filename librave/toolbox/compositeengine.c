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
#include "polarvolume.h"
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
  composite_engine_getLonLat_fun getLonLat;
  composite_engine_selectRadarData_fun selectRadarData;
};

/*@{ Private functions */

static int CompositeEngineInternal_getLonLat(void* extradata, RaveCoreObject* object, ProjectionPipeline_t* pipeline, double herex, double herey, double* olon, double* olat);

static int CompositeEngineInternal_selectRadarData(void* extradata, RaveCoreObject* object, double olon, double olat, CompositeUtilValue_t* cvalues, int ncvalues);
/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int CompositeEngine_constructor(RaveCoreObject* obj)
{
  CompositeEngine_t* this = (CompositeEngine_t*)obj;
  this->getLonLat = CompositeEngineInternal_getLonLat;
  this->selectRadarData = CompositeEngineInternal_selectRadarData;
  return 1;
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
  this->getLonLat = src->getLonLat;
  this->selectRadarData = src->selectRadarData;  
  return 1;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void CompositeEngine_destructor(RaveCoreObject* obj)
{
  // CompositeEngine_t* this = (CompositeEngine_t*)obj;
}

static int CompositeEngineInternal_getLonLat(void* extradata, RaveCoreObject* object, ProjectionPipeline_t* pipeline, double herex, double herey, double* olon, double* olat)
{
  return ProjectionPipeline_fwd(pipeline, herex, herey, olon, olat);
}

static int CompositeEngineInternal_selectRadarData(void* extradata, RaveCoreObject* object, double olon, double olat, CompositeUtilValue_t* cvalues, int ncvalues)
{
  double dist = 0.0, maxdist = 0.0, rdist = 0.0;
  int cindex = 0;
  
  if (RAVE_OBJECT_CHECK_TYPE(object, &PolarVolume_TYPE)) {
    dist = PolarVolume_getDistance((PolarVolume_t*)object, olon, olat);
    maxdist = PolarVolume_getMaxDistance((PolarVolume_t*)object);
  } else if (RAVE_OBJECT_CHECK_TYPE(object, &PolarScan_TYPE)) {
    dist = PolarScan_getDistance((PolarScan_t*)object, olon, olat);
    maxdist = PolarScan_getMaxDistance((PolarScan_t*)object);
  }
  if (dist <= maxdist) {
    double originaldist = dist;
    if (CompositeInternal_nearestPosition(composite, obj, olon, olat, &navinfo)) {
      double originaldist = dist;
      rdist = dist; /* Remember distance to radar */

      if (composite->method == CompositeSelectionMethod_HEIGHT) {
        dist = navinfo.actual_height;
      }

      for (cindex = 0; cindex < nparam; cindex++) {
        RaveValueType otype = RaveValueType_NODATA;
        double ovalue = 0.0, qivalue = 0.0;
        CompositeInternal_getValueAtPosition(composite, obj, cvalues[cindex].name, &navinfo, &otype, &ovalue, &qivalue);

        if (otype == RaveValueType_DATA || otype == RaveValueType_UNDETECT) {
          if (cvalues[cindex].vtype != RaveValueType_DATA && cvalues[cindex].vtype != RaveValueType_UNDETECT) {
            /* First time */
            cvalues[cindex].vtype = otype;
            cvalues[cindex].value = ovalue;
            cvalues[cindex].mindist = dist;
            cvalues[cindex].radardist = rdist;
            cvalues[cindex].radarindex = i;
            cvalues[cindex].navinfo = navinfo;
            cvalues[cindex].qivalue = qivalue;
          } else if (
            composite->qiFieldName != NULL &&
              ((qivalue > cvalues[cindex].qivalue) ||
              (qivalue == cvalues[cindex].qivalue && dist < cvalues[cindex].mindist))) {
            cvalues[cindex].vtype = otype;
            cvalues[cindex].value = ovalue;
            cvalues[cindex].mindist = dist;
            cvalues[cindex].radardist = rdist;
            cvalues[cindex].radarindex = i;
            cvalues[cindex].navinfo = navinfo;
            cvalues[cindex].qivalue = qivalue;
          } else if (composite->qiFieldName == NULL && dist < cvalues[cindex].mindist) {
            cvalues[cindex].vtype = otype;
            cvalues[cindex].value = ovalue;
            cvalues[cindex].mindist = dist;
            cvalues[cindex].radardist = rdist;
            cvalues[cindex].radarindex = i;
            cvalues[cindex].navinfo = navinfo;
            cvalues[cindex].qivalue = qivalue;
          }
        }
      }
    }
  }
  return 0;
}

int CompositeEngine_setLonLatFunction(CompositeEngine_t* self, composite_engine_getLonLat_fun getLonLat)
{
  self->getLonLat = getLonLat;
}

int CompositeEngine_setSelectRadarDataFunction(CompositeEngine_t* self, composite_engine_selectRadarData_fun selectRadarData)
{
  self->selectRadarData = selectRadarData;
}

int CompositeEngineFunction_getLonLat(CompositeEngine_t* self, void* extradata, RaveCoreObject* object, ProjectionPipeline_t* pipeline, double herex, double herey, double* olon, double* olat)
{
  return self->getLonLat(extradata, object, pipeline, herex, herey, olon, olat);
}

int CompositeEngineFunction_selectRadarData(CompositeEngine_t* self, void* extradata, RaveCoreObject* object, double olon, double olat, CompositeUtilValue_t* cvalues, int ncvalues)
{
  return self->selectRadarData(extradata, object, olon, olat, cvalues, ncvalues);
}

Cartesian_t* CompositeEngine_generate(CompositeEngine_t* self, CompositeArguments_t* arguments, void* extradata)
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

  // if (!CompositeUtils_addQualityFlagsToCartesian(arguments, cartesian, ACQVA_QUALITY_FLAG_DEFINITIONS, &nqualityflags)) {
  //   RAVE_ERROR0("Failed to add quality flags to product");
  //   goto fail;
  // }

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
        // RaveCoreObject* obj = NULL;
        // ProjectionPipeline_t* pipeline = NULL;
        // obj = RAVE_OBJECT_COPY(pipelineBinding[i].object);
        // pipeline = RAVE_OBJECT_COPY(pipelineBinding[i].pipeline);

        /* We will go from surface coords into the lonlat projection assuming that a polar volume uses a lonlat projection*/
        if (!CompositeEngineFunction_getLonLat(self, extradata, pipelineBinding[i].object, pipelineBinding[i].pipeline, herex, herey, &olon, &olat)) {
          RAVE_WARNING0("Failed to get radar data for wanted coordinate");
        } else {
          if (!CompositeEngineFunction_selectRadarData(self, extradata, pipelineBinding[i].object, olon, olat, cvalues, nentries)) {
            RAVE_ERROR0("Failed to get radar data");
          }
        }
        // if (!ProjectionPipeline_fwd(pipeline, herex, herey, &olon, &olat)) {
        //   RAVE_WARNING0("Failed to transform from composite into polar coordinates");
        // } else {
        //   double dist = 0.0;
        //   double maxdist = 0.0;
        //   if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
        //     dist = PolarVolume_getDistance((PolarVolume_t*)obj, olon, olat);
        //     maxdist = PolarVolume_getMaxDistance((PolarVolume_t*)obj);
        //   } else {
        //     RAVE_ERROR0("ACQVA currently only handles polar volumes");
        //     goto fail;
        //   }
        //   if (dist <= maxdist) {
        //     double height=0.0, elangle=0.0;
        //     int ray=0, bin=0, eindex=0;
        //     PolarNavigationInfo navinfo;
        //     if (AcqvaCompositeGeneratorFactoryInternal_findLowestUsableValue(self, (PolarVolume_t*)obj, 
        //       olon, olat, "se.smhi.acqva", &height, &elangle, &ray, &bin, &eindex, &navinfo)) {
        //       for (cindex = 0; cindex < nentries; cindex++) {
        //         RaveValueType otype = RaveValueType_NODATA;
        //         double v = 0.0;
        //         otype = PolarVolume_getConvertedParameterValueAt((PolarVolume_t*)obj, cvalues[cindex].name, eindex, bin, ray, &v);
        //         if (otype != RaveValueType_NODATA) {
        //           if (cvalues[cindex].mindist > height) {
        //             cvalues[cindex].mindist = height;
        //             cvalues[cindex].value = v;
        //             cvalues[cindex].vtype = otype;
        //             cvalues[cindex].navinfo = navinfo;
        //             cvalues[cindex].radarindex = i;
        //             cvalues[cindex].radardist = cvalues[cindex].navinfo.actual_range;
        //           }
        //         }
        //       }
        //     }
        //   }
        // }
        // RAVE_OBJECT_RELEASE(pipeline);
        // RAVE_OBJECT_RELEASE(obj);
      }

      // for (cindex = 0; cindex < nentries; cindex++) {
      //   double vvalue = cvalues[cindex].value;
      //   int vtype = cvalues[cindex].vtype;
      //   CartesianParam_setConvertedValue(cvalues[cindex].parameter, x, y, vvalue, vtype);
      //   if ((vtype == RaveValueType_DATA || vtype == RaveValueType_UNDETECT) &&
      //       cvalues[cindex].radarindex >= 0 && nqualityflags > 0) {
      //     AcqvaCompositeGeneratorFactoryInternal_fillQualityInformation(arguments, x, y, &cvalues[cindex]);
      //   }        
      // }
    }
  }

  result = RAVE_OBJECT_COPY(cartesian);
fail:
  RAVE_OBJECT_RELEASE(cartesian);
  CompositeUtils_freeCompositeValueParameters(&cvalues, nentries);
  CompositeUtils_releaseRaveObjectBinding(&pipelineBinding, nbindings);
  return result;
}

/*@} End of Interface functions */

RaveCoreObjectType CompositeEngine_TYPE = {
    "CompositeEngine",
    sizeof(CompositeEngine_t),
    CompositeEngine_constructor,
    CompositeEngine_destructor,
    CompositeEngine_copyconstructor
};

