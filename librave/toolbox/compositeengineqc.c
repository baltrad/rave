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
 * Provides some utilities
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-31
 */
 #include "compositeengineqc.h"
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
 #include <limits.h>
 #include <math.h>
 

/*@{ Probability of overshooting quality control functions  */

/**
 * Overshooting quality field gain
 */
#define HOWTASK_POO_GAIN (1.0/UCHAR_MAX)

/**
 * Overshooting quality field offset
 */
 #define HOWTASK_POO_OFFSET 0.0

/**
 * Overshooting quality data type
 */
#define HOWTASK_POO_DATATYPE RaveDataType_UCHAR

/**
 * Overshooting quality field how/task name
 */
#define HOWTASK_POO_NAME "se.smhi.detector.poo"

/**
 * The QC overshooting handler when generating poo fields.
 */
typedef struct _CompositeEngineOvershootingQcHandler_t {
  RAVE_OBJECT_HEAD /**< Always on top */
  COMPOSITE_ENGINE_QC_HANDLER_HEAD /**< the qc handler head */
  RaveObjectHashTable_t* poofields; /**< the poo fields */
} CompositeEngineOvershootingQcHandler_t;

static const char* CompositeEngineOvershootingQcHandler_getQualityFieldName(CompositeEngineQcHandler_t* self);

static int CompositeEngineOvershootingQcHandler_initialize(CompositeEngineQcHandler_t* self, void* extradata, RaveProperties_t* properties, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings);

static int CompositeEngineOvershootingQcHandler_getQualityValue(CompositeEngineQcHandler_t* self, void* extradata, CompositeArguments_t* arguments, RaveCoreObject* object, const char* quantity, const char* qiFieldName, PolarNavigationInfo* navinfo, double* qivalue);

static CompositeQualityFlagDefinition_t* CompositeEngineOvershootingQcHandler_getFlagDefinition(CompositeEngineQcHandler_t* self);

static int CompositeEngineOvershootingQcHandler_constructor(RaveCoreObject* obj)
{
  CompositeEngineOvershootingQcHandler_t* this = (CompositeEngineOvershootingQcHandler_t*)obj;
  this->getQualityFieldName = CompositeEngineOvershootingQcHandler_getQualityFieldName;
  this->getQualityValue = CompositeEngineOvershootingQcHandler_getQualityValue;
  this->initialize = CompositeEngineOvershootingQcHandler_initialize;
  this->getFlagDefinition = CompositeEngineOvershootingQcHandler_getFlagDefinition;
  this->poofields = NULL;
  return 1;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int CompositeEngineOvershootingQcHandler_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  CompositeEngineOvershootingQcHandler_t* this = (CompositeEngineOvershootingQcHandler_t*)obj;
  CompositeEngineOvershootingQcHandler_t* src = (CompositeEngineOvershootingQcHandler_t*)srcobj;
  this->getQualityFieldName = CompositeEngineOvershootingQcHandler_getQualityFieldName;
  this->getQualityValue = CompositeEngineOvershootingQcHandler_getQualityValue;
  this->initialize = CompositeEngineOvershootingQcHandler_initialize;
  this->getFlagDefinition = CompositeEngineOvershootingQcHandler_getFlagDefinition;
  this->poofields = NULL;

  if (src->poofields != NULL) {
    this->poofields = RAVE_OBJECT_CLONE(src->poofields);
    if (this->poofields == NULL) {
      RAVE_ERROR0("Failed to clone poo fields");
      goto fail;
    }
  }
  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->poofields);
  return 0;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void CompositeEngineOvershootingQcHandler_destructor(RaveCoreObject* obj)
{
  CompositeEngineOvershootingQcHandler_t* this = (CompositeEngineOvershootingQcHandler_t*)obj;
  RAVE_OBJECT_RELEASE(this->poofields);
}

static const char* CompositeEngineOvershootingQcHandler_getQualityFieldName(CompositeEngineQcHandler_t* self)
{
  return HOWTASK_POO_NAME;
}

static int CompositeEngineOvershootingQcHandler_initialize(CompositeEngineQcHandler_t* self, void* extradata, RaveProperties_t* properties, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings)
{
  CompositeEngineOvershootingQcHandler_t* this = (CompositeEngineOvershootingQcHandler_t*)self;  
  RAVE_OBJECT_RELEASE(this->poofields);
  if (CompositeArguments_hasQualityFlag(arguments, "se.smhi.detector.poo")) {
    this->poofields = CompositeEngineQc_getQualityScanFields(bindings, nbindings, "se.smhi.detector.poo");
  }
  return 1;
}

static int CompositeEngineOvershootingQcHandler_getQualityValue(CompositeEngineQcHandler_t* self, void* extradata, CompositeArguments_t* arguments, RaveCoreObject* object, const char* quantity, const char* qfieldName, PolarNavigationInfo* navinfo, double* qivalue)
{
  int result = 0;
  PolarScan_t* pooscan = NULL;
  CompositeEngineOvershootingQcHandler_t* this = (CompositeEngineOvershootingQcHandler_t*)self;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((object != NULL), "object == NULL");
  RAVE_ASSERT((navinfo != NULL), "navinfo == NULL");

  if (strcmp("se.smhi.detector.poo", qfieldName) == 0) {
    /* We know how to handle poo */
    if (RAVE_OBJECT_CHECK_TYPE(object, &PolarVolume_TYPE) && navinfo->ei >= 0 && navinfo->ri >= 0 && navinfo->ai >= 0) {
      const char* source = PolarVolume_getSource((PolarVolume_t*)object);
      if (source != NULL) {
        pooscan = (PolarScan_t*)RaveObjectHashTable_get(this->poofields, source);
      }
    } else if (RAVE_OBJECT_CHECK_TYPE(object, &PolarScan_TYPE) && navinfo->ri >= 0 && navinfo->ai >= 0) {
      const char* source = PolarScan_getSource((PolarScan_t*)object);
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

      *qivalue = (value - HOWTASK_POO_OFFSET) / HOWTASK_POO_GAIN;

      result = 1;
    }
  }

  RAVE_OBJECT_RELEASE(pooscan);
  return result;
}

static CompositeQualityFlagDefinition_t* CompositeEngineOvershootingQcHandler_getFlagDefinition(CompositeEngineQcHandler_t* self)
{
  return CompositeUtils_createQualityFlagDefinition(HOWTASK_POO_NAME, HOWTASK_POO_DATATYPE, HOWTASK_POO_OFFSET, HOWTASK_POO_GAIN);
}


/**
 * Type definition to use when creating a rave object.
 */
RaveCoreObjectType CompositeEngineOvershootingQcHandler_TYPE = {
  "CompositeEngineOvershootingQcHandler",
  sizeof(CompositeEngineOvershootingQcHandler_t),
  CompositeEngineOvershootingQcHandler_constructor,
  CompositeEngineOvershootingQcHandler_destructor,
  CompositeEngineOvershootingQcHandler_copyconstructor
};
/*@} End of Probability of overshooting quality control functions */

/*@{ Common functions used in quality control tasks  */
 RaveObjectHashTable_t* CompositeEngineQc_getQualityScanFields(CompositeEngineObjectBinding_t* bindings, int nbindings, const char* qualityFieldName)
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
       PolarScan_t* scan = PolarVolume_findAnyScanWithQualityFieldByHowTask((PolarVolume_t*)bindings[i].object, qualityFieldName);
       if (scan != NULL) {
         RaveField_t* field = PolarScan_findAnyQualityFieldByHowTask(scan, qualityFieldName);
         if (field != NULL) {
           PolarScan_t* newscan = PolarScan_createFromScanAndField(scan, field);
           if (newscan == NULL || !RaveObjectHashTable_put(scans, PolarScan_getSource(newscan), (RaveCoreObject*)newscan)) {
             RAVE_ERROR0("Failed to scan to hash table");
             status = 0;
           }
           RAVE_OBJECT_RELEASE(newscan);
         }
         RAVE_OBJECT_RELEASE(field);
       }
       RAVE_OBJECT_RELEASE(scan);
     }
   }
   result = RAVE_OBJECT_COPY(scans);
done:
   RAVE_OBJECT_RELEASE(scans);
   return result;
}

/*@{ End of Common functions used in quality control tasks  */ 
