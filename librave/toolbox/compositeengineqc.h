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
 * Contains definitions useful when adding qc-handling to composite factories
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-02-26
 */
 #ifndef COMPOSITE_ENGINE_QC_H
 #define COMPOSITE_ENGINE_QC_H
 #include "cartesian.h"
 #include "cartesianvolume.h"
 #include "compositearguments.h"
 #include "rave_attribute.h"
 #include "rave_object.h"
 #include "rave_types.h"
 #include "rave_value.h"
 #include "raveobject_list.h"
 #include "raveobject_hashtable.h"
 #include "rave_properties.h"
 #include "projection_pipeline.h"
 #include "composite_utils.h"
 #include "compositeenginebase.h"
 #include <strings.h>
 
/*@{ CompositeEngineQcHandler  */
struct _CompositeEngineQcHandler_t;

typedef const char*(*composite_engine_qc_handler_getQualityFieldName_fun)(struct _CompositeEngineQcHandler_t* self);

typedef int(*composite_engine_qc_handler_initialize_fun)(struct _CompositeEngineQcHandler_t* self, void* extradata, RaveProperties_t* properties, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings);

typedef int(*composite_engine_qc_handler_getQualityValue_fun)(struct _CompositeEngineQcHandler_t* self, void* extradata, CompositeArguments_t* arguments, RaveCoreObject* object, const char* quantity, const char* qiFieldName, PolarNavigationInfo* navinfo, double* qivalue);

typedef CompositeQualityFlagDefinition_t*(*composite_engine_qc_handler_getFlagDefinition_fun)(struct _CompositeEngineQcHandler_t* self);

#define COMPOSITE_ENGINE_QC_HANDLER_HEAD \
  composite_engine_qc_handler_getQualityFieldName_fun getQualityFieldName; \
  composite_engine_qc_handler_initialize_fun initialize; \
  composite_engine_qc_handler_getQualityValue_fun getQualityValue; \
  composite_engine_qc_handler_getFlagDefinition_fun getFlagDefinition;

/**
 * CompositeEngineQcHandler
 */
typedef struct _CompositeEngineQcHandler_t {
  RAVE_OBJECT_HEAD /** Always on top */
  COMPOSITE_ENGINE_QC_HANDLER_HEAD /**< qc specifics */
} CompositeEngineQcHandler_t;


/**
 * Macro expansion for calling the name function
 * @param[in] self - self
 * @returns the unique name for this algorithm
 */
 #define CompositeEngineQcHandler_getQualityFieldName(self) \
   ((CompositeEngineQcHandler_t*)self)->getQualityFieldName((CompositeEngineQcHandler_t*)self)

/**
 * Macro expansion for calling the initialize function
 * @param[in] self - self
 * @returns 1 on success otherwise 0
 */
 #define CompositeEngineQcHandler_initialize(self, extradata, properties, arguments, bindings, nbindings) \
   ((CompositeEngineQcHandler_t*)self)->initialize((CompositeEngineQcHandler_t*)self, extradata, properties, arguments, bindings, nbindings)

/**
 * Macro expansion for calling the getQualityValue function
 * @param[in] self - self
 * @returns 1 on success otherwise 0
 */
 #define CompositeEngineQcHandler_getQualityValue(self, extradata, args, obj, quantity, qfieldname, navinfo,  v) \
   ((CompositeEngineQcHandler_t*)self)->getQualityValue((CompositeEngineQcHandler_t*)self, extradata, args, obj, quantity, qfieldname, navinfo,  v)

/**
 * Macro expansion for calling the getFlagDefinition function
 * @param[in] self - self
 * @returns the flag definition on sucess otherwise NULL
 */
 #define CompositeEngineQcHandler_getFlagDefinition(self) \
   ((CompositeEngineQcHandler_t*)self)->getFlagDefinition((CompositeEngineQcHandler_t*)self)

/*@{ Probability of overshooting quality control functions  */
typedef struct _CompositeEngineOvershootingQcHandler_t CompositeEngineOvershootingQcHandler_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType CompositeEngineOvershootingQcHandler_TYPE;


 /*@} End of Probability of overshooting quality control functions */

/*@{ Common functions used in quality control tasks  */
 /**
 * Will traverse all objects in the list and atempt to find a scan that contains a
 * quality field that has got a how/task value == qualityFieldName.
 * All scans that contains such a field will get a scan set in the resulting
 * hash table with the quality data set as the default (and only) parameter.
 * @param[in] composite - the composite
 * @return a hash table
 */
 RaveObjectHashTable_t* CompositeEngineQc_getQualityScanFields(CompositeEngineObjectBinding_t* bindings, int nbindings, const char* qualityFieldName);
/*@{ End of Common functions used in quality control tasks  */ 

 #endif /* COMPOSITE_ENGINE_QC_H */
 