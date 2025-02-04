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
 * Provides base functionality for creating composites.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-31
 */
#ifndef COMPOSITEENGINE_H
#define COMPOSITEENGINE_H

#include "projection_pipeline.h"
#include "rave_object.h"
#include "rave_types.h"
#include "cartesian.h"
#include "composite_utils.h"
#include "compositearguments.h"
#include "limits.h"

/**
 * Defines a Composite engine
 */
typedef struct _CompositeEngine_t CompositeEngine_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType CompositeEngine_TYPE;

//self, extradata, pipelineBinding[i].object, pipelineBinding[i].pipeline, herex, herey, &olon, &olat
typedef int(*composite_engine_getLonLat_fun)(void* extradata, RaveCoreObject* object, ProjectionPipeline_t* pipeline, double herex, double herey, double* olon, double* olat);

typedef int(*composite_engine_selectRadarData_fun)(void* extradata, RaveCoreObject* object, double olon, double olat, CompositeUtilValue_t* cvalues, int ncvalues);

typedef int(*composite_engine_isUsable_fun)(void* extradata, RaveCoreObject* object, ProjectionPipeline_t* pipeline, double herex, double herey, double* olon, double* olat);

/**
 * This delegates the call to the set lon-lat function.
 * @param[in] self - self
 * @param[in] extradata - the extradata, normally the calling instance
 * @param[in] object - the object to use for getting lon/lat
 * @param[in] pipeline - the projection pipeline
 * @param[in] herex - cartesian surface x
 * @param[in] herey - cartesian surface y
 * @param[out] olon - the longitude
 * @param[out] olat - the latitude
 * @return 1 on success otherwise 0
 */
int CompositeEngineFunction_getLonLat(CompositeEngine_t* self, void* extradata, RaveCoreObject* object, ProjectionPipeline_t* pipeline, double herex, double herey, double* olon, double* olat);

/**
 * @param[in] self - self
 * @param[in] extradata - the extradata, normally the calling instance
 * @param[in] object - the object to use for getting lon/lat
 * @param[in] olon - the longitude
 * @param[in] olat - the latitude
 * @param[in,out] cvalues - the composite values that should be filled in
 * @param[in] ncvalues - number of values in cvalues
 */
int CompositeEngineFunction_selectRadarData(CompositeEngine_t* self, void* extradata, RaveCoreObject* object, double olon, double olat, CompositeUtilValue_t* cvalues, int ncvalues);

/**
 * Sets the lon-lat function. Default is to use standard vol/scan information.
 */
int CompositeEngine_setLonLatFunction(CompositeEngine_t* self, composite_engine_getLonLat_fun getLonLat);

/**
 * Sets the select radar data function. Default is to use standard nearest information.
 */
int CompositeEngine_setSelectRadarDataFunction(CompositeEngine_t* self, composite_engine_selectRadarData_fun selectRadarData);

/**
 * Sets the lon-lat function. Default is to use standard vol/scan information.
 */
int CompositeEngine_isUsable(CompositeEngine_t* self, composite_engine_isUsable_fun isUsable);

/**
 * Generates the composite using a basic approach
 */
Cartesian_t* CompositeEngine_generate(CompositeEngine_t* self, CompositeArguments_t* arguments, void* extradata);

#endif /* COMPOSITEENGINE_H */
