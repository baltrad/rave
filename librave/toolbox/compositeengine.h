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
 * Forward declaration of composite radar data struct
 */
struct CompositeEngineRadarData_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType CompositeEngine_TYPE;

/** The resolution to use for scaling the distance from pixel to used radar. */
/** By multiplying the values in the distance field by 2000, we get the value in unit meters. */
#define COMPOSITE_ENGINE_DISTANCE_TO_RADAR_RESOLUTION 2000.0

/** Same for height, scaled to 100 m resolution up to 25.5 km */
#define COMPOSITE_ENGINE_HEIGHT_RESOLUTION 100.0

/** The name of the task for specifying distance to radar */
#define COMPOSITE_ENGINE_DISTANCE_TO_RADAR_HOW_TASK "se.smhi.composite.distance.radar"

/** The name of the task for specifying height above sea level */
#define COMPOSITE_ENGINE_HEIGHT_ABOVE_SEA_HOW_TASK "se.smhi.composite.height.radar"

/** The name of the task for indexing the radars used */
#define COMPOSITE_ENGINE_RADAR_INDEX_HOW_TASK "se.smhi.composite.index.radar"

#define COMPOSITE_ENGINE_DEFAULT_QUALITY_FIELDS_GAIN   (1.0/UCHAR_MAX)

#define COMPOSITE_ENGINE_DEFAULT_QUALITY_FIELDS_OFFSET 0.0

/**
 * Function pointer used during composite generation (\ref CompositeEngine_generate).
 * Called when translating a surface coordinate into a lon/lat position. Will be called for each pixel in the resulting cartesian product.
 * @param[in] engine - self
 * @param[in] extradata - the extradata passed to the \ref CompositeEngine_generate function.
 * @param[in] object - the object corresponding to the pipeline
 * @param[in] pipeline - between cartesian->polar object.
 * @param[in] herex - the surface x coordinate
 * @param[in] herey - the surface y coordinate
 * @param[out] olon - out longitude in radians
 * @param[out] olat - out latitude in radians
 */
typedef int(*composite_engine_getLonLat_fun)(CompositeEngine_t* engine, void* extradata, RaveCoreObject* object, ProjectionPipeline_t* pipeline, double herex, double herey, double* olon, double* olat);

/**
 * Function pointer used during composite generation (\ref CompositeEngine_generate).
 * Called when identifying the radar values to use. Will fill the cvalues struct for each affected parameter if the polar object data should be used..
 * @param[in] engine - self
 * @param[in] extradata - the extradata passed to the \ref CompositeEngine_generate function.
 * @param[in] arguments -the arguments used when calling \ref CompositeEngine_generate function.
 * @param[in] object - the rave object to process
 * @param[in] 
 * @param[in] herex - the surface x coordinate
 * @param[in] herey - the surface y coordinate
 * @param[out] olon - out longitude in radians
 * @param[out] olat - out latitude in radians
 */
typedef int(*composite_engine_selectRadarData_fun)(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, RaveCoreObject* object, int index, double olon, double olat, struct CompositeEngineRadarData_t* cvalues, int ncvalues);

/**
 * Function pointer used during composite generation (\ref CompositeEngine_generate) from the default selectRadarData function. This function is
 * registered for each quantity. 
 * @param[in] extradata - the extradata passed to the \ref CompositeEngine_generate function.
 * @param[in] arguments -the arguments used when calling \ref CompositeEngine_generate function.
 * @param[in] object - the rave object to process
 * @param[in] quantity - the quantity
 * @param[in] navinfo - navigation info for position in radar data
 * @param[in] qiFieldName - name of the quality field if that also should be retrieved
 * @param[out] otype - the rave type of value
 * @param[out] ovalue - the value
 * @param[out] qivalue - the quality value
 * @return 1 on success otherwise 0
 */
typedef int(*composite_engine_getPolarValueAtPosition_fun)(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, RaveCoreObject* object, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue);

typedef int(*composite_engine_setRadarData_fun)(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, Cartesian_t* cartesian, double olon, double olat, long x, long y, struct CompositeEngineRadarData_t* cvalues, int ncvalues);

typedef int(*composite_engine_addQualityFlagsToCartesian_fun)(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, Cartesian_t* cartesian);

typedef int(*composite_engine_fillQualityInformation_fun)(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, long x, long y, CartesianParam_t* param, double radardist, int radarindex, PolarNavigationInfo* info);

typedef struct CompositeEngineRadarData_t {
  RaveValueType vtype; /**< value type */
  double value;       /**< value */
  double mindist;     /**< min distance */
  double radardist;   /**< distance to radar */
  int radarindex;     /**< radar index in list of radars */
  const char* name;   /**< name of quantity */
  PolarNavigationInfo navinfo; /**< the navigation info */
  CartesianParam_t* parameter; /**< the cartesian parameter */
  double qivalue;     /**< quality value */
  composite_engine_getPolarValueAtPosition_fun getPolarValueAtPosition; /**< specific get polar value at position  */
  void* extradata; /**< will be provided to the extradata pointer in the function call */
} CompositeEngineRadarData_t;

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
 * @param[in] arguments - the arguments
 * @param[in] object - the object to use for getting lon/lat
 * @param[in] index - index of the object in the list (used for radarindexing)
 * @param[in] olon - the longitude
 * @param[in] olat - the latitude
 * @param[in,out] cvalues - the composite values that should be filled in
 * @param[in] ncvalues - number of values in cvalues
 */
int CompositeEngineFunction_selectRadarData(CompositeEngine_t* self, void* extradata, CompositeArguments_t* arguments, RaveCoreObject* object, int index, double olon, double olat, CompositeEngineRadarData_t* cvalues, int ncvalues);

/**
 * @param[in] self - self
 * @param[in] extradata - the extradata, normally the calling instance
 * @param[in] arguments - the arguments
 * @param[in] cartesian - the cartesian object to write to
 * @param[in] olon - the longitude
 * @param[in] olat - the latitude
 * @param[in] x - x coordinate in cartesian
 * @param[in] y - y coordinate in cartesian
 * @param[in,out] cvalues - the composite values that should be filled in
 * @param[in] ncvalues - number of values in cvalues
 */
int CompositeEngineFunction_setRadarData(CompositeEngine_t* self, void* extradata, CompositeArguments_t* arguments, Cartesian_t* cartesian, double olon, double olat, long x, long y, CompositeEngineRadarData_t* cvalues, int ncvalues);

/**
 * Adds the quality flags to the cartesian product.

 * @param[in] self - self
 * @param[in] extradata - the extradata, normally the calling instance
 * @param[in] arguments - the arguments
 * @param[in] cartesian - the cartesian object to write to
 * @return 1 on success otherwise false;
 */
int CompositeEngineFunction_addQualityFlagsToCartesian(CompositeEngine_t* self, void* extradata, CompositeArguments_t* arguments, Cartesian_t* cartesian);

/**
 * Fills the quality information into the product

 * @param[in] self - self
 * @param[in] extradata - the extradata, normally the calling instance
 * @param[in] arguments - the arguments
 * @param[in] cartesian - the cartesian object to write to
 * @return 1 on success otherwise false;
 */
int CompositeEngineFunction_fillQualityInformation(CompositeEngine_t* self, void* extradata, CompositeArguments_t* arguments, long x, long y, CartesianParam_t* param, double radardist, int radarindex, PolarNavigationInfo* info);

/**
 * Sets the lon-lat function. Default is to use standard vol/scan information.
 */
int CompositeEngine_setLonLatFunction(CompositeEngine_t* self, composite_engine_getLonLat_fun getLonLat);

/**
 * Sets the select radar data function. Default is to use standard nearest information.
 */
int CompositeEngine_setSelectRadarDataFunction(CompositeEngine_t* self, composite_engine_selectRadarData_fun selectRadarData);

/**
 * Sets the default polarValueAtPosition function,
 * @param[in] self - self
 * @param[in] getPolarValueAtPosition - the function that will be called for the specified quantity
 * @return 1 on success otherwise 0
 */
int CompositeEngine_setDefaultPolarValueAtPositionFunction(CompositeEngine_t* self, composite_engine_getPolarValueAtPosition_fun getPolarValueAtPosition);

/**
 * Registers a polar value at position function for the specified quantity. This is using the composite utils definition in order to speed up calls instead of looking up
 * for each quantity. This function will use navinfo to retrieve data from wanted position. If getPolarValueAtPosition is NULL, the mapping is removed.
 * @param[in] self - self
 * @param[in] quantity - the quantity that should trigger this call (can be NULL)
 * @param[in] getPolarValueAtPosition - the function that will be called for the specified quantity
 * @return 1 on success otherwise 0
 */
int CompositeEngine_registerPolarValueAtPositionFunction(CompositeEngine_t* self, const char* quantity, composite_engine_getPolarValueAtPosition_fun getPolarValueAtPosition);

/**
 * Sets the set radar data function. Default is to set data and quality information.
 */
int CompositeEngine_setSetRadarDataFunction(CompositeEngine_t* self, composite_engine_setRadarData_fun setRadarData);

/**
 * Sets the add quality flags to cartesian that will add the quality flags to the cartesian product.
 */
int CompositeEngine_setAddQualityFlagsToCartesianFunction(CompositeEngine_t* self, composite_engine_addQualityFlagsToCartesian_fun addQualityFlagsToCartesian);

/**
 * The fill quality information function
 */
int CompositeEngine_setFillQualityInformationFunction(CompositeEngine_t* self, composite_engine_fillQualityInformation_fun fillQualityInformation);

/**
 * Calculates the lon/lat coordinate using the provided pipeline.
 * @param[in] engine - the engine
 * @param[in] object - the object
 * @param[in] pipeline - the pipeline
 * @param[in] herex - the x surface coordinate
 * @param[in] herey - the y surface coordinate
 * @param[out] olon - the longitude in radians
 * @param[out] olat - the latitude in radians
 * @return 1 on success otherwise 0
 */
int CompositeEngineUtility_getLonLat(CompositeEngine_t* engine, RaveCoreObject* object, ProjectionPipeline_t* pipeline, double herex, double herey, double* olon, double* olat);

/**
 * Fetches the radar data for specified position using NEAREST.
 * @param[in] engine - the engine
 * @param[in] arguments - the arguments
 * @param[in] object - the object
 * @param[in] index - the index in object list (this has to be set in order to get proper radar indexing)
 * @param[out] olon - the longitude in radians
 * @param[out] olat - the latitude in radians
 * @param[in,out] cvalues - the composite parameter values
 * @param[in] ncvalues - number of entries in cvalues
 * @return 1 on success otherwise 0
 */
int CompositeEngineUtility_selectRadarData(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, RaveCoreObject* object, int index, double olon, double olat, CompositeEngineRadarData_t* cvalues, int ncvalues);

/**
 * Fetches the polar value from specified position for specified quantity. 
 */
int CompositeEngineUtility_getPolarValueAtPosition(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, RaveCoreObject* object, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue);

/**
 * Sets the radar data for specified position using NEAREST. Will also invoke fillQualityInformation function.
 * @param[in] engine - the engine
 * @param[in] extradata - the extradata passed to the fillQualityInformation
 * @param[in] arguments - the arguments
 * @param[in] cartesian - the cartesian to fill
 * @param[in] olon - the longitude in radians
 * @param[in] olat - the latitude in radians
 * @param[in] x - the x position in cartesian
 * @param[in] y - the y position in cartesian
 * @param[in] cvalues - the composite parameter values
 * @param[in] ncvalues - number of entries in cvalues
 * @return 1 on success otherwise 0
 */
int CompositeEngineUtility_setRadarData(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, Cartesian_t* cartesian, double olon, double olat, long x, long y, CompositeEngineRadarData_t* cvalues, int ncvalues);

/**
 * Creates the quality fields in the cartesian product using arguments and default parameters.
 * The default configuration for the quality flags is:
 * static CompositeQualityFlagSettings_t COMPOSITE_ENGINE_QUALITY_FLAG_DEFINITIONS[] = {
 *   {COMPOSITE_ENGINE_DISTANCE_TO_RADAR_HOW_TASK, RaveDataType_UCHAR, 0.0, COMPOSITE_ENGINE_DISTANCE_TO_RADAR_RESOLUTION},
 *   {COMPOSITE_ENGINE_HEIGHT_ABOVE_SEA_HOW_TASK, RaveDataType_UCHAR, 0.0, COMPOSITE_ENGINE_HEIGHT_RESOLUTION},
 *   {COMPOSITE_ENGINE_RADAR_INDEX_HOW_TASK, RaveDataType_UCHAR, 0.0, 1.0},
 *   {NULL, RaveDataType_UNDEFINED, 0.0, 0.0}
 * };
 * @param[in] engine - the engine
 * @param[in] arguments - the arguments
 * @param[in] cartesian - the cartesian to fill
 * @return 1 on success otherwise 0
 */
int CompositeEngineUtility_addQualityFlagsToCartesian(CompositeEngine_t* engine, CompositeArguments_t* arguments, Cartesian_t* cartesian);

/**
 * Fills the quality information for the provided parameter. 
 * @param[in] engine - the engine
 * @param[in] arguments - the arguments
 * @param[in] x - the x position in cartesian
 * @param[in] y - the y position in cartesian
 * @param[in] param - the cartesian parameter to fill
 * @param[in] radardist - the distance to the radar for the specified point
 * @param[in] radarindex - the radar index in list of objects affected
 * @param[in] navinfo - the navigation info used
 * @return 1 on success otherwise 0
 */
int CompositeEngineUtility_fillQualityInformation(CompositeEngine_t* self, CompositeArguments_t* arguments, long x, long y, CartesianParam_t* param, double radardist, int radarindex, PolarNavigationInfo* navinfo);

/**
 * Creates an array of CompositeEngineRadarData_t. The array length will be the same as number of
 * parameters and the parameter in the struct will be associated with corresponding parameter
 * in the cartesian product.
 * @param[in] arguments - the arguments structure
 * @param[in] cartesian - the cartesian that was created from the arguments
 * @param[in] nentries - number of entries in the returned array
 * @return the array on success or NULL on failure
 */
CompositeEngineRadarData_t* CompositeEngineUtility_createRadarData(CompositeArguments_t* arguments, Cartesian_t* cartesian, int* nentries);

/**
 * Resets the array of CompositeEngineRadarData_t except the CartesianParam parameter.
 * @param[in] arguments - the arguments structure
 * @param[in] cvalues - pointer at the array
 * @param[in] nparam - number of parameters
 */
void CompositeEngineUtility_resetRadarData(CompositeArguments_t* arguments, CompositeEngineRadarData_t* cvalues, int nentries);

/**
 * Frees the CompositeEngineRadarData_t and ensures that all associated parameters are released.
 * @param[in,out] cvalues - the cvalue struct.
 * @param[in] nparam - number of entries in the array
 */
void CompositeEngineUtility_freeRadarData(CompositeEngineRadarData_t** cvalues, int nparam);

/**
 * Generates the composite using a basic approach
 */
Cartesian_t* CompositeEngine_generate(CompositeEngine_t* self, CompositeArguments_t* arguments, void* extradata);

#endif /* COMPOSITEENGINE_H */
