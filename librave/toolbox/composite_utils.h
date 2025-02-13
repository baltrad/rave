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
 * Contains various utility functions when creating composite factories.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-17
 */
#ifndef COMPOSITE_UTILS_H
#define COMPOSITE_UTILS_H
#include "cartesian.h"
#include "cartesianvolume.h"
#include "compositearguments.h"
#include "rave_attribute.h"
#include "rave_object.h"
#include "rave_types.h"
#include "raveobject_list.h"
#include "raveobject_hashtable.h"
#include "projection_pipeline.h"
#include <strings.h>

/**
 * Function pointer that can be used if you want to redirect a call to getting polar value data to a specific function for a specific quantity.
 *
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
typedef int(*composite_utils_getPolarValueAtPosition_fun)(void* extradata, CompositeArguments_t* arguments, RaveCoreObject* object, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue);

/**
 * This structure can be used to keep track of what value that should be used
 * when calculating what value to be used for a position.
 */
typedef struct CompositeUtilValue_t {
  RaveValueType vtype; /**< value type */
  double value;       /**< value */
  double mindist;     /**< min distance */
  double radardist;   /**< distance to radar */
  int radarindex;     /**< radar index in list of radars */
  const char* name;   /**< name of quantity */
  PolarNavigationInfo navinfo; /**< the navigation info */
  CartesianParam_t* parameter; /**< the cartesian parameter */
  double qivalue;     /**< quality value */
  composite_utils_getPolarValueAtPosition_fun getPolarValueAtPosition; /**< specific get polar value at position  */
  void* extradata; /**< will be provided to the extradata pointer in the function call */
} CompositeUtilValue_t;

/**
 * Utility class that helps out keeping a polar object associated with other objects
 * like a projection pipeline.
 */
typedef struct CompositeRaveObjectBinding_t {
  RaveCoreObject* object; /**< the rave object */
  ProjectionPipeline_t* pipeline; /**< the projection pipeline */
} CompositeRaveObjectBinding_t;

/**
 * Can be used to define if a quality flag requires different gain/offset or datatype than the default values
 * which are: UCHAR, offset=0.0 and gain = 255/UCHAR_MAX
 */
typedef struct CompositeQualityFlagSettings_t {
  const char* qualityFieldName;
  RaveDataType datatype;
  double offset;
  double gain;
} CompositeQualityFlagSettings_t;

/**
 * Validates the arguments so that it is possible to create a cartesian product
 * from the existing information.
 * @param[in] arguments - the argument object
 * @return 1 if possible to create the cartesian from arguments
 */
int CompositeUtils_isValidCartesianArguments(CompositeArguments_t* arguments);

/**
 * Creates a cartesian product from the arguments.
 * @param[in] arguments - the argument object
 * @returns The created cartesian product or NULL if not possible
 */
Cartesian_t* CompositeUtils_createCartesianFromArguments(CompositeArguments_t* arguments);

/**
 * Creates an array of CompositeUtilValue_t. The array length will be the same as number of
 * parameters and the parameter in the struct will be associated with corresponding parameter
 * in the cartesian product.
 * @param[in] arguments - the arguments structure
 * @param[in] cartesian - the cartesian that was created from the arguments
 * @param[in] nentries - number of entries in the composite util value struct
 * @return the array on success or NULL on failure
 */
CompositeUtilValue_t* CompositeUtils_createCompositeValues(CompositeArguments_t* arguments, Cartesian_t* cartesian, int* nentries);

/**
 * Resets the array of composite values except the CartesianParam parameter.
 * @param[in] arguments - the arguments structure
 * @param[in] cvalues - pointer at the array
 * @param[in] nparam - number of parameters
 */
void CompositeUtils_resetCompositeValues(CompositeArguments_t* arguments, CompositeUtilValue_t* cvalues, int nentries);

/**
 * Frees the composite value array and ensures that all associated parameters are released.
 * @param[in,out] cvalues - the cvalue struct.
 * @param[in] nparam - number of entries in the array
 */
void CompositeUtils_freeCompositeValueParameters(CompositeUtilValue_t** cvalues, int nparam);

/**
 * Tries to get the projection from the provided RaveCoreObject. Currently supported object types
 * are PolarScan, PolarVolume, Cartesian and CartesianVolume.
 * @param[in] obj - the object
 * @return the projection if possible, otherwise NULL
 */
Projection_t* CompositeUtils_getProjection(RaveCoreObject* obj);

/**
 * Tries to get the source from the provided RaveCoreObject. Currently supported object types
 * are PolarScan, PolarVolume, Cartesian and CartesianVolume.
 * @param[in] obj - the object
 * @param[in,out] source - the source
 * @param[in] nlen - length of source array
 * @return string length (excluding the terminating \0)
 */
int CompositeUtils_getObjectSource(RaveCoreObject* obj, char* source, int nlen);

/**
 * Creates the binding between radar objects and the pipelines that are relevant when creating composites.
 * The order of the binding will be the same as the objects in the arguments at time the object is 
 * @param[in] arguments - the arguments (containing the radar objects)
 * @param[in] cartesian - the target composite 
 * @param[out] nobject - the number of items in the returned array
 * @return the array of bindings or NULL on failure
 */
CompositeRaveObjectBinding_t* CompositeUtils_createRaveObjectBinding(CompositeArguments_t* arguments, Cartesian_t* cartesian, int* nobjects);

/**
 * Releases the objects and then deallocates the array
 * @param[in,out] arr - the array to release
 * @param[in] nobjects - number of items in array
 */
void CompositeUtils_releaseRaveObjectBinding(CompositeRaveObjectBinding_t** arr, int nobjects);

/**
 * Returns the offset, gain and datatype for the specified flag that has been defined in the settings. If the flagname doesn't exist in settings
 * or if settings is NULL. Offset, gain and datatype will be set to 0.0, 1.0/UCHAR_MAX and RaveDataType_UCHAR.
 * @param[in] settings - the array of settings
 * @param[in] flagname - name of the quality field
 * @param[out] offset - offset, may be NULL.
 * @param[out] gain - gain, may be NULL.
 * @param[out] datatype - datatype, may be NULL.
 */
void CompositeUtils_getQualityFlagSettings(CompositeQualityFlagSettings_t* settings, const char* flagname, double* offset, double* gain, RaveDataType* datatype);

/**
 * Creates all quality flags specified in the arguments and add them to the cartesian product.
 * @param[in] arguments - the arguments (containing the list of quality flags)
 * @param[in] cartesian - the product to which the quality fields should be added
 * @param[in] settings - the settings of specific quality flags. Should be defined as 
 * CompositeQualityFlagSettings_t settings[] = {
 *   {<qualityflag>, <datatype>, <offset>, <gain>},
 *   ....
 *   {NULL, RaveDataType_UNDEFINED, 0.0, 0.0}
 * };
 * since the loop will break at the first qualityflag == NULL.
 * @return 1 on success otherwise 0
 */
int CompositeUtils_addQualityFlagsToCartesian(CompositeArguments_t* arguments, Cartesian_t* cartesian, CompositeQualityFlagSettings_t* settings);

/**
 * Adds what/gain and what/offset to the RaveField.
 * @param[in] field - the field that should get gain & offset set
 * @param[in] gain - gain
 * @param[in] offset - offset
 * @return 1 on success otherwise 0
 */
int CompositeUtils_addGainAndOffsetToField(RaveField_t* field, double gain, double offset);

/**
 * Creates a quality field.
 * @param[in] howtaskvaluestr - the how/task string
 * @param[in] xsize - xsize
 * @param[in] ysize - ysize
 * @param[in] datatype - the data type
 * @param[in] gain - the gain
 * @param[in] offset - the offset
 * @return the created field on success otherwise NULL
 */
RaveField_t* CompositeUtils_createQualityField(const char* howtaskvaluestr, int xsize, int ysize, RaveDataType datatype, double gain, double offset);

/**
 * Gets the value(s) at the specified position for the specified quantity.
 * @param[in] obj - the object
 * @param[in] quantity - the quantity
 * @param[in] nav - the navigation information
 * @param[in] qiFieldName - the name of the quality field (may be NULL)
 * @param[out] type - the value type
 * @param[out] value - the value
 * @param[out] qualityValue - the quality value, may be NULL
 * @return 1 on success or 0 if value not could be retrieved
 */
int CompositeUtils_getPolarValueAtPosition(RaveCoreObject* obj, const char* quantity, PolarNavigationInfo* nav, const char* qiFieldName, RaveValueType* type, double* value, double* qualityValue);

/**
 * Gets the quality value at the specified position for the specified quantity and quality field in a polar object.
 * @param[in] obj - the object (Must be PolarScan or PolarVolume)
 * @param[in] quantity - the quantity
 * @param[in] qualityField - the quality field
 * @param[in] nav - the navigation information
 * @param[out] value - the value
 * @return 1 on success or 0 if value not could be retrieved
 */
int CompositeUtils_getPolarQualityValueAtPosition(RaveCoreObject* obj, const char* quantity, const char* qualityField, PolarNavigationInfo* nav, double* value);

/**
 * Returns the vertical max value for the specified quantity at the provided lon/lat position.
 * If no suitable value is found, vtype and vvalue will be left as is.
 *
 * @param[in] object - the polar object (MUST NOT BE NULL)
 * @param[in] quantity - the parameter
 * @param[in] qiFieldName - the quality field name (if qiv should be set), may be NULL
 * @param[in] lon - longitude in radians
 * @param[in] lat - latitude in radians
 * @param[out] vtype - the value type (MUST NOT BE NULL)
 * @param[out] vvalue - the value (MUST NOT BE NULL)
 * @param[out] navinfo - the navigation information (MAY BE NULL)
 * @param[out] qiv - the quality value (MAY BE NULL)
 * @return 1 on success or 0 on failure.
 */
int CompositeUtils_getVerticalMaxValue(
  RaveCoreObject* object,
  const char* quantity,
  const char* qiFieldName,
  double lon,
  double lat,
  RaveValueType* vtype,
  double* vvalue,
  PolarNavigationInfo* navinfo,
  double* qiv);



/**
 * Clones a RaveList of strings.
 * @param[in] inlist - the list to clone
 * @return the cloned list on success, otherwise NULL
 */
RaveList_t* CompositeUtils_cloneRaveListStrings(RaveList_t* inlist);

#endif /* COMPOSITE_UTILS_H */
