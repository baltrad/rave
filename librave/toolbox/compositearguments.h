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
 * The arguments that should be passed on to the composite generator.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-10-11
 */
#ifndef COMPOSITE_ARGUMENTS_H
#define COMPOSITE_ARGUMENTS_H
#include "rave_proj.h"
#include "area.h"
#include "projection.h"
#include "rave_object.h"
#include "rave_datetime.h"
#include "rave_attribute.h"
#include "raveobject_list.h"
#include "odim_sources.h"

/**
 * Defines a Geographical Area
 */
typedef struct _CompositeArguments_t CompositeArguments_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType CompositeArguments_TYPE;

typedef enum Rave_CompositingMethod {
  Rave_CompositingProduct_PPI,    /**< PPI */
  Rave_CompositingProduct_CAPPI,  /**< CAPPI */
  Rave_CompositingProduct_PCAPPI, /**< PCAPPI */
  Rave_CompositingProduct_ETOP,   /**< ETOP */
  Rave_CompositingProduct_MAX,    /**< MAX */
  Rave_CompositingProduct_RR,     /**< RR */
  Rave_CompositingProduct_PMAX,   /**< PMAX */
  Rave_CompositingProduct_ACQVA,  /**< ACQVA */
  Rave_CompositingProduct_UNDEFINED  /**< Not defined*/
} Rave_CompositingProduct;


/**
 * Converts a method into a string.
 * @param[in] method - the \ref Rave_CompositingMethod
 * @returns a const char defining the string
 */
const char* CompositeArguments_productToString(Rave_CompositingProduct product);

/**
 * Converts a string into a product.
 * @param[in] method - the string
 * @returns the \ref Rave_CompositingMethod or Rave_CompositingMethod_UNDEFINED if not possible to translate
 */
Rave_CompositingProduct CompositeArguments_stringToProduct(const char* product);

/**
 * Sets the sources registry for lookup purposes
 * @param[in] args - self
 * @param[in] sources - the sources
 */
void CompositeArguments_setSources(CompositeArguments_t* args, OdimSources_t* sources);

/**
 * Returns the sources registry.
 * @param[in] args - self
 * @return the odim sources or NULL if none is set
 */
OdimSources_t* CompositeArguments_getSources(CompositeArguments_t* args);

/**
 * Sets the compositing method to use when creating the composite. Note, this should be
 * set as a const char since a plugin might be able to provide support for a non-predefined
 * method. 
 * 
 * Each method will have it's own requirements and as such it is up to the plugin to validate
 * and allow for the combination.
 *
 * Height/Elevation angle and range are used in combination with the products.
 * PPI requires elevation angle
 * CAPPI, PCAPPI and PMAX requires height above sea level
 * PMAX also requires range in meters
 *
 * @param[in] args - self
 * @param[in] product - the method to use
 * @return 1 on success otherwise 0
 */
int CompositeArguments_setProduct(CompositeArguments_t* args, const char* product);

/**
 * Returns the compositing product
 * @returns the compositing product
 */
const char* CompositeArguments_getProduct(CompositeArguments_t* args);

/**
 * Sets the area to use when creating the composite.
 * @param[in] args - self
 * @param[in] area - the area
 * @return 1 on success otherwise 0
 */
int CompositeArguments_setArea(CompositeArguments_t* args, Area_t* area);

/**
 * Returns the area.
 * @param[in] args - self
 * @return the area
 */
Area_t* CompositeArguments_getArea(CompositeArguments_t* args);

/**
 * Sets the nominal time.
 * @param[in] args - self
 * @param[in] value - the time in the format HHmmss
 * @returns 1 on success, otherwise 0
 */
int CompositeArguments_setTime(CompositeArguments_t* args, const char* value);

/**
 * Returns the nominal time.
 * @param[in] args - self
 * @returns the nominal time (or NULL if there is none)
 */
const char* CompositeArguments_getTime(CompositeArguments_t* args);

/**
 * Sets the nominal date.
 * @param[in] args - self
 * @param[in] value - the date in the format YYYYMMDD
 * @returns 1 on success, otherwise 0
 */
int CompositeArguments_setDate(CompositeArguments_t* args, const char* value);

/**
 * Returns the nominal date.
 * @param[in] args - self
 * @returns the nominal time (or NULL if there is none)
 */
const char* CompositeArguments_getDate(CompositeArguments_t* args);

/**
 * Sets the height that should be used when generating a composite.
 * @param[in] args - self
 * @param[in] height - the height
 */
void CompositeArguments_setHeight(CompositeArguments_t* args, double height);

/**
 * Returns the height that is used for composite generation.
 * @param[in] args - self
 * @returns the height
 */
double CompositeArguments_getHeight(CompositeArguments_t* args);

/**
 * Sets the elevation angle that should be used when generating a
 * composite as PPI.
 * @param[in] args - self
 * @param[in] angle - the angle in radians
 */
void CompositeArguments_setElevationAngle(CompositeArguments_t* args, double angle);

/**
 * Returns the elevation angle that is used for composite generation.
 * @param[in] args - self
 * @returns the elevation angle in radians
 */
double CompositeArguments_getElevationAngle(CompositeArguments_t* args);

/**
 * Sets the range that should be used when generating the Pseudo MAX. This range
 * is the limit in meters for when the vertical max should be used. When outside
 * this range, the PCAPPI value is used instead.
 *
 * @param[in] args - self
 * @param[in] angle - the range in meters
 */
void CompositeArguments_setRange(CompositeArguments_t* args, double range);

/**
 * Returns the range in meters
 * @param[in] args - self
 * @returns the range in meters
 */
double CompositeArguments_getRange(CompositeArguments_t* args);

/**
 * Sets the strategy id. Will help when determining which plugin/factory to use for
 * product generation.
 * @param[in] args - self
 * @param[in] strategy - the strategy id
 * @return 1 on success, otherwise 0
 */
int CompositeArguments_setStrategy(CompositeArguments_t* args, const char* strategy);

/**
 * Returns the strategy.
 * @param[in] args - self
 * @return the strategy or NULL
 */
const char* CompositeArguments_getStrategy(CompositeArguments_t* args);

/**
 * Adds a rave attriargumentbute to the composite arguments.
 * instead.
 * @param[in] args - self
 * @param[in] argument - the argument
 * @return 1 on success otherwise 0
 */
int CompositeArguments_addArgument(CompositeArguments_t* args, RaveAttribute_t* argument);

/**
 * Returns the rave argument that is named accordingly.
 * @param[in] args - self
 * @param[in] name - the name of the argument
 * @returns the attribute if found otherwise NULL
 */
RaveAttribute_t* CompositeArguments_getArgument(CompositeArguments_t* args, const char* name);

/**
 * Returns if there is a rave argument that is named accordingly.
 * @param[in] args - self
 * @param[in] name - the name of the argument
 * @returns if argument exists or not
 */
int CompositeArguments_hasArgument(CompositeArguments_t* args, const char* name);

/**
 * Adds a parameter to be processed.
 * @param[in] args - self
 * @param[in] quantity - the parameter quantity
 * @param[in] gain - the gain to be used for the parameter
 * @param[in] offset - the offset to be used for the parameter
 * @param[in] datatype - the type of data \ref #RaveDataType
 * @param[in] nodata - the nodata value to use
 * @param[in] undetect - the undetect value to use
 * @return 1 on success
 */
int CompositeArguments_addParameter(CompositeArguments_t* args, const char* quantity, double gain, double offset, RaveDataType datatype, double nodata, double undetect);

/**
 * Returns if this composite generator is going to process specified parameter
 * @param[in] args - self
 * @param[in] quantity - the parameter quantity
 * @return 1 if yes otherwise 0
 */
int CompositeArguments_hasParameter(CompositeArguments_t* args, const char* quantity);

/**
 * Returns the parameter at specified index
 * @param[in] args - self
 * @param[in] parameter - the name of the parameter
 * @param[out] gain - the gain to be used for the parameter (MAY BE NULL)
 * @param[out] offset - the offset to be used for the parameter (MAY BE NULL)
 * @param[out] datatype - the datatype to be used for the parameter (MAY BE NULL)
 * @param[out] nodata - the nodata to be used for the parameter (MAY BE NULL)
 * @param[out] undetect - the undetect to be used for the parameter (MAY BE NULL)
 * @return 1 if parameter found, otherwise 0
 */
int CompositeArguments_getParameter(CompositeArguments_t* args, const char* parameter, double* gain, double* offset, RaveDataType* datatype, double* nodata, double* undetect);

/**
 * Returns the number of parameters to be processed
 * @param[in] composite - self
 * @return the number of parameters
 */
int CompositeArguments_getParameterCount(CompositeArguments_t* args);

/**
 * Returns the parameter at specified index
 * @param[in] composite - self
 * @param[in] index - the index
 * @param[out] gain - the gain to be used for the parameter (MAY BE NULL)
 * @param[out] offset - the offset to be used for the parameter (MAY BE NULL)
 * @param[out] datatype - the datatype to be used for the parameter (MAY BE NULL)
 * @param[out] nodata - the nodata to be used for the parameter (MAY BE NULL)
 * @param[out] undetect - the undetect to be used for the parameter (MAY BE NULL)
 * @return the parameter name, NULL otherwise
 */
const char* CompositeArguments_getParameterAtIndex(CompositeArguments_t* args, int index, double* gain, double* offset, RaveDataType* datatype, double* nodata, double* undetect);

/**
 * Returns the parameter name at specified index.
 * @param[in] composite - self
 * @param[in] index - the index
 * @return the parameter name if found, NULL otherwise
 */
const char* CompositeArguments_getParameterName(CompositeArguments_t* args, int index);

/**
 * Adds a rave object to the arguments
 * @param[in] args - self
 * @param[in] object - a rave object.
 * @return 1 on success, otherwise 0
 */
int CompositeArguments_addObject(CompositeArguments_t* args, RaveCoreObject* object);

/**
 * Returns the number of objects
 * @param[in] args - self
 * @return number of objects
 */
int CompositeArguments_getNumberOfObjects(CompositeArguments_t* args);

/**
 * Returns the object at specified index
 * @param[in] args - self
 * @param[in] index - position in list of objects
 * @return the found object or NULL if not valid index
 */
RaveCoreObject* CompositeArguments_getObject(CompositeArguments_t* args, int index);

/**
 * Adds a quality flag that should be generated during processing.
 * @param[in] args - self
 * @param[in] flag - the quality flag that should be passed on
 * @return 1 on success, otherwise 0
 */
int CompositeArguments_addQualityFlag(CompositeArguments_t* args, const char* flag);

/**
 * Sets the quality flags that should be used.
 * @param[in] args - self
 * @param[in] flags - an array of quality flags
 * @param[in] nrflags - number of flags
 * @return 1 on success otherwise 0
 */
int CompositeArguments_setQualityFlags(CompositeArguments_t* args, const char* flags[], int nrflags);

/*
 * Removes the quality flag with provided name
 * @param[in] args - self
 * @param[in] flag - the quality flag that should be removed
 * @return 1 on success, otherwise 0
 */
int CompositeArguments_removeQualityFlag(CompositeArguments_t* args, const char* flag);

/*
 * Removes the quality flag at specified index
 * @param[in] args - self
 * @param[in] index - the index
 * @return 1 on success, otherwise 0
 */
int CompositeArguments_removeQualityFlagAt(CompositeArguments_t* args, int index);

/**
 * Returns the number of quality flags.
 * @param[in] args - self
 * @return number of quality flags
 */
int CompositeArguments_getNumberOfQualityFlags(CompositeArguments_t* args);

/**
 * Returns the name of the quality flag at specified position.
 * @param[in] args - self
 * @param[in] index - index of quality flag
 * @return quality flag on success, otherwise NULL
 */
const char* CompositeArguments_getQualityFlagAt(CompositeArguments_t* args, int index);

/**
 * Creates the radar index mapping from the objects and the sources. If no sources
 * instance a best effort will be done creating the index mapping.
 *
 * @param[in] args - self
 * @return 1 on success, otherwise 0
 */
int CompositeArguments_createRadarIndexMapping(CompositeArguments_t* args);

/**
 * Returns the registered radar indexes.
 * @param[in] args - self
 * @return a list of keys
 */
RaveList_t* CompositeArguments_getRadarIndexKeys(CompositeArguments_t* args);

/**
 * Returns the index for the specified key.
 * @param[in] args - self
 * @param[in] key - the key
 * @return the radar index (1..N). Will return 1 if not found 
 */
int CompositeArguments_getRadarIndexValue(CompositeArguments_t* args, const char* key);

/**
 * Creates a radar index value for the specified key. If there already is a key the
 * currently set index will be returned.
 * @param[in] args - self
 * @param[in] key - the key
 * @return the index value created or the existing value for the key or 0 on failure.
 */
int CompositeArguments_createRadarIndex(CompositeArguments_t* args, const char* key);

#endif /* COMPOSITE_ARGUMENTS_H */
