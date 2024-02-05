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
 * Provides functionality for creating composites according to the acqva method
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-01-17
 */
#ifndef ACQVA_H
#define ACQVA_H

#include "rave_object.h"
#include "rave_types.h"
#include "cartesian.h"
#include "area.h"
#include "raveobject_hashtable.h"
#include "limits.h"

#define ACQVA_QUALITY_FIELDS_GAIN   (1.0/UCHAR_MAX)
#define ACQVA_QUALITY_FIELDS_OFFSET 0.0

/**
 * Defines a Acqva composite generator
 */
typedef struct _Acqva_t Acqva_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType Acqva_TYPE;

/**
 * Adds one RaveCoreObject. The only supported types are volumes (& possibly scans) due to the nature of the algorithm.
 * @param[in] self - self
 * @param[in] object - the item to be added to the composite
 * @returns 1 on success, otherwise 0
 */
int Acqva_add(Acqva_t* self, RaveCoreObject* object);

/**
 * Returns the number of objects this composite will process
 * @param[in] self - self
 * @return the number of objects
 */
int Acqva_getNumberOfObjects(Acqva_t* self);

/**
 * Return the object at position index.
 * @param[in] self - self
 * @param[in] index - the index, should be >= 0 and < getNumberOfObjects
 * @return the object or NULL if outside range
 */
RaveCoreObject* Acqva_get(Acqva_t* self, int index);

/**
 * Return the radar index value that has been assigned to the object as position index.
 * @param[in] self - self
 * @param[in] index - the index, should be >= 0 and < getNumberOfObjects
 * @return the radar index or 0 if no index has been assigned yet.
 */
int Acqva_getRadarIndexValue(Acqva_t* self, int index);

/**
 * Adds a parameter to be processed.
 * @param[in] self - self
 * @param[in] quantity - the parameter quantity
 * @param[in] gain - the gain to be used for the parameter
 * @param[in] offset - the offset to be used for the parameter
 * @return 1 on success
 */
int Acqva_addParameter(Acqva_t* self, const char* quantity, double gain, double offset);

/**
 * Returns if this composite generator is going to process specified parameter
 * @param[in] self - self
 * @param[in] quantity - the parameter quantity
 * @return 1 if yes otherwise 0
 */
int Acqva_hasParameter(Acqva_t* self, const char* quantity);

/**
 * Returns the number of parameters to be processed
 * @param[in] self - self
 * @return the number of parameters
 */
int Acqva_getParameterCount(Acqva_t* self);

/**
 * Returns the parameter at specified index
 * @param[in] self - self
 * @param[in] index - the index
 * @param[out] gain - the gain to be used for the parameter (MAY BE NULL)
 * @param[out] offset - the offset to be used for the parameter (MAY BE NULL)
 * @return the parameter name
 */
const char* Acqva_getParameter(Acqva_t* self, int index, double* gain, double* offset);

/**
 * Sets the nominal time.
 * @param[in] self - self
 * @param[in] value - the time in the format HHmmss
 * @returns 1 on success, otherwise 0
 */
int Acqva_setTime(Acqva_t* self, const char* value);

/**
 * Returns the nominal time.
 * @param[in] self - self
 * @returns the nominal time (or NULL if there is none)
 */
const char* Acqva_getTime(Acqva_t* self);

/**
 * Sets the nominal date.
 * @param[in] self - self
 * @param[in] value - the date in the format YYYYMMDD
 * @returns 1 on success, otherwise 0
 */
int Acqva_setDate(Acqva_t* self, const char* value);

/**
 * Returns the nominal date.
 * @param[in] self - self
 * @returns the nominal time (or NULL if there is none)
 */
const char* Acqva_getDate(Acqva_t* self);

/**
 * If you want the objects included in the composite to have a specific index value when generating the
 * quality field se.smhi.composite.index.radar, then you can provide a hash table that maps source with
 * a RaveAttribute_t containing a long value. The source should be the full source as defined in the
 * added objects. The indexes must be unique values, preferrably starting from 1. If there is a mapping
 * missing, the default behaviour is to take first available integer closest to 1.
 *
 * Note, that in order to the mapping to take, this call must be performed after all the objects has
 * been added to the generator and before calling \ref Acqva_generate.
 *
 * @param[in] self - self
 * @param[in] mapping - the source - index mapping
 * @return 1 on success, otherwise 0.
 */
int Acqva_applyRadarIndexMapping(Acqva_t* self, RaveObjectHashTable_t* mapping);

/**
 * Generates a composite according to the configured parameters in the composite structure.
 * @param[in] self - self
 * @param[in] area - the area that should be used for defining the composite.
 * @param[in] qualityflags - A list of char pointers identifying how/task values in the quality fields of the polar data.
 *            Each entry in this list will result in the atempt to generate a corresponding quality field
 *            in the resulting cartesian product. (MAY BE NULL)
 * @returns the generated composite.
 */
Cartesian_t* Acqva_generate(Acqva_t* self, Area_t* area, RaveList_t* qualityflags);

#endif /* ACQVA_H */
