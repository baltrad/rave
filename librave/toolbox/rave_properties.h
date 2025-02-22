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
 * Property handling
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-02-14
 */
#ifndef RAVE_PROPERTIES_H
#define RAVE_PROPERTIES_H
#include "rave_value.h"
#include "rave_object.h"
#include "odim_sources.h"

/**
 * Defines the area registry
 */
typedef struct _RaveProperties_t RaveProperties_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RaveProperties_TYPE;

/**
 * Simplified loading function, takes filename and a projection registry
 * @param[in] filename - the area file name
 * @param[in] pRegistry - the projection registry
 * @returns an area registry
 */
//RaveProperties_t* RaveProperties_load(const char* filename);

/**
 * Sets the property in the table
 * @param[in] self - self
 * @param[in] name - name of property
 * @param[in] value - the value
 */
int RaveProperties_set(RaveProperties_t* self, const char* name, RaveValue_t* value);

/**
 * Returns the value with specified name.
 * @param[in] self - self
 * @param[in] name - name of property
 * @return the value or NULL if not found
 */
RaveValue_t* RaveProperties_get(RaveProperties_t* self, const char* name);

/**
 * Returns if the set contains the specified property
 * @param[in] self - self
 * @param[in] name - name of property
 * @return 1 if found, otherwise 0
 */
int RaveProperties_hasProperty(RaveProperties_t* self, const char* name);

/**
 * Removes the property from the set
 * @param[in] self - self
 * @param[in] name - name of property
 */
void RaveProperties_remove(RaveProperties_t* self, const char* name);

/**
 * Returns the number of properties
 * @param[in] self - self
 * @return the number of properties
 */
 int RaveProperties_size(RaveProperties_t* self);

 /**
  * Sets the odim sources to the rave properties
  * @param[in] self - self
  * @param[in] sources - the odim sources
  */
 void RaveProperties_setOdimSources(RaveProperties_t* self, OdimSources_t* sources);

 /**
  * @param[in] self - self
  * @return the odim sources
  */
 OdimSources_t* RaveProperties_getOdimSources(RaveProperties_t* self);

#endif /* RAVE_PROPERTIES */
