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
 * Used for defining a legend that can be added to a parameter.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-02-10
 */
#ifndef RAVE_LEGEND_H
#define RAVE_LEGEND_H

#include "rave_object.h"
#include "rave_types.h"

/**
 * Defines a attribute tablee
 */
typedef struct _RaveLegend_t RaveLegend_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RaveLegend_TYPE;

int RaveLegend_addValue(RaveLegend_t* self, const char* key, const char* value);

int RaveLegend_size(RaveLegend_t* self);

int RaveLegend_exists(RaveLegend_t* self, const char* key);

const char* RaveLegend_getValue(RaveLegend_t* self, const char* key);

const char* RaveLegend_getValueAt(RaveLegend_t* self, int index);

const char* RaveLegend_getNameAt(RaveLegend_t* self, int index);

int RaveLegend_clear(RaveLegend_t* self);

int RaveLegend_remove(RaveLegend_t* self, const char* key);

int RaveLegend_removeAt(RaveLegend_t* self, int index);

int RaveLegend_maxKeyLength(RaveLegend_t* self);

int RaveLegend_maxValueLength(RaveLegend_t* self);

#endif /* RAVE_ATTRIBUTE_TABLE_H */
