/* --------------------------------------------------------------------
Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Utilities when working with ODIM H5 files.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2011-09-30
 */
#ifndef ODIM_IO_UTILITIES_H
#define ODIM_IO_UTILITIES_H

#include "rave_field.h"
#include "rave_object.h"
#include "raveobject_list.h"
#include "hlhdf.h"
#include <stdarg.h>

/**
 * Struct that can be used when passing around objects and associated nodelist
 * between the writing and loading functions.
 *
 */
typedef struct OdimIoUtilityArg {
  HL_NodeList* nodelist;  /**< the nodelist */
  RaveCoreObject* object; /**< the object */
} OdimIoUtilityArg;

/**
 * Adds a rave field to a nodelist.
 *
 * @param[in] field - the field
 * @param[in] nodelist - the hlhdf node list
 * @param[in] fmt - the varargs format string
 * @param[in] ... - the varargs
 * @return 1 on success otherwise 0
 */
int OdimIoUtilities_addRaveField(RaveField_t* field, HL_NodeList* nodelist, const char* fmt, ...);

/**
 * Adds a list of quality fields (RaveField_t) to a nodelist.
 *
 * @param[in] fields - the list of fields
 * @param[in] nodelist - the hlhdf node list
 * @param[in] fmt - the varargs format string
 * @param[in] ... - the varargs
 * @return 1 on success otherwise 0
 */
int OdimIoUtilities_addQualityFields(RaveObjectList_t* fields, HL_NodeList* nodelist, const char* fmt, ...);

/**
 * Loads a rave field. A rave field can be just about anything with a mapping
 * between attributes and a dataset.
 * @param[in] nodelist - the hlhdf node list
 * @param[in] fmt - the variable argument list string format
 * @param[in] ... - the variable argument list
 * @return a rave field on success otherwise NULL
 */
RaveField_t* OdimIoUtilities_loadField(HL_NodeList* nodelist, const char* fmt, ...);

#endif /* ODIM_IO_UTILITIES_H */
