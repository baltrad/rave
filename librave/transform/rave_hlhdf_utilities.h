/* --------------------------------------------------------------------
Copyright (C) 2009 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Utilities for working with H5 files
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-09-10
 */
#ifndef RAVE_HLHDF_UTILITIES_H
#define RAVE_HLHDF_UTILITIES_H
#include "hlhdf.h"
#include "rave_utilities.h"
#include "rave_types.h"
#include <stdarg.h>

/**
 * ODIM Version string
 */
#define RAVE_ODIM_VERSION_2_0_STR "ODIM_H5/V2_0"

/**
 * ODIM H5 rad version string
 */
#define RAVE_ODIM_H5RAD_VERSION_2_0_STR "H5rad 2.0"

/**
 * Verifies if the file contains a node with the name as specified by the variable
 * argument list.
 * @param[in] nodelist - the hlhdf nodelist
 * @param[in] fmt    - the variable argument format specifier
 * @param[in] ...    - the variable argument list
 * @returns 1 if the node could be found, otherwise 0
 */
int RaveHL_hasNodeByName(HL_NodeList* nodelist, const char* fmt, ...);

/**
 * Creates a group node in the node list.
 * @param[in] nodelist - the node list
 * @param[in] fmt - the variable argument format
 * @param[in] ... - the arguments.
 */
int RaveHL_createGroup(HL_NodeList* nodelist, const char* fmt, ...);

/**
 * Adds a string value to a nodelist
 * @param[in] nodelist - the hlhdf node list
 * @param[in] value - the string value
 * @param[in] fmt - the varargs format string
 * @returns 1 on success otherwise 0
 */
int RaveHL_createStringValue(HL_NodeList* nodelist, const char* value, const char* fmt, ...);

/**
 * Puts an attribute in the nodelist as a hlhdf node.
 * @param[in] nodelist - the node list
 * @param[in] attribute - the attribute, the name of the attribute will be used as attr-member
 * @param[in] fmt - the root name, specified as a varargs
 * @param[in] ... - the varargs list
 * @returns 1 on success otherwise 0
 */
int RaveHL_addAttribute(HL_NodeList* nodelist, RaveAttribute_t* attribute, const char* fmt, ...);

/**
 * Stores the attributes from the object into the nodelist
 * name/how/..., name/where/... and name/what/... If the groups
 * name/how, name/where, name/what does not exist they will be created
 * for you. All other groups you will have to add your self.
 *
 * @param[in] nodelist - the hlhdf list
 * @param[in] name - the name of the object
 * @param[in] object - the object to fill
 */
int RaveHL_addAttributes(HL_NodeList* nodelist, RaveObjectList_t* attributes, const char* name);

/**
 * Creates a dataset with the provided 2-dimensional array.
 * @param[in] nodelist - the node list
 * @param[in] data - the data
 * @param[in] xsize - the xsize
 * @param[in] ysize - the ysize
 * @param[in] dataType - the type of data
 * @param[in] fmt - the variable argument format
 * @param[in] ... - the arguments.
 * @returns 1 on success, otherwise 0
 */
int RaveHL_createDataset(HL_NodeList* nodelist, void* data, long xsize, long ysize, RaveDataType dataType, const char* fmt, ...);

/**
 * Adds a data field to the node list according to ODIM H5. If data type is
 * UCHAR, the nessecary attributes for viewing in HdfView will also be added.
 * The name will always be <root>/data since that is according to ODIM H5
 * as well.
 * @param[in] nodelist - the node list that should get nodes added
 * @param[in] data - the array data
 * @param[in] xsize - the xsize
 * @param[in] ysize - the ysize
 * @param[in] dataType - type of data
 * @param[in] fmt - the varargs format
 * @param[in] ... - the vararg list
 * @returns 1 on success otherwise 0
 */
int RaveHL_addData(
  HL_NodeList* nodelist,
  void* data,
  long xsize,
  long ysize,
  RaveDataType dataType,
  const char* fmt, ...);


/**
 * Translates a rave data type into a hlhdf format specifier
 * @param[in] format - the rave data type
 * @returns the hlhdf format specifier
 */
HL_FormatSpecifier RaveHL_raveToHlhdfType(RaveDataType format);

/**
 * Translates a hlhdf format specified into a rave data type.
 * @param[in] format - the hlhdf format specified
 * @returns the RaveDataType
 */
RaveDataType RaveHL_hlhdfToRaveType(HL_FormatSpecifier format);

#endif /* RAVE_HLHDF_UTILITIES_H */
