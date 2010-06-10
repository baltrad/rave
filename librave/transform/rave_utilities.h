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
 * Contains various utility functions that makes life easier when working
 * with the rave framework.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-06-10
 */
#ifndef RAVE_UTILITIES_H
#define RAVE_UTILITIES_H
#include "rave_attribute.h"
#include "raveobject_list.h"

/**
 * Adds a long attribute to an object list.
 * @param[in] l - the list
 * @param[in] name - the name of the attribute
 * @param[in] value - the long
 * @returns 1 on success otherwise 0
 */
int RaveUtilities_addLongAttributeToList(RaveObjectList_t* l, const char* name, long value);

/**
 * Adds a double attribute to an object list.
 * @param[in] l - the list
 * @param[in] name - the name of the attribute
 * @param[in] value - the double
 * @returns 1 on success otherwise 0
 */
int RaveUtilities_addDoubleAttributeToList(RaveObjectList_t* l, const char* name, double value);

/**
 * Adds a string attribute to an object list.
 * @param[in] l - the list
 * @param[in] name - the name of the attribute
 * @param[in] value - the string
 * @returns 1 on success otherwise 0
 */
int RaveUtilities_addStringAttributeToList(RaveObjectList_t* l, const char* name, const char* value);

#endif /* RAVE_UTILITIES_H */
