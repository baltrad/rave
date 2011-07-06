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
#include "raveobject_hashtable.h"

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

/**
 * Replaces the content of a attribute in the object list. If the attribute does not exist a new
 * one will be created.
 * @param[in] l - the list
 * @param[in] name - the name of the attribute
 * @param[in] value - the long
 * @returns 1 on success otherwise 0
 */
int RaveUtilities_replaceLongAttributeInList(RaveObjectList_t* l, const char* name, long value);

/**
 * Replaces the content of a attribute in the object list. If the attribute does not exist a new
 * one will be created.
 * @param[in] l - the list
 * @param[in] name - the name of the attribute
 * @param[in] value - the double
 * @returns 1 on success otherwise 0
 */
int RaveUtilities_replaceDoubleAttributeInList(RaveObjectList_t* l, const char* name, double value);

/**
 * Replaces the content of a attribute in the object list. If the attribute does not exist a new
 * one will be created.
 * @param[in] l - the list
 * @param[in] name - the name of the attribute
 * @param[in] value - the string
 * @returns 1 on success otherwise 0
 */
int RaveUtilities_replaceStringAttributeInList(RaveObjectList_t* l, const char* name, const char* value);

/**
 * Removes the rave attribute with specified name from the list
 * @param[in] l - the list
 * @param[in] name - the name
 */
void RaveUtilities_removeAttributeFromList(RaveObjectList_t* l, const char* name);

/**
 * Gets the double value from a rave attribute that resides in a hash table
 * @param[in] h - the hash table
 * @param[in] name - the name
 * @param[in,out] v - the value
 * @returns 1 on success otherwise 0
 */
int RaveUtilities_getRaveAttributeDoubleFromHash(RaveObjectHashTable_t* h, const char* name, double* v);

/**
 * Returns if the character is a whitespace character or not,
 * i.e. ' ', '\t', '\r' or '\n'
 * @param[in] c - the character to check
 * @returns true if character is a whitespace otherwise 0
 */
int RaveUtilities_iswhitespace(char c);

/**
 * Trims the text from all leading and trailing whitespaces.
 * @param[in] str - the string to trim
 * @param[in] len - the length of the string to trim
 * @returns a new trimmed string, release with RAVE_FREE
 */
char* RaveUtilities_trimText(const char* str, int len);

/**
 * Returns a list of tokens delimited by 'c'. The tokens will
 * be trimmed from any leading and trailing whitespaces.
 * @param[in] str - the string to tokenize
 * @param[in] c - the delimiter
 * @returns a list of tokens, use @ref RaveList_freeAndDestroy to delete
 */
RaveList_t* RaveUtilities_getTrimmedTokens(const char* str, int c);

/**
 * Returns if xml support is activated or not since expat support
 * is optional and ought to be tested.
 * @returns 0 if xml isn't supported in the build, otherwise 1
 */
int RaveUtilities_isXmlSupported(void);

#endif /* RAVE_UTILITIES_H */
