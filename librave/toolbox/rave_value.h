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
 * A value object that can represent standard types like int, double, .. It can also contain arrays, lists and hashtables
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-02-14
 */
#ifndef RAVE_VALUE_H
#define RAVE_VALUE_H
#include "rave_object.h"
#include "raveobject_hashtable.h"
#include <stdarg.h>

typedef enum RaveValue_Type {
  RaveValue_Type_Undefined = -1,  /**< Undefined */
  RaveValue_Type_String = 0,      /**< String */
  RaveValue_Type_Long = 1,        /**< Long */
  RaveValue_Type_Double = 2,      /**< Double */
  RaveValue_Type_Boolean = 3,     /**< Boolean */
  RaveValue_Type_Null = 4,        /**< Null */
  RaveValue_Type_Data2D = 5,      /**< 2D array */
  RaveValue_Type_Hashtable = 6,   /**< Hash table */
  RaveValue_Type_List = 7
} RaveValue_Type;

/**
 * Defines a rave value
 */
typedef struct _RaveValue_t RaveValue_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RaveValue_TYPE;

/**
 * Returns the format for this attribute
 * @param[in] self - self
 * @returns the format
 */
RaveValue_Type RaveValue_type(RaveValue_t* self);

/**
 * Resets the value object
 * @param[in] self - self
 */
void RaveValue_reset(RaveValue_t* self);

/**
 * Creates a string rave value.
 * @param[in] value - the string
 * @returns the value on success otherwise NULL
 */
RaveValue_t* RaveValue_createString(const char* value);

/**
 * Creates a string rave value from a varargs list
 * @param[in] value - the string
 * @returns the value on success otherwise NULL
 */
RaveValue_t* RaveValue_createStringFmt(const char* fmt, ...);

/**
 * Sets a string value in self
 * @param[in] self - self
 * @param[in] value - the value
 * @return 1 on success otherwise 0
 */
int RaveValue_setString(RaveValue_t* self, const char* value);

/**
 * Returns the value as a string.
 * @param[in] self - self
 * @param[out] value - the internal 0-terminated string, DO NOT RELEASE memory
 * @returns 1 on success or 0 if format of the data not is a string
 */
int RaveValue_getString(RaveValue_t* self, char** value);

/**
 * Returns the string value. NOTE! It up to user to ensure that value actually is a string otherwise behavior will be undefined. 
 * @param[in] self - self
 * @return the string value or NULL
 */
const char* RaveValue_toString(RaveValue_t* self);

RaveValue_t* RaveValueString_trim(RaveValue_t* self);

/**
 * Creates a substring from self
 * @param[in] self - self
 * @param[in] start - the start position to create string from
 * @param[in] len - the number of bytes to copy (if start + len is >= length of string, then only string from start -> end is copied)
 * @return a substring
 */
RaveValue_t* RaveValueString_substring(RaveValue_t* self, int start, int len);

/**
 * Tokenizes a string into a string array of tokens.
 * @param[in] self - self
 * @param[in] delim - the delimiter
 * @return a string array (list of strings)
 */
RaveValue_t* RaveValueString_tokenize(RaveValue_t* self, int delim);

/**
 * Creates a long rave value.
 * @param[in] value - the long
 * @returns the value on success otherwise NULL
 */
RaveValue_t* RaveValue_createLong(long value);

/**
 * Sets the value as a long.
 * @param[in] self - self
 * @param[in] value - the value
 */
void RaveValue_setLong(RaveValue_t* self, long value);

/**
 * Returns the value as a long.
 * @param[in] self - self
 * @param[out] value - the long value
 * @returns 1 on success or 0 if format of the data not is a long
 */
int RaveValue_getLong(RaveValue_t* self, long* value);

/**
 * Returns the long value. NOTE! It up to user to ensure that value actually is a long otherwise behavior will be undefined. 
 * @param[in] self - self
 * @return the long value or 0
 */
long RaveValue_toLong(RaveValue_t* self);


/**
 * Creates a double rave value.
 * @param[in] value - the double
 * @returns the value on success otherwise NULL
 */
RaveValue_t* RaveValue_createDouble(double value);

/**
 * Sets the value as a double.
 * @param[in] self - self
 * @param[in] value - the value
 */
void RaveValue_setDouble(RaveValue_t* self, double value);

/**
 * Returns the value as a double.
 * @param[in] self - self
 * @param[out] value - the double value
 * @returns 1 on success or 0 if format of the data not is a double
 */
int RaveValue_getDouble(RaveValue_t* self, double* value);

/**
 * Returns the double value. NOTE! It up to user to ensure that value actually is a double otherwise behavior will be undefined. 
 * @param[in] self - self
 * @return the double value or -
 */
double RaveValue_toDouble(RaveValue_t* self);

/**
 * Creates a boolean rave value.
 * @param[in] value - the boolean. 0 means false, other values means true
 * @returns the value on success otherwise NULL
 */
 RaveValue_t* RaveValue_createBoolean(int value);

 /**
  * Sets the value as a boolean.
  * @param[in] self - self
  * @param[in] value - the value
  */
 void RaveValue_setBoolean(RaveValue_t* self, int value);
 
 /**
  * Returns the value as a boolean.
  * @param[in] self - self
  * @param[out] value - the boolean value
  * @returns 1 on success or 0 if format of the data not is a boolean
  */
 int RaveValue_getBoolean(RaveValue_t* self, int* value);
 
 /**
  * Returns the boolean value. NOTE! It up to user to ensure that value actually is a boolean otherwise behavior will be undefined. 
  * @param[in] self - self
  * @return the boolean or -
  */
 int RaveValue_toBoolean(RaveValue_t* self);

 /**
 * Creates a null rave value.
 * @param[in] value - the null value
 * @returns the value on success otherwise NULL
 */
RaveValue_t* RaveValue_createNull();

/**
 * Sets the value as a null value.
 * @param[in] self - self
 */
void RaveValue_setNull(RaveValue_t* self);

/**
 * Returns if the value is null or not
 * @param[in] self - self
 * @return 1 if null, otherwise 0
 */
int RaveValue_isNull(RaveValue_t* self);

/**
 * Returns if the rave value can be represented as a string array (list of strings only)
 * @param[in] self - self
 * @return 1 if possible otherwise 0
 */
int RaveValue_isStringArray(RaveValue_t* self);

/**
 * Creates a string array rave value.
 * @param[in] value - the string array
 * @param[in] len - the length of the array
 * @returns the value on success otherwise NULL
 */
RaveValue_t* RaveValue_createStringArray(const char** value, int len);

/**
 * Sets the value as a simple 1-dimensional string array.
 * @param[in] self - self
 * @param[in] value - the value
 * @param[in] len - the number of doubles in the array
 * @returns 1 on success otherwise 0
 */
int RaveValue_setStringArray(RaveValue_t* self, const char** value, int len);

/**
 * Returns the value as a string array.
 * @param[in] self - self
 * @param[out] value - the a string array. Release memory.
 * @param[out] len - the number of values in the array
 * @returns 1 on success or 0 if format of the data not is a double array
 */
int RaveValue_getStringArray(RaveValue_t* self, char*** value, int* len);

/**
 * Returns if the rave value can be represented as a long array (list of long only)
 * @param[in] self - self
 * @return 1 if possible otherwise 0
 */
 int RaveValue_isLongArray(RaveValue_t* self);

/**
 * Creates a long array rave value.
 * @param[in] value - the long array
 * @param[in] len - the length of the array
 * @returns the value on success otherwise NULL
 */
RaveValue_t* RaveValue_createLongArray(long* value, int len);

/**
 * Sets the value as a simple 1-dimensional long array.
 * @param[in] self - self
 * @param[in] value - the value
 * @param[in] len - the number of longs in the array
 * @returns 1 on success otherwise 0
 */
int RaveValue_setLongArray(RaveValue_t* self, long* value, int len);

/**
 * Returns the value as a long array.
 * @param[in] self - self
 * @param[out] value - the long array, Release memory
 * @param[out] len - the number of values in the array
 * @returns 1 on success or 0 if format of the data not is a long array
 */
int RaveValue_getLongArray(RaveValue_t* self, long** value, int* len);

/**
 * Returns if the rave value can be represented as a double array (list of double only)
 * @param[in] self - self
 * @return 1 if possible otherwise 0
 */
 int RaveValue_isDoubleArray(RaveValue_t* self);

/**
 * Creates a double array rave value.
 * @param[in] value - the double array
 * @param[in] len - the length of the array
 * @returns the value on success otherwise NULL
 */
RaveValue_t* RaveValue_createDoubleArray(double* value, int len);

/**
 * Sets the value as a simple 1-dimensional double array.
 * @param[in] self - self
 * @param[in] value - the value
 * @param[in] len - the number of doubles in the array
 * @returns 1 on success otherwise 0
 */
int RaveValue_setDoubleArray(RaveValue_t* self, double* value, int len);

/**
 * Returns the value as a double array.
 * @param[in] self - self
 * @param[out] value - the double array, Release memory
 * @param[out] len - the number of values in the array
 * @returns 1 on success or 0 if format of the data not is a double array
 */
int RaveValue_getDoubleArray(RaveValue_t* self, double** value, int* len);

/**
 * Creates a rave value hash table.
 * @param[in] hashtable - the hashtable
 * @return the rave value on success otherwise NULL
 */
RaveValue_t* RaveValue_createHashTable(RaveObjectHashTable_t* hashtable);

/**
 * Sets the value as a hash table.
 * @param[in] self - self
 * @param[in] table - the object hash table.
 * @return 1 on success or 0 if not settable
 */
int RaveValue_setHashTable(RaveValue_t* self, RaveObjectHashTable_t* table);

/**
 * Returns the hash table if possible.
 * @param[in] self
 * @param[out] table - a reference to the hash table
 * @return 1 if value is a hash table otherwise 0
 */
int RaveValue_getHashTable(RaveValue_t* self, RaveObjectHashTable_t** table);

/**
 * NOTE! It up to user to ensure that value actually is a hash table otherwise behavior will be undefined. 
 * @param[in] self - self
 * @return the hash table or NULL
 */
RaveObjectHashTable_t* RaveValue_toHashTable(RaveValue_t* self);

/**
 * If rave value is hash, then this provides same functionality as \ref RaveObjectHashTable_put.
 * @param[in] self - self
 * @param[in] key - the key
 * @param[in] value - the rave value
 * @return 1 if ok, otherwise 0
 */
int RaveValueHash_put(RaveValue_t* self, const char* key, RaveValue_t* value);

/**
 * If rave value is hash, then this provides same functionality as \ref RaveObjectHashTable_get.
 * @param[in] self - self
 * @param[in] key - the key
 * @return the rave value or NULL if not possible
 */
 RaveValue_t* RaveValueHash_get(RaveValue_t* self, const char* key);


/**
 * If rave value is hash, then this provides same functionality as \ref RaveObjectHashTable_size.
 * @param[in] self - self
 * @returns the number of items in this self.
 */
int RaveValueHash_size(RaveValue_t* self);

/**
 * If rave value is hash, then this provides same functionality as \ref RaveObjectHashTable_exists.
 * @param[in] self - self
 * @param[in] key - the key
 * @return 1 if exists, otherwise 0
 */
int RaveValueHash_exists(RaveValue_t* self, const char* key);

/**
 * If rave value is hash, then this provides same functionality as \ref RaveObjectHashTable_remove except that the value is destrouyed.
 * @param[in] self - self
 * @param[in] key - the key
 */
void RaveValueHash_remove(RaveValue_t* self, const char* key);

/**
 * If rave value is hash, then this provides same functionality as \ref RaveObjectHashTable_clear.
 * @param[in] self - self
 */
void RaveValueHash_clear(RaveValue_t* self);

/**
 * If rave value is hash, then this provides same functionality as \ref RaveObjectHashTable_keys.
 * @param[in] self - self
 * @return the keys or NULL
 */
RaveList_t* RaveValueHash_keys(RaveValue_t* self);

/**
 * Creates a rave value list.
 * @param[in] rlist - the rave list
 * @return the rave value on success otherwise NULL
 */
RaveValue_t* RaveValue_createList(RaveObjectList_t* rlist);

 /**
  * Sets the value as a rave list.
  * @param[in] self - self
  * @param[in] rlist - the object list.
  * @return 1 on success or 0 if not settable
  */
int RaveValue_setList(RaveValue_t* self, RaveObjectList_t* rlist);
 
 /**
  * Returns the rave list if possible
  * @param[in] self
  * @param[out] rlist - a reference to the rave list
  * @return 1 if value is a rave list otherwise 0
  */
int RaveValue_getList(RaveValue_t* self, RaveObjectList_t** rlist);
 
 /**
  * NOTE! It up to user to ensure that value actually is a rave list otherwise behavior will be undefined. 
  * @param[in] self - self
  * @return the rave list or NULL
  */
RaveObjectList_t* RaveValue_toList(RaveValue_t* self);

/**
 * If rave value is list, then this provides same functionality as \ref RaveObjectList_add
 * @param[in] self - self
 * @param[in] value - the value
 * @return 1 on success otherwise 0
 */
int RaveValueList_add(RaveValue_t* self, RaveValue_t* value);

/**
 * If rave value is list, then this provides same functionality as \ref RaveObjectList_insert
 * @param[in] self - self
 * @param[in] index - the index where to insert the value
 * @param[in] value - the value
 * @return 1 on success otherwise 0
 */
int RaveValueList_insert(RaveValue_t* self, int index, RaveValue_t* value);

/**
 * Creates a rave value string from the list of strings and adds the separator
 * between each entry.
 * @param[in] self - the list
 * @param[in] sep - the separator
 * @return a rave value string
 */
RaveValue_t* RaveValueList_join(RaveValue_t* self, const char* sep);

/**
 * If rave value is list, then this provides same functionality as \ref RaveObjectList_size
 * @param[in] self - self
 * @return the size of the list
 */
int RaveValueList_size(RaveValue_t* self);

/**
 * If rave value is list, then this provides same functionality as \ref RaveObjectList_get
 * @param[in] self - self
 * @param[in] index - the index
 * @return the value if possible otherwise NULL
 */
RaveValue_t* RaveValueList_get(RaveValue_t* self, int index);

/**
 * If rave value is list, then this provides same functionality as \ref RaveObjectList_release
 * @param[in] self - self
 * @param[in] index - the index
 */
void RaveValueList_release(RaveValue_t* self, int index);

/**
 * If rave value is list, then this provides same functionality as \ref RaveObjectList_clear
 * @param[in] self - self
 */
 void RaveValueList_clear(RaveValue_t* self);

/**
 * Translates self to a json representation.
 * @param[in] self - self
 * @return the json representation of self.
 */
char* RaveValue_toJSON(RaveValue_t* self);

/**
 * Support function for translating a json structure to a rave value.
 * @param[in] json - the json string
 * @return the rave value
 */
RaveValue_t* RaveValue_fromJSON(const char* json);

/**
 * Loads a JSON object from a file. 
 * @param[in] filename - the filename
 * @return the read JSON object
 */
RaveValue_t* RaveValue_loadJSON(const char* filename);

#endif /* RAVE_VALUE_H */

