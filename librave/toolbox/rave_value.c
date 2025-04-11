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
 * Data value abstraction
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-02-14
 */
#include "rave_value.h"
#include "rave_debug.h"
#include "rave_alloc.h"

#ifdef RAVE_JSON_SUPPORTED
#include <arraylist.h>
#include <json_object.h>
#include <json.h>
#endif

#include <string.h>
#include "rave_list.h"
#include "rave_object.h"
#include "rave_data2d.h"
#include "raveobject_hashtable.h"
#include "raveobject_list.h"
#include <errno.h>
#include <stdio.h>

/**
 * Represents one scan in a volume.
 */
struct _RaveValue_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveValue_Type type; /**< the value type */
  char* stringValue;   /**< the string value */
  long longValue;      /**< the long value */
  double doubleValue;  /**< the double value */
  int booleanValue;    /**< the boolean value */
  RaveObjectHashTable_t* hashtable; /**< the hash table */
  RaveObjectList_t* list; /**< the rave list */
};

/*@{ Private functions */
static void RaveValueInternal_freeStringArray(char*** sdata, int len)
{
  if (sdata != NULL) {
    if (*sdata != NULL) {
      int i;
      for (i = 0; i < len; i++) {
        RAVE_FREE((*sdata)[i]);
      }
      RAVE_FREE(*sdata);
      *sdata = NULL;
    }
  }
}

/**
 * Constructor.
 */
static int RaveValue_constructor(RaveCoreObject* obj)
{
  RaveValue_t* this = (RaveValue_t*)obj;
  this->type = RaveValue_Type_Undefined;
  this->stringValue = NULL;
  this->longValue = 0;
  this->doubleValue = 0.0;
  this->booleanValue = 0;
  this->hashtable = NULL;
  this->list = NULL;
  return 1;
}

static int RaveValue_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RaveValue_t* this = (RaveValue_t*)obj;
  RaveValue_t* src = (RaveValue_t*)srcobj;
  this->type = RaveValue_Type_Undefined;
  this->longValue = src->longValue;
  this->doubleValue = src->doubleValue;
  this->booleanValue = src->booleanValue;
  this->stringValue = NULL;
  this->hashtable = NULL;
  this->list = NULL;

  if (src->stringValue != NULL && !RaveValue_setString(this, (const char*)src->stringValue)) {
    RAVE_ERROR0("Failed to clone string");
    goto fail;
  }

  if (src->hashtable != NULL) {
    this->hashtable = RAVE_OBJECT_CLONE(src->hashtable);
    if (this->hashtable == NULL) {
      RAVE_ERROR0("Failed to clone hashtable");
      goto fail;
    }
  }

  if (src->list != NULL) {
    this->list = RAVE_OBJECT_CLONE(src->list);
    if (this->list == NULL) {
      RAVE_ERROR0("Failed to clone list");
      goto fail;
    }
  }

  return 1;
fail:
  RAVE_FREE(this->stringValue);
  RAVE_OBJECT_RELEASE(this->hashtable);
  RAVE_OBJECT_RELEASE(this->list);
  return 0;
}

/**
 * Destructor.
 */
static void RaveValue_destructor(RaveCoreObject* obj)
{
  RaveValue_t* this = (RaveValue_t*)obj;
  RAVE_FREE(this->stringValue);
  RAVE_OBJECT_RELEASE(this->hashtable);
  RAVE_OBJECT_RELEASE(this->list);
}

/*@} End of Private functions */

/*@{ Interface functions */
RaveValue_Type RaveValue_type(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->type;
}

void RaveValue_reset(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->type = RaveValue_Type_Undefined;
  RAVE_FREE(self->stringValue);
  self->longValue = 0;
  self->doubleValue = 0.0;
  self->booleanValue = 0;
  RAVE_OBJECT_RELEASE(self->hashtable); 
  RAVE_OBJECT_RELEASE(self->list);
}


RaveValue_t* RaveValue_createString(const char* value)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);
  if (result != NULL) {
    if (!RaveValue_setString(result, value)) {
      RAVE_OBJECT_RELEASE(result);
    }
  }
  return result;
}

int RaveValue_setString(RaveValue_t* self, const char* value)
{
  char* tdata = NULL;
  RAVE_ASSERT((self != NULL), "attr == NULL");

  if (value != NULL) {
    tdata = RAVE_STRDUP(value);
    if (tdata == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for string");
      goto fail;
    }
  }
  RaveValue_reset(self);

  if (tdata != NULL) {
    self->stringValue = tdata;
  }
  self->type = RaveValue_Type_String;
  return 1;
fail:
  return 0;
}

int RaveValue_getString(RaveValue_t* self, char** value)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  if (self->type == RaveValue_Type_String) {
    *value = self->stringValue;
    result = 1;
  }
  return result;
}

const char* RaveValue_toString(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "attr == NULL");
  return (const char*)self->stringValue;
}

RaveValue_t* RaveValue_createLong(long value)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);

  if (result != NULL) {
    RaveValue_setLong(result, value);
  }

  return result;
}

void RaveValue_setLong(RaveValue_t* self, long value)
{
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RaveValue_reset(self);
  self->longValue = value;
  self->type = RaveValue_Type_Long;
}

int RaveValue_getLong(RaveValue_t* self, long* value)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  if (self->type == RaveValue_Type_Long) {
    *value = self->longValue;
    result = 1;
  }
  return result;
}

long RaveValue_toLong(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "attr == NULL");
  return self->longValue;
}

RaveValue_t* RaveValue_createDouble(double value)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);

  if (result != NULL) {
    RaveValue_setDouble(result, value);
  }

  return result;

}

void RaveValue_setDouble(RaveValue_t* self, double value)
{
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RaveValue_reset(self);
  self->doubleValue = value;
  self->type = RaveValue_Type_Double;
}

int RaveValue_getDouble(RaveValue_t* self, double* value)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  if (self->type == RaveValue_Type_Double) {
    *value = self->doubleValue;
    result = 1;
  }
  return result;
}

double RaveValue_toDouble(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "attr == NULL");
  return self->doubleValue;
}

RaveValue_t* RaveValue_createBoolean(int value)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);

  if (result != NULL) {
    RaveValue_setBoolean(result, value);
  }

  return result;
}

void RaveValue_setBoolean(RaveValue_t* self, int value)
{
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RaveValue_reset(self);
  self->booleanValue = value;
  self->type = RaveValue_Type_Boolean;
}
 
int RaveValue_getBoolean(RaveValue_t* self, int* value)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  if (self->type == RaveValue_Type_Boolean) {
    *value = self->booleanValue;
    result = 1;
  }
  return result;
}
 
int RaveValue_toBoolean(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "attr == NULL");
  return self->booleanValue;
}

RaveValue_t* RaveValue_createNull()
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);

  if (result != NULL) {
    RaveValue_setNull(result);
  }

  return result;
}

void RaveValue_setNull(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "attr == NULL");
  RaveValue_reset(self);
  self->type = RaveValue_Type_Null;
}

int RaveValue_isNull(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "attr == NULL");
  return (self->type == RaveValue_Type_Null) ? 1 : 0;
}

int RaveValue_isStringArray(RaveValue_t* self)
{
  int nlen = 0, i = 0;
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  if (self->type == RaveValue_Type_List) {
    nlen = RaveObjectList_size(self->list);
    for (i = 0; i < nlen; i++) {
      RaveValue_t* obj = (RaveValue_t*)RaveObjectList_get(self->list, i);
      if (obj == NULL || (RaveValue_type(obj) != RaveValue_Type_String && RaveValue_type(obj) != RaveValue_Type_Null)) {
        RAVE_OBJECT_RELEASE(obj);
        goto fail;
      }
      RAVE_OBJECT_RELEASE(obj);
    }
  }
  result = 1;
fail:
  return result;
}


RaveValue_t* RaveValue_createStringArray(const char** value, int len)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);
  if (result != NULL) {
    if (!RaveValue_setStringArray(result, value, len)) {
      RAVE_OBJECT_RELEASE(result);
    }
  }
  return result;
}

int RaveValue_setStringArray(RaveValue_t* self, const char** value, int len)
{
  RaveObjectList_t* rlist = NULL;
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  rlist = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (rlist != NULL) {
    int i = 0;
    for (i = 0; i < len; i++) {
      RaveValue_t* v = NULL;
      if (value[i] != NULL) {
        v = RaveValue_createString(value[i]);
      } else {
        v = RaveValue_createNull();
      }
      if (v == NULL || !RaveObjectList_add(rlist, (RaveCoreObject*)v)) {
        RAVE_ERROR0("Failed to create rave value from string/null");
        RAVE_OBJECT_RELEASE(v);
        goto fail;
      }
      RAVE_OBJECT_RELEASE(v);
    }
  } else {
    RAVE_ERROR0("Failed to allocate object list");
    goto fail;
  }

  result = RaveValue_setList(self, rlist);
fail:
  RAVE_OBJECT_RELEASE(rlist);
  return result;
}

int RaveValue_getStringArray(RaveValue_t* self, char*** value, int* len)
{
  int result = 0;
  int nlen = 0;
  char** rvalue = NULL;

  RAVE_ASSERT((self != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  RAVE_ASSERT((len != NULL), "len == NULL");
  if (self->type == RaveValue_Type_List) {
    int i = 0;
    nlen = RaveObjectList_size(self->list);
    rvalue = RAVE_MALLOC(sizeof(char*) * nlen);
    if (rvalue == NULL) {
      RAVE_ERROR0("Failed to allocate memory");
      goto fail;
    }
    memset(rvalue, 0, sizeof(char*)*nlen);

    for (i = 0; i < nlen; i++) {
      RaveValue_t* rv = (RaveValue_t*)RaveObjectList_get(self->list, i);
      if (rv == NULL || (RaveValue_type(rv) != RaveValue_Type_String && RaveValue_type(rv) != RaveValue_Type_Null)) {
        RAVE_ERROR0("Failed to handle list item as string");
        RAVE_OBJECT_RELEASE(rv);
        goto fail;
      }
      if (RaveValue_type(rv) == RaveValue_Type_String) {
        rvalue[i] = RAVE_STRDUP(RaveValue_toString(rv));
        if (rvalue[i] == NULL) {
          RAVE_ERROR0("Failed to duplicate string");
          RAVE_OBJECT_RELEASE(rv);
          goto fail;
        }
      } else {
        rvalue[i] = NULL;
      }
      RAVE_OBJECT_RELEASE(rv);
    }
    *value = rvalue;
    rvalue = NULL;
    *len = nlen;
    result = 1;
  }

fail:
  if (rvalue != NULL) {
    RaveValueInternal_freeStringArray(&rvalue, nlen); 
  }
  return result;  
}

int RaveValue_isLongArray(RaveValue_t* self)
{
  int nlen = 0, i = 0;
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  if (self->type == RaveValue_Type_List) {
    nlen = RaveObjectList_size(self->list);
    for (i = 0; i < nlen; i++) {
      RaveValue_t* obj = (RaveValue_t*)RaveObjectList_get(self->list, i);
      if (obj == NULL || RaveValue_type(obj) != RaveValue_Type_Long) {
        RAVE_OBJECT_RELEASE(obj);
        goto fail;
      }
      RAVE_OBJECT_RELEASE(obj);
    }
  }
  result = 1;
fail:
  return result;
}

RaveValue_t* RaveValue_createLongArray(long* value, int len)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);
  if (result != NULL) {
    if (!RaveValue_setLongArray(result, value, len)) {
      RAVE_OBJECT_RELEASE(result);
    }
  }
  return result;
}

int RaveValue_setLongArray(RaveValue_t* self, long* value, int len)
{
  RaveObjectList_t* rlist = NULL;
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  rlist = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (rlist != NULL) {
    int i = 0;
    for (i = 0; i < len; i++) {
      RaveValue_t* v = RaveValue_createLong(value[i]);
      if (v == NULL || !RaveObjectList_add(rlist, (RaveCoreObject*)v)) {
        RAVE_ERROR0("Failed to create rave value from long");
        RAVE_OBJECT_RELEASE(v);
        goto fail;
      }
      RAVE_OBJECT_RELEASE(v);
    }
  } else {
    RAVE_ERROR0("Failed to allocate object list");
    goto fail;
  }

  result = RaveValue_setList(self, rlist);
fail:
  RAVE_OBJECT_RELEASE(rlist);
  return result;  
}

int RaveValue_getLongArray(RaveValue_t* self, long** value, int* len)
{
  int result = 0;
  int nlen = 0;
  long* rvalue = NULL;

  RAVE_ASSERT((self != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  RAVE_ASSERT((len != NULL), "len == NULL");
  if (self->type == RaveValue_Type_List) {
    int i = 0;
    nlen = RaveObjectList_size(self->list);
    rvalue = RAVE_MALLOC(sizeof(long) * nlen);
    if (rvalue == NULL) {
      RAVE_ERROR0("Failed to allocate memory");
      goto fail;
    }
    memset(rvalue, 0, sizeof(long)*nlen);

    for (i = 0; i < nlen; i++) {
      RaveValue_t* rv = (RaveValue_t*)RaveObjectList_get(self->list, i);
      if (rv == NULL || (RaveValue_type(rv) != RaveValue_Type_Long)) {
        RAVE_ERROR0("Failed to handle list item as long");
        RAVE_OBJECT_RELEASE(rv);
        goto fail;
      }
      RAVE_OBJECT_RELEASE(rv);
      rvalue[i] = RaveValue_toLong(rv);
    }
    *value = rvalue;
    rvalue = NULL;
    *len = nlen;
    result = 1;
  }

fail:
  if (rvalue != NULL) {
    RAVE_FREE(rvalue);
  }
  return result;  
}

int RaveValue_isDoubleArray(RaveValue_t* self)
{
  int nlen = 0, i = 0;
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  if (self->type == RaveValue_Type_List) {
    nlen = RaveObjectList_size(self->list);
    for (i = 0; i < nlen; i++) {
      RaveValue_t* obj = (RaveValue_t*)RaveObjectList_get(self->list, i);
      if (obj == NULL || (RaveValue_type(obj) != RaveValue_Type_Double && RaveValue_type(obj) != RaveValue_Type_Long)) {
        RAVE_OBJECT_RELEASE(obj);
        goto fail;
      }
      RAVE_OBJECT_RELEASE(obj);
    }
  }
  result = 1;
fail:
  return result;
}

RaveValue_t* RaveValue_createDoubleArray(double* value, int len)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);
  if (result != NULL) {
    if (!RaveValue_setDoubleArray(result, value, len)) {
      RAVE_OBJECT_RELEASE(result);
    }
  }
  return result;
}

int RaveValue_setDoubleArray(RaveValue_t* self, double* value, int len)
{
  RaveObjectList_t* rlist = NULL;
  int result = 0;
  RAVE_ASSERT((self != NULL), "attr == NULL");
  rlist = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (rlist != NULL) {
    int i = 0;
    for (i = 0; i < len; i++) {
      RaveValue_t* v = RaveValue_createDouble(value[i]);
      if (v == NULL || !RaveObjectList_add(rlist, (RaveCoreObject*)v)) {
        RAVE_ERROR0("Failed to create rave value from double");
        RAVE_OBJECT_RELEASE(v);
        goto fail;
      }
      RAVE_OBJECT_RELEASE(v);
    }
  } else {
    RAVE_ERROR0("Failed to allocate object list");
    goto fail;
  }

  result = RaveValue_setList(self, rlist);
fail:
  RAVE_OBJECT_RELEASE(rlist);
  return result;  
}

int RaveValue_getDoubleArray(RaveValue_t* self, double** value, int* len)
{
  int result = 0;
  int nlen = 0;
  double* rvalue = NULL;

  RAVE_ASSERT((self != NULL), "attr == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  RAVE_ASSERT((len != NULL), "len == NULL");
  if (self->type == RaveValue_Type_List) {
    int i = 0;
    nlen = RaveObjectList_size(self->list);
    rvalue = RAVE_MALLOC(sizeof(double) * nlen);
    if (rvalue == NULL) {
      RAVE_ERROR0("Failed to allocate memory");
      goto fail;
    }
    memset(rvalue, 0, sizeof(long)*nlen);

    for (i = 0; i < nlen; i++) {
      RaveValue_t* rv = (RaveValue_t*)RaveObjectList_get(self->list, i);
      if (rv == NULL || (RaveValue_type(rv) != RaveValue_Type_Long && RaveValue_type(rv) != RaveValue_Type_Double)) {
        RAVE_ERROR0("Failed to handle list item as double");
        RAVE_OBJECT_RELEASE(rv);
        goto fail;
      }
      if (RaveValue_type(rv) == RaveValue_Type_Long) {
        rvalue[i] = (double)RaveValue_toLong(rv);
      } else {
        rvalue[i] = RaveValue_toDouble(rv);
      }
      RAVE_OBJECT_RELEASE(rv);
    }
    *value = rvalue;
    rvalue = NULL;
    *len = nlen;
    result = 1;
  }

fail:
  if (rvalue != NULL) {
    RAVE_FREE(rvalue);
  }
  return result;  
}

RaveValue_t* RaveValue_createHashTable(RaveObjectHashTable_t* hashtable)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);
  if (result != NULL) {
    RaveObjectHashTable_t* tmp = NULL;
    if (hashtable != NULL) {
      tmp = RAVE_OBJECT_COPY(hashtable);
    } else {
      tmp = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
      if (tmp == NULL) {
        RAVE_ERROR0("Failed to create object hash table");
        RAVE_OBJECT_RELEASE(result);
        return NULL;
      }
    }
    if (!RaveValue_setHashTable(result, tmp)) {
      RAVE_OBJECT_RELEASE(result);
    }
    RAVE_OBJECT_RELEASE(tmp);
  } else {
    RAVE_ERROR0("Faild to create rave value");
  }
  return result;
}

static int RaveValueInternal_verifyHashTable(RaveObjectHashTable_t* table)
{
  int result = 0, i = 0, nlen = 0;
  RaveObjectList_t* values = NULL;
  if (table != NULL) {
    values = RaveObjectHashTable_values(table);
    if (values != NULL) {
      result = 1;
      nlen = RaveObjectList_size(values);
      for (i = 0; result && i < nlen; i++) {
        RaveCoreObject* obj = RaveObjectList_get(values, i);
        if (!RAVE_OBJECT_CHECK_TYPE(obj, &RaveValue_TYPE)) {
          RAVE_ERROR0("Not a rave value in hash");
          result = 0;
        }
        RAVE_OBJECT_RELEASE(obj);
      }
    }
    RAVE_OBJECT_RELEASE(values);
  }
  return result;
}

int RaveValue_setHashTable(RaveValue_t* self, RaveObjectHashTable_t* table)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RaveValue_reset(self);
  if (table != NULL && RaveValueInternal_verifyHashTable(table)) {
    self->hashtable = RAVE_OBJECT_COPY(table);
    self->type = RaveValue_Type_Hashtable;
  }
  return 1;
}

int RaveValue_getHashTable(RaveValue_t* self, RaveObjectHashTable_t** table)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->type == RaveValue_Type_Hashtable) {
    RAVE_OBJECT_RELEASE(*table);
    *table = RAVE_OBJECT_COPY(self->hashtable);
    return 1;
  }
  return 0;
}

RaveObjectHashTable_t* RaveValue_toHashTable(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RAVE_OBJECT_COPY(self->hashtable);
}

int RaveValueHash_put(RaveValue_t* self, const char* key, RaveValue_t* value)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");

  if (self->type == RaveValue_Type_Hashtable) {
    result = RaveObjectHashTable_put(self->hashtable, key, (RaveCoreObject*)value);
  }
  return result;
}

RaveValue_t* RaveValueHash_get(RaveValue_t* self, const char* key)
{
  RaveValue_t* obj = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");

  if (self->type == RaveValue_Type_Hashtable) {
    RaveCoreObject* t = RaveObjectHashTable_get(self->hashtable, key);
    if (RAVE_OBJECT_CHECK_TYPE(t, &RaveValue_TYPE)) {
      obj = RAVE_OBJECT_COPY(t);
    }
    RAVE_OBJECT_RELEASE(t);
  }
  return obj;
}
 
int RaveValueHash_exists(RaveValue_t* self, const char* key)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");

  if (self->type == RaveValue_Type_Hashtable) {
    result = RaveObjectHashTable_exists(self->hashtable, key);
  }
  return result;

}
 
void RaveValueHash_remove(RaveValue_t* self, const char* key)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->type == RaveValue_Type_Hashtable) {
    RaveCoreObject* obj = RaveObjectHashTable_remove(self->hashtable, key);
    RAVE_OBJECT_RELEASE(obj);
  }
}
 
void RaveValueHash_clear(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->type == RaveValue_Type_Hashtable) {
    RaveObjectHashTable_clear(self->hashtable);
  }
}
 
RaveList_t* RaveValueHash_keys(RaveValue_t* self)
{
  RaveList_t* result = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->type == RaveValue_Type_Hashtable) {
    result = RaveObjectHashTable_keys(self->hashtable);
  }
  return result;
}
 
RaveValue_t* RaveValue_createList(RaveObjectList_t* rlist)
{
  RaveValue_t* result = RAVE_OBJECT_NEW(&RaveValue_TYPE);
  if (result != NULL) {
    if (!RaveValue_setList(result, rlist)) {
      RAVE_OBJECT_RELEASE(result);
    }
  }
  return result;
}

int RaveValue_setList(RaveValue_t* self, RaveObjectList_t* rlist)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RaveValue_reset(self);
  if (rlist != NULL) {
    self->list = RAVE_OBJECT_COPY(rlist);
    self->type = RaveValue_Type_List;
  }
  return 1;

}

int RaveValue_getList(RaveValue_t* self, RaveObjectList_t** rlist)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->type == RaveValue_Type_List) {
    RAVE_OBJECT_RELEASE(*rlist);
    *rlist = RAVE_OBJECT_COPY(self->list);
    return 1;
  }
  return 0;
}

RaveObjectList_t* RaveValue_toList(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RAVE_OBJECT_COPY(self->list);
}

int RaveValueList_add(RaveValue_t* self, RaveValue_t* value)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->type == RaveValue_Type_List) {
    result = RaveObjectList_add(self->list, (RaveCoreObject*)value);
  }
  return result;
}

int RaveValueList_insert(RaveValue_t* self, int index, RaveValue_t* value)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->type == RaveValue_Type_List) {
    result = RaveObjectList_insert(self->list, index, (RaveCoreObject*)value);
  }
  return result;
}

int RaveValueList_size(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->type == RaveValue_Type_List) {
    return RaveObjectList_size(self->list);
  }
  return 0;
}

RaveValue_t* RaveValueList_get(RaveValue_t* self, int index)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->type == RaveValue_Type_List) {
    return (RaveValue_t*)RaveObjectList_get(self->list, index);
  }
  return NULL;
}

void RaveValueList_release(RaveValue_t* self, int index)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->type == RaveValue_Type_List) {
    RaveObjectList_release(self->list, index);
  }
}

void RaveValueList_clear(RaveValue_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->type == RaveValue_Type_List) {
    RaveObjectList_clear(self->list);
  }
}

#ifdef RAVE_JSON_SUPPORTED
static json_object* RaveValueInternal_toJsonObject(RaveValue_t* self)
{
  json_object* obj = NULL;
  if (RaveValue_type(self) == RaveValue_Type_Boolean) {
    obj = json_object_new_boolean(RaveValue_toBoolean(self));
  } else if (RaveValue_type(self) == RaveValue_Type_Long) {
    obj = json_object_new_int64(RaveValue_toLong(self));
  } else if (RaveValue_type(self) == RaveValue_Type_Double) {
    obj = json_object_new_double(RaveValue_toDouble(self));
  } else if (RaveValue_type(self) == RaveValue_Type_Null) {
    obj = json_object_new_null();
  } else if (RaveValue_type(self) == RaveValue_Type_String) {
    obj = json_object_new_string(RaveValue_toString(self));
  } else if (RaveValue_type(self) == RaveValue_Type_Hashtable) {
    RaveObjectHashTable_t* hashtable = NULL;
    RaveList_t* keys = NULL;
    obj = json_object_new_object();
    if (obj == NULL) {
      RAVE_ERROR0("Failed to create json object class");
      goto fail;
    }
    hashtable = RaveValue_toHashTable(self);
    keys = RaveObjectHashTable_keys(hashtable);
    if (keys != NULL) {
      int i = 0, nlen = RaveList_size(keys);
      for (i = 0; i < nlen; i++) {
        const char* key = (const char*)RaveList_get(keys, i);
        RaveCoreObject* robj = RaveObjectHashTable_get(hashtable, key);
        RaveValue_t* rvalue = (RaveValue_t*)robj;
        if (rvalue != NULL) {
          json_object* nobj = RaveValueInternal_toJsonObject(rvalue);
          if (json_object_object_add(obj, key, nobj) < 0) {
            RAVE_ERROR0("Failed to add instance to object class");
            json_object_put(nobj);
            json_object_put(obj);
            RAVE_OBJECT_RELEASE(hashtable);
            RAVE_OBJECT_RELEASE(rvalue);
            goto fail;
          }
        }
        RAVE_OBJECT_RELEASE(rvalue);
      }
      RaveList_freeAndDestroy(&keys);
    }
    RAVE_OBJECT_RELEASE(hashtable);
  } else if (RaveValue_type(self) == RaveValue_Type_List) {
    RaveObjectList_t* list = NULL;
    int i = 0, nlen = 0;
    obj = json_object_new_array();
    if (obj == NULL) {
      RAVE_ERROR0("Failed to create json array class");
      goto fail;
    }
    list = RaveValue_toList(self);
    nlen = RaveObjectList_size(list);
    for (i = 0; i < nlen; i++) {
      RaveValue_t* rvalue = (RaveValue_t*)RaveObjectList_get(list, i);
      if (rvalue != NULL) {
        json_object* nobj = RaveValueInternal_toJsonObject(rvalue);
        if (json_object_array_add(obj, nobj) < 0) {
          RAVE_ERROR0("Failed to add instance to list class");
          json_object_put(nobj);
          json_object_put(obj);
          RAVE_OBJECT_RELEASE(list);
          RAVE_OBJECT_RELEASE(rvalue);
          goto fail;
        }
      }
      RAVE_OBJECT_RELEASE(rvalue);
    }
    RAVE_OBJECT_RELEASE(list);
  }

fail:
  return obj;
}
#endif

char* RaveValue_toJSON(RaveValue_t* self)
{
  char* result = NULL;
#ifdef RAVE_JSON_SUPPORTED
  json_object* obj = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  obj = RaveValueInternal_toJsonObject(self);
  json_c_set_serialization_double_format("%.12g", JSON_C_OPTION_GLOBAL); /* Allow 12 decimal precision */
  const char* jsonstr = json_object_to_json_string(obj);
  if (jsonstr != NULL) {
    result = RAVE_STRDUP(jsonstr);
  }
  json_object_put(obj);
#endif
  return result;
}

#ifdef RAVE_JSON_SUPPORTED
static RaveValue_t* RaveValueInternal_fromJsonObject(json_object* jobject)
{
  RaveValue_t* value = NULL;
  RaveValue_t* result = NULL;

  switch (json_object_get_type(jobject)) {
    case json_type_boolean: {
      value = RaveValue_createBoolean(json_object_get_boolean(jobject));
      if (value == NULL) {
        RAVE_ERROR0("Failed to create boolean");
        goto done;
      }
      break;
    }
    case json_type_double: {
      value = RaveValue_createDouble(json_object_get_double(jobject));
      if (value == NULL) {
        RAVE_ERROR0("Failed to create boolean");
        goto done;
      }
      break;
    }
    case json_type_int: {
      value = RaveValue_createLong(json_object_get_int(jobject));
      if (value == NULL) {
        RAVE_ERROR0("Failed to create long");
        goto done;
      }
      break;
    }
    case json_type_null: {
      value = RaveValue_createNull();
      if (value == NULL) {
        RAVE_ERROR0("Failed to create null");
        goto done;
      }
      break;
    }
    case json_type_string: {
      value = RaveValue_createString(json_object_get_string(jobject));
      if (value == NULL) {
        RAVE_ERROR0("Failed to create string");
        goto done;
      }
      break;
    }
    case json_type_object: {
      RaveObjectHashTable_t* hashtable = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
      if (hashtable != NULL) {
        json_object_object_foreach(jobject, key, val) {
          RaveValue_t* newvalue = RaveValueInternal_fromJsonObject(val);
          if (newvalue == NULL || !RaveObjectHashTable_put(hashtable, key, (RaveCoreObject*)newvalue)) {
            RAVE_ERROR1("Failed to add object %s to hash", newvalue);
            RAVE_OBJECT_RELEASE(newvalue);
            RAVE_OBJECT_RELEASE(hashtable);
            goto done;
          }
          RAVE_OBJECT_RELEASE(newvalue);
        }
        value = RaveValue_createHashTable(hashtable);
        if (value == NULL) {
          RAVE_ERROR0("Failed to create hashtable");
          RAVE_OBJECT_RELEASE(hashtable);
          goto done;
        }
        RAVE_OBJECT_RELEASE(hashtable);
      } else {
        RAVE_ERROR0("Failed to create hashtable");
        goto done;
      }
      break;
    }
    case json_type_array: {
      RaveObjectList_t* rlist = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
      if (rlist != NULL) {
        int i = 0, nlen = json_object_array_length(jobject);
        for (i = 0; i < nlen; i++) {
          json_object* obj = json_object_array_get_idx(jobject, i);
          RaveValue_t* newvalue = RaveValueInternal_fromJsonObject(obj);
          if (newvalue == NULL || !RaveObjectList_add(rlist, (RaveCoreObject*)newvalue)) {
            RAVE_ERROR1("Failed to add object %s to list", newvalue);
            RAVE_OBJECT_RELEASE(newvalue);
            RAVE_OBJECT_RELEASE(rlist);
            goto done;
          }
          RAVE_OBJECT_RELEASE(newvalue);
        }
          
        value = RaveValue_createList(rlist);
        if (value == NULL) {
          RAVE_ERROR0("Failed to create list");
          RAVE_OBJECT_RELEASE(rlist);
          goto done;
        }
        RAVE_OBJECT_RELEASE(rlist);
      }
      break;
    }
    default:
      RAVE_ERROR0("Should never arrive here....");
      break;
  }

  result = RAVE_OBJECT_COPY(value);
done:
  RAVE_OBJECT_RELEASE(value);
  return result;  
}
#endif

RaveValue_t* RaveValue_fromJSON(const char* json)
{
  RaveValue_t* result = NULL;
#ifdef RAVE_JSON_SUPPORTED
  json_object *root = json_tokener_parse(json);
  if (root == NULL) {
    RAVE_ERROR1("Could not create root json from %s", json);
    return NULL;
  }

  if (json_object_get_type(root)!=json_type_object) {
    RAVE_ERROR1("JSON must be defined as an object, i.e. encapsulated by {}. Json string is '%s'", json);
    return NULL;
  }

  result = RaveValueInternal_fromJsonObject(root);

  json_object_put(root);
#endif
  return result;
}


/*@} End of Interface functions */

RaveCoreObjectType RaveValue_TYPE = {
  "RaveValue",
  sizeof(RaveValue_t),
  RaveValue_constructor,
  RaveValue_destructor,
  RaveValue_copyconstructor
};
