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
 * Provides support for property handling
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-02-14
 */
#include "rave_properties.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_value.h"
#include "raveobject_hashtable.h"
#include "odim_sources.h"
#include <string.h>
#include <stdio.h>

/**
 * Represents the registry
 */
struct _RaveProperties_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveValue_t* properties; /**< the property mapping as a rave value */
  OdimSources_t* sources; /**< odim sources */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int RaveProperties_constructor(RaveCoreObject* obj)
{
  RaveProperties_t* this = (RaveProperties_t*)obj;
  this->sources = NULL;
  this->properties = RaveValue_createHashTable(NULL);
  if (this->properties == NULL) {
    goto error;
  }


  return 1;
error:
  RAVE_OBJECT_RELEASE(this->properties);
  return 0;
}

/**
 * Copy constructor
 */
static int RaveProperties_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RaveProperties_t* this = (RaveProperties_t*)obj;
  RaveProperties_t* src = (RaveProperties_t*)srcobj;
  this->sources = NULL;
  this->properties = RAVE_OBJECT_CLONE(src->properties);
  if (this->properties == NULL) {
    goto error;
  }
  if (src->sources != NULL) {
    this->sources = RAVE_OBJECT_CLONE(src->sources);
    if (this->sources == NULL) {
      RAVE_ERROR0("Failed to clone sources");
      goto error;
    }
  }
  return 1;
error:
  RAVE_OBJECT_RELEASE(this->properties);
  RAVE_OBJECT_RELEASE(this->sources);
  return 0;
}

/**
 * Destructor
 */
static void RaveProperties_destructor(RaveCoreObject* obj)
{
  RaveProperties_t* this = (RaveProperties_t*)obj;
  RAVE_OBJECT_RELEASE(this->properties);
  RAVE_OBJECT_RELEASE(this->sources);
}
/*@} End of Private functions */

/*@{ Interface functions */
RaveProperties_t* RaveProperties_load(const char* filename)
{
#ifdef RAVE_JSON_SUPPORTED
  RaveProperties_t* result = NULL;
  RaveValue_t* value = RaveValue_loadJSON(filename);
  if (value != NULL && RaveValue_type(value) == RaveValue_Type_Hashtable) {
    result = RAVE_OBJECT_NEW(&RaveProperties_TYPE);
    if (result != NULL) {
      RAVE_OBJECT_RELEASE(result->properties);
      result->properties = RAVE_OBJECT_COPY(value);
    }
#ifdef RAVE_XML_SUPPORTED
    if (RaveValueHash_exists(value, "odim.sources")) {
      RaveValue_t* t = RaveValueHash_get(value, "odim.sources");
      if (RaveValue_type(t) == RaveValue_Type_String) {
        result->sources = OdimSources_load(RaveValue_toString(t));
      }
      RAVE_OBJECT_RELEASE(t);
    }
#endif    
  } else {
    RAVE_ERROR1("Could not load JSON file %s", filename);
  }
  RAVE_OBJECT_RELEASE(value);
  return result;
#else
  RAVE_WARNING0("rave is not built with json support");
  return NULL;
#endif
}


int RaveProperties_set(RaveProperties_t* self, const char* name, RaveValue_t* value)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (name == NULL || value == NULL) {
    RAVE_ERROR0("Invalid arguments");
    return 0;
  }
  return RaveValueHash_put(self->properties, name, value);
}

RaveValue_t* RaveProperties_get(RaveProperties_t* self, const char* name)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveValueHash_get(self->properties, name);
}

int RaveProperties_hasProperty(RaveProperties_t* self, const char* name)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveValueHash_exists(self->properties, name);
}

void RaveProperties_remove(RaveProperties_t* self, const char* name)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RaveValueHash_remove(self->properties, name);
}

int RaveProperties_size(RaveProperties_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveValueHash_size(self->properties);
}

void RaveProperties_setOdimSources(RaveProperties_t* self, OdimSources_t* sources)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_OBJECT_RELEASE(self->sources);
  if (sources != NULL) {
    self->sources = RAVE_OBJECT_COPY(sources);
  }
}

OdimSources_t* RaveProperties_getOdimSources(RaveProperties_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->sources != NULL) {
    return RAVE_OBJECT_COPY(self->sources);
  }
  return NULL;
}

/*@} End of Interface functions */

RaveCoreObjectType RaveProperties_TYPE = {
    "RaveProperties",
    sizeof(RaveProperties_t),
    RaveProperties_constructor,
    RaveProperties_destructor,
    RaveProperties_copyconstructor
};
