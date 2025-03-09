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
#include "raveobject_hashtable.h"
#include "odim_sources.h"
//#include "rave_simplexml.h"
//#include "rave_utilities.h"
//#include "expat.h"
#include <string.h>

/**
 * Represents the registry
 */
struct _RaveProperties_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveObjectHashTable_t* properties; /**< the property mapping */
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
  this->properties = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
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
int RaveProperties_set(RaveProperties_t* self, const char* name, RaveValue_t* value)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (name == NULL || value == NULL) {
    RAVE_ERROR0("Invalid arguments");
    return 0;
  }
  return RaveObjectHashTable_put(self->properties, name, (RaveCoreObject*)value);
}

RaveValue_t* RaveProperties_get(RaveProperties_t* self, const char* name)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (RaveValue_t*)RaveObjectHashTable_get(self->properties, name);
}

int RaveProperties_hasProperty(RaveProperties_t* self, const char* name)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveObjectHashTable_exists(self->properties, name);
}

void RaveProperties_remove(RaveProperties_t* self, const char* name)
{
  RaveCoreObject* obj = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  obj = RaveObjectHashTable_remove(self->properties, name);
  RAVE_OBJECT_RELEASE(obj);
}

int RaveProperties_size(RaveProperties_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveObjectHashTable_size(self->properties);
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
