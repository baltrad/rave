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
 * Defines an odim source
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-15
 */
#include "odim_source.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>
#include <stdio.h>

/**
 * Represents the area
 */
struct _OdimSource_t {
  RAVE_OBJECT_HEAD /** Always on top */
  char* nod;     /**< the NOD */
  char* wmo;     /**< the WMO */
  char* wigos;   /**< the WIGOS */
  char* plc;     /**< the PLC */
  char* rad;     /**< the RAD */
  char* cccc;    /**< the CCCC */
  char* org;     /**< the ORG */
  char* source;  /**< the source string */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int OdimSource_constructor(RaveCoreObject* obj)
{
  OdimSource_t* this = (OdimSource_t*)obj;
  this->nod = NULL;
  this->wmo = NULL;
  this->wigos = NULL;
  this->plc = NULL;
  this->rad = NULL;
  this->cccc = NULL;
  this->org = NULL;
  this->source = NULL;
  return 1;
}

/**
 * Copy constructor
 */
static int OdimSource_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  OdimSource_t* this = (OdimSource_t*)obj;
  OdimSource_t* src = (OdimSource_t*)srcobj;

  OdimSource_constructor(obj); // First just initialize everything like the constructor

  if (!OdimSource_setNod(this, src->nod)) {
    goto error;
  }
  if (!OdimSource_setWmo(this, src->wmo)) {
    goto error;
  }
  if (!OdimSource_setWigos(this, src->wigos)) {
    goto error;
  }
  if (!OdimSource_setPlc(this, src->plc)) {
    goto error;
  }
  if (!OdimSource_setRad(this, src->rad)) {
    goto error;
  }
  if (!OdimSource_setCccc(this, src->cccc)) {
    goto error;
  }
  if (!OdimSource_setOrg(this, src->org)) {
    goto error;
  }
  if (src->source != NULL) {
    this->source = RAVE_STRDUP(src->source);
    if (this->source == NULL) {
      goto error;
    }
  }
  return 1;
error:
  RAVE_FREE(this->nod);
  RAVE_FREE(this->wmo);
  RAVE_FREE(this->wigos);
  RAVE_FREE(this->plc);
  RAVE_FREE(this->rad);
  RAVE_FREE(this->cccc);
  RAVE_FREE(this->org);
  RAVE_FREE(this->source);
  return 0;
}

/**
 * Destroys the odim source
 * @param[in] obj - the the OdimSource_t instance
 */
static void OdimSource_destructor(RaveCoreObject* obj)
{
  OdimSource_t* this = (OdimSource_t*)obj;
  RAVE_FREE(this->nod);
  RAVE_FREE(this->wmo);
  RAVE_FREE(this->wigos);
  RAVE_FREE(this->plc);
  RAVE_FREE(this->rad);
  RAVE_FREE(this->cccc);
  RAVE_FREE(this->org);
  RAVE_FREE(this->source);
}
/*@} End of Private functions */

/*@{ Interface functions */
char* OdimSource_getIdFromOdimSource(const char* source, const char* id)
{
  char* result = NULL;
  if (source != NULL && id != NULL) {
    char* p = strstr(source, id);
    if (p != NULL) {
      int len = 0;
      char* pbrk = NULL;
      p += strlen(id);
      len = strlen(p);
      pbrk = strpbrk((const char*)p, ",");

      if (pbrk != NULL) {
        len = pbrk - p;
      }

      result = RAVE_MALLOC(sizeof(char) * (len + 1));
      if (result != NULL) {
        strncpy(result, p, len);
        result[len] = '\0';
      }
    }
  }
  return result;
}

char* OdimSource_getIdFromOdimSourceInclusive(const char* source, const char* id)
{
  char* result = NULL;
  if (source != NULL && id != NULL) {
    char* p = strstr(source, id);
    if (p != NULL) {
      int len = 0;
      char* pbrk = NULL;
      len = strlen(p);
      pbrk = strpbrk((const char*)p, ",");

      if (pbrk != NULL) {
        len = pbrk - p;
      }

      result = RAVE_MALLOC(sizeof(char) * (len + 1));
      if (result != NULL) {
        strncpy(result, p, len);
        result[len] = '\0';
      }
    }
  }
  return result;
}


OdimSource_t* OdimSource_create(const char* nod, const char* wmo, const char* wigos, const char* plc, const char* rad, const char* cccc, const char* org)
{
  OdimSource_t *result = NULL, *source = NULL;
  if (nod == NULL) {
    RAVE_ERROR0("Must specify at least nod in order to create a odim source");
    return NULL;
  }
  source = RAVE_OBJECT_NEW(&OdimSource_TYPE);
  if (source != NULL) {
    if (!OdimSource_setNod(source, nod)) {
      goto fail;
    }
    if (!OdimSource_setWmo(source, wmo)) {
      goto fail;
    }
    if (!OdimSource_setWigos(source, wigos)) {
      goto fail;
    }
    if (!OdimSource_setPlc(source, plc)) {
      goto fail;
    }
    if (!OdimSource_setRad(source, rad)) {
      goto fail;
    }
    if (!OdimSource_setCccc(source, cccc)) {
      goto fail;
    }
    if (!OdimSource_setOrg(source, org)) {
      goto fail;
    }
  }
  result = RAVE_OBJECT_COPY(source);
fail:
  RAVE_OBJECT_RELEASE(source);
  return result;
}

int OdimSource_setNod(OdimSource_t* self, const char* sid)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_FREE(self->nod);
  RAVE_FREE(self->source);
  if (sid != NULL) {
    self->nod = RAVE_STRDUP(sid);
    if (self->nod == NULL) {
      RAVE_CRITICAL0("Failure when copying nod");
      return 0;
    }
  }
  return 1;
}

const char* OdimSource_getNod(OdimSource_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->nod;
}

int OdimSource_setWmo(OdimSource_t* self, const char* sid)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_FREE(self->wmo);
  RAVE_FREE(self->source);
  if (sid != NULL) {
    self->wmo = RAVE_STRDUP(sid);
    if (self->wmo == NULL) {
      RAVE_CRITICAL0("Failure when copying wmo");
      return 0;
    }
  }
  return 1;
}

const char* OdimSource_getWmo(OdimSource_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->wmo;
}

int OdimSource_setWigos(OdimSource_t* self, const char* sid)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_FREE(self->wigos);
  RAVE_FREE(self->source);
  if (sid != NULL) {
    self->wigos = RAVE_STRDUP(sid);
    if (self->wigos == NULL) {
      RAVE_CRITICAL0("Failure when copying wigos");
      return 0;
    }
  }
  return 1;
}

const char* OdimSource_getWigos(OdimSource_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->wigos;
}

int OdimSource_setPlc(OdimSource_t* self, const char* sid)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_FREE(self->plc);
  RAVE_FREE(self->source);
  if (sid != NULL) {
    self->plc = RAVE_STRDUP(sid);
    if (self->plc == NULL) {
      RAVE_CRITICAL0("Failure when copying plc");
      return 0;
    }
  }
  return 1;
}

const char* OdimSource_getPlc(OdimSource_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->plc;
}

int OdimSource_setRad(OdimSource_t* self, const char* sid)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_FREE(self->rad);
  RAVE_FREE(self->source);
  if (sid != NULL) {
    self->rad = RAVE_STRDUP(sid);
    if (self->rad == NULL) {
      RAVE_CRITICAL0("Failure when copying rad");
      return 0;
    }
  }
  return 1;
}

const char* OdimSource_getRad(OdimSource_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->rad;
}

int OdimSource_setCccc(OdimSource_t* self, const char* sid)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_FREE(self->cccc);
  if (sid != NULL) {
    self->cccc = RAVE_STRDUP(sid);
    if (self->cccc == NULL) {
      RAVE_CRITICAL0("Failure when copying cccc");
      return 0;
    }
  }
  return 1;
}

const char* OdimSource_getCccc(OdimSource_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->cccc;
}

int OdimSource_setOrg(OdimSource_t* self, const char* sid)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_FREE(self->org);
  if (sid != NULL) {
    self->org = RAVE_STRDUP(sid);
    if (self->org == NULL) {
      RAVE_CRITICAL0("Failure when copying org");
      return 0;
    }
  }
  return 1;
}

const char* OdimSource_getOrg(OdimSource_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->org;
}

const char* OdimSource_getSource(OdimSource_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->source == NULL) {
    char buff[1024];
    char* ptr = buff;
    int len = 1024, sz = 0;
    sz = snprintf(ptr, len, "NOD:%s", self->nod);
    ptr += sz;
    len -= sz;
    if (self->wmo != NULL && strcmp("00000", self->wmo) != 0) {
      sz = snprintf(ptr, len, ",WMO:%s", self->wmo);
      ptr += sz;
      len -= sz;
    }
    if (self->rad != NULL) {
      sz = snprintf(ptr, len, ",RAD:%s", self->rad);
      ptr += sz;
      len -= sz;
    }
    if (self->plc != NULL) {
      sz = snprintf(ptr, len, ",PLC:%s", self->plc);
      ptr += sz;
      len -= sz;
    }
    if (self->wigos != NULL) {
      sz = snprintf(ptr, len, ",WIGOS:%s", self->wigos);
      ptr += sz;
      len -= sz;
    }

    self->source = RAVE_STRDUP(buff);
  }
  return self->source;
}

/*@} End of Interface functions */

RaveCoreObjectType OdimSource_TYPE = {
    "OdimSource",
    sizeof(OdimSource_t),
    OdimSource_constructor,
    OdimSource_destructor,
    OdimSource_copyconstructor
};
