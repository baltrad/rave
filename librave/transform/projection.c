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
 * Wrapper around PROJ.4
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-20
 */
#include "projection.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>

/**
 * Represents one projection
 */
struct _Projection_t {
  char* id;
  char* description;
  char* definition;
  PJ* pj;

  int debug;
  void* voidPtr;
  long ps_refCount;
};

/*@{ Private functions */
/**
 * Destroys the projection
 * @param[in] projection - the projection to destroy
 */
static void Projection_destroy(Projection_t* projection)
{
  if (projection != NULL) {
    RAVE_FREE(projection->id);
    RAVE_FREE(projection->description);
    RAVE_FREE(projection->definition);
    if (projection->pj != NULL) {
      pj_free(projection->pj);
    }
    RAVE_FREE(projection);
  }
}
/*@} End of Private functions */

/*@{ Interface functions */
Projection_t* Projection_new(const char* id, const char* description, const char* definition)
{
  Projection_t* result = NULL;

  if (definition == NULL) {
    RAVE_ERROR0("Must at least provide definition when creating a projection");
    return NULL;
  }

  result = RAVE_MALLOC(sizeof(Projection_t));
  if (result != NULL) {
    result->id = NULL;
    result->description = NULL;
    result->definition = NULL;
    result->pj = NULL;
    result->debug = 0;
    result->voidPtr = NULL;
    result->ps_refCount = 1;

    if (id != NULL) {
      result->id = RAVE_STRDUP(id);
      if (result->id == NULL) {
        RAVE_ERROR0("Failed to duplicate id");
        goto error;
      }
    }
    if (description != NULL) {
      result->description = RAVE_STRDUP(description);
      if (result->description == NULL) {
        RAVE_ERROR0("Failed to duplicate description");
        goto error;
      }
    }

    result->definition = RAVE_STRDUP(definition);
    if (result->definition == NULL) {
      RAVE_ERROR0("Failed to duplicate definition");
      goto error;
    }

    result->pj = pj_init_plus(definition);
    if (result->pj == NULL) {
      RAVE_ERROR0("Failed to create projection");
      goto error;
    }
  }

  return result;
error:
  if (result != NULL) {
    Projection_destroy(result);
  }
  return NULL;
}

void Projection_release(Projection_t* projection)
{
  if (projection != NULL) {
    projection->ps_refCount--;
    if (projection->ps_refCount <= 0) {
      Projection_destroy(projection);
    }
  }
}

Projection_t* Projection_copy(Projection_t* projection)
{
  RAVE_ASSERT((projection != NULL), "projection was NULL");
  projection->ps_refCount++;
  return projection;
}

const char* Projection_getID(Projection_t* projection)
{
  RAVE_ASSERT((projection != NULL), "projection was NULL");
  return (const char*)projection->id;
}

const char* Projection_getDescription(Projection_t* projection)
{
  RAVE_ASSERT((projection != NULL), "projection was NULL");
  return (const char*)projection->description;
}

const char* Projection_getDefinition(Projection_t* projection)
{
  RAVE_ASSERT((projection != NULL), "projection was NULL");
  return (const char*)projection->definition;
}

int Projection_transform(Projection_t* projection, Projection_t* tgt, double* x, double* y, double* z)
{
  int pjv = 0;
  int result = 1;
  double vx,vy;
  RAVE_ASSERT((projection != NULL), "projection was NULL");
  RAVE_ASSERT((tgt != NULL), "target projection was NULL");
  RAVE_ASSERT((x != NULL), "x was NULL");
  RAVE_ASSERT((y != NULL), "y was NULL");
  vx = *x;
  vy = *y;
  if ((pjv = pj_transform(projection->pj, tgt->pj, 1, 1, x, y, z)) != 0)
  {
    RAVE_ERROR1("Transform failed with pj_errno: %d\n", pjv);
    result = 0;
  }

  return result;
}

int Projection_inv(Projection_t* projection, double* x, double* y)
{
  int result = 1;
  projUV in,out;
  RAVE_ASSERT((projection != NULL), "projection was NULL");
  RAVE_ASSERT((x != NULL), "x was NULL");
  RAVE_ASSERT((y != NULL), "y was NULL");
  in.u = *x;
  in.v = *y;
  out = pj_inv(in, projection->pj);
  *x = out.u;
  *y = out.v;
  return result;
}

void Projection_setVoidPtr(Projection_t* projection, void* ptr)
{
  RAVE_ASSERT((projection != NULL), "projection was NULL");
  projection->voidPtr = ptr;
}

void* Projection_getVoidPtr(Projection_t* projection)
{
  RAVE_ASSERT((projection != NULL), "projection was NULL");
  return projection->voidPtr;
}

void Projection_setDebug(Projection_t* projection, int debug)
{
  RAVE_ASSERT((projection != NULL), "projection was NULL");
  projection->debug = debug;
}
/*@} End of Interface functions */
