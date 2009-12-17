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
  RAVE_OBJECT_HEAD /** Always on top */
  int initialized;     /**< if this instance has been initialized */
  char* id;            /**< the id of this projection */
  char* description;   /**< the description */
  char* definition;    /**< the proj.4 definition string */
  PJ* pj;              /**< the proj.4 instance */
};

/*@{ Private functions */
static int Projection_constructor(RaveCoreObject* obj)
{
  Projection_t* projection = (Projection_t*)obj;
  projection->initialized = 0;
  projection->id = NULL;
  projection->description = NULL;
  projection->definition = NULL;
  projection->pj = NULL;
  return 1;
}

/**
 * Copy constructor.
 */
static int Projection_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  int result = 0;
  Projection_t* this = (Projection_t*)obj;
  Projection_t* src = (Projection_t*)srcobj;

  this->initialized = 0;
  this->id = NULL;
  this->description = NULL;
  this->definition = NULL;
  this->pj = NULL;

  result = Projection_init(this, src->id, src->description, src->definition);

  return result;
}

/**
 * Destroys the projection
 * @param[in] projection - the projection to destroy
 */
static void Projection_destructor(RaveCoreObject* obj)
{
  Projection_t* projection = (Projection_t*)obj;
  if (projection != NULL) {
    RAVE_FREE(projection->id);
    RAVE_FREE(projection->description);
    RAVE_FREE(projection->definition);
    if (projection->pj != NULL) {
      pj_free(projection->pj);
    }
  }
}
/*@} End of Private functions */

/*@{ Interface functions */
int Projection_init(Projection_t* projection, const char* id, const char* description, const char* definition)
{
  int result = 0;
  RAVE_ASSERT((projection != NULL), "projection was NULL");
  RAVE_ASSERT((projection->initialized == 0), "projection was already initalized");
  if (id == NULL || description == NULL || definition == NULL) {
    RAVE_ERROR0("One of id, description or definition was NULL when initializing");
    return 0;
  }
  projection->id = RAVE_STRDUP(id);
  projection->description = RAVE_STRDUP(description);
  projection->definition = RAVE_STRDUP(definition);
  projection->pj = pj_init_plus(definition);
  if (projection->id == NULL || projection->description == NULL ||
      projection->definition == NULL || projection->pj == NULL) {
    if (projection->id == NULL) {
      RAVE_ERROR0("Could not set id");
    }
    if (projection->description == NULL) {
      RAVE_ERROR0("Could not set description");
    }
    if (projection->definition == NULL) {
      RAVE_ERROR0("Could not set definition");
    }
    if (projection->pj == NULL) {
      RAVE_ERROR0("Failed to create projection");
    }
    RAVE_FREE(projection->id);
    RAVE_FREE(projection->description);
    RAVE_FREE(projection->definition);
    if (projection->pj != NULL) {
      pj_free(projection->pj);
    }
  } else {
    result = 1;
    projection->initialized = 1;
  }
  return result;
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

int Projection_inv(Projection_t* projection, double x, double y, double* lon, double* lat)
{
  int result = 1;
  projUV in,out;
  RAVE_ASSERT((projection != NULL), "projection was NULL");
  RAVE_ASSERT((lon != NULL), "lon was NULL");
  RAVE_ASSERT((lat != NULL), "lat was NULL");
  in.u = x;
  in.v = y;
  out = pj_inv(in, projection->pj);
  *lon = out.u;
  *lat = out.v;
  return result;
}

int Projection_fwd(Projection_t* projection, double lon, double lat, double* x, double* y)
{
  int result = 1;
  projUV in,out;
  RAVE_ASSERT((projection != NULL), "projection was NULL");
  RAVE_ASSERT((x != NULL), "x was NULL");
  RAVE_ASSERT((y != NULL), "y was NULL");
  in.u = lon;
  in.v = lat;
  out = pj_fwd(in, projection->pj);
  *x = out.u;
  *y = out.v;
  return result;
}

/*@} End of Interface functions */

RaveCoreObjectType Projection_TYPE = {
    "Projection",
    sizeof(Projection_t),
    Projection_constructor,
    Projection_destructor,
    Projection_copyconstructor
};
