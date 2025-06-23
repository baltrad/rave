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
 * Defines an tiledef, the extent, projection, etc.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#include "tiledef.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>

/**
 * Represents the tiledef
 */
struct _TileDef_t {
  RAVE_OBJECT_HEAD /** Always on top */

  char* id;        /**< the id */
  char* areaid;        /**< the areaid */

  double llX;      /**< lower left x-coordinate */
  double llY;      /**< lower left y-coordinate */
  double urX;      /**< upper right x-coordinate */
  double urY;      /**< upper right y-coordinate */

};



/*@{ Private functions */
/**
 * Constructor.
 */
static int TileDef_constructor(RaveCoreObject* obj)
{
  TileDef_t* this = (TileDef_t*)obj;
  this->id = NULL;
  this->areaid=NULL;
  this->llX = 0.0L;
  this->llY = 0.0L;
  this->urX = 0.0L;
  this->urY = 0.0L;
  return 1;
}

/**
 * Copy constructor
 */
static int TileDef_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  TileDef_t* this = (TileDef_t*)obj;
  TileDef_t* src = (TileDef_t*)srcobj;

  TileDef_constructor(obj); // First just initialize everything like the constructor

  this->llX = src->llX;
  this->llY = src->llY;
  this->urX = src->urX;
  this->urY = src->urY;

  if (!TileDef_setID(this, src->id)) {
    goto error;
  }
  if (!TileDef_setAreaID(this, src->areaid)) {
    goto error;
  }
  return 1;
error:
  RAVE_FREE(this->id);
  RAVE_FREE(this->areaid);
  return 0;
}

/**
 * Destroys the tiledef
 * @param[in] obj - the the TileDef_t instance
 */
static void TileDef_destructor(RaveCoreObject* obj)
{
  TileDef_t* tiledef = (TileDef_t*)obj;
  if (tiledef != NULL) {
    RAVE_FREE(tiledef->id);
    RAVE_FREE(tiledef->areaid);
  }
}
/*@} End of Private functions */

/*@{ Interface functions */
int TileDef_setID(TileDef_t* tiledef, const char* id)
{
  RAVE_ASSERT((tiledef != NULL), "tiledef was NULL");
  RAVE_FREE(tiledef->id);
  if (id != NULL) {
    tiledef->id = RAVE_STRDUP(id);
    if (tiledef->id == NULL) {
      RAVE_CRITICAL0("Failure when copying id");
      return 0;
    }
  }
  return 1;
}

const char* TileDef_getID(TileDef_t* tiledef)
{
  RAVE_ASSERT((tiledef != NULL), "tiledef was NULL");
  return (const char*)tiledef->id;
}

/*@{ Interface functions */
int TileDef_setAreaID(TileDef_t* tiledef, const char* id)
{
  RAVE_ASSERT((tiledef != NULL), "tiledef was NULL");
  RAVE_FREE(tiledef->areaid);
  if (id != NULL) {
    tiledef->areaid = RAVE_STRDUP(id);
    if (tiledef->areaid == NULL) {
      RAVE_CRITICAL0("Failure when copying areaid");
      return 0;
    }
  }
  return 1;
}

const char* TileDef_getAreaID(TileDef_t* tiledef)
{
  RAVE_ASSERT((tiledef != NULL), "tiledef was NULL");
  return (const char*)tiledef->areaid;
}

void TileDef_setExtent(TileDef_t* tiledef, double llX, double llY, double urX, double urY)
{
  RAVE_ASSERT((tiledef != NULL), "tiledef was NULL");
  tiledef->llX = llX;
  tiledef->llY = llY;
  tiledef->urX = urX;
  tiledef->urY = urY;
}

void TileDef_getExtent(TileDef_t* tiledef, double* llX, double* llY, double* urX, double* urY)
{
  RAVE_ASSERT((tiledef != NULL), "tiledef was NULL");
  if (llX != NULL) {
    *llX = tiledef->llX;
  }
  if (llY != NULL) {
    *llY = tiledef->llY;
  }
  if (urX != NULL) {
    *urX = tiledef->urX;
  }
  if (urY != NULL) {
    *urY = tiledef->urY;
  }
}

/*@} End of Interface functions */

RaveCoreObjectType TileDef_TYPE = {
    "TileDef",
    sizeof(TileDef_t),
    TileDef_constructor,
    TileDef_destructor,
    TileDef_copyconstructor
};
