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
 * Provides functionality for creating composites.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-01-19
 */
#include "composite.h"
#include "polarvolume.h"
#include "raveobject_list.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>

/**
 * Represents the cartesian product.
 */
struct _Composite_t {
  RAVE_OBJECT_HEAD /** Always on top */

  RaveObjectList_t* list;
};

/*@{ Private functions */
/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int Composite_constructor(RaveCoreObject* obj)
{
  Composite_t* this = (Composite_t*)obj;
  this->list = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (this->list == NULL) {
    return 0;
  }

  return 1;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int Composite_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  Composite_t* this = (Composite_t*)obj;
  Composite_t* src = (Composite_t*)srcobj;
  this->list = RAVE_OBJECT_COPY(src->list);
  if (this->list == NULL) {
    return 0;
  }
  return 1;
}


/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void Composite_destructor(RaveCoreObject* obj)
{
  Composite_t* this = (Composite_t*)obj;
  RAVE_OBJECT_RELEASE(this->list);
}
/*@} End of Private functions */

/*@{ Interface functions */
int Composite_add(Composite_t* composite, RaveCoreObject* object)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  RAVE_ASSERT((object != NULL), "object == NULL");

  if (!RAVE_OBJECT_CHECK_TYPE(object, &PolarVolume_TYPE)) {
    RAVE_ERROR0("Providing an object that not is a PolarVolume during composite generation");
    return 0;
  }
  return RaveObjectList_add(composite->list, object);
}

Cartesian_t* Composite_nearest(Composite_t* composite, Area_t* area, double height)
{
  Cartesian_t* result = NULL;
  Projection_t* projection = NULL;

  int x = 0, y = 0, i = 0, xsize = 0, ysize = 0, nradars = 0;
  result = RAVE_OBJECT_NEW(&Cartesian_TYPE);
  if (!Cartesian_init(result, area, RaveDataType_UCHAR)) {
    goto fail;
  }
  xsize = Cartesian_getXSize(result);
  ysize = Cartesian_getYSize(result);
  projection = Cartesian_getProjection(result);
  nradars = RaveObjectList_size(composite->list);

  for (y = 0; y < ysize; y++) {
    for (x = 0; x < xsize; x++) {
      for (i = 0; i < nradars; i++) {

      }
    }
  }

fail:
  RAVE_OBJECT_RELEASE(result);
  return result;
}
/*@} End of Interface functions */

RaveCoreObjectType Composite_TYPE = {
    "Composite",
    sizeof(Composite_t),
    Composite_constructor,
    Composite_destructor,
    Composite_copyconstructor
};

