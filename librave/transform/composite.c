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
#include "rave_datetime.h"
#include <string.h>

/**
 * Represents the cartesian product.
 */
struct _Composite_t {
  RAVE_OBJECT_HEAD /** Always on top */
  Rave_ProductType ptype; /**< the product type, default PCAPPI */
  double height; /**< the height when generating pcapppi, cappi, default 1000 */
  char* paramname; /**< the parameter name */
  RaveDateTime_t* datetime;  /**< the date and time */
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
  this->ptype = Rave_ProductType_PCAPPI;
  this->height = 1000.0;
  this->paramname = NULL;
  this->list = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  this->datetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  if (this->list == NULL || this->datetime == NULL || !Composite_setQuantity(this, "DBZH")) {
    goto error;
  }
  return 1;
error:
  RAVE_FREE(this->paramname);
  RAVE_OBJECT_RELEASE(this->list);
  RAVE_OBJECT_RELEASE(this->datetime);
  return 0;
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
  this->ptype = src->ptype;
  this->height = src->height;
  this->paramname = NULL;
  this->list = RAVE_OBJECT_CLONE(src->list);
  this->datetime = RAVE_OBJECT_CLONE(src->datetime);
  if (this->list == NULL || this->datetime == NULL || !Composite_setQuantity(this, Composite_getQuantity(src))) {
    goto error;
  }

  return 1;
error:
  RAVE_FREE(this->paramname);
  RAVE_OBJECT_RELEASE(this->list);
  RAVE_OBJECT_RELEASE(this->datetime);
  return 0;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void Composite_destructor(RaveCoreObject* obj)
{
  Composite_t* this = (Composite_t*)obj;
  RAVE_OBJECT_RELEASE(this->list);
  RAVE_OBJECT_RELEASE(this->datetime);
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

void Composite_setProduct(Composite_t* composite, Rave_ProductType type)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  if (type == Rave_ProductType_PCAPPI) {
    composite->ptype = type;
  } else {
    RAVE_ERROR0("Only supported algorithm right now is PCAPPI");
  }
}

Rave_ProductType Composite_getProduct(Composite_t* composite)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return composite->ptype;
}

void Composite_setHeight(Composite_t* composite, double height)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  composite->height = height;
}

double Composite_getHeight(Composite_t* composite)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return composite->height;
}

int Composite_setQuantity(Composite_t* composite, const char* quantity)
{
  int result = 0;
  char* tmp = NULL;
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  if (quantity == NULL) {
    return 0;
  }
  tmp = RAVE_STRDUP(quantity);
  if (tmp != NULL) {
    RAVE_FREE(composite->paramname);
    composite->paramname = tmp;
    result = 1;
  }
  return result;
}

const char* Composite_getQuantity(Composite_t* composite)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return (const char*)composite->paramname;
}

int Composite_setTime(Composite_t* composite, const char* value)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return RaveDateTime_setTime(composite->datetime, value);
}

const char* Composite_getTime(Composite_t* composite)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return RaveDateTime_getTime(composite->datetime);
}

int Composite_setDate(Composite_t* composite, const char* value)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return RaveDateTime_setDate(composite->datetime, value);
}

const char* Composite_getDate(Composite_t* composite)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return RaveDateTime_getDate(composite->datetime);
}

Cartesian_t* Composite_nearest(Composite_t* composite, Area_t* area)
{
  Cartesian_t* result = NULL;
  Projection_t* projection = NULL;
  RaveAttribute_t* prodpar = NULL;

  int x = 0, y = 0, i = 0, xsize = 0, ysize = 0, nradars = 0;
  double v = 0.0L;
  RaveValueType vtype = RaveValueType_UNDEFINED;

  RAVE_ASSERT((composite != NULL), "composite == NULL");
  RAVE_ASSERT((area != NULL), "area == NULL");


  result = RAVE_OBJECT_NEW(&Cartesian_TYPE);
  if (!Cartesian_init(result, area, RaveDataType_UCHAR)) {
    goto fail;
  }
  prodpar = RaveAttributeHelp_createDouble("what/prodpar", composite->height);
  if (prodpar == NULL) {
    goto fail;
  }

  Cartesian_setObjectType(result, Rave_ObjectType_COMP);
  Cartesian_setProduct(result, composite->ptype);
  if (!Cartesian_addAttribute(result, prodpar)) {
    goto fail;
  }
  if (!Cartesian_setQuantity(result, Composite_getQuantity(composite))) {
    goto fail;
  }
  if (Composite_getTime(composite) != NULL) {
    if (!Cartesian_setTime(result, Composite_getTime(composite))) {
      goto fail;
    }
  }
  if (Composite_getDate(composite) != NULL) {
    if (!Cartesian_setDate(result, Composite_getDate(composite))) {
      goto fail;
    }
  }
  if (!Cartesian_setSource(result, Area_getID(area))) {
    goto fail;
  }

  Cartesian_setNodata(result, 255.0);
  Cartesian_setUndetect(result, 0.0);
  xsize = Cartesian_getXSize(result);
  ysize = Cartesian_getYSize(result);
  projection = Cartesian_getProjection(result);
  nradars = RaveObjectList_size(composite->list);

  for (y = 0; y < ysize; y++) {
    double herey = Cartesian_getLocationY(result, y);
    for (x = 0; x < xsize; x++) {
      PolarVolume_t* pvol = NULL;
      double herex = Cartesian_getLocationX(result, x);
      double olon = 0.0, olat = 0.0, pvollon = 0.0, pvollat = 0.0;
      double mindist = 1e10;

      for (i = 0, mindist=1e10; i < nradars; i++) {
        PolarVolume_t* vol = NULL;
        Projection_t* volproj = NULL;

        vol = (PolarVolume_t*)RaveObjectList_get(composite->list, i);
        if (vol != NULL) {
          volproj = PolarVolume_getProjection(vol);
        }

        if (volproj != NULL) {
          /* We will go from surface coords into the lonlat projection assuming that a polar volume uses a lonlat projection*/
          if (!Projection_transformx(projection, volproj, herex, herey, 0.0, &olon, &olat, NULL)) {
            RAVE_WARNING0("Failed to transform from composite into polar coordinates");
          } else {
            double dist = PolarVolume_getDistance(vol, olon, olat);
            if (dist < mindist) {
              mindist = dist;
              RAVE_OBJECT_RELEASE(pvol);
              pvol = RAVE_OBJECT_COPY(vol);
              pvollon = olon;
              pvollat = olat;
            }
          }
        }
        RAVE_OBJECT_RELEASE(vol);
        RAVE_OBJECT_RELEASE(volproj);
      }

      if (pvol != NULL) {
        if (composite->ptype == Rave_ProductType_PCAPPI) {
          //vtype = PolarVolume_getNearest(pvol, pvollon, pvollat, composite->height, 0, &v);
          vtype = PolarVolume_getNearestParameterValue(pvol,
                                                       Composite_getQuantity(composite),
                                                       pvollon,
                                                       pvollat,
                                                       Composite_getHeight(composite),
                                                       0,
                                                       &v);

        } else {
          vtype = RaveValueType_NODATA;
        }
        if (vtype == RaveValueType_NODATA) {
          Cartesian_setValue(result, x, y, Cartesian_getNodata(result));
        } else if (vtype == RaveValueType_UNDETECT) {
          Cartesian_setValue(result, x, y, Cartesian_getUndetect(result));
        } else {
          Cartesian_setValue(result, x, y, v);
        }
      }
      RAVE_OBJECT_RELEASE(pvol);
    }
  }

  RAVE_OBJECT_RELEASE(projection);
  RAVE_OBJECT_RELEASE(prodpar);
  return result;
fail:
  RAVE_OBJECT_RELEASE(projection);
  RAVE_OBJECT_RELEASE(prodpar);
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

