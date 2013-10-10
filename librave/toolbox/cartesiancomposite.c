/* --------------------------------------------------------------------
Copyright (C) 2013 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Defines the functions available when creating composites from cartesian products.
 *
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2013-10-01
 */
#include "cartesiancomposite.h"
#include "cartesianparam.h"
#include "cartesian.h"
#include "raveobject_list.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_datetime.h"
#include <string.h>
#include "raveobject_hashtable.h"

/**
 * Represents the cartesian composite generator.
 */
struct _CartesianComposite_t {
  RAVE_OBJECT_HEAD /** Always on top */
  CompositeSelectionMethod_t method; /**< selection method, default CompositeSelectionMethod_NEAREST */
  /*RaveList_t* parameters;*/ /**< the parameters to generate */
  RaveObjectList_t* list; /**< the list of cartesian objects */
  RaveDateTime_t* datetime;  /**< the date and time */

  char* quantity; /**< the quantity to make a composite of */
  double offset; /**< the offset for the data */
  double gain; /**< the gain for the data */
};

/** The name of the task for specifying distance to radar */
#define DISTANCE_TO_RADAR_HOW_TASK "se.smhi.composite.distance.radar"

/*@{ Private functions */
/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int CartesianComposite_constructor(RaveCoreObject* obj)
{
  CartesianComposite_t* this = (CartesianComposite_t*)obj;
  this->method = CompositeSelectionMethod_NEAREST;
  this->datetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  this->list = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  this->quantity = RAVE_STRDUP("DBZH");
  this->offset = 0.0;
  this->gain = 1.0;
  if (this->list == NULL || this->datetime == NULL || this->quantity == NULL) {
    goto error;
  }
  return 1;
error:
  RAVE_OBJECT_RELEASE(this->list);
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_FREE(this->quantity);
  return 0;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int CartesianComposite_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  CartesianComposite_t* this = (CartesianComposite_t*)obj;
  CartesianComposite_t* src = (CartesianComposite_t*)srcobj;
  this->method = src->method;
  this->list = RAVE_OBJECT_CLONE(src->list);
  this->datetime = RAVE_OBJECT_CLONE(src->datetime);
  this->quantity = RAVE_STRDUP(src->quantity); /* Assuming that we never let a quantity be set to NULL */
  this->offset = src->offset;
  this->gain = src->gain;
  if (this->datetime == NULL || this->list == NULL || this->quantity == NULL) {
    goto error;
  }

  return 1;
error:
  RAVE_OBJECT_RELEASE(this->list);
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_FREE(this->quantity);
  return 0;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void CartesianComposite_destructor(RaveCoreObject* obj)
{
  CartesianComposite_t* this = (CartesianComposite_t*)obj;
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_OBJECT_RELEASE(this->list);
  RAVE_FREE(this->quantity);
}

/**
 * Creates the resulting composite image.
 * @param[in] self - self
 * @param[in] area - the area the composite image(s) should have
 * @returns the cartesian on success otherwise NULL
 */
//static Cartesian_t* CartesianComposite_createCompositeImage(CartesianComposite_t* self, Area_t* area)
//{
//  Cartesian_t *result = NULL, *cartesian = NULL;
//  RaveAttribute_t* prodpar = NULL;
//  CartesianParam_t* cp = NULL;
//
//  RAVE_ASSERT((self != NULL), "self == NULL");
//
//  cartesian = RAVE_OBJECT_NEW(&Cartesian_TYPE);
//  if (cartesian == NULL) {
//    goto done;
//  }
//  Cartesian_init(cartesian, area);
//
//  prodpar = RaveAttributeHelp_createString("what/prodpar", "some relevant information");
//  if (prodpar == NULL) {
//    goto done;
//  }
//
//  Cartesian_setObjectType(cartesian, Rave_ObjectType_COMP);
//  Cartesian_setProduct(cartesian, Rave_ProductType_COMP);
//  if (!Cartesian_addAttribute(cartesian, prodpar)) {
//    goto done;
//  }
//  if (CartesianComposite_getTime(self) != NULL) {
//    if (!Cartesian_setTime(cartesian, CartesianComposite_getTime(self))) {
//      goto done;
//    }
//  }
//  if (CartesianComposite_getDate(self) != NULL) {
//    if (!Cartesian_setDate(cartesian, CartesianComposite_getDate(self))) {
//      goto done;
//    }
//  }
//  if (!Cartesian_setSource(cartesian, Area_getID(area))) {
//    goto done;
//  }
//
//  cp = Cartesian_createParameter(cartesian, self->quantity, RaveDataType_UCHAR);
//  if (cp == NULL) {
//    goto done;
//  }
//  CartesianParam_setNodata(cp, 255.0);
//  CartesianParam_setUndetect(cp, 0.0);
//  CartesianParam_setGain(cp, self->gain);
//  CartesianParam_setOffset(cp, self->offset);
//  RAVE_OBJECT_RELEASE(cp);
//
//  result = RAVE_OBJECT_COPY(cartesian);
//done:
//  RAVE_OBJECT_RELEASE(cartesian);
//  RAVE_OBJECT_RELEASE(prodpar);
//  return result;
//}
/*@} End of Private functions */

/*@{ Interface functions */
int CartesianComposite_add(CartesianComposite_t* self, Cartesian_t* o)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((o != NULL), "o == NULL");
  return RaveObjectList_add(self->list, (RaveCoreObject*)o);
}

int CartesianComposite_setTime(CartesianComposite_t* self, const char* value)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_setTime(self->datetime, value);
}

const char* CartesianComposite_getTime(CartesianComposite_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_getTime(self->datetime);
}

int CartesianComposite_setDate(CartesianComposite_t* self, const char* value)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_setDate(self->datetime, value);
}

const char* CartesianComposite_getDate(CartesianComposite_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_getDate(self->datetime);
}

int CartesianComposite_setQuantity(CartesianComposite_t* self, const char* quantity)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (quantity != NULL) {
    char* tmp = RAVE_STRDUP(quantity);
    if (tmp != NULL) {
      RAVE_FREE(self->quantity);
      self->quantity = tmp;
      result = 1;
    }
  } else {
    RAVE_INFO0("Quantity can not be NULL");
  }
  return result;
}

const char* CartesianComposite_getQuantity(CartesianComposite_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->quantity;
}

void CartesianComposite_setGain(CartesianComposite_t* self, double gain)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (gain != 0.0) {
    self->gain = gain;
  }
}

double CartesianComposite_getGain(CartesianComposite_t* self)
{
  RAVE_ASSERT((self != NULL), "cartesian == NULL");
  return self->gain;
}

void CartesianComposite_setOffset(CartesianComposite_t* self, double offset)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->offset = offset;
}

double CartesianComposite_getOffset(CartesianComposite_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->offset;
}


Cartesian_t* CartesianComposite_nearest(CartesianComposite_t* self, Area_t* area)
{
  return NULL;
}

/*@} End of Interface functions */

RaveCoreObjectType CartesianComposite_TYPE = {
    "CartesianComposite",
    sizeof(CartesianComposite_t),
    CartesianComposite_constructor,
    CartesianComposite_destructor,
    CartesianComposite_copyconstructor
};
