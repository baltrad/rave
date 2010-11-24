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
 * Defines the functions available when working with cartesian products
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-16
 */
#include "cartesian.h"
#include "area.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_datetime.h"
#include "rave_data2d.h"
#include "raveobject_hashtable.h"
#include "rave_utilities.h"
#include <string.h>

/**
 * Represents the cartesian product.
 */
struct _Cartesian_t {
  RAVE_OBJECT_HEAD /** Always on top */

  // Where
  double xscale;     /**< xscale */
  double yscale;     /**< yscale */

  Rave_ProductType product;   /**< product */
  Rave_ObjectType objectType; /**< object type */

  double llX;        /**< lower left x-coordinate */
  double llY;        /**< lower left y-coordinate */
  double urX;        /**< upper right x-coordinate */
  double urY;        /**< upper right x-coordinate */

  // What
  char* quantity;            /**< what does this data represent */
  RaveDateTime_t* datetime;  /**< the date and time */
  RaveDateTime_t* startdatetime;  /**< the start date and time */
  RaveDateTime_t* enddatetime;  /**< the end date and time */

  char* source;              /**< where does this data come from */
  double gain;       /**< gain when scaling, default 1 */
  double offset;     /**< offset when scaling, default 0 */
  double nodata;     /**< nodata */
  double undetect;   /**< undetect */

  Projection_t* projection; /**< the projection */

  RaveData2D_t* data;   /**< 2 dimensional data array */

  RaveObjectHashTable_t* attrs; /**< attributes */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int Cartesian_constructor(RaveCoreObject* obj)
{
  Cartesian_t* this = (Cartesian_t*)obj;
  this->xscale = 0.0;
  this->yscale = 0.0;
  this->llX = 0.0;
  this->llY = 0.0;
  this->urX = 0.0;
  this->urY = 0.0;
  this->quantity = NULL;
  this->datetime = NULL;
  this->product = Rave_ProductType_UNDEFINED;
  this->objectType = Rave_ObjectType_IMAGE;
  this->source = NULL;
  this->gain = 1.0;
  this->offset = 0.0;
  this->nodata = 0.0;
  this->undetect = 0.0;
  this->projection = NULL;
  this->data = NULL;
  this->datetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  this->startdatetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  this->enddatetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  this->data = RAVE_OBJECT_NEW(&RaveData2D_TYPE);
  this->attrs = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  if (this->datetime == NULL || this->data == NULL || this->attrs == NULL ||
      this->startdatetime == NULL || this->enddatetime == NULL) {
    goto fail;
  }

  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->data);
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_OBJECT_RELEASE(this->attrs);
  RAVE_OBJECT_RELEASE(this->startdatetime);
  RAVE_OBJECT_RELEASE(this->enddatetime);
  return 0;
}

/**
 * Copy constructor.
 */
static int Cartesian_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  Cartesian_t* this = (Cartesian_t*)obj;
  Cartesian_t* src = (Cartesian_t*)srcobj;
  this->xscale = src->xscale;
  this->yscale = src->yscale;
  this->llX = src->llX;
  this->llY = src->llY;
  this->urX = src->urX;
  this->urY = src->urY;
  this->product = src->product;
  this->objectType = src->objectType;
  this->gain = src->gain;
  this->offset = src->offset;
  this->nodata = src->nodata;
  this->undetect = src->undetect;
  this->source = NULL;
  this->quantity = NULL;
  this->projection = NULL;
  this->datetime = NULL;
  this->startdatetime = NULL;
  this->enddatetime = NULL;
  this->data = NULL;

  Cartesian_setQuantity(this, Cartesian_getQuantity(src));

  this->datetime = RAVE_OBJECT_CLONE(src->datetime);
  this->startdatetime = RAVE_OBJECT_CLONE(src->startdatetime);
  this->enddatetime = RAVE_OBJECT_CLONE(src->enddatetime);
  this->data = RAVE_OBJECT_CLONE(src->data);
  this->attrs = RAVE_OBJECT_CLONE(src->attrs);

  if (this->datetime == NULL || this->data == NULL || this->attrs == NULL ||
      this->startdatetime == NULL || this->enddatetime == NULL) {
    goto fail;
  }

  Cartesian_setSource(this, Cartesian_getSource(src));

  if (src->projection != NULL) {
    this->projection = RAVE_OBJECT_CLONE(src->projection);
    if (this->projection == NULL) {
      goto fail;
    }
  }

  return 1;
fail:
  RAVE_FREE(this->source);
  RAVE_FREE(this->quantity);
  RAVE_OBJECT_RELEASE(this->data);
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_OBJECT_RELEASE(this->startdatetime);
  RAVE_OBJECT_RELEASE(this->enddatetime);
  RAVE_OBJECT_RELEASE(this->attrs);
  RAVE_OBJECT_RELEASE(this->projection);
  return 0;
}


/**
 * Destroys the cartesian product
 * @param[in] scan - the cartesian product to destroy
 */
static void Cartesian_destructor(RaveCoreObject* obj)
{
  Cartesian_t* cartesian = (Cartesian_t*)obj;
  if (cartesian != NULL) {
    RAVE_OBJECT_RELEASE(cartesian->projection);
    RAVE_OBJECT_RELEASE(cartesian->datetime);
    RAVE_OBJECT_RELEASE(cartesian->startdatetime);
    RAVE_OBJECT_RELEASE(cartesian->enddatetime);
    RAVE_FREE(cartesian->source);
    RAVE_FREE(cartesian->quantity);
    RAVE_OBJECT_RELEASE(cartesian->data);
    RAVE_OBJECT_RELEASE(cartesian->attrs);
  }
}

/*@} End of Private functions */

/*@{ Interface functions */
int Cartesian_setTime(Cartesian_t* cartesian, const char* value)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  return RaveDateTime_setTime(cartesian->datetime, value);
}

const char* Cartesian_getTime(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  return RaveDateTime_getTime(cartesian->datetime);
}

int Cartesian_setDate(Cartesian_t* cartesian, const char* value)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  return RaveDateTime_setDate(cartesian->datetime, value);
}

const char* Cartesian_getDate(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  return RaveDateTime_getDate(cartesian->datetime);
}

int Cartesian_setStartTime(Cartesian_t* cartesian, const char* value)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  return RaveDateTime_setTime(cartesian->startdatetime, value);
}

const char* Cartesian_getStartTime(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  if (RaveDateTime_getTime(cartesian->startdatetime) == NULL) {
    return RaveDateTime_getTime(cartesian->datetime);
  }
  return RaveDateTime_getTime(cartesian->startdatetime);
}

int Cartesian_setStartDate(Cartesian_t* cartesian, const char* value)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  return RaveDateTime_setDate(cartesian->startdatetime, value);
}

const char* Cartesian_getStartDate(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  if (RaveDateTime_getDate(cartesian->startdatetime) == NULL) {
    return RaveDateTime_getDate(cartesian->datetime);
  }
  return RaveDateTime_getDate(cartesian->startdatetime);
}

int Cartesian_setEndTime(Cartesian_t* cartesian, const char* value)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  return RaveDateTime_setTime(cartesian->enddatetime, value);
}

const char* Cartesian_getEndTime(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  if (RaveDateTime_getTime(cartesian->enddatetime) == NULL) {
    return RaveDateTime_getTime(cartesian->datetime);
  }
  return RaveDateTime_getTime(cartesian->enddatetime);
}

int Cartesian_setEndDate(Cartesian_t* cartesian, const char* value)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  return RaveDateTime_setDate(cartesian->enddatetime, value);
}

const char* Cartesian_getEndDate(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  if (RaveDateTime_getDate(cartesian->enddatetime) == NULL) {
    return RaveDateTime_getDate(cartesian->datetime);
  }
  return RaveDateTime_getDate(cartesian->enddatetime);
}

int Cartesian_setSource(Cartesian_t* cartesian, const char* value)
{
  char* tmp = NULL;
  int result = 0;
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  if (value != NULL) {
    tmp = RAVE_STRDUP(value);
    if (tmp != NULL) {
      RAVE_FREE(cartesian->source);
      cartesian->source = tmp;
      tmp = NULL;
      result = 1;
    }
  } else {
    RAVE_FREE(cartesian->source);
    result = 1;
  }
  return result;
}

const char* Cartesian_getSource(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return (const char*)cartesian->source;
}

int Cartesian_setObjectType(Cartesian_t* cartesian, Rave_ObjectType type)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  if (type == Rave_ObjectType_IMAGE || type == Rave_ObjectType_COMP) {
    cartesian->objectType = type;
    return 1;
  }
  return 0;
}

Rave_ObjectType Cartesian_getObjectType(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->objectType;
}

long Cartesian_getXSize(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return RaveData2D_getXsize(cartesian->data);
}

long Cartesian_getYSize(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return RaveData2D_getYsize(cartesian->data);
}

void Cartesian_setAreaExtent(Cartesian_t* cartesian, double llX, double llY, double urX, double urY)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->llX = llX;
  cartesian->llY = llY;
  cartesian->urX = urX;
  cartesian->urY = urY;
}

void Cartesian_getAreaExtent(Cartesian_t* cartesian, double* llX, double* llY, double* urX, double* urY)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  if (llX != NULL) {
    *llX = cartesian->llX;
  }
  if (llY != NULL) {
    *llY = cartesian->llY;
  }
  if (urX != NULL) {
    *urX = cartesian->urX;
  }
  if (urY != NULL) {
    *urY = cartesian->urY;
  }
}

void Cartesian_setXScale(Cartesian_t* cartesian, double xscale)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->xscale = xscale;
}

double Cartesian_getXScale(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->xscale;
}

void Cartesian_setYScale(Cartesian_t* cartesian, double yscale)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->yscale = yscale;
}

double Cartesian_getYScale(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->yscale;
}

int Cartesian_setProduct(Cartesian_t* cartesian, Rave_ProductType type)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->product = type;
  return 1;
}

Rave_ProductType Cartesian_getProduct(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->product;
}

double Cartesian_getLocationX(Cartesian_t* cartesian, long x)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->llX + cartesian->xscale * (double)x;
}

double Cartesian_getLocationY(Cartesian_t* cartesian, long y)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->urY - cartesian->yscale * (double)y;
}

long Cartesian_getIndexX(Cartesian_t* cartesian, double x)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  RAVE_ASSERT((cartesian->xscale != 0.0), "xcale == 0.0, would result in Division by zero");
  return (long)((x - cartesian->llX) / cartesian->xscale);
}

long Cartesian_getIndexY(Cartesian_t* cartesian, double y)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  RAVE_ASSERT((cartesian->yscale != 0.0), "ycale == 0.0, would result in Division by zero");
  return (long)((cartesian->urY - y)/cartesian->yscale);

}

RaveDataType Cartesian_getDataType(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return RaveData2D_getType(cartesian->data);
}

int Cartesian_setQuantity(Cartesian_t* cartesian, const char* quantity)
{
  int result = 0;
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  if (quantity != NULL) {
    char* tmp = RAVE_STRDUP(quantity);
    if (tmp != NULL) {
      RAVE_FREE(cartesian->quantity);
      cartesian->quantity = tmp;
      result = 1;
    }
  } else {
    RAVE_FREE(cartesian->quantity);
    result = 1;
  }
  return result;
}

const char* Cartesian_getQuantity(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return (const char*)cartesian->quantity;
}

void Cartesian_setGain(Cartesian_t* cartesian, double gain)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  if (gain != 0.0) {
    cartesian->gain = gain;
  }
}

double Cartesian_getGain(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->gain;
}

void Cartesian_setOffset(Cartesian_t* cartesian, double offset)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->offset = offset;
}

double Cartesian_getOffset(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->offset;
}

void Cartesian_setNodata(Cartesian_t* cartesian, double nodata)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->nodata = nodata;
}

double Cartesian_getNodata(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->nodata;
}

void Cartesian_setUndetect(Cartesian_t* cartesian, double undetect)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->undetect = undetect;
}

double Cartesian_getUndetect(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->undetect;
}

void Cartesian_setProjection(Cartesian_t* cartesian, Projection_t* projection)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  RAVE_OBJECT_RELEASE(cartesian->projection);
  if (projection != NULL) {
    cartesian->projection = RAVE_OBJECT_COPY(projection);
  }
}

Projection_t* Cartesian_getProjection(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  if (cartesian->projection != NULL) {
    return RAVE_OBJECT_COPY(cartesian->projection);
  }
  return NULL;
}

const char* Cartesian_getProjectionString(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  if (cartesian->projection != NULL) {
    return Projection_getDefinition(cartesian->projection);
  }
  return NULL;
}

int Cartesian_setData(Cartesian_t* cartesian, long xsize, long ysize, void* data, RaveDataType type)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  return RaveData2D_setData(cartesian->data, xsize, ysize, data, type);
}

int Cartesian_createData(Cartesian_t* cartesian, long xsize, long ysize, RaveDataType type)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  return RaveData2D_createData(cartesian->data, xsize, ysize, type);
}

void* Cartesian_getData(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  return RaveData2D_getData(cartesian->data);
}

int Cartesian_setValue(Cartesian_t* cartesian, long x, long y, double v)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return RaveData2D_setValue(cartesian->data, x, y, v);
}

int Cartesian_setConvertedValue(Cartesian_t* cartesian, long x, long y, double v)
{
  double value = v;
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  if (value != cartesian->undetect && value != cartesian->nodata) {
    if (cartesian->gain != 0.0) {
      value = (v - cartesian->offset)/cartesian->gain;
    } else {
      RAVE_ERROR0("gain is 0.0 => division by zero error");
      return 0;
    }
  }
  return RaveData2D_setValue(cartesian->data, x, y, value);
}

RaveValueType Cartesian_getValue(Cartesian_t* cartesian, long x, long y, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  double value = 0.0;
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");

  value = cartesian->nodata;

  if (RaveData2D_getValue(cartesian->data, x, y, &value)) {
    result = RaveValueType_DATA;
    if (value == cartesian->nodata) {
      result = RaveValueType_NODATA;
    } else if (value == cartesian->undetect) {
      result = RaveValueType_UNDETECT;
    }
  }

  if (v != NULL) {
    *v = value;
  }

  return result;
}

RaveValueType Cartesian_getConvertedValue(Cartesian_t* cartesian, long x, long y, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");

  result = Cartesian_getValue(cartesian, x, y, v);
  if (result == RaveValueType_DATA && v != NULL) {
    *v = (*v) * cartesian->gain + cartesian->offset;
  }
  return result;
}

int Cartesian_init(Cartesian_t* cartesian, Area_t* area, RaveDataType datatype)
{
  double llX = 0.0L, llY = 0.0L, urX = 0.0L, urY = 0.0L;
  Projection_t* projection = NULL;
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  RAVE_ASSERT((area != NULL), "area == NULL");

  Cartesian_setXScale(cartesian, Area_getXScale(area));
  Cartesian_setYScale(cartesian, Area_getYScale(area));
  projection = Area_getProjection(area);
  Cartesian_setProjection(cartesian, projection);
  Area_getExtent(area, &llX, &llY, &urX, &urY);
  Cartesian_setAreaExtent(cartesian, llX, llY, urX, urY);
  RAVE_OBJECT_RELEASE(projection);
  return Cartesian_createData(cartesian, Area_getXSize(area), Area_getYSize(area), datatype);
}

RaveValueType Cartesian_getMean(Cartesian_t* cartesian, long x, long y, int N, double* v)
{
  RaveValueType xytype = RaveValueType_NODATA;
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");

  xytype = Cartesian_getValue(cartesian, x, y, v);
  if (xytype == RaveValueType_DATA) {
    long xk = 0, yk = 0;
    double sum = 0.0L;
    int pts = 0;
    int k = N/2;
    double value = 0.0L;

    for (yk = -k; yk < k; yk++) {
      for (xk = -k; xk < k; xk++) {
        xytype = Cartesian_getValue(cartesian, xk + x, yk + y, &value);
        if (xytype == RaveValueType_DATA) {
          sum += value;
          pts++;
        }
      }
    }
    *v = sum / (double)pts; // we have at least 1 at pts so division by zero will not occur
  }

  return xytype;
}

int Cartesian_isTransformable(Cartesian_t* cartesian)
{
  int result = 0;
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");

  if (RaveData2D_hasData(cartesian->data) &&
      cartesian->projection != NULL &&
      cartesian->xscale > 0 &&
      cartesian->yscale > 0) {
    result = 1;
  }
  return result;
}

int Cartesian_addAttribute(Cartesian_t* cartesian, RaveAttribute_t* attribute)
{
  const char* name = NULL;
  char* aname = NULL;
  char* gname = NULL;
  int result = 0;
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  RAVE_ASSERT((attribute != NULL), "attribute == NULL");

  name = RaveAttribute_getName(attribute);
  if (name != NULL) {
    if (!RaveAttributeHelp_extractGroupAndName(name, &gname, &aname)) {
      RAVE_ERROR1("Failed to extract group and name from %s", name);
      goto done;
    }
    if (strcasecmp("how", gname)==0 &&
      strchr(aname, '/') == NULL) {
      result = RaveObjectHashTable_put(cartesian->attrs, name, (RaveCoreObject*)attribute);
    } else if (strcasecmp("what/prodpar", name)==0) {
      result = RaveObjectHashTable_put(cartesian->attrs, name, (RaveCoreObject*)attribute);
    } else {
      RAVE_WARNING1("You are not allowed to add dynamic attributes in other groups than 'how': '%s'", name);
      goto done;
    }
  }

done:
  RAVE_FREE(aname);
  RAVE_FREE(gname);
  return result;
}

RaveAttribute_t* Cartesian_getAttribute(Cartesian_t* cartesian, const char* name)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  if (name == NULL) {
    RAVE_ERROR0("Trying to get an attribute with NULL name");
    return NULL;
  }
  return (RaveAttribute_t*)RaveObjectHashTable_get(cartesian->attrs, name);
}

RaveList_t* Cartesian_getAttributeNames(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  return RaveObjectHashTable_keys(cartesian->attrs);
}

RaveObjectList_t* Cartesian_getAttributeValues(Cartesian_t* cartesian)
{
  RaveObjectList_t* result = NULL;
  RaveObjectList_t* attrs = NULL;

  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");

  attrs = RaveObjectHashTable_values(cartesian->attrs);
  if (attrs == NULL) {
    goto error;
  }
  result = RAVE_OBJECT_CLONE(attrs);
error:
  RAVE_OBJECT_RELEASE(attrs);
  return result;
}

int Cartesian_hasAttribute(Cartesian_t* cartesian, const char* name)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  return RaveObjectHashTable_exists(cartesian->attrs, name);
}

/*@} End of Interface functions */

RaveCoreObjectType Cartesian_TYPE = {
    "Cartesian",
    sizeof(Cartesian_t),
    Cartesian_constructor,
    Cartesian_destructor,
    Cartesian_copyconstructor
};

