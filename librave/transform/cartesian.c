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
  char* source;              /**< where does this data come from */
  double gain;       /**< gain when scaling */
  double offset;     /**< offset when scaling */
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
  this->gain = 0.0;
  this->offset = 0.0;
  this->nodata = 0.0;
  this->undetect = 0.0;
  this->projection = NULL;
  this->data = NULL;
  this->datetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  this->data = RAVE_OBJECT_NEW(&RaveData2D_TYPE);
  this->attrs = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  if (this->datetime == NULL || this->data == NULL || this->attrs == NULL) {
    goto fail;
  }

  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->data);
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_OBJECT_RELEASE(this->attrs);
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
  this->data = NULL;

  Cartesian_setQuantity(this, Cartesian_getQuantity(src));

  this->datetime = RAVE_OBJECT_CLONE(src->datetime);
  this->data = RAVE_OBJECT_CLONE(src->data);
  this->attrs = RAVE_OBJECT_CLONE(src->attrs);

  if (this->datetime == NULL || this->data == NULL || this->attrs == NULL) {
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
    RAVE_FREE(cartesian->source);
    RAVE_FREE(cartesian->quantity);
    RAVE_OBJECT_RELEASE(cartesian->data);
    RAVE_OBJECT_RELEASE(cartesian->attrs);
  }
}

static int CartesianInternal_setProjectionDefinition(Cartesian_t* cartesian, const char* projdef)
{
  int result = 0;
  Projection_t* projection = NULL;

  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");

  projection = RAVE_OBJECT_NEW(&Projection_TYPE);
  if (projection == NULL) {
    RAVE_ERROR0("Could not create projection");
    goto error;
  }

  if (!Projection_init(projection, "raveio-projection", "autoloaded projection", projdef)) {
    RAVE_ERROR0("Could not initialize projection");
    goto error;
  }

  Cartesian_setProjection(cartesian, projection);

  result = 1;
error:
  RAVE_OBJECT_RELEASE(projection);
  return result;
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
  cartesian->gain = gain;
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
    if (strcasecmp("what/date", name)==0 ||
        strcasecmp("what/time", name)==0 ||
        strcasecmp("what/source", name)==0 ||
        strcasecmp("where/projdef", name)==0 ||
        strcasecmp("what/quantity", name)==0 ||
        strcasecmp("what/product", name)==0) {
      // Strings
      char* value = NULL;
      if (!RaveAttribute_getString(attribute, &value)) {
        RAVE_ERROR1("Failed to extract %s as a string", name);
        goto done;
      }
      if (strcasecmp("what/date", name)==0) {
        result = Cartesian_setDate(cartesian, value);
      } else if (strcasecmp("what/time", name)==0) {
        result = Cartesian_setTime(cartesian, value);
      } else if (strcasecmp("what/source", name)==0) {
        result = Cartesian_setSource(cartesian, value);
      } else if (strcasecmp("where/projdef", name)==0) {
        result = CartesianInternal_setProjectionDefinition(cartesian, value);
      } else if (strcasecmp("what/quantity", name)==0) {
        result = Cartesian_setQuantity(cartesian, value);
      } else if (strcasecmp("what/product", name)==0) {
        result = Cartesian_setProduct(cartesian, RaveTypes_getProductTypeFromString(value));
      }
    } else if (strcasecmp("what/gain", name)==0 ||
               strcasecmp("what/nodata", name)==0 ||
               strcasecmp("what/offset", name)==0 ||
               strcasecmp("what/undetect", name)==0 ||
               strcasecmp("where/xscale", name)==0 ||
               strcasecmp("where/yscale", name)==0) {
      /* Double values */
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR1("Failed to extract %s as double", name);
      }
      if (strcasecmp("what/gain", name)==0) {
        Cartesian_setGain(cartesian, value);
      } else if (strcasecmp("what/nodata", name)==0) {
        Cartesian_setNodata(cartesian, value);
      } else if (strcasecmp("what/offset", name)==0) {
        Cartesian_setOffset(cartesian, value);
      } else if (strcasecmp("what/undetect", name)==0) {
        Cartesian_setUndetect(cartesian, value);
      } else if (strcasecmp("where/xscale", name)==0) {
        Cartesian_setXScale(cartesian, value);
      } else if (strcasecmp("where/yscale", name)==0) {
        Cartesian_setYScale(cartesian, value);
      }
    } else {
      if (!RaveAttributeHelp_extractGroupAndName(name, &gname, &aname)) {
        RAVE_ERROR1("Failed to extract group and name from %s", name);
        goto done;
      }
      if ((strcasecmp("how", gname)==0 ||
           strcasecmp("what", gname)==0 ||
           strcasecmp("where", gname)==0) &&
        strchr(aname, '/') == NULL) {
        result = RaveObjectHashTable_put(cartesian->attrs, name, (RaveCoreObject*)attribute);
      }
    }

    // Verify if it is possible to generate the area extent.
    if (strcasecmp("where/LL_lon", name)==0 ||
        strcasecmp("where/LL_lat", name)==0 ||
        strcasecmp("where/UR_lon", name)==0 ||
        strcasecmp("where/UR_lat", name)==0 ||
        strcasecmp("where/projdef", name)==0) {
      if (RaveObjectHashTable_exists(cartesian->attrs, "where/LL_lon") &&
          RaveObjectHashTable_exists(cartesian->attrs, "where/LL_lat") &&
          RaveObjectHashTable_exists(cartesian->attrs, "where/UR_lon") &&
          RaveObjectHashTable_exists(cartesian->attrs, "where/UR_lat") &&
          cartesian->projection != NULL) {
        double LL_lon = 0.0, LL_lat = 0.0, UR_lon = 0.0, UR_lat = 0.0;
        result = 0; /* reset result to 0 again since we need to be able to create an extent */

        if (RaveUtilities_getRaveAttributeDoubleFromHash(cartesian->attrs, "where/LL_lon", &LL_lon) &&
            RaveUtilities_getRaveAttributeDoubleFromHash(cartesian->attrs, "where/LL_lat", &LL_lat) &&
            RaveUtilities_getRaveAttributeDoubleFromHash(cartesian->attrs, "where/UR_lon", &UR_lon) &&
            RaveUtilities_getRaveAttributeDoubleFromHash(cartesian->attrs, "where/UR_lat", &UR_lat)) {
          double llX = 0.0L, llY = 0.0L, urX = 0.0L, urY = 0.0;
          if (!Projection_fwd(cartesian->projection, LL_lon * M_PI/180.0, LL_lat * M_PI/180.0, &llX, &llY)) {
            RAVE_ERROR0("Could not generate XY pair for LL");
            goto done;
          }

          if (!Projection_fwd(cartesian->projection, UR_lon * M_PI/180.0, UR_lat * M_PI/180.0, &urX, &urY)) {
            RAVE_ERROR0("Could not generate XY pair for UR");
            goto done;
          }
          result = 1;
          Cartesian_setAreaExtent(cartesian, llX, llY, urX, urY);
        }
      }
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

RaveObjectList_t* Cartesian_getAttributeValues(Cartesian_t* cartesian, Rave_ObjectType otype)
{
  RaveObjectList_t* result = NULL;
  RaveObjectList_t* tableattrs = NULL;

  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");

  if (otype != Rave_ObjectType_COMP && otype != Rave_ObjectType_CVOL && otype != Rave_ObjectType_IMAGE) {
    RAVE_ERROR1("Can not get attribute values for object type = %s\n", RaveTypes_getStringFromObjectType(otype));
    goto error;
  }

  tableattrs = RaveObjectHashTable_values(cartesian->attrs);
  if (tableattrs == NULL) {
    goto error;
  }
  result = RAVE_OBJECT_CLONE(tableattrs);
  if (result == NULL) {
    goto error;
  }

  if (otype == Rave_ObjectType_COMP || otype == Rave_ObjectType_IMAGE) {
    if (cartesian->projection != NULL) {
      if (!RaveUtilities_addStringAttributeToList(result, "where/projdef", Projection_getDefinition(cartesian->projection))) {
        goto error;
      }
      if (!CartesianHelper_addLonLatExtentToAttributeList(result, cartesian->projection, cartesian->llX, cartesian->llY, cartesian->urX, cartesian->urY)) {
        goto error;
      }
    }

    if (!RaveUtilities_addStringAttributeToList(result, "what/date", Cartesian_getDate(cartesian)) ||
        !RaveUtilities_addStringAttributeToList(result, "what/time", Cartesian_getTime(cartesian)) ||
        !RaveUtilities_addStringAttributeToList(result, "what/source", Cartesian_getSource(cartesian)) ||
        !RaveUtilities_addDoubleAttributeToList(result, "where/xscale", Cartesian_getXScale(cartesian)) ||
        !RaveUtilities_addDoubleAttributeToList(result, "where/yscale", Cartesian_getYScale(cartesian)) ||
        !RaveUtilities_replaceLongAttributeInList(result, "where/xsize", Cartesian_getXSize(cartesian)) ||
        !RaveUtilities_replaceLongAttributeInList(result, "where/ysize", Cartesian_getYSize(cartesian))) {
      goto error;
    }

    // prodpar is dataset specific.. so it should only be there for images in volumes.
    RaveUtilities_removeAttributeFromList(result, "what/prodpar");

  } else if (otype == Rave_ObjectType_CVOL) {
    if (!RaveUtilities_addDoubleAttributeToList(result, "what/gain", Cartesian_getGain(cartesian)) ||
        !RaveUtilities_addDoubleAttributeToList(result, "what/nodata", Cartesian_getNodata(cartesian)) ||
        !RaveUtilities_addDoubleAttributeToList(result, "what/offset", Cartesian_getOffset(cartesian)) ||
        !RaveUtilities_addDoubleAttributeToList(result, "what/undetect", Cartesian_getUndetect(cartesian)) ||
        !RaveUtilities_addStringAttributeToList(result, "what/quantity", Cartesian_getQuantity(cartesian)) ||
        !RaveUtilities_replaceStringAttributeInList(result, "what/product",
                                                    RaveTypes_getStringFromProductType(Cartesian_getProduct(cartesian)))) {
      goto error;
    }
    if (RaveDateTime_getDate(cartesian->datetime) != NULL) {
      const char* dtdate = RaveDateTime_getDate(cartesian->datetime);
      if (!RaveObjectHashTable_exists(cartesian->attrs, "what/startdate") &&
          !RaveUtilities_replaceStringAttributeInList(result, "what/startdate", dtdate)) {
        goto error;
      }
      if (!RaveObjectHashTable_exists(cartesian->attrs, "what/enddate") &&
          !RaveUtilities_replaceStringAttributeInList(result, "what/enddate", dtdate)) {
        goto error;
      }
    }
    if (RaveDateTime_getTime(cartesian->datetime) != NULL) {
      const char* dttime = RaveDateTime_getTime(cartesian->datetime);
      if (!RaveObjectHashTable_exists(cartesian->attrs, "what/starttime") &&
          !RaveUtilities_replaceStringAttributeInList(result, "what/starttime", dttime)) {
        goto error;
      }
      if (!RaveObjectHashTable_exists(cartesian->attrs, "what/endtime") &&
          !RaveUtilities_replaceStringAttributeInList(result, "what/endtime", dttime)) {
        goto error;
      }
    }
  }


  RAVE_OBJECT_RELEASE(tableattrs);
  return result;
error:
  RAVE_OBJECT_RELEASE(result);
  RAVE_OBJECT_RELEASE(tableattrs);
  return NULL;
}

int Cartesian_hasAttribute(Cartesian_t* cartesian, const char* name)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  return RaveObjectHashTable_exists(cartesian->attrs, name);
}

/**
 * Validates that all nessecary attributes are set when the cartesian
 * is worked as a standalone product
 * @param[in] cartesian - self
 * @returns 1 if valid, otherwise 0
 */
static int CartesianInternal_isValidImage(Cartesian_t* cartesian)
{
  int result = 0;

  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");

  if (Cartesian_getDate(cartesian) == NULL ||
      Cartesian_getTime(cartesian) == NULL ||
      Cartesian_getSource(cartesian) == NULL) {
    RAVE_INFO0("Date, Time and Source must be set");
    goto done;
  }

  if (cartesian->projection == NULL) {
    RAVE_INFO0("Projection must be defined for cartesian");
    goto done;
  }

  if (Cartesian_getXSize(cartesian) == 0 ||
      Cartesian_getYSize(cartesian) == 0 ||
      Cartesian_getXScale(cartesian) == 0.0 ||
      Cartesian_getYScale(cartesian) == 0.0) {
    RAVE_INFO0("x/y sizes and scales must be defined");
    goto done;
  }

  if (Cartesian_getProduct(cartesian) == Rave_ProductType_UNDEFINED) {
    RAVE_INFO0("product type must be defined");
    goto done;
  }

  if (Cartesian_getQuantity(cartesian) == NULL) {
    RAVE_INFO0("Quantity must be defined");
    goto done;
  }

  if (Cartesian_getData(cartesian) == NULL) {
    RAVE_INFO0("Data must be set");
    goto done;
  }

  result = 1;
done:
  return result;
}

/**
 * Validates that all nessecary attributes are set when the cartesian
 * belongs to a volume.
 * @param[in] cartesian - self
 * @returns 1 if valid, otherwise 0
 */
static int CartesianInternal_isValidCvol(Cartesian_t* cartesian)
{
  int result = 0;

  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");

  // We must either have date & time or at least some startdate and starttime as
  // an attribute.
  if ((Cartesian_getDate(cartesian) == NULL ||
       Cartesian_getTime(cartesian) == NULL) &&
      (!Cartesian_hasAttribute(cartesian, "what/startdate") ||
       !Cartesian_hasAttribute(cartesian, "what/starttime"))) {
    RAVE_INFO0("Date, Time or what/startdate, what/starttime must be set");
    goto done;
  }
  if (Cartesian_getXSize(cartesian) == 0 ||
      Cartesian_getYSize(cartesian) == 0 ||
      Cartesian_getXScale(cartesian) == 0.0 ||
      Cartesian_getYScale(cartesian) == 0.0) {
    RAVE_INFO0("x/y sizes and scales must be defined");
    goto done;
  }

  if (Cartesian_getProduct(cartesian) == Rave_ProductType_UNDEFINED) {
    RAVE_INFO0("product type must be defined");
    goto done;
  }
  if (Cartesian_getQuantity(cartesian) == NULL) {
    RAVE_INFO0("Quantity must be defined");
    goto done;
  }
  if (Cartesian_getData(cartesian) == NULL) {
    RAVE_INFO0("Data must be set");
    goto done;
  }

  result = 1;
done:
  return result;
}

int Cartesian_isValid(Cartesian_t* cartesian, Rave_ObjectType otype)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  if (otype == Rave_ObjectType_IMAGE) {
    return CartesianInternal_isValidImage(cartesian);
  } else if (otype == Rave_ObjectType_COMP) {
    return CartesianInternal_isValidCvol(cartesian);
  } else if (otype == Rave_ObjectType_CVOL) {
    return CartesianInternal_isValidCvol(cartesian);
  } else {
    RAVE_INFO0("Cartesian does not support other than COMP, CVOL and IMAGE");
    return 0;
  }
}

int CartesianHelper_addLonLatExtentToAttributeList(RaveObjectList_t* list, Projection_t* projection, double llX, double llY, double urX, double urY)
{
  int result = 0;

  RAVE_ASSERT((list != NULL), "list == NULL");
  RAVE_ASSERT((projection != NULL), "projection == NULL");

  if (llX != 0.0 && llY != 0.0 && urX != 0.0 && urY != 0.0) {
    // Generate the correct corner coordinates.
    double LL_lat = 0.0, LL_lon = 0.0, LR_lat = 0.0, LR_lon = 0.0;
    double UL_lat = 0.0, UL_lon = 0.0, UR_lat = 0.0, UR_lon = 0.0;

    if (!Projection_inv(projection, llX, llY, &LL_lon, &LL_lat) ||
        !Projection_inv(projection, llX, urY, &UL_lon, &UL_lat) ||
        !Projection_inv(projection, urX, urY, &UR_lon, &UR_lat) ||
        !Projection_inv(projection, urX, llY, &LR_lon, &LR_lat)) {
      RAVE_ERROR0("Failed to translate surface extent into lon/lat corner pairs\n");
      goto done;
    }

    if (!RaveUtilities_replaceDoubleAttributeInList(list, "where/LL_lat", LL_lat * 180.0/M_PI) ||
        !RaveUtilities_replaceDoubleAttributeInList(list, "where/LL_lon", LL_lon * 180.0/M_PI) ||
        !RaveUtilities_replaceDoubleAttributeInList(list, "where/LR_lat", LR_lat * 180.0/M_PI) ||
        !RaveUtilities_replaceDoubleAttributeInList(list, "where/LR_lon", LR_lon * 180.0/M_PI) ||
        !RaveUtilities_replaceDoubleAttributeInList(list, "where/UL_lat", UL_lat * 180.0/M_PI) ||
        !RaveUtilities_replaceDoubleAttributeInList(list, "where/UL_lon", UL_lon * 180.0/M_PI) ||
        !RaveUtilities_replaceDoubleAttributeInList(list, "where/UR_lat", UR_lat * 180.0/M_PI) ||
        !RaveUtilities_replaceDoubleAttributeInList(list, "where/UR_lon", UR_lon * 180.0/M_PI)) {
      goto done;
    }
  }

  result = 1;
done:
  return result;
}


/*@} End of Interface functions */

RaveCoreObjectType Cartesian_TYPE = {
    "Cartesian",
    sizeof(Cartesian_t),
    Cartesian_constructor,
    Cartesian_destructor,
    Cartesian_copyconstructor
};

