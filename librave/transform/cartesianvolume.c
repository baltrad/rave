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
#include "cartesianvolume.h"
#include "area.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_datetime.h"
#include "rave_data2d.h"
#include "raveobject_hashtable.h"
#include "rave_utilities.h"
#include <string.h>

/**
 * Represents the cartesian volume
 */
struct _CartesianVolume_t {
  RAVE_OBJECT_HEAD /** Always on top */
  Rave_ObjectType type; /**< what type of cartesian volume this is, COMP, CVOL or anything else */
  RaveDateTime_t* datetime;  /**< the date and time */
  char* source;              /**< where does this data come from */
  RaveObjectList_t* images;  /**< the list of images */
  RaveObjectHashTable_t* attrs; /**< attributes */
  Projection_t* projection;     /**< this volumes projection definition */
  double xscale;                /**< x scale */
  double yscale;                /**< y scale */
  double llX;                   /**< lower left x-coordinate */
  double llY;                   /**< lower left y-coordinate */
  double urX;                   /**< upper right x-coordinate */
  double urY;                   /**< upper right x-coordinate */
  long xsize;                   /**< xsize */
  long ysize;                   /**< ysize */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int CartesianVolume_constructor(RaveCoreObject* obj)
{
  CartesianVolume_t* this = (CartesianVolume_t*)obj;
  this->type = Rave_ObjectType_CVOL;
  this->source = NULL;
  this->datetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  this->images = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  this->attrs = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  this->projection = NULL;
  this->xscale = 0.0;
  this->yscale = 0.0;
  this->xsize = 0;
  this->ysize = 0;
  this->llX = 0.0;
  this->llY = 0.0;
  this->urX = 0.0;
  this->urY = 0.0;
  if (this->datetime == NULL || this->images == NULL || this->attrs == NULL) {
    goto error;
  }
  return 1;
error:
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_OBJECT_RELEASE(this->images);
  RAVE_OBJECT_RELEASE(this->attrs);
  return 0;
}

/**
 * Copy constructor.
 */
static int CartesianVolume_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  CartesianVolume_t* this = (CartesianVolume_t*)obj;
  CartesianVolume_t* src = (CartesianVolume_t*)srcobj;
  this->type = src->type;
  this->source = NULL;
  this->datetime = RAVE_OBJECT_CLONE(src->datetime);
  this->images = RAVE_OBJECT_CLONE(src->images);
  this->attrs = RAVE_OBJECT_CLONE(src->attrs);
  this->xscale = src->xscale;
  this->yscale = src->yscale;
  this->xsize = src->xsize;
  this->ysize = src->ysize;
  this->llX = src->llX;
  this->llY = src->llY;
  this->urX = src->urX;
  this->urY = src->urY;
  if (this->datetime == NULL || this->images == NULL || this->attrs == NULL) {
    goto error;
  }
  if (src->projection != NULL) {
    this->projection = RAVE_OBJECT_CLONE(src->projection);
    if (this->projection == NULL) {
      goto error;
    }
  }
  if (!CartesianVolume_setSource(this, CartesianVolume_getSource(src))) {
    goto error;
  }
  return 1;
error:
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_OBJECT_RELEASE(this->images);
  RAVE_FREE(this->source);
  RAVE_OBJECT_RELEASE(this->attrs);
  return 0;
}

/**
 * Destroys the cartesian product
 * @param[in] scan - the cartesian product to destroy
 */
static void CartesianVolume_destructor(RaveCoreObject* obj)
{
  CartesianVolume_t* this = (CartesianVolume_t*)obj;
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_OBJECT_RELEASE(this->images);
  RAVE_OBJECT_RELEASE(this->attrs);
  RAVE_FREE(this->source);
  RAVE_OBJECT_RELEASE(this->projection);
}

static int CartesianVolumeInternal_setProjectionDefinition(CartesianVolume_t* cvol, const char* projdef)
{
  int result = 0;
  Projection_t* projection = NULL;

  RAVE_ASSERT((cvol != NULL), "cvol == NULL");

  projection = RAVE_OBJECT_NEW(&Projection_TYPE);
  if (projection == NULL) {
    RAVE_ERROR0("Could not create projection");
    goto error;
  }

  if (!Projection_init(projection, "raveio-projection", "autoloaded projection", projdef)) {
    RAVE_ERROR0("Could not initialize projection");
    goto error;
  }

  CartesianVolume_setProjection(cvol, projection);

  result = 1;
error:
  RAVE_OBJECT_RELEASE(projection);
  return result;
}
/*@} End of Private functions */

/*@{ Interface functions */
int CartesianVolume_setTime(CartesianVolume_t* cvol, const char* value)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  return RaveDateTime_setTime(cvol->datetime, value);
}

const char* CartesianVolume_getTime(CartesianVolume_t* cvol)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  return RaveDateTime_getTime(cvol->datetime);
}

int CartesianVolume_setDate(CartesianVolume_t* cvol, const char* value)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  return RaveDateTime_setDate(cvol->datetime, value);
}

const char* CartesianVolume_getDate(CartesianVolume_t* cvol)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  return RaveDateTime_getDate(cvol->datetime);
}

int CartesianVolume_setSource(CartesianVolume_t* cvol, const char* value)
{
  char* tmp = NULL;
  int result = 0;
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  if (value != NULL) {
    tmp = RAVE_STRDUP(value);
    if (tmp != NULL) {
      RAVE_FREE(cvol->source);
      cvol->source = tmp;
      tmp = NULL;
      result = 1;
    }
  } else {
    RAVE_FREE(cvol->source);
    result = 1;
  }
  return result;
}

const char* CartesianVolume_getSource(CartesianVolume_t* cvol)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  return (const char*)cvol->source;
}

int CartesianVolume_setObjectType(CartesianVolume_t* cvol, Rave_ObjectType type)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  if (type == Rave_ObjectType_CVOL || type == Rave_ObjectType_COMP) {
    cvol->type = type;
    return 1;
  }
  return 0;
}

Rave_ObjectType CartesianVolume_getObjectType(CartesianVolume_t* cvol)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  return cvol->type;
}

void CartesianVolume_setProjection(CartesianVolume_t* cvol, Projection_t* projection)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  RAVE_OBJECT_RELEASE(cvol->projection);
  if (projection != NULL) {
    cvol->projection = RAVE_OBJECT_COPY(projection);
  }
}

Projection_t* CartesianVolume_getProjection(CartesianVolume_t* cvol)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  return RAVE_OBJECT_COPY(cvol->projection);
}

void CartesianVolume_setXScale(CartesianVolume_t* cvol, double xscale)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  cvol->xscale = xscale;
}

double CartesianVolume_getXScale(CartesianVolume_t* cvol)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  return cvol->xscale;
}

void CartesianVolume_setYScale(CartesianVolume_t* cvol, double yscale)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  cvol->yscale = yscale;
}

double CartesianVolume_getYScale(CartesianVolume_t* cvol)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  return cvol->yscale;
}

long CartesianVolume_getXSize(CartesianVolume_t* cvol)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  return cvol->xsize;
}

long CartesianVolume_getYSize(CartesianVolume_t* cvol)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  return cvol->ysize;
}

void CartesianVolume_setAreaExtent(CartesianVolume_t* cvol, double llX, double llY, double urX, double urY)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  cvol->llX = llX;
  cvol->llY = llY;
  cvol->urX = urX;
  cvol->urY = urY;
}

void CartesianVolume_getAreaExtent(CartesianVolume_t* cvol, double* llX, double* llY, double* urX, double* urY)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  if (llX != NULL) {
    *llX = cvol->llX;
  }
  if (llY != NULL) {
    *llY = cvol->llY;
  }
  if (urX != NULL) {
    *urX = cvol->urX;
  }
  if (urY != NULL) {
    *urY = cvol->urY;
  }
}

int CartesianVolume_addImage(CartesianVolume_t* cvol, Cartesian_t* image)
{
  int result = 0;
  Projection_t* projection = NULL;

  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  RAVE_ASSERT((image != NULL), "image == NULL");

  if ((cvol->xsize != 0 && Cartesian_getXSize(image) != cvol->xsize) ||
      (cvol->ysize != 0 && Cartesian_getYSize(image) != cvol->ysize)) {
    RAVE_ERROR0("Conflicting sizes between cartesian images in volume\n");
    goto done;
  }

  if (RaveObjectList_add(cvol->images, (RaveCoreObject*)image)) {
    cvol->xsize = Cartesian_getXSize(image);
    cvol->ysize = Cartesian_getYSize(image);
    result = 1;
  }

  if (result == 1) {
    double llX=0.0, llY=0.0, urX=0.0, urY=0.0;
    // Adjust the image to get properties from this volume if missing..
    projection = Cartesian_getProjection(image);
    if (projection == NULL && cvol->projection != NULL) {
      Cartesian_setProjection(image, cvol->projection);
    }
    RAVE_OBJECT_RELEASE(projection);

    if (Cartesian_getXScale(image)==0.0) {
      Cartesian_setXScale(image, cvol->xscale);
    }

    if (Cartesian_getYScale(image)==0.0) {
      Cartesian_setYScale(image, cvol->yscale);
    }

    if (Cartesian_getDate(image) == NULL || Cartesian_getTime(image) == NULL) {
      RaveAttribute_t* stime = Cartesian_getAttribute(image, "what/starttime");
      RaveAttribute_t* sdate = Cartesian_getAttribute(image, "what/startdate");
      if (stime != NULL && sdate != NULL) {
        if (Cartesian_getTime(image) == NULL) {
          char* value = NULL;
          if (RaveAttribute_getString(stime, &value)) {
            Cartesian_setTime(image, value);
          }
        }
        if (Cartesian_getDate(image) == NULL) {
          char* value = NULL;
          if (RaveAttribute_getString(sdate, &value)) {
            Cartesian_setDate(image, value);
          }
        }
      } else {
        Cartesian_setTime(image, CartesianVolume_getTime(cvol));
        Cartesian_setDate(image, CartesianVolume_getDate(cvol));
      }
      RAVE_OBJECT_RELEASE(stime);
      RAVE_OBJECT_RELEASE(sdate);
    }

    if (Cartesian_getSource(image) == NULL) {
      Cartesian_setSource(image, CartesianVolume_getSource(cvol));
    }

    Cartesian_getAreaExtent(image, &llX, &llY, &urX, &urY);
    if (llX == 0.0 && llY == 0.0 && urX == 0.0 && urY == 0.0) {
      Cartesian_setAreaExtent(image, cvol->llX, cvol->llY, cvol->urX, cvol->urY);
    }
  }

done:
  return result;
}

Cartesian_t* CartesianVolume_getImage(CartesianVolume_t* cvol, int index)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  return (Cartesian_t*)RaveObjectList_get(cvol->images, index);
}

int CartesianVolume_getNumberOfImages(CartesianVolume_t* cvol)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  return RaveObjectList_size(cvol->images);
}

int CartesianVolume_addAttribute(CartesianVolume_t* cvol, RaveAttribute_t* attribute)
{
  const char* name = NULL;
  char* aname = NULL;
  char* gname = NULL;
  int result = 0;
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  RAVE_ASSERT((attribute != NULL), "attribute == NULL");

  name = RaveAttribute_getName(attribute);
  if (name != NULL) {
    if (strcasecmp("what/date", name)==0 ||
        strcasecmp("what/time", name)==0 ||
        strcasecmp("what/source", name)==0 ||
        strcasecmp("where/projdef", name)==0) {
      char* value = NULL;
      if (!RaveAttribute_getString(attribute, &value)) {
        RAVE_ERROR1("Failed to extract %s as a string", name);
        goto done;
      }
      if (strcasecmp("what/date", name)==0) {
        result = CartesianVolume_setDate(cvol, value);
      } else if (strcasecmp("what/time", name)==0) {
        result = CartesianVolume_setTime(cvol, value);
      } else if (strcasecmp("what/source", name)==0) {
        result = CartesianVolume_setSource(cvol, value);
      } else if (strcasecmp("where/projdef", name)==0) {
        result = CartesianVolumeInternal_setProjectionDefinition(cvol, value);
      }
    } else if (strcasecmp("where/xscale", name)==0 ||
               strcasecmp("where/yscale", name)==0) {
      double value = 0.0;
      result = RaveAttribute_getDouble(attribute, &value);
      if (strcasecmp("where/xscale", name)==0) {
        CartesianVolume_setXScale(cvol, value);
      } else if (strcasecmp("where/yscale", name)==0) {
        CartesianVolume_setYScale(cvol, value);
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
        result = RaveObjectHashTable_put(cvol->attrs, name, (RaveCoreObject*)attribute);
      }
    }

    // Verify if it is possible to generate the area extent.
    if (strcasecmp("where/LL_lon", name)==0 ||
        strcasecmp("where/LL_lat", name)==0 ||
        strcasecmp("where/UR_lon", name)==0 ||
        strcasecmp("where/UR_lat", name)==0 ||
        strcasecmp("where/projdef", name)==0) {
      if (RaveObjectHashTable_exists(cvol->attrs, "where/LL_lon") &&
          RaveObjectHashTable_exists(cvol->attrs, "where/LL_lat") &&
          RaveObjectHashTable_exists(cvol->attrs, "where/UR_lon") &&
          RaveObjectHashTable_exists(cvol->attrs, "where/UR_lat") &&
          cvol->projection != NULL) {
        double LL_lon = 0.0, LL_lat = 0.0, UR_lon = 0.0, UR_lat = 0.0;
        result = 0; /* reset result to 0 again since we need to be able to create an extent */

        if (RaveUtilities_getRaveAttributeDoubleFromHash(cvol->attrs, "where/LL_lon", &LL_lon) &&
            RaveUtilities_getRaveAttributeDoubleFromHash(cvol->attrs, "where/LL_lat", &LL_lat) &&
            RaveUtilities_getRaveAttributeDoubleFromHash(cvol->attrs, "where/UR_lon", &UR_lon) &&
            RaveUtilities_getRaveAttributeDoubleFromHash(cvol->attrs, "where/UR_lat", &UR_lat)) {
          double llX = 0.0L, llY = 0.0L, urX = 0.0L, urY = 0.0;
          if (!Projection_fwd(cvol->projection, LL_lon * M_PI/180.0, LL_lat * M_PI/180.0, &llX, &llY)) {
            RAVE_ERROR0("Could not generate XY pair for LL");
            goto done;
          }

          if (!Projection_fwd(cvol->projection, UR_lon * M_PI/180.0, UR_lat * M_PI/180.0, &urX, &urY)) {
            RAVE_ERROR0("Could not generate XY pair for UR");
            goto done;
          }
          result = 1;
          CartesianVolume_setAreaExtent(cvol, llX, llY, urX, urY);
        }
      }
    }
  }

done:
  RAVE_FREE(aname);
  RAVE_FREE(gname);
  return result;
}

RaveAttribute_t* CartesianVolume_getAttribute(CartesianVolume_t* cvol,  const char* name)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  if (name == NULL) {
    RAVE_ERROR0("Trying to get an attribute with NULL name");
    return NULL;
  }
  return (RaveAttribute_t*)RaveObjectHashTable_get(cvol->attrs, name);
}

RaveList_t* CartesianVolume_getAttributeNames(CartesianVolume_t* cvol)
{
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  return RaveObjectHashTable_keys(cvol->attrs);
}

RaveObjectList_t* CartesianVolume_getAttributeValues(CartesianVolume_t* cvol)
{
  RaveObjectList_t* result = NULL;
  RaveObjectList_t* tableattrs = NULL;

  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  tableattrs = RaveObjectHashTable_values(cvol->attrs);
  if (tableattrs == NULL) {
    goto error;
  }
  result = RAVE_OBJECT_CLONE(tableattrs);
  if (result == NULL) {
    goto error;
  }
  if (!RaveUtilities_addStringAttributeToList(result, "what/date", CartesianVolume_getDate(cvol)) ||
      !RaveUtilities_addStringAttributeToList(result, "what/time", CartesianVolume_getTime(cvol)) ||
      !RaveUtilities_addStringAttributeToList(result, "what/source", CartesianVolume_getSource(cvol)) ||
      !RaveUtilities_addDoubleAttributeToList(result, "where/xscale", CartesianVolume_getXScale(cvol)) ||
      !RaveUtilities_addDoubleAttributeToList(result, "where/yscale", CartesianVolume_getYScale(cvol)) ||
      !RaveUtilities_replaceLongAttributeInList(result, "where/xsize", CartesianVolume_getXSize(cvol)) ||
      !RaveUtilities_replaceLongAttributeInList(result, "where/ysize", CartesianVolume_getYSize(cvol))) {
    goto error;
  }

  // Add projection + extent if possible
  if (cvol->projection != NULL) {
    if (!RaveUtilities_addStringAttributeToList(result, "where/projdef", Projection_getDefinition(cvol->projection))) {
      goto error;
    }
    if (!CartesianHelper_addLonLatExtentToAttributeList(result, cvol->projection, cvol->llX, cvol->llY, cvol->urX, cvol->urY)) {
      goto error;
    }
  }

  RAVE_OBJECT_RELEASE(tableattrs);
  return result;
error:
  RAVE_OBJECT_RELEASE(result);
  RAVE_OBJECT_RELEASE(tableattrs);
  return NULL;
}

int CartesianVolume_isValid(CartesianVolume_t* cvol)
{
  int result = 0;
  int ncartesians = 0;
  int i = 0;
  RAVE_ASSERT((cvol != NULL), "cvol == NULL");
  if (CartesianVolume_getDate(cvol) == NULL ||
      CartesianVolume_getTime(cvol) == NULL ||
      CartesianVolume_getSource(cvol) == NULL) {
    RAVE_INFO0("date, time and source MUST be defined");
    goto done;
  }

  ncartesians = RaveObjectList_size(cvol->images);
  if (ncartesians <= 0) {
    RAVE_INFO0("a cartesian volume must at least contains one product");
    goto done;
  }

  result = 1; // on error, let result become 0 and hence break the loop
  for (i = 0; result == 1 && i < ncartesians; i++) {
    Cartesian_t* cartesian = (Cartesian_t*)RaveObjectList_get(cvol->images, i);
    result = Cartesian_isValid(cartesian, cvol->type);
    RAVE_OBJECT_RELEASE(cartesian);
  }

  result = 1;
done:
  return result;
}

/*@} End of Interface functions */

RaveCoreObjectType CartesianVolume_TYPE = {
    "CartesianVolume",
    sizeof(CartesianVolume_t),
    CartesianVolume_constructor,
    CartesianVolume_destructor,
    CartesianVolume_copyconstructor
};
