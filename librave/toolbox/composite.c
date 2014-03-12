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
#include "raveobject_hashtable.h"

/**
 * Represents the cartesian product.
 */
struct _Composite_t {
  RAVE_OBJECT_HEAD /** Always on top */
  Rave_ProductType ptype; /**< the product type, default PCAPPI */
  CompositeSelectionMethod_t method; /**< selection method, default CompositeSelectionMethod_NEAREST */
  double height; /**< the height when generating pcapppi, cappi, pmax default 1000 */
  double elangle; /**< the elevation angle when generating ppi, default 0.0 */
  double range;  /*< the range when generating pmax, default = 500000 meters */
  RaveList_t* parameters; /**< the parameters to generate */
  RaveDateTime_t* datetime;  /**< the date and time */
  RaveObjectList_t* list;
  CompositeAlgorithm_t* algorithm; /**< the specific algorithm */
};

/**
 * Structure for keeping track on parameters that should be composited.
 */
typedef struct CompositingParameter_t {
  char* name;    /**< quantity */
  double gain;   /**< gain to be used in composite data*/
  double offset; /**< offset to be used in composite data*/
} CompositingParameter_t;

/**
 * Structure for keeping track on values / parameter
 */
typedef struct CompositeValues_t {
  RaveValueType vtype; /**< value type */
  double value;       /**< value */
  double mindist;     /**< min distance */
  double radardist;   /**< distance to radar */
  int radarindex;     /**< radar index in list of radars */
  PolarNavigationInfo navinfo; /**< navigation info for this parameter */
  const char* name;   /**< name of quantity */
  CartesianParam_t* parameter; /**< the cartesian parameter */
} CompositeValues_t;

/** The resolution to use for scaling the distance from pixel to used radar */
#define DISTANCE_TO_RADAR_RESOLUTION 2000.0

/** The name of the task for specifying distance to radar */
#define DISTANCE_TO_RADAR_HOW_TASK "se.smhi.composite.distance.radar"

/*@{ Private functions */
/**
 * Creates a parameter that should be composited
 * @param[in] name - quantity
 * @param[in] gain - gain
 * @param[in] offset - offset
 * @return the parameter or NULL on failure
 */
static CompositingParameter_t* CompositeInternal_createParameter(const char* name, double gain, double offset)
{
  CompositingParameter_t* result = NULL;
  if (name != NULL) {
    result = RAVE_MALLOC(sizeof(CompositingParameter_t));
    if (result != NULL) {
      result->name = RAVE_STRDUP(name);
      result->gain = gain;
      result->offset = offset;
      if (result->name == NULL) {
        RAVE_FREE(result);
        result = NULL;
      }
    }
  }
  return result;
}

/**
 * Frees the parameter including its members
 * @param[in] p - the parameter to release
 */
static void CompositeInternal_freeParameter(CompositingParameter_t* p)
{
  if (p != NULL) {
    RAVE_FREE(p->name);
    RAVE_FREE(p);
  }
}

/**
 * Releases the complete parameter list including its items
 * @param[in] p - a pointer to the parameter list
 */
static void CompositeInternal_freeParameterList(RaveList_t** p)
{
  if (p != NULL && *p != NULL) {
    CompositingParameter_t* cp = RaveList_removeLast(*p);
    while (cp != NULL) {
      CompositeInternal_freeParameter(cp);
      cp = RaveList_removeLast(*p);
    }
    RAVE_OBJECT_RELEASE(*p);
  }
}

/**
 * Clones a parameter
 * @param[in] p - parameter to clone
 * @return the clone or NULL on failure
 */
static CompositingParameter_t* CompositeInternal_cloneParameter(CompositingParameter_t* p)
{
  CompositingParameter_t* result = NULL;
  if (p != NULL) {
    result = RAVE_MALLOC(sizeof(CompositingParameter_t));
    if (result != NULL) {
      result->name = RAVE_STRDUP(p->name);
      result->gain = p->gain;
      result->offset = p->offset;
      if (result->name == NULL) {
        RAVE_FREE(result);
        result = NULL;
      }
    }
  }
  return result;
}

/**
 * Clones a parameter list
 * @param[in] p - the parameter list to clone
 * @return the clone or NULL on failure
 */
static RaveList_t* CompositeInternal_cloneParameterList(RaveList_t* p)
{
  int len = 0, i = 0;
  RaveList_t *result = NULL, *clone = NULL;;
  if (p != NULL) {
    clone = RAVE_OBJECT_NEW(&RaveList_TYPE);
    if (clone != NULL) {
      len = RaveList_size(p);
      for (i = 0; i < len; i++) {
        CompositingParameter_t* cp = RaveList_get(p, i);
        CompositingParameter_t* cpclone = CompositeInternal_cloneParameter(cp);
        if (cpclone == NULL || !RaveList_add(clone, cpclone)) {
          CompositeInternal_freeParameter(cpclone);
          goto done;
        }
      }
    }
  }

  result = RAVE_OBJECT_COPY(clone);
done:
  RAVE_OBJECT_RELEASE(clone);
  return result;
}

/**
 * Returns a pointer to the internall stored parameter in the composite.
 * @param[in] composite - composite
 * @param[in] quantity - the parameter
 * @return the found parameter or NULL if not found
 */
static CompositingParameter_t* CompositeInternal_getParameterByName(Composite_t* composite, const char* quantity)
{
  int len = 0, i = 0;
  if (quantity != NULL) {
    len = RaveList_size(composite->parameters);
    for (i = 0; i < len; i++) {
      CompositingParameter_t* cp = RaveList_get(composite->parameters, i);
      if (strcmp(cp->name, quantity) == 0) {
        return cp;
      }
    }
  }
  return NULL;
}

/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int Composite_constructor(RaveCoreObject* obj)
{
  Composite_t* this = (Composite_t*)obj;
  this->ptype = Rave_ProductType_PCAPPI;
  this->method = CompositeSelectionMethod_NEAREST;
  this->height = 1000.0;
  this->elangle = 0.0;
  this->range = 500000.0;
  this->parameters = NULL;
  this->algorithm = NULL;
  this->list = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  this->datetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  this->parameters = RAVE_OBJECT_NEW(&RaveList_TYPE);
  if (this->list == NULL || this->parameters == NULL || this->datetime == NULL) {
    goto error;
  }
  return 1;
error:
  CompositeInternal_freeParameterList(&this->parameters);
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
  this->method = src->method;
  this->height = src->height;
  this->elangle = src->elangle;
  this->range = src->range;
  this->algorithm = NULL;
  this->parameters = CompositeInternal_cloneParameterList(src->parameters);
  this->list = RAVE_OBJECT_CLONE(src->list);
  this->datetime = RAVE_OBJECT_CLONE(src->datetime);

  if (this->list == NULL || this->datetime == NULL || this->parameters == NULL) {
    goto error;
  }

  if (src->algorithm != NULL) {
    this->algorithm = RAVE_OBJECT_CLONE(src->algorithm);
    if (this->algorithm == NULL) {
      goto error;
    }
  }

  return 1;
error:
  CompositeInternal_freeParameterList(&this->parameters);
  RAVE_OBJECT_RELEASE(this->list);
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_OBJECT_RELEASE(this->algorithm);
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
  CompositeInternal_freeParameterList(&this->parameters);
  RAVE_OBJECT_RELEASE(this->algorithm);
}

/**
 * Creates an array of CompositeValues_t with length nparam.
 * @param[in] nparam - the number of items in the array
 * @returns the array on success or NULL on failure
 */
static CompositeValues_t* CompositeInternal_createCompositeValues(int nparam)
{
  CompositeValues_t* result = NULL;
  if (nparam > 0) {
    result = RAVE_MALLOC(sizeof(CompositeValues_t) * nparam);
    if (result == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for composite values");
    } else {
      memset(result, 0, sizeof(CompositeValues_t) * nparam);
    }
  }
  return result;
}

/**
 * Resets the array of composite values except the CartesianParam parameter
 * and the dToRadar field.
 * @param[in] nparam - number of parameters
 * @param[in] p - pointer at the array
 */
static void CompositeInternal_resetCompositeValues(Composite_t* composite, int nparam, CompositeValues_t* p)
{
  int i = 0;
  for (i = 0; i < nparam; i++) {
    p[i].mindist = 1e10;
    p[i].radarindex = -1;
    p[i].navinfo.ei = -1;
    p[i].navinfo.ai = -1;
    p[i].navinfo.ri = -1;
    p[i].vtype = RaveValueType_NODATA;
    p[i].name = (const char*)((CompositingParameter_t*)RaveList_get(composite->parameters, i))->name;
  }
}

/**
 * Returns the position that is closest to the specified lon/lat according to
 * the composites attributes like type/elevation/height/etc.
 * @param[in] composite - self
 * @param[in] object - the data object
 * @param[in] plon - the longitude
 * @param[in] plat - the latitude
 * @param[out] nav - the navigation information
 * @return 1 if hit or 0 if outside
 */
static int CompositeInternal_nearestPosition(
  Composite_t* composite,
  RaveCoreObject* object,
  double plon,
  double plat,
  PolarNavigationInfo* nav)
{
  int result = 0;

  RAVE_ASSERT((composite != NULL), "composite == NULL");
  RAVE_ASSERT((nav != NULL), "nav == NULL");
  if (object != NULL) {
    if (RAVE_OBJECT_CHECK_TYPE(object, &PolarScan_TYPE)) {
      if (composite->ptype == Rave_ProductType_PPI ||
          composite->ptype == Rave_ProductType_PCAPPI ||
          composite->ptype == Rave_ProductType_PMAX) {
        result = PolarScan_getNearestNavigationInfo((PolarScan_t*)object, plon, plat, nav);
      }
    } else if (RAVE_OBJECT_CHECK_TYPE(object, &PolarVolume_TYPE)) {
      if (composite->ptype == Rave_ProductType_PCAPPI ||
          composite->ptype == Rave_ProductType_CAPPI ||
          composite->ptype == Rave_ProductType_PMAX) {
        int insidee = (composite->ptype == Rave_ProductType_PCAPPI || composite->ptype == Rave_ProductType_PMAX)?0:1;
        result = PolarVolume_getNearestNavigationInfo((PolarVolume_t*)object,
                                                      plon,
                                                      plat,
                                                      Composite_getHeight(composite),
                                                      insidee,
                                                      nav);
      } else if (composite->ptype == Rave_ProductType_PPI) {
        PolarScan_t* scan = PolarVolume_getScanClosestToElevation((PolarVolume_t*)object,
                                                                  Composite_getElevationAngle(composite),
                                                                  0);
        if (scan == NULL) {
          RAVE_ERROR1("Failed to fetch scan nearest to elevation %g",
                      Composite_getElevationAngle(composite));
          goto done;
        }
        result = PolarScan_getNearestNavigationInfo((PolarScan_t*)scan, plon, plat, nav);
        nav->ei = PolarVolume_indexOf((PolarVolume_t*)object, scan);
        RAVE_OBJECT_RELEASE(scan);
      }
    }
  }

done:
  return result;
}

/**
 * Gets the value at the specified position for the specified quantity.
 * @param[in] composite - self
 * @param[in] obj - the object
 * @param[in] quantity - the quantity
 * @param[in] nav - the navigation information
 * @param[out] type - the value type
 * @param[out] value - the value
 * @return 1 on success or 0 if value not could be retrieved
 */
static int CompositeInternal_getValueAtPosition(
  Composite_t* composite,
  RaveCoreObject* obj,
  const char* quantity,
  PolarNavigationInfo* nav,
  RaveValueType* type,
  double* value)
{
  int result = 0;
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  RAVE_ASSERT((nav != NULL), "nav == NULL");
  RAVE_ASSERT((type != NULL), "type == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");

  if (obj != NULL) {
    if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
      *type = PolarScan_getConvertedParameterValue((PolarScan_t*)obj, quantity, nav->ri, nav->ai, value);
    } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
      *type = PolarVolume_getConvertedParameterValueAt((PolarVolume_t*)obj, quantity, nav->ei, nav->ri, nav->ai, value);
    } else {
      RAVE_WARNING0("Unsupported object type");
      goto done;
    }
  }

  result = 1;
done:
  return result;
}

/**
 * Returns the vertical max value for the specified quantity at the provided lon/lat position.
 * If no suitable value is found, vtype and vvalue will be left as is.
 *
 * @param[in] self - self
 * @param[in] radarindex - the index of the radar object in the composite list
 * @param[in] quantity - the parameter
 * @param[in] lon - longitude in radians
 * @param[in] lat - latitude in radians
 * @param[out] vtype - the value type (MUST NOT BE NULL)
 * @param[out] vvalue - the value (MUST NOT BE NULL)
 * @param[out] navinfo - the navigation information (MAY BE NULL)
 * @return 1 on success or 0 on failure.
 */
static int CompositeInternal_getVerticalMaxValue(
  Composite_t* self,
  int radarindex,
  const char* quantity,
  double lon,
  double lat,
  RaveValueType* vtype,
  double* vvalue,
  PolarNavigationInfo* navinfo)
{
  int result = 0;
  RaveCoreObject* obj = NULL;
  PolarNavigationInfo info;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((vtype != NULL), "vtype == NULL");
  RAVE_ASSERT((vvalue != NULL), "vvalue == NULL");

  obj = RaveObjectList_get(self->list, radarindex);
  if (obj == NULL) {
    goto done;
  }

  if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
    *vtype = PolarScan_getNearestConvertedParameterValue((PolarScan_t*)obj, quantity, lon, lat, vvalue, &info);
  } else {
    *vtype = PolarVolume_getConvertedVerticalMaxValue((PolarVolume_t*)obj, quantity, lon, lat, vvalue, &info);
  }

  if (navinfo != NULL) {
    *navinfo = info;
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(obj);
  return result;
}

/**
 * Adds quality flags to the composite.
 * @apram[in] self - self
 * @param[in] image - the image to add quality flags to
 * @param[in] qualityflags - a list of strings identifying the how/task value in the quality fields
 * @return 1 on success otherwise 0
 */
static int CompositeInternal_addQualityFlags(Composite_t* self, Cartesian_t* image, RaveList_t* qualityflags)
{
  int result = 0;
  int nqualityflags = 0;
  RaveField_t* field = NULL;
  RaveAttribute_t* howtaskattribute = NULL;
  RaveAttribute_t* gainattribute = NULL;
  CartesianParam_t* param = NULL;
  RaveList_t* paramNames = NULL;

  int xsize = 0, ysize = 0;
  int i = 0, j = 0;
  int nparam = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((image != NULL), "image == NULL");

  xsize = Cartesian_getXSize(image);
  ysize = Cartesian_getYSize(image);

  paramNames = Cartesian_getParameterNames(image);
  if (paramNames == NULL) {
    goto done;
  }

  nparam = RaveList_size(paramNames);

  if (qualityflags != NULL) {
    nqualityflags = RaveList_size(qualityflags);
  }

  for (i = 0; i < nqualityflags; i++) {
    char* howtaskvaluestr = (char*)RaveList_get(qualityflags, i);
    field = RAVE_OBJECT_NEW(&RaveField_TYPE);
    howtaskattribute = RaveAttributeHelp_createString("how/task", howtaskvaluestr);
    if (field == NULL ||
        howtaskattribute == NULL ||
        !RaveField_createData(field, xsize, ysize, RaveDataType_UCHAR) ||
        !RaveField_addAttribute(field, howtaskattribute)) {
      RAVE_ERROR0("Failed to create quality field");
      goto done;
    }
    if (strcmp(DISTANCE_TO_RADAR_HOW_TASK, howtaskvaluestr) == 0) {
      gainattribute = RaveAttributeHelp_createDouble("what/gain", DISTANCE_TO_RADAR_RESOLUTION);
      if (gainattribute == NULL ||
          !RaveField_addAttribute(field, gainattribute)) {
        RAVE_ERROR0("Failed to create gain attribute for quality field");
        goto done;
      }
      RAVE_OBJECT_RELEASE(gainattribute);
      gainattribute = RaveAttributeHelp_createDouble("what/offset", 0.0);
      if (gainattribute == NULL ||
          !RaveField_addAttribute(field, gainattribute)) {
        RAVE_ERROR0("Failed to create offset attribute for quality field");
        goto done;
      }
    }

    for (j = 0; j < nparam; j++) {
      param = Cartesian_getParameter(image, (const char*)RaveList_get(paramNames, j));
      if (param != NULL) {
        RaveField_t* cfield = RAVE_OBJECT_CLONE(field);
        if (cfield == NULL ||
            !CartesianParam_addQualityField(param, cfield)) {
          RAVE_OBJECT_RELEASE(cfield);
          RAVE_ERROR0("Failed to add quality field");
          goto done;
        }
        RAVE_OBJECT_RELEASE(cfield);
      }
      RAVE_OBJECT_RELEASE(param);
    }

    RAVE_OBJECT_RELEASE(field);
    RAVE_OBJECT_RELEASE(howtaskattribute);
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(field);
  RAVE_OBJECT_RELEASE(howtaskattribute);
  RAVE_OBJECT_RELEASE(gainattribute);
  RAVE_OBJECT_RELEASE(param);
  RaveList_freeAndDestroy(&paramNames);
  return result;
}

/**
 * Uses the navigation information and fills all associated cartesian quality
 * with the composite objects quality fields.
 * @param[in] composite - self
 * @param[in] x - x coordinate
 * @param[in] y - y coordinate
 * @param[in] radardist - distance to radar for this position
 * @param[in] radarindex - the object to use in the composite
 * @param[in] navinfo - the navigational information
 */
static void CompositeInternal_fillQualityInformation(
  Composite_t* composite,
  int x,
  int y,
  CartesianParam_t* param,
  double radardist,
  int radarindex,
  PolarNavigationInfo* navinfo)
{
  int nfields = 0, i = 0;
  const char* quantity;
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  RAVE_ASSERT((param != NULL), "param == NULL");
  RAVE_ASSERT((navinfo != NULL), "navinfo == NULL");

  nfields = CartesianParam_getNumberOfQualityFields(param);
  quantity = CartesianParam_getQuantity(param);

  for (i = 0; i < nfields; i++) {
    RaveField_t* field = NULL;
    RaveAttribute_t* attribute = NULL;
    char* name = NULL;
    double v = 0.0;

    field = CartesianParam_getQualityField(param, i);
    if (field != NULL) {
      attribute = RaveField_getAttribute(field, "how/task");
    }
    if (attribute != NULL) {
      RaveAttribute_getString(attribute, &name);
    }

    if (name != NULL) {
      RaveCoreObject* obj = RaveObjectList_get(composite->list, radarindex);
      if (obj != NULL) {
        if (strcmp(DISTANCE_TO_RADAR_HOW_TASK, name) == 0) {
          RaveField_setValue(field, x, y, radardist/DISTANCE_TO_RADAR_RESOLUTION);
        } else if (composite->algorithm != NULL && CompositeAlgorithm_supportsFillQualityInformation(composite->algorithm, name)) {
          // If the algorithm indicates that it is able to support the provided how/task field, then do so
          if (!CompositeAlgorithm_fillQualityInformation(composite->algorithm, obj, name, quantity, field, x, y, navinfo)) {
            RaveField_setValue(field, x, y, 0.0);
          }
        } else {
          if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
            if (navinfo->ei >= 0 && navinfo->ri >= 0 && navinfo->ai >= 0 &&
                PolarVolume_getQualityValueAt((PolarVolume_t*)obj, quantity, navinfo->ei, navinfo->ri, navinfo->ai, name, &v)) {
              RaveField_setValue(field, x, y, v);
            } else {
              RaveField_setValue(field, x, y, 0.0); // No data found
            }
          } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
            if (navinfo->ri >= 0 && navinfo->ai >= 0 &&
                PolarScan_getQualityValueAt((PolarScan_t*)obj, quantity, navinfo->ri, navinfo->ai, name, &v)) {
              RaveField_setValue(field, x, y , v);
            } else {
              RaveField_setValue(field, x, y, 0.0); // No data found
            }
          }
        }
      }
      RAVE_OBJECT_RELEASE(obj);
    }
    RAVE_OBJECT_RELEASE(field);
    RAVE_OBJECT_RELEASE(attribute);
  }
}

/**
 * Creates the resulting composite image.
 * @param[in] self - self
 * @param[in] area - the area the composite image(s) should have
 * @returns the cartesian on success otherwise NULL
 */
static Cartesian_t* CompositeInternal_createCompositeImage(Composite_t* self, Area_t* area)
{
  Cartesian_t *result = NULL, *cartesian = NULL;
  RaveAttribute_t* prodpar = NULL;
  int nparam = 0, i = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  cartesian = RAVE_OBJECT_NEW(&Cartesian_TYPE);
  if (cartesian == NULL) {
    goto done;
  }
  Cartesian_init(cartesian, area);

  nparam = Composite_getParameterCount(self);
  if (nparam <= 0) {
    RAVE_ERROR0("You can not generate a composite without specifying at least one parameter");
    goto done;
  }

  if (self->ptype == Rave_ProductType_CAPPI ||
      self->ptype == Rave_ProductType_PCAPPI) {
    prodpar = RaveAttributeHelp_createDouble("what/prodpar", self->height);
  } else if (self->ptype == Rave_ProductType_PMAX) {
    char s[256];
    sprintf(s, "%f,%f",self->height,self->range);
    prodpar = RaveAttributeHelp_createString("what/prodpar", s);
  } else {
    prodpar = RaveAttributeHelp_createDouble("what/prodpar", self->elangle * 180.0/M_PI);
  }
  if (prodpar == NULL) {
    goto done;
  }

  Cartesian_setObjectType(cartesian, Rave_ObjectType_COMP);
  Cartesian_setProduct(cartesian, self->ptype);
  if (!Cartesian_addAttribute(cartesian, prodpar)) {
    goto done;
  }
  if (Composite_getTime(self) != NULL) {
    if (!Cartesian_setTime(cartesian, Composite_getTime(self))) {
      goto done;
    }
  }
  if (Composite_getDate(self) != NULL) {
    if (!Cartesian_setDate(cartesian, Composite_getDate(self))) {
      goto done;
    }
  }
  if (!Cartesian_setSource(cartesian, Area_getID(area))) {
    goto done;
  }

  for (i = 0; i < nparam; i++) {
    double gain = 0.0, offset = 0.0;
    const char* name = Composite_getParameter(self, i, &gain, &offset);
    CartesianParam_t* cp = Cartesian_createParameter(cartesian, name, RaveDataType_UCHAR);
    if (cp == NULL) {
      goto done;
    }
    CartesianParam_setNodata(cp, 255.0);
    CartesianParam_setUndetect(cp, 0.0);
    CartesianParam_setGain(cp, gain);
    CartesianParam_setOffset(cp, offset);
    RAVE_OBJECT_RELEASE(cp);
  }

  result = RAVE_OBJECT_COPY(cartesian);
done:
  RAVE_OBJECT_RELEASE(cartesian);
  RAVE_OBJECT_RELEASE(prodpar);
  return result;
}

/**
 * Returns the projection object that belongs to this obj.
 * @param[in] obj - the rave core object instance
 * @return the projection or NULL if there is no projection instance
 */
static Projection_t* CompositeInternal_getProjection(RaveCoreObject* obj)
{
  Projection_t* result = NULL;
  if (obj != NULL) {
    if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
      result = PolarVolume_getProjection((PolarVolume_t*)obj);
    } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
      result = PolarScan_getProjection((PolarScan_t*)obj);
    }
  }
  return result;
}

static int CompositeInternal_getDistances(RaveCoreObject* obj, double lon, double lat, double* distance, double* maxdistance)
{
  int result = 0;
  RAVE_ASSERT((distance != NULL), "distance == NULL");
  RAVE_ASSERT((maxdistance != NULL), "maxdistance == NULL");
  if (obj != NULL) {
    if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
      *distance = PolarVolume_getDistance((PolarVolume_t*)obj, lon, lat);
      *maxdistance = PolarVolume_getMaxDistance((PolarVolume_t*)obj);
      result = 1;
    } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
      *distance = PolarScan_getDistance((PolarScan_t*)obj, lon, lat);
      *maxdistance = PolarScan_getMaxDistance((PolarScan_t*)obj);
      result = 1;
    }
  }
  return result;
}

/**
 * Pure max is a quite difference composite generator that does not care about proximity to ground
 * or radar or anything else. It only cares about maximum value at the specific position so we handle
 * this as a separate scheme instead of trying to mix into _nearest.
 *
 * This processing scheme does not support algorithm.process but it support algorithm.fillQualityInformation.
 *
 * @param[in] composite - self
 * @param[in] area - the area we are working with
 * @param[in] qualityflags - the quality flags we want to have set
 * @return the cartesian product
 */
static Cartesian_t* Composite_nearest_max(Composite_t* composite, Area_t* area, RaveList_t* qualityflags)
{
  Cartesian_t* result = NULL;
  Projection_t* projection = NULL;
  PolarNavigationInfo navinfo;
  CompositeValues_t* cvalues = NULL;
  int x = 0, y = 0, i = 0, xsize = 0, ysize = 0, nradars = 0;
  int nqualityflags = 0;
  int nparam = 0;

  RAVE_ASSERT((composite != NULL), "composite == NULL");
  if (area == NULL) {
    RAVE_ERROR0("Trying to generate composite with NULL area");
    goto fail;
  }

  nparam = Composite_getParameterCount(composite);
  if (nparam <= 0) {
    RAVE_ERROR0("You can not generate a composite without specifying at least one parameter");
    goto fail;
  }

  result = CompositeInternal_createCompositeImage(composite, area);
  if (result == NULL) {
    goto fail;
  }

  if ((cvalues = CompositeInternal_createCompositeValues(nparam)) == NULL) {
    goto fail;
  }

  for (i = 0; i < nparam; i++) {
    const char* name = Composite_getParameter(composite, i, NULL, NULL);
    cvalues[i].parameter = Cartesian_getParameter(result, name); // Keep track on parameters
    if (cvalues[i].parameter == NULL) {
      RAVE_ERROR0("Failure in parameter handling\n");
      goto fail;
    }
  }

  xsize = Cartesian_getXSize(result);
  ysize = Cartesian_getYSize(result);
  projection = Cartesian_getProjection(result);
  nradars = RaveObjectList_size(composite->list);

  if (qualityflags != NULL) {
    nqualityflags = RaveList_size(qualityflags);
    if (!CompositeInternal_addQualityFlags(composite, result, qualityflags)) {
      goto fail;
    }
  }

  if (composite->algorithm != NULL) {
    if (!CompositeAlgorithm_initialize(composite->algorithm, composite)) {
      goto fail;
    }
  }

  for (y = 0; y < ysize; y++) {
    double herey = Cartesian_getLocationY(result, y);
    for (x = 0; x < xsize; x++) {
      int cindex = 0;
      double herex = Cartesian_getLocationX(result, x);
      double olon = 0.0, olat = 0.0;

      CompositeInternal_resetCompositeValues(composite, nparam, cvalues);
      if (composite->algorithm != NULL) {
        CompositeAlgorithm_reset(composite->algorithm, x, y);
      }

      for (i = 0; i < nradars; i++) {
        RaveCoreObject* obj = NULL;
        Projection_t* objproj = NULL;

        obj = RaveObjectList_get(composite->list, i);
        if (obj != NULL) {
          objproj = CompositeInternal_getProjection(obj);
        }

        if (objproj != NULL) {
          /* We will go from surface coords into the lonlat projection assuming that a polar volume uses a lonlat projection*/
          if (!Projection_transformx(projection, objproj, herex, herey, 0.0, &olon, &olat, NULL)) {
            RAVE_WARNING0("Failed to transform from composite into polar coordinates");
          } else {
            double dist = 0.0;
            double maxdist = 0.0;

            // We only use distance & max distance to speed up processing but it isn't used for anything else
            // in the pure vertical max implementation.
            if (CompositeInternal_getDistances(obj, olon, olat, &dist, &maxdist) && dist <= maxdist) {
              for (cindex = 0; cindex < nparam; cindex++) {
                RaveValueType otype = RaveValueType_NODATA;
                double ovalue = 0.0;
                CompositeInternal_getVerticalMaxValue(composite, i, cvalues[cindex].name, olon, olat, &otype, &ovalue, &navinfo);
                if (otype == RaveValueType_DATA || otype == RaveValueType_UNDETECT) {
                  if ((cvalues[cindex].vtype != RaveValueType_DATA && cvalues[cindex].vtype != RaveValueType_UNDETECT) ||
                      (cvalues[cindex].vtype == RaveValueType_UNDETECT && otype == RaveValueType_DATA) ||
                      (cvalues[cindex].vtype == RaveValueType_DATA && otype == RaveValueType_DATA && ovalue > cvalues[cindex].value)) {
                    cvalues[cindex].vtype = otype;
                    cvalues[cindex].value = ovalue;
                    cvalues[cindex].mindist = dist;
                    cvalues[cindex].radardist = dist;
                    cvalues[cindex].radarindex = i;
                    cvalues[cindex].navinfo = navinfo;
                  }
                }
              }
            }
          }
        }
        RAVE_OBJECT_RELEASE(obj);
        RAVE_OBJECT_RELEASE(objproj);
      }

      for (cindex = 0; cindex < nparam; cindex++) {
        double vvalue = cvalues[cindex].value;
        double vtype = cvalues[cindex].vtype;
        if (vtype == RaveValueType_NODATA) {
          CartesianParam_setConvertedValue(cvalues[cindex].parameter, x, y, CartesianParam_getNodata(cvalues[cindex].parameter));
        } else {
          if (vtype == RaveValueType_UNDETECT) {
            CartesianParam_setConvertedValue(cvalues[cindex].parameter, x, y, CartesianParam_getUndetect(cvalues[cindex].parameter));
          } else {
            CartesianParam_setConvertedValue(cvalues[cindex].parameter, x, y, vvalue);
          }
        }
        if ((vtype == RaveValueType_DATA || vtype == RaveValueType_UNDETECT) &&
            cvalues[cindex].radarindex >= 0 && nqualityflags > 0) {
          PolarNavigationInfo info = cvalues[cindex].navinfo;
          CompositeInternal_fillQualityInformation(composite, x, y, cvalues[cindex].parameter,
                                                   cvalues[cindex].radardist, cvalues[cindex].radarindex, &info);
        }
      }
    }
  }

  for (i = 0; cvalues != NULL && i < nparam; i++) {
    RAVE_OBJECT_RELEASE(cvalues[i].parameter);
  }
  RAVE_FREE(cvalues);
  RAVE_OBJECT_RELEASE(projection);
  return result;
fail:
  for (i = 0; cvalues != NULL && i < nparam; i++) {
    RAVE_OBJECT_RELEASE(cvalues[i].parameter);
  }
  RAVE_FREE(cvalues);
  RAVE_OBJECT_RELEASE(projection);
  RAVE_OBJECT_RELEASE(result);
  return result;

}


/*@} End of Private functions */

/*@{ Interface functions */
int Composite_add(Composite_t* composite, RaveCoreObject* object)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  RAVE_ASSERT((object != NULL), "object == NULL");

  if (!RAVE_OBJECT_CHECK_TYPE(object, &PolarVolume_TYPE) &&
      !RAVE_OBJECT_CHECK_TYPE(object, &PolarScan_TYPE)) {
    RAVE_ERROR0("Providing an object that not is a PolarVolume nor a PolarScan during composite generation");
    return 0;
  }
  return RaveObjectList_add(composite->list, object);
}

int Composite_getNumberOfObjects(Composite_t* composite)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return RaveObjectList_size(composite->list);
}

RaveCoreObject* Composite_get(Composite_t* composite, int index)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return RaveObjectList_get(composite->list, index);
}

void Composite_setProduct(Composite_t* composite, Rave_ProductType type)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  if (type == Rave_ProductType_PCAPPI ||
      type == Rave_ProductType_CAPPI ||
      type == Rave_ProductType_PPI ||
      type == Rave_ProductType_PMAX ||
      type == Rave_ProductType_MAX) {
    composite->ptype = type;
  } else {
    RAVE_ERROR0("Only supported algorithms are PPI, CAPPI, PCAPPI, PMAX and MAX");
  }
}

Rave_ProductType Composite_getProduct(Composite_t* composite)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return composite->ptype;
}

int Composite_setSelectionMethod(Composite_t* self, CompositeSelectionMethod_t method)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (method >= CompositeSelectionMethod_NEAREST && method <= CompositeSelectionMethod_HEIGHT) {
    self->method = method;
    result = 1;
  }
  return result;
}

CompositeSelectionMethod_t Composite_getSelectionMethod(Composite_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->method;
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

void Composite_setElevationAngle(Composite_t* composite, double angle)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  composite->elangle = angle;
}

double Composite_getElevationAngle(Composite_t* composite)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return composite->elangle;
}

void Composite_setRange(Composite_t* self, double range)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->range = range;
}

double Composite_getRange(Composite_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->range;
}

int Composite_addParameter(Composite_t* composite, const char* quantity, double gain, double offset)
{
  int result = 0;
  CompositingParameter_t* param = NULL;

  RAVE_ASSERT((composite != NULL), "composite == NULL");

  param = CompositeInternal_getParameterByName(composite, quantity);
  if (param != NULL) {
    param->gain = gain;
    param->offset = offset;
    result = 1;
  } else {
    param = CompositeInternal_createParameter(quantity, gain, offset);
    if (param != NULL) {
      result = RaveList_add(composite->parameters, param);
      if (!result) {
        CompositeInternal_freeParameter(param);
      }
    }
  }
  return result;
}

int Composite_hasParameter(Composite_t* composite, const char* quantity)
{
  int result = 0;
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  if (quantity != NULL) {
    int i = 0;
    int len = RaveList_size(composite->parameters);
    for (i = 0; result == 0 && i < len ; i++) {
      CompositingParameter_t* s = RaveList_get(composite->parameters, i);
      if (s != NULL && s->name != NULL && strcmp(quantity, s->name) == 0) {
        result = 1;
      }
    }
  }
  return result;
}

int Composite_getParameterCount(Composite_t* composite)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return RaveList_size(composite->parameters);
}

const char* Composite_getParameter(Composite_t* composite, int index, double* gain, double* offset)
{
  CompositingParameter_t* param = NULL;
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  param = RaveList_get(composite->parameters, index);
  if (param != NULL) {
    if (gain != NULL) {
      *gain = param->gain;
    }
    if (offset != NULL) {
      *offset = param->offset;
    }
    return (const char*)param->name;
  }
  return NULL;
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

Cartesian_t* Composite_nearest(Composite_t* composite, Area_t* area, RaveList_t* qualityflags)
{
  Cartesian_t* result = NULL;
  Projection_t* projection = NULL;
  PolarNavigationInfo navinfo;
  CompositeValues_t* cvalues = NULL;
  int x = 0, y = 0, i = 0, xsize = 0, ysize = 0, nradars = 0;
  int nqualityflags = 0;
  int nparam = 0;

  RAVE_ASSERT((composite != NULL), "composite == NULL");

  if (composite->ptype == Rave_ProductType_MAX) { // Special handling of the max algorithm.
    return Composite_nearest_max(composite, area, qualityflags);
  }

  if (area == NULL) {
    RAVE_ERROR0("Trying to generate composite with NULL area");
    goto fail;
  }

  nparam = Composite_getParameterCount(composite);
  if (nparam <= 0) {
    RAVE_ERROR0("You can not generate a composite without specifying at least one parameter");
    goto fail;
  }

  result = CompositeInternal_createCompositeImage(composite, area);
  if (result == NULL) {
    goto fail;
  }

  if ((cvalues = CompositeInternal_createCompositeValues(nparam)) == NULL) {
    goto fail;
  }

  for (i = 0; i < nparam; i++) {
    const char* name = Composite_getParameter(composite, i, NULL, NULL);
    cvalues[i].parameter = Cartesian_getParameter(result, name); // Keep track on parameters
    if (cvalues[i].parameter == NULL) {
      RAVE_ERROR0("Failure in parameter handling\n");
      goto fail;
    }
  }

  xsize = Cartesian_getXSize(result);
  ysize = Cartesian_getYSize(result);
  projection = Cartesian_getProjection(result);
  nradars = RaveObjectList_size(composite->list);

  if (qualityflags != NULL) {
    nqualityflags = RaveList_size(qualityflags);
    if (!CompositeInternal_addQualityFlags(composite, result, qualityflags)) {
      goto fail;
    }
  }

  if (composite->algorithm != NULL) {
    if (!CompositeAlgorithm_initialize(composite->algorithm, composite)) {
      goto fail;
    }
  }

  for (y = 0; y < ysize; y++) {
    double herey = Cartesian_getLocationY(result, y);
    for (x = 0; x < xsize; x++) {
      int cindex = 0;
      double herex = Cartesian_getLocationX(result, x);
      double olon = 0.0, olat = 0.0;

      CompositeInternal_resetCompositeValues(composite, nparam, cvalues);
      if (composite->algorithm != NULL) {
        CompositeAlgorithm_reset(composite->algorithm, x, y);
      }

      for (i = 0; i < nradars; i++) {
        RaveCoreObject* obj = NULL;
        Projection_t* objproj = NULL;

        obj = RaveObjectList_get(composite->list, i);
        if (obj != NULL) {
          if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
            objproj = PolarVolume_getProjection((PolarVolume_t*)obj);
          } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
            objproj = PolarScan_getProjection((PolarScan_t*)obj);
          }
        }

        if (objproj != NULL) {
          /* We will go from surface coords into the lonlat projection assuming that a polar volume uses a lonlat projection*/
          if (!Projection_transformx(projection, objproj, herex, herey, 0.0, &olon, &olat, NULL)) {
            RAVE_WARNING0("Failed to transform from composite into polar coordinates");
          } else {
            double dist = 0.0;
            double maxdist = 0.0;
            double rdist = 0.0;
            if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
              dist = PolarVolume_getDistance((PolarVolume_t*)obj, olon, olat);
              maxdist = PolarVolume_getMaxDistance((PolarVolume_t*)obj);
            } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
              dist = PolarScan_getDistance((PolarScan_t*)obj, olon, olat);
              maxdist = PolarScan_getMaxDistance((PolarScan_t*)obj);
            }
            if (dist <= maxdist) {
              if (CompositeInternal_nearestPosition(composite, obj, olon, olat, &navinfo)) {
                double originaldist = dist;
                rdist = dist; /* Remember distance to radar */

                if (composite->method == CompositeSelectionMethod_HEIGHT) {
                  dist = navinfo.actual_height;
                }

                for (cindex = 0; cindex < nparam; cindex++) {
                  RaveValueType otype = RaveValueType_NODATA;
                  double ovalue = 0.0;
                  CompositeInternal_getValueAtPosition(composite, obj, cvalues[cindex].name, &navinfo, &otype, &ovalue);

                  if (composite->algorithm != NULL && CompositeAlgorithm_supportsProcess(composite->algorithm)) {
                    if (CompositeAlgorithm_process(composite->algorithm, obj, cvalues[cindex].name, olon, olat, originaldist, &otype, &ovalue, &cvalues[cindex].navinfo)) {
                      cvalues[cindex].vtype = otype;
                      cvalues[cindex].value = ovalue;
                      cvalues[cindex].mindist = originaldist;
                      cvalues[cindex].radardist = rdist;
                      cvalues[cindex].radarindex = i;
                      cvalues[cindex].navinfo = navinfo;
                    }
                  } else {
                    if (otype == RaveValueType_DATA || otype == RaveValueType_UNDETECT) {
                      if ((cvalues[cindex].vtype != RaveValueType_DATA && cvalues[cindex].vtype != RaveValueType_UNDETECT) ||
                          dist < cvalues[cindex].mindist) {
                        cvalues[cindex].vtype = otype;
                        cvalues[cindex].value = ovalue;
                        cvalues[cindex].mindist = dist;
                        cvalues[cindex].radardist = rdist;
                        cvalues[cindex].radarindex = i;
                        cvalues[cindex].navinfo = navinfo;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        RAVE_OBJECT_RELEASE(obj);
        RAVE_OBJECT_RELEASE(objproj);
      }

      for (cindex = 0; cindex < nparam; cindex++) {
        double vvalue = cvalues[cindex].value;
        double vtype = cvalues[cindex].vtype;
        PolarNavigationInfo info = cvalues[cindex].navinfo;

        if (vtype == RaveValueType_NODATA) {
          CartesianParam_setConvertedValue(cvalues[cindex].parameter, x, y, CartesianParam_getNodata(cvalues[cindex].parameter));
        } else {
          if (composite->ptype == Rave_ProductType_PMAX && cvalues[cindex].radardist < composite->range) {
            RaveValueType ntype = RaveValueType_NODATA;
            double nvalue = 0.0;
            if (vtype == RaveValueType_UNDETECT) {
              /* Undetect should not affect navigation information */
              CompositeInternal_getVerticalMaxValue(composite, cvalues[cindex].radarindex, cvalues[cindex].name, olon, olat, &ntype, &nvalue, NULL);
            } else {
              CompositeInternal_getVerticalMaxValue(composite, cvalues[cindex].radarindex, cvalues[cindex].name, olon, olat, &ntype, &nvalue, &info);
            }
            if (ntype != RaveValueType_NODATA) {
              vtype = ntype;
              vvalue = nvalue;
            } else {
              /* If we find nodata then we really should use the original navigation information since there must be something wrong */
              info = cvalues[cindex].navinfo;
            }
          }

          if (vtype == RaveValueType_UNDETECT) {
            CartesianParam_setConvertedValue(cvalues[cindex].parameter, x, y, CartesianParam_getUndetect(cvalues[cindex].parameter));
          } else {
            CartesianParam_setConvertedValue(cvalues[cindex].parameter, x, y, vvalue);
          }
        }
        if ((vtype == RaveValueType_DATA || vtype == RaveValueType_UNDETECT) &&
            cvalues[cindex].radarindex >= 0 && nqualityflags > 0) {
          CompositeInternal_fillQualityInformation(composite, x, y, cvalues[cindex].parameter, cvalues[cindex].radardist, cvalues[cindex].radarindex, &info);
        }
      }
    }
  }

  for (i = 0; cvalues != NULL && i < nparam; i++) {
    RAVE_OBJECT_RELEASE(cvalues[i].parameter);
  }
  RAVE_FREE(cvalues);
  RAVE_OBJECT_RELEASE(projection);
  return result;
fail:
  for (i = 0; cvalues != NULL && i < nparam; i++) {
    RAVE_OBJECT_RELEASE(cvalues[i].parameter);
  }
  RAVE_FREE(cvalues);
  RAVE_OBJECT_RELEASE(projection);
  RAVE_OBJECT_RELEASE(result);
  return result;
}

void Composite_setAlgorithm(Composite_t* composite, CompositeAlgorithm_t* algorithm)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  RAVE_OBJECT_RELEASE(composite->algorithm);
  composite->algorithm = RAVE_OBJECT_COPY(algorithm);
}

CompositeAlgorithm_t* Composite_getAlgorithm(Composite_t* composite)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return RAVE_OBJECT_COPY(composite->algorithm);
}

/*@} End of Interface functions */

RaveCoreObjectType Composite_TYPE = {
    "Composite",
    sizeof(Composite_t),
    Composite_constructor,
    Composite_destructor,
    Composite_copyconstructor
};

