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
  CompositeSelectionMethod_t method; /**< selection method, default CompositeSelectionMethod_NEAREST */
  double height; /**< the height when generating pcapppi, cappi, default 1000 */
  double elangle; /**< the elevation angle when generating ppi, default 0.0 */
  char* paramname; /**< the parameter name, default DBZH */
  double offset;     /**< the offset, default 0 */
  double gain;     /**< the gain, default 1 */
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
  this->method = CompositeSelectionMethod_NEAREST;
  this->height = 1000.0;
  this->elangle = 0.0;
  this->paramname = NULL;
  this->offset = 0;
  this->gain = 1;
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
  this->method = src->method;
  this->height = src->height;
  this->elangle = src->elangle;
  this->paramname = NULL;
  this->offset = src->offset;
  this->gain = src->gain;
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
  RAVE_FREE(this->paramname);
}

static void CompositeInternal_nearestValue(Composite_t* composite, RaveCoreObject* object,
  double plon, double plat, RaveValueType* type, double* value, PolarNavigationInfo* nav)
{
  RAVE_ASSERT((type != NULL), "type == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");

  *type = RaveValueType_NODATA;
  *value = 0.0;

  if (object != NULL) {
    if (RAVE_OBJECT_CHECK_TYPE(object, &PolarScan_TYPE)) {
      // We could support CAPPI on scans as well but that is something
      // for the future.
      //
      if (composite->ptype == Rave_ProductType_PPI ||
          composite->ptype == Rave_ProductType_PCAPPI) {
        *type = PolarScan_getNearestConvertedParameterValue((PolarScan_t*)object,
                                                            Composite_getQuantity(composite),
                                                            plon,
                                                            plat,
                                                            value,
                                                            nav);
      } else {
        *type = RaveValueType_NODATA;
      }
    } else  if (RAVE_OBJECT_CHECK_TYPE(object, &PolarVolume_TYPE)){
      if (composite->ptype == Rave_ProductType_PCAPPI ||
          composite->ptype == Rave_ProductType_CAPPI) {
        int insidee = (composite->ptype == Rave_ProductType_PCAPPI)?0:1;
        *type = PolarVolume_getNearestConvertedParameterValue((PolarVolume_t*)object,
                                                              Composite_getQuantity(composite),
                                                              plon,
                                                              plat,
                                                              Composite_getHeight(composite),
                                                              insidee,
                                                              value,
                                                              nav);
      } else if (composite->ptype == Rave_ProductType_PPI) {
        PolarScan_t* scan = PolarVolume_getScanClosestToElevation((PolarVolume_t*)object,
                                                                  Composite_getElevationAngle(composite),
                                                                  0);
        if (scan == NULL) {
          RAVE_ERROR1("Failed to fetch scan nearest to elevation %g",
                      Composite_getElevationAngle(composite));
          return;
        }
        *type = PolarScan_getNearestConvertedParameterValue(scan,
                                                            Composite_getQuantity(composite),
                                                            plon,
                                                            plat,
                                                            value,
                                                            nav);
        RAVE_OBJECT_RELEASE(scan);
      } else {
        *type = RaveValueType_NODATA;
      }
    } else {
      *type = RaveValueType_NODATA;
    }
  }
}

/**
 * Adds quality flags to the composite.
 * @param[in] image - the image to add quality flags to
 * @param[in] qualityflags - a list of strings identifying the how/task value in the quality fields
 * @return 1 on success otherwise 0
 */
static int CompositeInternal_addQualityFlags(Cartesian_t* image, RaveList_t* qualityflags)
{
  int result = 1;
  int nqualityflags = 0;
  RaveField_t* field = NULL;
  RaveAttribute_t* howtaskattribute = NULL;
  int xsize = 0, ysize = 0;
  int i = 0;

  RAVE_ASSERT((image != NULL), "image == NULL");

  xsize = Cartesian_getXSize(image);
  ysize = Cartesian_getYSize(image);

  if (qualityflags != NULL) {
    nqualityflags = RaveList_size(qualityflags);
  }

  for (i = 0; result == 1 && i < nqualityflags; i++) {
    field = RAVE_OBJECT_NEW(&RaveField_TYPE);
    char* howtaskvaluestr = (char*)RaveList_get(qualityflags, i);
    howtaskattribute = RaveAttributeHelp_createString("how/task", howtaskvaluestr);
    if (field == NULL ||
        howtaskattribute == NULL ||
        !RaveField_createData(field, xsize, ysize, RaveDataType_UCHAR) ||
        !RaveField_addAttribute(field, howtaskattribute) ||
        !Cartesian_addQualityField(image, field)) {
      RAVE_ERROR0("Failed to add quality field to the cartesian product");
      result = 0;
    }

    RAVE_OBJECT_RELEASE(field);
    RAVE_OBJECT_RELEASE(howtaskattribute);
  }

  return result;
}

/**
 * Uses the navigation information and fills all associated cartesian quality
 * with the composite objects quality fields.
 * @param[in] composite - self
 * @param[in] x - x coordinate
 * @param[in] y - y coordinate
 * @param[in] radarindex - the object to use in the composite
 * @param[in] navinfo - the navigational information
 */
static void CompositeInternal_fillQualityInformation(
  Composite_t* composite,
  int x, int y,
  Cartesian_t* cartesian,
  int radarindex,
  PolarNavigationInfo* navinfo)
{
  int nfields = 0, i = 0;
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  RAVE_ASSERT((navinfo != NULL), "navinfo == NULL");
  nfields = Cartesian_getNumberOfQualityFields(cartesian);
  for (i = 0; i < nfields; i++) {
    RaveField_t* field = NULL;
    RaveAttribute_t* attribute = NULL;
    char* name = NULL;
    double v = 0.0;

    field = Cartesian_getQualityField(cartesian, i);
    if (field != NULL) {
      attribute = RaveField_getAttribute(field, "how/task");
    }
    if (attribute != NULL) {
      RaveAttribute_getString(attribute, &name);
    }

    if (name != NULL) {
      RaveCoreObject* obj = RaveObjectList_get(composite->list, radarindex);
      if (obj != NULL) {
        if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
          if (navinfo->ei >= 0 && navinfo->ri >= 0 && navinfo->ai >= 0 &&
              PolarVolume_getQualityValueAt((PolarVolume_t*)obj, Composite_getQuantity(composite), navinfo->ei, navinfo->ri, navinfo->ai, name, &v)) {
            RaveField_setValue(field, x, y, v);
          } else {
            RaveField_setValue(field, x, y, 0.0); // No data found
          }
        } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
          if (navinfo->ri >= 0 && navinfo->ai >= 0 &&
              PolarScan_getQualityValueAt((PolarScan_t*)obj, Composite_getQuantity(composite), navinfo->ri, navinfo->ai, name, &v)) {
            RaveField_setValue(field, x, y , v);
          } else {
            RaveField_setValue(field, x, y, 0.0); // No data found
          }
        }
      }
      RAVE_OBJECT_RELEASE(obj);
    }
    RAVE_OBJECT_RELEASE(field);
    RAVE_OBJECT_RELEASE(attribute);
  }
}

#ifdef KALLE
static RaveObjectHashTable_t* CompositeInternal_getPooScanFields(Composite_t* composite)
{
  RaveObjectList_t* result = NULL;
  RaveObjectHashTable_t* scans = NULL;
  int nrobjs = 0, i = 0;

  scans = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  if (scans == NULL) {
    RAVE_ERROR0("Failed to allocate memory for object hash table");
    goto done;
  }
  nrobjs = RaveObjectList_size(composite->list);
  for (i = 0; i < nrobjs; i++) {
    RaveCoreObject* obj = RaveObjectList_get(composite->list, i);
    if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
      RaveField_t* field = PolarScan_findQualityFieldByHowTask((PolarScan_t*)obj, "se.smhi.detector.poo");
      if (field != NULL) {
        PolarScan_t* scan = PolarScan_fromField()
      }
    }
  }

  result = RAVE_OBJECT_COPY(scans);
done:
  RAVE_OBJECT_RELEASE(scans);
  return result;
}
#endif
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

void Composite_setProduct(Composite_t* composite, Rave_ProductType type)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  if (type == Rave_ProductType_PCAPPI ||
      type == Rave_ProductType_CAPPI ||
      type == Rave_ProductType_PPI) {
    composite->ptype = type;
  } else {
    RAVE_ERROR0("Only supported algorithms are PPI, CAPPI and PCAPPI");
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
  return composite->elangle;
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

void Composite_setOffset(Composite_t* composite, double offset)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  composite->offset = offset;
}

double Composite_getOffset(Composite_t* composite)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return composite->offset;
}

void Composite_setGain(Composite_t* composite, double gain)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  composite->gain = gain;
}

double Composite_getGain(Composite_t* composite)
{
  RAVE_ASSERT((composite != NULL), "composite == NULL");
  return composite->gain;
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
  RaveAttribute_t* prodpar = NULL;
#ifdef KALLE
  RaveObjectList_t* scanlist = NULL;
#endif
  PolarNavigationInfo navinfo;
  int x = 0, y = 0, i = 0, xsize = 0, ysize = 0, nradars = 0;
  int nqualityflags = 0;

  RAVE_ASSERT((composite != NULL), "composite == NULL");
  RAVE_ASSERT((area != NULL), "area == NULL");

  result = RAVE_OBJECT_NEW(&Cartesian_TYPE);
  if (!Cartesian_init(result, area, RaveDataType_UCHAR)) {
    goto fail;
  }
  if (composite->ptype == Rave_ProductType_CAPPI ||
      composite->ptype == Rave_ProductType_PCAPPI) {
    prodpar = RaveAttributeHelp_createDouble("what/prodpar", composite->height);
  } else {
    prodpar = RaveAttributeHelp_createDouble("what/prodpar", composite->elangle * 180.0/M_PI);
  }
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
  Cartesian_setOffset(result, Composite_getOffset(composite));
  Cartesian_setGain(result, Composite_getGain(composite));

  xsize = Cartesian_getXSize(result);
  ysize = Cartesian_getYSize(result);
  projection = Cartesian_getProjection(result);
  nradars = RaveObjectList_size(composite->list);

  if (qualityflags != NULL) {
    nqualityflags = RaveList_size(qualityflags);
  }

  if (!CompositeInternal_addQualityFlags(result, qualityflags)) {
    goto fail;
  }
#ifdef KALLE
  if (composite->method == CompositeSelectionMethod_POO) {
    scanlist = CompositeInternal_getPooScanFields(composite);
  }
#endif
  for (y = 0; y < ysize; y++) {
    double herey = Cartesian_getLocationY(result, y);
    for (x = 0; x < xsize; x++) {
      double herex = Cartesian_getLocationX(result, x);
      double olon = 0.0, olat = 0.0;
      double mindist = 1e10;
      double v = 0.0L;
      RaveValueType vtype = RaveValueType_NODATA;
      int radarindex = -1;
      PolarNavigationInfo rememberednav;
      rememberednav.ei = -1;
      rememberednav.ai = -1;
      rememberednav.ri = -1;

      for (i = 0, mindist=1e10; i < nradars; i++) {
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
            if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
              dist = PolarVolume_getDistance((PolarVolume_t*)obj, olon, olat);
              maxdist = PolarVolume_getMaxDistance((PolarVolume_t*)obj);
            } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
              dist = PolarScan_getDistance((PolarScan_t*)obj, olon, olat);
              maxdist = PolarScan_getMaxDistance((PolarScan_t*)obj);
            }
            if (dist <= maxdist) {
              RaveValueType otype = RaveValueType_NODATA;
              double ovalue = 0.0;
              CompositeInternal_nearestValue(composite, obj, olon, olat, &otype, &ovalue, &navinfo);

              if (composite->method == CompositeSelectionMethod_HEIGHT) {
                dist = navinfo.actual_height; // We are determining pixel by height above sea level
              }

              if (otype == RaveValueType_DATA || otype == RaveValueType_UNDETECT) {
                if ((vtype != RaveValueType_DATA && vtype != RaveValueType_UNDETECT) || dist < mindist) {
                  vtype = otype;
                  v = ovalue;
                  mindist = dist;
                  radarindex = i;
                  rememberednav = navinfo;
                }
              }
            }
          }
        }
        RAVE_OBJECT_RELEASE(obj);
        RAVE_OBJECT_RELEASE(objproj);
      }

      if (vtype == RaveValueType_NODATA) {
        Cartesian_setConvertedValue(result, x, y, Cartesian_getNodata(result));
      } else if (vtype == RaveValueType_UNDETECT) {
        Cartesian_setConvertedValue(result, x, y, Cartesian_getUndetect(result));
      } else {
        Cartesian_setConvertedValue(result, x, y, v);
      }

      if ((vtype == RaveValueType_DATA || vtype == RaveValueType_UNDETECT) && radarindex >= 0 && nqualityflags > 0) {
        CompositeInternal_fillQualityInformation(composite, x, y, result, radarindex, &rememberednav);
      }
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

