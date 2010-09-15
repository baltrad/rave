/* --------------------------------------------------------------------
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Cartesian ODIM decorator
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-09-09
 */
#include "cartesian_odim_io.h"
#include "rave_hlhdf_utilities.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>

/**
 * Represents the adaptor
 */
struct _CartesianOdimIO_t {
  RAVE_OBJECT_HEAD /** Always on top */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int CartesianOdimIO_constructor(RaveCoreObject* obj)
{
  //CartesianOdimIO_t* this = (CartesianOdimIO_t*)obj;
  return 1;
}

/**
 * Copy constructor
 */
static int CartesianOdimIO_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  //CartesianOdimIO_t* this = (CartesianOdimIO_t)obj;
  //CartesianOdimIO_t* src = (CartesianOdimIO_t*)srcobj;
  return 1;
}

/**
 * Destroys the object
 * @param[in] obj - the instance
 */
static void CartesianOdimIO_destructor(RaveCoreObject* obj)
{
  //CartesianOdimIO_t* area = (CartesianOdimIO_t*)obj;
}

/**
 * Adds a cartesian image (belonging to a volume) to a node list.
 * @param[in] cvol - the cartesian image to be added to a node list
 * @param[in] nodelist - the nodelist the nodes should be added to
 * @returns 1 on success otherwise 0
 */
static int CartesianOdimIOInternal_addCartesianImageToNodeList(Cartesian_t* cartesian, HL_NodeList* nodelist, const char* fmt, ...)
{
  int result = 0;
  char nodeName[1024];
  RaveObjectList_t* attributes = NULL;
  va_list ap;
  int n = 0;

  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n < 0 || n >= 1024) {
    RAVE_ERROR1("Failed to create image name from fmt=%s", fmt);
    goto done;
  }

  if (!RaveHL_hasNodeByName(nodelist, nodeName)) {
    if (!RaveHL_createGroup(nodelist, nodeName)) {
      goto done;
    }
  }

  if (!RaveHL_hasNodeByName(nodelist, "%s/data1", nodeName)) {
    if (!RaveHL_createGroup(nodelist,"%s/data1", nodeName)) {
      goto done;
    }
  }

  attributes = Cartesian_getAttributeValues(cartesian);
  if (attributes == NULL) {
    goto done;
  }

  if (!RaveUtilities_addDoubleAttributeToList(attributes, "what/gain", Cartesian_getGain(cartesian)) ||
      !RaveUtilities_addDoubleAttributeToList(attributes, "what/nodata", Cartesian_getNodata(cartesian)) ||
      !RaveUtilities_addDoubleAttributeToList(attributes, "what/offset", Cartesian_getOffset(cartesian)) ||
      !RaveUtilities_addDoubleAttributeToList(attributes, "what/undetect", Cartesian_getUndetect(cartesian)) ||
      !RaveUtilities_addStringAttributeToList(attributes, "what/quantity", Cartesian_getQuantity(cartesian)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/product",
                                                  RaveTypes_getStringFromProductType(Cartesian_getProduct(cartesian)))) {
    goto done;
  }
  if (Cartesian_getDate(cartesian) != NULL) {
    const char* dtdate = Cartesian_getDate(cartesian);
    if (!Cartesian_hasAttribute(cartesian, "what/startdate") &&
        !RaveUtilities_replaceStringAttributeInList(attributes, "what/startdate", dtdate)) {
      goto done;
    }
    if (!Cartesian_hasAttribute(cartesian, "what/enddate") &&
        !RaveUtilities_replaceStringAttributeInList(attributes, "what/enddate", dtdate)) {
      goto done;
    }
  }
  if (Cartesian_getTime(cartesian) != NULL) {
    const char* dttime = Cartesian_getTime(cartesian);
    if (!Cartesian_hasAttribute(cartesian, "what/starttime") &&
        !RaveUtilities_replaceStringAttributeInList(attributes, "what/starttime", dttime)) {
      goto done;
    }
    if (!Cartesian_hasAttribute(cartesian, "what/endtime") &&
        !RaveUtilities_replaceStringAttributeInList(attributes, "what/endtime", dttime)) {
      goto done;
    }
  }

  if (attributes == NULL || !RaveHL_addAttributes(nodelist, attributes, nodeName)) {
    goto done;
  }

  if (!RaveHL_addData(nodelist,
                      Cartesian_getData(cartesian),
                      Cartesian_getXSize(cartesian),
                      Cartesian_getYSize(cartesian),
                      Cartesian_getDataType(cartesian),
                      "%s/data1", nodeName)) {
    goto done;
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(attributes);
  return result;
}

/*@} End of Private functions */

/*@{ Interface functions */
int CartesianOdimIO_fillImage(CartesianOdimIO_t* self, HL_NodeList* nodelist, Cartesian_t* cartesian)
{
  int result = 0;
  RaveObjectList_t* attributes = NULL;
  Rave_ObjectType otype = Rave_ObjectType_UNDEFINED;
  Projection_t* projection = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");

  if (!Cartesian_isValid(cartesian, Rave_ObjectType_IMAGE) &&
      !Cartesian_isValid(cartesian, Rave_ObjectType_COMP)) {
    goto done;
  }

  otype = Cartesian_getObjectType(cartesian);
  projection = Cartesian_getProjection(cartesian);

  if (otype != Rave_ObjectType_COMP && otype != Rave_ObjectType_IMAGE) {
    RAVE_WARNING1("CartesianOdimIO_fillImage does not support objectType = %d\n", otype);
    goto done;
  }

  if (!RaveHL_hasNodeByName(nodelist, "/Conventions")) {
    if (!RaveHL_createStringValue(nodelist, RAVE_ODIM_VERSION_2_0_STR, "/Conventions")) {
      goto done;
    }
  }

  attributes = Cartesian_getAttributeValues(cartesian);
  if (attributes != NULL) {
    const char* objectType = RaveTypes_getStringFromObjectType(Cartesian_getObjectType(cartesian));
    if (!RaveUtilities_addStringAttributeToList(attributes, "what/object", objectType) ||
        !RaveUtilities_addStringAttributeToList(attributes, "what/version", RAVE_ODIM_H5RAD_VERSION_2_0_STR)) {
      RAVE_ERROR0("Failed to add what/object or what/version to attributes");
      goto done;
    }
  } else {
    RAVE_ERROR0("Failed to aquire attributes for cartesian product");
    goto done;
  }

  if (projection != NULL) {
    double llX = 0.0, llY = 0.0, urX = 0.0, urY = 0.0;
    if (!RaveUtilities_addStringAttributeToList(attributes, "where/projdef", Projection_getDefinition(projection))) {
      goto done;
    }
    Cartesian_getAreaExtent(cartesian, &llX, &llY, &urX, &urY);
    if (!CartesianHelper_addLonLatExtentToAttributeList(attributes, projection, llX, llY, urX, urY)) {
      goto done;
    }
  }

  if (!RaveUtilities_addStringAttributeToList(attributes, "what/date", Cartesian_getDate(cartesian)) ||
      !RaveUtilities_addStringAttributeToList(attributes, "what/time", Cartesian_getTime(cartesian)) ||
      !RaveUtilities_addStringAttributeToList(attributes, "what/source", Cartesian_getSource(cartesian)) ||
      !RaveUtilities_addDoubleAttributeToList(attributes, "where/xscale", Cartesian_getXScale(cartesian)) ||
      !RaveUtilities_addDoubleAttributeToList(attributes, "where/yscale", Cartesian_getYScale(cartesian)) ||
      !RaveUtilities_replaceLongAttributeInList(attributes, "where/xsize", Cartesian_getXSize(cartesian)) ||
      !RaveUtilities_replaceLongAttributeInList(attributes, "where/ysize", Cartesian_getYSize(cartesian))) {
    goto done;
  }

  // prodpar is dataset specific.. so it should only be there for images in volumes.
  RaveUtilities_removeAttributeFromList(attributes, "what/prodpar");

  if (attributes == NULL || !RaveHL_addAttributes(nodelist, attributes, "")) {
    goto done;
  }

  RAVE_OBJECT_RELEASE(attributes);

  attributes = Cartesian_getAttributeValues(cartesian);

  if (!RaveUtilities_addDoubleAttributeToList(attributes, "what/gain", Cartesian_getGain(cartesian)) ||
      !RaveUtilities_addDoubleAttributeToList(attributes, "what/nodata", Cartesian_getNodata(cartesian)) ||
      !RaveUtilities_addDoubleAttributeToList(attributes, "what/offset", Cartesian_getOffset(cartesian)) ||
      !RaveUtilities_addDoubleAttributeToList(attributes, "what/undetect", Cartesian_getUndetect(cartesian)) ||
      !RaveUtilities_addStringAttributeToList(attributes, "what/quantity", Cartesian_getQuantity(cartesian)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/product",
                                                  RaveTypes_getStringFromProductType(Cartesian_getProduct(cartesian)))) {
    goto done;
  }

  if (Cartesian_getDate(cartesian) != NULL) {
    const char* dtdate = Cartesian_getDate(cartesian);
    if (!Cartesian_hasAttribute(cartesian, "what/startdate") &&
        !RaveUtilities_replaceStringAttributeInList(attributes, "what/startdate", dtdate)) {
      goto done;
    }
    if (!Cartesian_hasAttribute(cartesian, "what/enddate") &&
        !RaveUtilities_replaceStringAttributeInList(attributes, "what/enddate", dtdate)) {
      goto done;
    }
  }
  if (Cartesian_getTime(cartesian) != NULL) {
    const char* dttime = Cartesian_getTime(cartesian);
    if (!Cartesian_hasAttribute(cartesian, "what/starttime") &&
        !RaveUtilities_replaceStringAttributeInList(attributes, "what/starttime", dttime)) {
      goto done;
    }
    if (!Cartesian_hasAttribute(cartesian, "what/endtime") &&
        !RaveUtilities_replaceStringAttributeInList(attributes, "what/endtime", dttime)) {
      goto done;
    }
  }

  if (!RaveHL_createGroup(nodelist,"/dataset1")) {
    goto done;
  }

  if (attributes == NULL || !RaveHL_addAttributes(nodelist, attributes, "/dataset1")) {
    goto done;
  }

  if (!RaveHL_createGroup(nodelist,"/dataset1/data1")) {
    goto done;
  }

  if (!RaveHL_addData(nodelist,
                      Cartesian_getData(cartesian),
                      Cartesian_getXSize(cartesian),
                      Cartesian_getYSize(cartesian),
                      Cartesian_getDataType(cartesian),
                      "/dataset1/data1")) {
    goto done;
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(attributes);
  RAVE_OBJECT_RELEASE(projection);
  return result;
}

int CartesianOdimIO_fillVolume(CartesianOdimIO_t* self, HL_NodeList* nodelist, CartesianVolume_t* volume)
{
  int result = 0;
  int nrImages = 0;
  int i = 0;
  RaveObjectList_t* attributes = NULL;
  Projection_t* projection = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((volume != NULL), "volume == NULL");

  // First verify that no bogus data is entered into the system.
  if (!CartesianVolume_isValid(volume)) {
    goto done;
  }

  if (!RaveHL_hasNodeByName(nodelist, "/Conventions")) {
    if (!RaveHL_createStringValue(nodelist, RAVE_ODIM_VERSION_2_0_STR, "/Conventions")) {
      goto done;
    }
  }

  attributes = CartesianVolume_getAttributeValues(volume);
  if (attributes != NULL) {
    const char* objectType = RaveTypes_getStringFromObjectType(CartesianVolume_getObjectType(volume));
    if (!RaveUtilities_addStringAttributeToList(attributes, "what/object", objectType) ||
        !RaveUtilities_addStringAttributeToList(attributes, "what/version", RAVE_ODIM_H5RAD_VERSION_2_0_STR)) {
      RAVE_ERROR0("Failed to add what/object or what/version to attributes");
      goto done;
    }
  } else {
    goto done;
  }

  if (!RaveUtilities_addStringAttributeToList(attributes, "what/date", CartesianVolume_getDate(volume)) ||
      !RaveUtilities_addStringAttributeToList(attributes, "what/time", CartesianVolume_getTime(volume)) ||
      !RaveUtilities_addStringAttributeToList(attributes, "what/source", CartesianVolume_getSource(volume)) ||
      !RaveUtilities_addDoubleAttributeToList(attributes, "where/xscale", CartesianVolume_getXScale(volume)) ||
      !RaveUtilities_addDoubleAttributeToList(attributes, "where/yscale", CartesianVolume_getYScale(volume)) ||
      !RaveUtilities_replaceLongAttributeInList(attributes, "where/xsize", CartesianVolume_getXSize(volume)) ||
      !RaveUtilities_replaceLongAttributeInList(attributes, "where/ysize", CartesianVolume_getYSize(volume))) {
    goto done;
  }

  // Add projection + extent if possible
  projection = CartesianVolume_getProjection(volume);
  if (projection != NULL) {
    double llX = 0.0, llY = 0.0, urX = 0.0, urY = 0.0;
    if (!RaveUtilities_addStringAttributeToList(attributes, "where/projdef", Projection_getDefinition(projection))) {
      goto done;
    }
    CartesianVolume_getAreaExtent(volume, &llX, &llY, &urX, &urY);
    if (!CartesianHelper_addLonLatExtentToAttributeList(attributes, projection, llX, llY, urX, urY)) {
      goto done;
    }
  }

  if (attributes == NULL || !RaveHL_addAttributes(nodelist, attributes, "")) {
    goto done;
  }

  result = 1;

  nrImages = CartesianVolume_getNumberOfImages(volume);
  for (i = 0; result == 1 && i < nrImages; i++) {
    Cartesian_t* image = CartesianVolume_getImage(volume, i);
    result = CartesianOdimIOInternal_addCartesianImageToNodeList(image, nodelist, "/dataset%d", (i+1));
    RAVE_OBJECT_RELEASE(image);
  }
done:
  RAVE_OBJECT_RELEASE(attributes);
  RAVE_OBJECT_RELEASE(projection);
  return result;
}

/*@} End of Interface functions */

RaveCoreObjectType CartesianOdimIO_TYPE = {
    "CartesianOdimIO",
    sizeof(CartesianOdimIO_t),
    CartesianOdimIO_constructor,
    CartesianOdimIO_destructor,
    CartesianOdimIO_copyconstructor
};
