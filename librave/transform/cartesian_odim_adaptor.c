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
#include "cartesian_odim_adaptor.h"
#include "rave_hlhdf_utilities.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>

/**
 * Represents the adaptor
 */
struct _CartesianOdimAdaptor_t {
  RAVE_OBJECT_HEAD /** Always on top */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int CartesianOdimAdaptor_constructor(RaveCoreObject* obj)
{
  //CartesianOdimAdaptor_t* this = (CartesianOdimAdaptor_t*)obj;
  return 1;
}

/**
 * Copy constructor
 */
static int CartesianOdimAdaptor_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  //CartesianOdimAdaptor_t* this = (CartesianOdimAdaptor_t)obj;
  //CartesianOdimAdaptor_t* src = (CartesianOdimAdaptor_t*)srcobj;
  return 1;
}

/**
 * Destroys the object
 * @param[in] obj - the instance
 */
static void CartesianOdimAdaptor_destructor(RaveCoreObject* obj)
{
  //CartesianOdimAdaptor_t* area = (CartesianOdimAdaptor_t*)obj;
}
/*@} End of Private functions */

/*@{ Interface functions */
int CartesianOdimAdaptor_fillImageInformation(CartesianOdimAdaptor_t* self, HL_NodeList* nodelist, Cartesian_t* cartesian)
{
  int result = 0;
  RaveObjectList_t* attributes = NULL;
  Rave_ObjectType otype = Rave_ObjectType_UNDEFINED;
  Projection_t* projection = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");

  otype = Cartesian_getObjectType(cartesian);
  projection = Cartesian_getProjection(cartesian);

  if (otype != Rave_ObjectType_COMP && otype != Rave_ObjectType_IMAGE) {
    RAVE_WARNING1("CartesianOdimAdaptor_fillImageInformation does not support objectType = %d\n", otype);
    goto done;
  }

  if (!RaveHL_createStringValue(nodelist, RAVE_ODIM_VERSION_2_0_STR, "/Conventions")) {
    goto done;
  }

  attributes = Cartesian_getAttributeValues(cartesian, Rave_ObjectType_IMAGE);
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

  attributes = Cartesian_getAttributeValues(cartesian, Rave_ObjectType_CVOL);
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

/*@} End of Interface functions */

RaveCoreObjectType CartesianOdimAdaptor_TYPE = {
    "CartesianOdimAdaptor",
    sizeof(CartesianOdimAdaptor_t),
    CartesianOdimAdaptor_constructor,
    CartesianOdimAdaptor_destructor,
    CartesianOdimAdaptor_copyconstructor
};
