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
 * Defines the functions available when transforming between different
 * types of products
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-20
 */
#include "transform.h"
#include "projection.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>

/**
 * Represents one transformator
 */
struct _Transform_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveTransformationMethod method;
};

/*@{ Private functions */
/**
 * Constructor
 */
static int Transform_constructor(RaveCoreObject* obj)
{
  Transform_t* transform = (Transform_t*)obj;
  transform->method = NEAREST;
  return 1;
}

/**
 * Destructor
 */
static void Transform_destructor(RaveCoreObject* obj)
{
}

/**
 * Internal routine to handle both cappis and pseudo-cappis since they are similar in behaviour.
 * @param[in] transform - the transformer instance
 * @param[in] pvol - the polar volume
 * @param[in] cartesian - the cartesian (resulting) product
 * @param[in] height - the altitude to create the cappi at
 * @param[in] insidee - the only difference between cappi and pseudo-cappi is if the range/height evaluates to an elevation that is inside or outside the min-max scan elevations.
 * @returns 1 on success otherwise 0
 */
static int Transform_cappis_internal(Transform_t* transform, PolarVolume_t* pvol, Cartesian_t* cartesian, double height, int insidee)
{
  int result = 0;
  long xsize = 0, ysize = 0, x = 0, y = 0;
  double cnodata = 0.0L, cundetect = 0.0L;
  Projection_t* sourcepj = NULL;
  Projection_t* targetpj = NULL;

  RAVE_ASSERT((transform != NULL), "transform was NULL");
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");

  if (!Cartesian_isTransformable(cartesian) ||
      !PolarVolume_isTransformable(pvol)) {
    RAVE_ERROR0("Cartesian product or polar volume is not possible to transform");
    goto done;
  }

  sourcepj = Cartesian_getProjection(cartesian);
  targetpj = PolarVolume_getProjection(pvol);
  cnodata = Cartesian_getNodata(cartesian);
  cundetect = Cartesian_getUndetect(cartesian);
  xsize = Cartesian_getXSize(cartesian);
  ysize = Cartesian_getYSize(cartesian);

  for (y = 0; y < ysize; y++) {
    double herey = Cartesian_getLocationY(cartesian, y);
    double tmpy = herey;
    for (x = 0; x < xsize; x++) {
      double herex = Cartesian_getLocationX(cartesian, x);
      herey = tmpy; // So that we can use herey over and over again
      RaveValueType valid = RaveValueType_NODATA;
      double v = 0.0L;
      if (!Projection_transform(sourcepj, targetpj, &herex, &herey, NULL)) {
        RAVE_ERROR0("Transform failed");
        goto done;
      }
      valid = PolarVolume_getNearest(pvol, herex, herey, height, insidee, &v);

      if (valid == RaveValueType_NODATA) {
        v = cnodata;
      } else if (valid == RaveValueType_UNDETECT) {
        v = cundetect;
      }
      Cartesian_setValue(cartesian, x, y, v);
    }
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(sourcepj);
  RAVE_OBJECT_RELEASE(targetpj);
  return result;
}
/*@} End of Private functions */

/*@{ Interface functions */
int Transform_setMethod(Transform_t* transform, RaveTransformationMethod method)
{
  int result = 0;
  RAVE_ASSERT((transform != NULL), "transform was NULL");
  if (method >= NEAREST && method <= INVERSE) {
    transform->method = method;
    result = 1;
  }
  return result;
}

RaveTransformationMethod Transform_getMethod(Transform_t* transform)
{
  RAVE_ASSERT((transform != NULL), "transform was NULL");
  return transform->method;
}

int Transform_ppi(Transform_t* transform, PolarScan_t* scan, Cartesian_t* cartesian)
{
  int result = 0;
  long xsize = 0, ysize = 0, x = 0, y = 0;
  double cnodata = 0.0L, cundetect = 0.0L;
  Projection_t* sourcepj = NULL;
  Projection_t* targetpj = NULL;

  RAVE_ASSERT((transform != NULL), "transform was NULL");
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");

  if (!Cartesian_isTransformable(cartesian) ||
      !PolarScan_isTransformable(scan)) {
    RAVE_ERROR0("Cartesian product or scan is not possible to transform");
    goto done;
  }

  sourcepj = Cartesian_getProjection(cartesian);
  targetpj = PolarScan_getProjection(scan);
  cnodata = Cartesian_getNodata(cartesian);
  cundetect = Cartesian_getUndetect(cartesian);
  xsize = Cartesian_getXSize(cartesian);
  ysize = Cartesian_getYSize(cartesian);

  for (y = 0; y < ysize; y++) {
    double herey = Cartesian_getLocationY(cartesian, y);
    double tmpy = herey;
    for (x = 0; x < xsize; x++) {
      double herex = Cartesian_getLocationX(cartesian, x);
      herey = tmpy; // So that we can use herey over and over again
      RaveValueType valid = RaveValueType_NODATA;
      double v = 0.0L;
      if (!Projection_transform(sourcepj, targetpj, &herex, &herey, NULL)) {
        RAVE_ERROR0("Transform failed");
        goto done;
      }
      valid = PolarScan_getNearest(scan, herex, herey, &v);

      if (valid == RaveValueType_NODATA) {
        v = cnodata;
      } else if (valid == RaveValueType_UNDETECT) {
        v = cundetect;
      }
      Cartesian_setValue(cartesian, x, y, v);
    }
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(sourcepj);
  RAVE_OBJECT_RELEASE(targetpj);
  return result;
}

int Transform_cappi(Transform_t* transform, PolarVolume_t* pvol, Cartesian_t* cartesian, double height)
{
  return Transform_cappis_internal(transform, pvol, cartesian, height, 1);
}

int Transform_pcappi(Transform_t* transform, PolarVolume_t* pvol, Cartesian_t* cartesian, double height)
{
  return Transform_cappis_internal(transform, pvol, cartesian, height, 0);
}

int Transform_ctoscan(Transform_t* transform, Cartesian_t* cartesian, PolarScan_t* scan)
{
  int result = 0;
#ifdef KALLE
  Projection_t* sourcepj = NULL;
  Projection_t* targetpj = NULL;
  double snodata = 0.0, sundetect = 0.0;
  long nbins = 0, nrays = 0;
  long bin = 0, ray = 0;

  RAVE_ASSERT((transform != NULL), "transform == NULL");
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  if (!Cartesian_isTransformable(cartesian) ||
      !PolarScan_isTransformable(scan)) {
    RAVE_ERROR0("Cartesian product or scan is not possible to transform");
    goto done;
  }

  sourcepj = PolarScan_getProjection(scan);
  targetpj = Cartesian_getProjection(cartesian);
  //snodata = PolarScan_getN
  //sundetect = PolarScan_getUndetect(scan);
  nbins = PolarScan_getNbins(scan);
  nrays = PolarScan_getNrays(scan);

  result = 1;
done:
  RAVE_OBJECT_RELEASE(sourcepj);
  RAVE_OBJECT_RELEASE(targetpj);
#endif
  return result;
}

int Transform_ctop(Transform_t* transform, Cartesian_t* cartesian, PolarVolume_t* pvol)
{
  int result = 0;
#ifdef KALLE
  RAVE_ASSERT((transform != NULL), "transform == NULL");
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");

  if (!Cartesian_isTransformable(cartesian) ||
      !PolarVolume_isTransformable(pvol)) {
    RAVE_ERROR0("Cartesian product or volume is not possible to transform");
    goto done;
  }

  sourcepj = PolarVolume_getProjection(pvol);
  targetpj = Cartesian_getProjection(cartesian);
  pnodata = PolarVolume_getNodata(pvol);
  pundetect = PolarVolume_getUndetect(pvol);
  xsize = Cartesian_getXSize(cartesian);
  ysize = Cartesian_getYSize(cartesian);
#endif
  return 0;
}
/*@} End of Interface functions */

RaveCoreObjectType Transform_TYPE = {
    "Transform",
    sizeof(Transform_t),
    Transform_constructor,
    Transform_destructor
};
