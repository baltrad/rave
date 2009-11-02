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
  RaveTransformationMethod method;

  long ps_refCount;
};

/*@{ Private functions */
/**
 * Destroys the transformer
 * @param[in] transform - the transformer to destroy
 */
static void Transform_destroy(Transform_t* transform)
{
  if (transform != NULL) {
    RAVE_FREE(transform);
  }
}
/*@} End of Private functions */

/*@{ Interface functions */
Transform_t* Transform_new(void)
{
  Transform_t* result = NULL;
  result = RAVE_MALLOC(sizeof(Transform_t));
  if (result != NULL) {
    result->method = NEAREST;
    result->ps_refCount = 1;
  }
  return result;
}

void Transform_release(Transform_t* transform)
{
  RAVE_ASSERT((transform != NULL), "transform was NULL");
  transform->ps_refCount--;
  if (transform->ps_refCount <= 0) {
    Transform_destroy(transform);
  }
}

Transform_t* Transform_copy(Transform_t* transform)
{
  RAVE_ASSERT((transform != NULL), "transform was NULL");
  transform->ps_refCount++;
  return transform;
}

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

int Transform_cappi(Transform_t* transform, PolarVolume_t* pvol, Cartesian_t* cartesian, double height)
{
  int result = 0;
  long xsize = 0, ysize = 0, x = 0, y = 0;
  double cnodata = 0.0L, cundetect = 0.0L;
  Projection_t* sourcepj = NULL;
  Projection_t* targetpj = NULL;

  RAVE_ASSERT((transform != NULL), "transform was NULL");
  RAVE_ASSERT((pvol != NULL), "pvol was NULL");
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");

  sourcepj = Cartesian_getProjection(cartesian);
  if (sourcepj == NULL) {
    RAVE_ERROR0("No projection in cartesian product");
    goto done;
  }

  targetpj = PolarVolume_getProjection(pvol);
  if (targetpj == NULL) {
    RAVE_ERROR0("No projection in polar volume");
    goto done;
  }
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
      valid = PolarVolume_getNearest(pvol, herex, herey, height, &v);

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
  Projection_release(sourcepj);
  Projection_release(targetpj);
  return result;
}
/*@} End of Interface functions */
